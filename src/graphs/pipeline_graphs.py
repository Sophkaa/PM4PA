"""Pydantic Graph definitions and node implementations for pipeline settings 1–5."""

from pathlib import Path
from typing import Type, Dict, Any, Union, List
from pydantic import BaseModel, ValidationError
import logging
import os
import traceback
from pydantic_graph import BaseNode, Graph, GraphRunContext, End

from ..models.state import ProcessState
from ..models.artifacts import (
    ProcessDraft, 
    ValidationResultSetting4
)
from ..models.bpmn import BPMNModelJsonNested
from ..agents.retrieval import extract_query_structure_and_expand
from ..agents.bpmn import run_draft_agent_with_structure, run_bpmn_agent_with_structure, run_bpmn_agent_revision
from ..agents.validation import (
    run_validation_agent_setting4, 
    run_scope_completeness_validator_setting5,
    run_factual_fidelity_validator_setting5,
    run_process_logic_validator_setting5,
    aggregate_validation_results_setting5
)
from ..agents.retrieval_bpmn import run_retrieval_bpmn_agent, run_retrieval_bpmn_agent_with_structure
from ..infrastructure.retrieval.keyword_search import (
    bm25_search,
    bm25_search_all,
    normalize_bm25_scores,
)
from ..infrastructure.vector_store.chroma_store import ChromaVectorStore
from ..infrastructure.api.openwebui_client import OpenWebUIClient
from ..config import settings

logger = logging.getLogger(__name__)


def _graph_status(msg: str) -> None:
    print(f"[graph] {msg}", flush=True)


def _graph_debug(msg: str) -> None:
    if settings.debug:
        print(f"[graph|debug] {msg}", flush=True)


def _graph_debug_exc(context: str, exc: BaseException) -> None:
    if settings.debug:
        print(f"[graph|debug] {context}: {exc}", flush=True)
        traceback.print_exc()


def _graph_verbose_enabled() -> bool:
    """Enable expensive debug dumps only when explicitly requested."""
    return os.getenv("GRAPH_VERBOSE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


class ProcessDependencies(BaseModel):
    """Dependencies for process graph nodes."""
    model_config = {"arbitrary_types_allowed": True}
    
    vector_store: ChromaVectorStore
    api_client: OpenWebUIClient


class RetrievalAndBpmnNode(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Combined retrieval and BPMN generation (for Setting 1)."""
    
    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> End[BPMNModelJsonNested]:
        """Execute combined retrieval and BPMN generation in one step."""
        _graph_status("setting 1: retrieval + BPMN generation")

        state = ctx.state
        deps = ctx.deps
        
        try:
            # Run combined retrieval and BPMN generation agent
            # Use file_filter from state if available (e.g., for filtering uploaded documents)
            file_filter = state.file_filter
            bpmn_model, retrieved_docs, retrieved_metas, relevance_scores = await run_retrieval_bpmn_agent(
                user_request=state.user_request,
                vector_store=deps.vector_store,
                api_client=deps.api_client,
                file_filter=file_filter
            )
            
            # Update state with retrieved documents and BPMN model
            state.retrieved_documents = retrieved_docs
            state.retrieved_metadatas = retrieved_metas
            state.relevance_scores = relevance_scores
            state.bpmn = bpmn_model
            
            # Add chunk_nr to each metadata (1-based enumeration)
            for idx, meta in enumerate(state.retrieved_metadatas):
                meta['chunk_nr'] = idx + 1
            
            _graph_debug(
                f"retrieval+bpmn: {len(retrieved_docs)} doc(s), "
                f"{len(bpmn_model.pools)} pool(s)"
                + (f", process={bpmn_model.process_name!r}" if bpmn_model.process_name else "")
            )

        except Exception as e:
            # On complete failure, set empty results
            state.retrieved_documents = []
            state.retrieved_metadatas = []
            state.retrieved_ids = []
            state.relevance_scores = []
            state.bpmn = BPMNModelJsonNested(pools=[])
            _graph_debug_exc("retrieval+bpmn node failed", e)
        
        # End directly with BPMN model (no validation)
        if state.bpmn:
            return End(state.bpmn)
        else:
            empty_bpmn = BPMNModelJsonNested(pools=[])
            return End(empty_bpmn)


async def _execute_enhanced_retrieval_setting2(
    state: ProcessState,
    deps: ProcessDependencies,
    top_k: int = None,
    n_results: int = None
) -> None:
    """
    Hybrid retrieval: query structure + expansion, BM25 over the collection, multi-query
    vector search, combined scoring, and top-k selection.

    Used by graph settings 2–5. Writes ``query_structure``, ``expanded_queries``, and the
    selected chunk lists on ``state``.

    ``top_k`` / ``n_results`` default to the setting-2 config keys; settings 3–5 pass
    overrides from their nodes.
    """
    final_top_k = top_k if top_k is not None else settings.retrieval_top_k
    final_n_results = n_results if n_results is not None else settings.retrieval_n_results
    query_structure, expanded_queries = await extract_query_structure_and_expand(state.user_request)
    state.query_structure = query_structure
    state.expanded_queries = expanded_queries
    
    all_docs_result = deps.vector_store.collection.get(
        where=state.file_filter,
        include=["documents", "metadatas"]
    )
    
    all_documents = all_docs_result.get("documents", [])
    all_metadatas = all_docs_result.get("metadatas", [])
    all_ids = all_docs_result.get("ids", [])
    
    if not all_documents:
        logger.warning("Vector store returned no documents for retrieval (check filter / ingestion).")
        state.retrieved_documents = []
        state.retrieved_metadatas = []
        state.retrieved_ids = []
        state.relevance_scores = []
        return
    
    # Generate keywords from structured fields and expanded queries
    # Note: expanded_queries already contains original query + keyTerms + synonyms
    # We add structured fields (process_name, domain, procedure_type) and remove duplicates
    keyword_sources = []
    
    # Add structured fields first (these are important for BM25)
    keyword_sources.append(query_structure.process_name)
    if query_structure.domain and query_structure.domain != "NICHT ANGEGEBEN":
        keyword_sources.append(query_structure.domain)
    if query_structure.procedure_type and query_structure.procedure_type != "NICHT ANGEGEBEN":
        keyword_sources.append(query_structure.procedure_type)
    
    # Add expanded queries (original query + keyTerms + synonyms), avoiding duplicates
    seen = set(k.lower() for k in keyword_sources)  # Case-insensitive duplicate detection
    for query in expanded_queries:
        if query and query.strip() and query.lower() not in seen:
            keyword_sources.append(query.strip())
            seen.add(query.lower())
    
    if settings.debug and _graph_verbose_enabled():
        print("[graph|debug] —— BM25 keywords & retrieval weights ——")
        print(f"         keywords ({len(keyword_sources)}): {keyword_sources}")
        print(
            f"         file_filter={state.file_filter!r}  bm25_top_n={settings.bm25_top_n}  "
            f"vec_w={settings.vector_weight}  bm25_w={settings.bm25_weight}  "
            f"threshold={settings.retrieval_score_threshold}  top_k={final_top_k}"
        )
    
    # Perform BM25 search on ALL documents to get scores for all
    bm25_scores_raw_all = bm25_search_all(
        queries=keyword_sources,
        documents=all_documents
    )
    
    # Normalize BM25 scores for all documents
    bm25_scores_normalized_all = normalize_bm25_scores(bm25_scores_raw_all)
    
    # Create mapping: document index -> BM25 score (for ALL documents)
    bm25_score_map = {idx: score for idx, score in enumerate(bm25_scores_normalized_all)}
    
    # Also get top-N for reporting purposes
    bm25_results = bm25_search(
        queries=keyword_sources,
        documents=all_documents,
        top_k=settings.bm25_top_n
    )
    bm25_indices = [idx for idx, _ in bm25_results]

    if settings.debug and _graph_verbose_enabled():
        print(
            f"[graph|debug] BM25: {len(all_documents)} doc(s) scored, "
            f"top-{len(bm25_indices)} indices (reference)"
        )
    
    all_vector_docs = []
    all_vector_metas = []
    all_vector_scores = []
    all_vector_ids = []
    seen_hashes = set()
    
    for query in expanded_queries:
        try:
            embeddings = await deps.api_client.get_embeddings([query])
            results = deps.vector_store.query(
                query_embeddings=embeddings,
                n_results=final_n_results,
                where=state.file_filter,
                include=["documents", "metadatas", "distances"]
            )
            docs = results.get("documents", [[]])[0] if results.get("documents") else []
            metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            dists = results.get("distances", [[]])[0] if results.get("distances") else []
            ids = results.get("ids", [[]])[0] if results.get("ids") else []
            
            for doc, meta, dist, doc_id in zip(docs, metas, dists, ids):
                doc_hash = hash(doc[:100])
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    all_vector_docs.append(doc)
                    all_vector_metas.append(meta)
                    vector_score = 1 - float(dist) if dist is not None else 0.0
                    all_vector_scores.append(vector_score)
                    all_vector_ids.append(doc_id)
                        
        except Exception as e:
            _graph_debug(f"vector query failed ({query[:40]!r}…): {e}")
            continue

    if settings.debug and _graph_verbose_enabled():
        print(f"[graph|debug] vector search: {len(all_vector_docs)} unique hit(s)")
    
    combined_scores = []
    for i, doc_id in enumerate(all_vector_ids):
        # Find the document index in all_documents
        try:
            doc_idx = all_ids.index(doc_id)
            bm25_score = bm25_score_map.get(doc_idx, 0.0)
        except ValueError:
            bm25_score = 0.0
        
        vector_score = all_vector_scores[i]
        combined_score = (
            settings.vector_weight * vector_score +
            settings.bm25_weight * bm25_score
        )
        combined_scores.append(combined_score)
    
    indices = list(range(len(combined_scores)))
    filtered = [i for i in indices if combined_scores[i] >= settings.retrieval_score_threshold]
    if not filtered:
        # Fallback: take top K without filter
        filtered = sorted(indices, key=lambda i: combined_scores[i], reverse=True)[:final_top_k]
    else:
        filtered = sorted(filtered, key=lambda i: combined_scores[i], reverse=True)[:final_top_k]
    
    # Store final results in state
    state.retrieved_documents = [all_vector_docs[i] for i in filtered]
    state.retrieved_metadatas = [all_vector_metas[i] for i in filtered]
    state.retrieved_ids = [all_vector_ids[i] for i in filtered]
    state.relevance_scores = [combined_scores[i] for i in filtered]
    
    # Add chunk_nr to each metadata (1-based enumeration)
    for idx, meta in enumerate(state.retrieved_metadatas):
        meta['chunk_nr'] = idx + 1
    
    if settings.debug and _graph_verbose_enabled():
        print("[graph|debug] —— final selection (scores + chunk text) ——")
        print(f"         selected {len(state.retrieved_documents)} chunk(s) after re-rank")
        for i, idx in enumerate(filtered, 1):
            doc = all_vector_docs[idx]
            meta = all_vector_metas[idx]
            doc_id = all_vector_ids[idx]
            combined_score = combined_scores[idx]
            vector_score = all_vector_scores[idx]
            try:
                doc_idx = all_ids.index(doc_id)
                bm25_score = bm25_score_map.get(doc_idx, 0.0)
            except ValueError:
                bm25_score = 0.0
            doc_name = meta.get("file_name", "Unknown") if meta else "Unknown"
            file_path = meta.get("file_path", "Unknown") if meta else "Unknown"
            subfolder = meta.get("subfolder", None) if meta else None
            page_number = meta.get("page_number", None) if meta else None
            chunk_index = meta.get("chunk_index", None) if meta else None
            print(f"         [{i}] {doc_name}  path={file_path}")
            if subfolder:
                print(f"             subfolder={subfolder}")
            if page_number is not None or chunk_index is not None:
                print(f"             page={page_number}  chunk_index={chunk_index}")
            print(
                f"             vec={vector_score:.4f}  bm25={bm25_score:.4f}  "
                f"combined={combined_score:.4f}"
            )
            print(f"             --- chunk text ({len(doc)} chars) ---")
            print(doc)
            print("             --- end chunk ---")

    _graph_status(f"hybrid retrieval: {len(state.retrieved_documents)} chunk(s) selected")


class EnhancedRetrievalNodeSetting2(BaseNode[ProcessState, ProcessDependencies]):
    """Enhanced retrieval with structured query extraction, BM25 pre-filtering, and hybrid search (for Setting 2)."""
    
    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'RelevanceEvaluationNodeSetting2':
        """Execute enhanced retrieval with query extraction, BM25 pre-filtering, and hybrid search."""
        _graph_status("setting 2: enhanced retrieval (query + BM25 + hybrid)")

        state = ctx.state
        deps = ctx.deps

        try:
            # Execute shared retrieval logic
            await _execute_enhanced_retrieval_setting2(state, deps)
        except Exception as e:
            # On complete failure, continue with empty results
            state.retrieved_documents = []
            state.retrieved_metadatas = []
            state.retrieved_ids = []
            state.relevance_scores = []
            _graph_debug_exc("enhanced retrieval (setting 2) failed", e)
        
        return RelevanceEvaluationNodeSetting2()


class EnhancedRetrievalNodeSetting3(BaseNode[ProcessState, ProcessDependencies]):
    """Enhanced retrieval with structured query extraction, BM25 pre-filtering, and hybrid search (for Setting 3)."""
    
    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'RelevanceEvaluationNodeSetting3':
        """Execute enhanced retrieval with query extraction, BM25 pre-filtering, and hybrid search."""
        _graph_status("setting 3: enhanced retrieval (query + BM25 + hybrid)")

        state = ctx.state
        deps = ctx.deps

        try:
            # Execute retrieval logic with Setting 3 parameters (top_k=25, n_results=30)
            await _execute_enhanced_retrieval_setting2(
                state, 
                deps,
                top_k=settings.retrieval_top_k,
                n_results=settings.retrieval_n_results
            )
        except Exception as e:
            # On complete failure, continue with empty results
            state.retrieved_documents = []
            state.retrieved_metadatas = []
            state.retrieved_ids = []
            state.relevance_scores = []
            _graph_debug_exc("enhanced retrieval (setting 3) failed", e)
        
        return RelevanceEvaluationNodeSetting3()


class GenerateBpmnFromRetrievalNode(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Generate BPMN model from retrieved documents (for Setting 2)."""
    
    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> End[BPMNModelJsonNested]:
        """Execute BPMN generation from retrieved documents with query structure."""
        _graph_status("setting 2: BPMN from retrieved chunks")

        state = ctx.state
        deps = ctx.deps

        if not state.retrieved_documents:
            # If no documents retrieved, create empty model
            state.bpmn = BPMNModelJsonNested(pools=[])
            logger.warning("No retrieved chunks; emitting empty BPMN.")
            return End(state.bpmn)
        
        try:
            # Run BPMN generation agent with query structure
            bpmn_model = await run_retrieval_bpmn_agent_with_structure(
                user_request=state.user_request,
                retrieved_documents=state.retrieved_documents,
                retrieved_metadatas=state.retrieved_metadatas,
                query_structure=state.query_structure,
                api_client=deps.api_client
            )
            
            state.bpmn = bpmn_model
            
            _graph_debug(
                f"BPMN from retrieval: {len(bpmn_model.pools)} pool(s)"
                + (f", process={bpmn_model.process_name!r}" if bpmn_model.process_name else "")
            )

        except Exception as e:
            # Create empty BPMN model on error
            state.bpmn = BPMNModelJsonNested(pools=[])
            _graph_debug_exc("BPMN from retrieval failed", e)
        
        if state.bpmn:
            return End(state.bpmn)
        else:
            empty_bpmn = BPMNModelJsonNested(pools=[])
            return End(empty_bpmn)


async def _create_draft_helper(state: ProcessState) -> None:
    """Build ``state.draft`` (and ``draft_user_prompt``) from retrieval context."""
    try:
        # Use the dedicated agent function with query_structure
        draft_result = await run_draft_agent_with_structure(
            user_request=state.user_request,
            retrieved_documents=state.retrieved_documents,
            retrieved_metadatas=state.retrieved_metadatas,
            expanded_queries=state.expanded_queries,
            query_structure=state.query_structure
        )
        state.draft, state.draft_user_prompt = draft_result
        
        if settings.debug and _graph_verbose_enabled():
            td = state.draft.text_description
            preview = (td[:800] + "…") if len(td) > 800 else td
            print("[graph|debug] —— draft text ——")
            print(f"         length={len(td)} chars")
            print(preview)

    except Exception as e:
        # Create a minimal draft on error
        state.draft = ProcessDraft(text_description=state.user_request)
        _graph_debug_exc("draft creation failed (using raw user request as draft)", e)


def _group_and_sort_chunks_by_document(
    docs: List[str],
    metas: List[Dict[str, Any]],
    ids: List[str],
    scores: List[float]
) -> tuple[List[str], List[Dict[str, Any]], List[str], List[float]]:
    """Group chunks by source file, sort by ``chunk_index``, prepend a header row per document."""
    # Group chunks by document (use file_path as key, fallback to file_name)
    doc_groups: Dict[str, List[tuple]] = {}
    
    for i, (doc, meta, chunk_id, score) in enumerate(zip(docs, metas, ids, scores)):
        # Determine document key
        doc_key = None
        if meta:
            doc_key = meta.get('file_path') or meta.get('file_name')
        
        if not doc_key:
            # Fallback: use chunk_id prefix or index
            doc_key = f"unknown_doc_{i}"
        
        if doc_key not in doc_groups:
            doc_groups[doc_key] = []
        
        # Store chunk with metadata for sorting
        chunk_index = meta.get('chunk_index', i) if meta else i
        doc_groups[doc_key].append((chunk_index, doc, meta, chunk_id, score))
    
    # Sort groups by document key (reverse alphabetically: Z -> A)
    sorted_doc_keys = sorted(doc_groups.keys(), reverse=True)
    
    # Build output lists
    grouped_docs = []
    grouped_metas = []
    grouped_ids = []
    grouped_scores = []
    
    for doc_key in sorted_doc_keys:
        chunks = doc_groups[doc_key]
        
        # Sort chunks within document by chunk_index
        chunks.sort(key=lambda x: x[0])
        
        # Get document metadata from first chunk
        first_meta = chunks[0][2] if chunks else {}
        
        # Build document header
        doc_name = (
            first_meta.get('file_name') or
            first_meta.get('document_title') or
            first_meta.get('title') or
            (Path(doc_key).name if doc_key else 'Unbekanntes Dokument')
        )
        
        # Create header (only document name)
        header = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        header += f"📄 DOKUMENT: {doc_name}\n"
        header += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Add header as first "chunk"
        grouped_docs.append(header)
        grouped_metas.append({
            'is_document_header': True,
            'document_name': doc_name,
        })
        grouped_ids.append(f"header_{doc_key}")
        grouped_scores.append(0.0)
        
        # Add all chunks from this document
        for chunk_index, doc, meta, chunk_id, score in chunks:
            grouped_docs.append(doc)
            grouped_metas.append(meta)
            grouped_ids.append(chunk_id)
            grouped_scores.append(score)
    
    return grouped_docs, grouped_metas, grouped_ids, grouped_scores


async def _execute_relevance_evaluation(
    state: ProcessState,
    deps: ProcessDependencies,
    top_n: int = None
) -> None:
    """
    LLM relevance scoring on retrieved chunks, optional synonym re-retrieval when
    high+medium counts are low, then top-N selection and grouping by source document.

    Updates ``state.relevance_evaluation`` and replaces the retrieved chunk lists with
    the filtered, document-grouped view passed to later nodes.

    ``top_n`` defaults to ``settings.relevance_top_n``.
    """
    from ..agents.relevance import evaluate_retrieval
    from ..agents.retrieval import generate_additional_synonyms

    final_top_n = top_n if top_n is not None else settings.relevance_top_n
    min_high_medium = settings.relevance_min_high_medium
    max_retries = settings.relevance_max_retries if settings.relevance_retry_with_new_synonyms else 0
    
    retry_count = 0
    
    while retry_count <= max_retries:
        if not state.retrieved_documents:
            _graph_debug("relevance: no chunks to evaluate, skip")
            return

        try:
            result = await evaluate_retrieval(
                query=state.user_request,
                query_structure=state.query_structure,
                chunks=state.retrieved_documents,
                metadatas=state.retrieved_metadatas,
                ids=state.retrieved_ids if state.retrieved_ids else None,
                top_n=final_top_n,
            )

            state.relevance_evaluation = result

            original_metadatas = state.retrieved_metadatas.copy()

            high_count = sum(1 for a in result.chunk_assessments if a.relevance == "high")
            medium_count = sum(1 for a in result.chunk_assessments if a.relevance == "medium")
            low_count = sum(1 for a in result.chunk_assessments if a.relevance == "low")
            none_count = sum(1 for a in result.chunk_assessments if a.relevance == "none")
            high_medium_count = high_count + medium_count

            _graph_debug(
                f"relevance batch: evaluated={len(result.chunk_assessments)}  "
                f"H/M/L/none={high_count}/{medium_count}/{low_count}/{none_count}  "
                f"H+M={high_medium_count} (min {min_high_medium})"
            )

            # Check if we have enough high/medium chunks
            if high_medium_count >= min_high_medium:
                _graph_debug(f"relevance: enough H+M chunks ({high_medium_count} >= {min_high_medium})")
                break
            
            # Too few relevant chunks - check if we should retry
            if retry_count < max_retries:
                _graph_debug(
                    f"relevance: low H+M ({high_medium_count} < {min_high_medium}), "
                    f"re-retrieve synonyms (try {retry_count + 1}/{max_retries})"
                )
                
                try:
                    # Generate new synonyms
                    new_query_structure, new_expanded_queries = await generate_additional_synonyms(
                        state.user_request,
                        state.query_structure
                    )
                    
                    # Update state with new query structure
                    state.query_structure = new_query_structure
                    state.expanded_queries = new_expanded_queries
                    
                    # Use retrieval parameters matching the current setting
                    top_k = settings.retrieval_top_k
                    n_results = settings.retrieval_n_results
                    await _execute_enhanced_retrieval_setting2(
                        state=state,
                        deps=deps,
                        top_k=top_k,
                        n_results=n_results
                    )
                    
                    # Check if new documents were found
                    if not state.retrieved_documents:
                        _graph_debug("relevance re-retrieve: no new chunks, stop retry")
                        break
                    
                    _graph_debug(f"relevance re-retrieve: {len(state.retrieved_documents)} chunk(s), re-evaluate")
                    retry_count += 1
                    continue  # Start next iteration with new documents
                    
                except Exception as e:
                    logger.warning("Relevance re-retrieval failed: %s", e)
                    _graph_debug_exc("re-retrieval", e)
                    break  # Stop retrying on error
            else:
                # Max retries reached
                _graph_debug(f"relevance: max retries ({max_retries}), using available chunks")
                break

        except Exception as e:
            logger.warning("Relevance evaluation failed: %s", e)
            _graph_debug_exc("relevance evaluation", e)
            return

    high_medium_assessments = [
        a for a in state.relevance_evaluation.chunk_assessments
        if a.relevance in ["high", "medium"]
    ]

    def sort_key(assessment):
        relevance_priority = {"high": 0, "medium": 1}
        return (relevance_priority.get(assessment.relevance, 999), -assessment.confidence)

    high_medium_assessments.sort(key=sort_key)
    selected = high_medium_assessments[:final_top_n]

    if not selected:
        logger.warning(
            "No high/medium chunks after relevance; keeping %s chunk(s) as fallback.",
            len(state.retrieved_documents),
        )
        if len(state.retrieved_documents) < settings.relevance_min_high_medium:
            logger.warning(
                "Chunk count %s below relevance_min_high_medium (%s).",
                len(state.retrieved_documents),
                settings.relevance_min_high_medium,
            )
        return

    if settings.debug and _graph_verbose_enabled():
        print("[graph|debug] —— chunk_nr mapping (first 10) ——")
        print(f"         chunks={len(state.retrieved_documents)}  assessments={len(selected)}")
        for i, assessment in enumerate(selected[:10]):
            idx = assessment.chunk_nr - 1
            valid = 0 <= idx < len(state.retrieved_documents)
            print(f"         [{i}] chunk_nr {assessment.chunk_nr} -> idx {idx}  {'OK' if valid else 'INVALID'}")
        if len(selected) > 10:
            print(f"         … +{len(selected) - 10} more")

    new_docs = []
    new_metas = []
    new_ids = []
    new_scores = []

    for assessment in selected:
        idx = assessment.chunk_nr - 1
        
        if not (0 <= idx < len(state.retrieved_documents)):
            _graph_debug(
                f"chunk_nr {assessment.chunk_nr} out of range (valid 1–{len(state.retrieved_documents)})"
            )
            continue
        if idx < len(state.retrieved_documents):
            new_docs.append(state.retrieved_documents[idx])
            new_metas.append(state.retrieved_metadatas[idx] if idx < len(state.retrieved_metadatas) else {})
            new_ids.append(state.retrieved_ids[idx] if idx < len(state.retrieved_ids) else f"chunk_{idx}")
            new_scores.append(state.relevance_scores[idx] if idx < len(state.relevance_scores) else 0.0)

    grouped_docs, grouped_metas, grouped_ids, grouped_scores = _group_and_sort_chunks_by_document(
        new_docs, new_metas, new_ids, new_scores
    )
    
    state.retrieved_documents = grouped_docs
    state.retrieved_metadatas = grouped_metas
    state.retrieved_ids = grouped_ids
    state.relevance_scores = grouped_scores

    actual_chunk_count = sum(1 for meta in grouped_metas if not meta.get('is_document_header', False))
    n_docs = len([m for m in grouped_metas if m.get('is_document_header', False)])
    _graph_status(f"relevance filter: {actual_chunk_count} chunk(s) from {n_docs} document(s) → next step")

    if len(new_docs) < settings.relevance_min_high_medium:
        logger.warning(
            "Only %s high/medium chunk(s) selected (min recommended %s).",
            len(new_docs),
            settings.relevance_min_high_medium,
        )

    if settings.debug and _graph_verbose_enabled():
        print("[graph|debug] —— relevance summary ——")
        print(f"         decision={state.relevance_evaluation.decision}")
        print(
            f"         evaluated={len(state.relevance_evaluation.chunk_assessments)}  "
            f"passed_groups={len(state.retrieved_documents)} (from {len(original_metadatas)} raw)"
        )
        high_count = sum(1 for a in state.relevance_evaluation.chunk_assessments if a.relevance == "high")
        medium_count = sum(1 for a in state.relevance_evaluation.chunk_assessments if a.relevance == "medium")
        low_count = sum(1 for a in state.relevance_evaluation.chunk_assessments if a.relevance == "low")
        none_count = sum(1 for a in state.relevance_evaluation.chunk_assessments if a.relevance == "none")
        print(f"         H/M/L/none={high_count}/{medium_count}/{low_count}/{none_count}")
        
        sorted_assessments = sorted(state.relevance_evaluation.chunk_assessments, key=lambda a: a.chunk_nr)
        
        print("         per-chunk assessments:")
        for i, assessment in enumerate(sorted_assessments):
            original_idx = assessment.chunk_nr - 1
            if 0 <= original_idx < len(original_metadatas):
                meta = original_metadatas[original_idx]
                file_name = meta.get('file_name', 'Unknown')
                chunk_index = meta.get('chunk_index', '?')
            else:
                file_name = 'Unknown'
                chunk_index = '?'
            
            print(
                f"         [{i+1}] chunk={assessment.chunk_nr} idx={original_idx}  "
                f"{file_name} #{chunk_index}  {assessment.relevance}  conf={assessment.confidence:.2f}"
            )
            if assessment.evidence_spans:
                print(f"             evidence ({len(assessment.evidence_spans)}):")
                for j, span in enumerate(assessment.evidence_spans):
                    print(f"               [{j+1}] {span.text}")
            else:
                print("             evidence: (none)")
            if assessment.why_not_relevant:
                print(f"             why_not_relevant: {assessment.why_not_relevant}")


class RelevanceEvaluationNodeSetting2(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Evaluate relevance of retrieved chunks and filter to top-N (Setting 2)."""

    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'GenerateBpmnFromRetrievalNode':
        _graph_status("setting 2: relevance filtering")
        state = ctx.state
        deps = ctx.deps
        await _execute_relevance_evaluation(state, deps)
        return GenerateBpmnFromRetrievalNode()


class RelevanceEvaluationNodeSetting3(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Evaluate relevance of retrieved chunks and filter to top-N (Setting 3)."""

    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'DraftProcessNodeSetting3':
        _graph_status("setting 3: relevance filtering")
        state = ctx.state
        deps = ctx.deps
        await _execute_relevance_evaluation(state, deps)
        return DraftProcessNodeSetting3()


class DraftProcessNodeSetting3(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Create a ProcessDraft from user request with query_structure (for Setting 3)."""
    
    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'GenerateBpmnNodeSetting3':
        """Execute draft creation step with explicit query_structure orientation."""
        _graph_status("setting 3: process draft")

        state = ctx.state
        
        # Use helper function
        await _create_draft_helper(state)
        
        return GenerateBpmnNodeSetting3()


async def _generate_bpmn_from_draft_helper(
    state: ProcessState,
    save_original: bool = False
) -> BPMNModelJsonNested:
    """Generate nested BPMN JSON from ``state.draft`` and ``state.query_structure``.

    When ``save_original`` is true, also assigns ``state.bpmn_original``.
    """
    if not state.draft:
        # If no draft, create one from user request
        state.draft = ProcessDraft(text_description=state.user_request)
    
    try:
        # Use the dedicated agent function with query_structure
        bpmn_model = await run_bpmn_agent_with_structure(
            draft_text=state.draft.text_description,
            query_structure=state.query_structure
        )
        
        state.bpmn = bpmn_model
        
        # Save original BPMN if requested
        if save_original:
            state.bpmn_original = bpmn_model
        
        if settings.debug:
            pn = bpmn_model.process_name or "unnamed"
            _graph_debug(f"BPMN from draft: {len(bpmn_model.pools)} pool(s), process={pn!r}")

        return bpmn_model
        
    except Exception as e:
        # Create empty BPMN model on error
        state.bpmn = BPMNModelJsonNested(pools=[])
        if save_original:
            state.bpmn_original = BPMNModelJsonNested(pools=[])
        _graph_debug_exc("BPMN generation from draft failed", e)
        return BPMNModelJsonNested(pools=[])


class GenerateBpmnNodeSetting3(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Generate BPMN model from draft with query_structure (for Setting 3)."""
    
    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> End[BPMNModelJsonNested]:
        """Execute BPMN generation step with explicit query_structure orientation."""
        _graph_status("setting 3: BPMN from draft")

        state = ctx.state
        
        # Use helper function (save_original=False for Setting 3)
        bpmn_model = await _generate_bpmn_from_draft_helper(state, save_original=False)
        
        # For Setting 3, end here
        if state.bpmn:
            return End(state.bpmn)
        else:
            empty_bpmn = BPMNModelJsonNested(pools=[])
            return End(empty_bpmn)


class ValidateBpmnNodeSetting4(BaseNode[ProcessState, ProcessDependencies]):
    """Setting 4: validate nested BPMN JSON against chunks; may loop to revision."""

    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> Union['GenerateBpmnRevisionNodeSetting4', End[BPMNModelJsonNested]]:
        """Execute validation step. Loop to revision if iteration_recommended and under max iterations."""
        state = ctx.state
        
        # Safety check: prevent infinite revision loops (check on every validation)
        if state.revision_iteration_count >= settings.max_revision_iterations:
            logger.warning(
                "Max revision iterations (%s) reached; stopping revision loop.",
                settings.max_revision_iterations,
            )
            _graph_status(f"setting 4: max revisions ({settings.max_revision_iterations}) reached, stop")
            return End(state.bpmn if state.bpmn else BPMNModelJsonNested(pools=[]))
        
        validation_number = state.revision_iteration_count + 1
        _graph_status(f"setting 4: BPMN validation (#{validation_number})")
        
        if not state.bpmn:
            # Create empty validation result
            validation_result = ValidationResultSetting4(
                missing_elements=[],
                hallucinated_elements=[],
                structural_issues=[],
                overall_assessment={"iteration_recommended": False},
                assessment_statement="No BPMN model to validate"
            )
            state.validation_setting4 = validation_result
            return End(state.bpmn if state.bpmn else BPMNModelJsonNested(pools=[]))
        
        try:
            import json
            nested_json = json.dumps(
                state.bpmn.model_dump(mode='json', exclude_none=True),
                indent=2,
                ensure_ascii=False
            )

            validation_result = await run_validation_agent_setting4(
                bpmn_json=nested_json,
                user_request=state.user_request,
                query_structure=state.query_structure,
                retrieved_documents=state.retrieved_documents or [],
                retrieved_metadatas=state.retrieved_metadatas or [],
            )
            
            # Store validation result
            state.validation_setting4 = validation_result
            
            iteration_recommended = validation_result.overall_assessment.get("iteration_recommended", False)
            _graph_status(
                f"setting 4: validation #{validation_number} — "
                f"missing={len(validation_result.missing_elements)}  "
                f"hallucinated={len(validation_result.hallucinated_elements)}  "
                f"revise={iteration_recommended}"
            )
            if settings.debug and _graph_verbose_enabled():
                print("[graph|debug] —— full validation JSON ——")
                print(validation_result.model_dump_json(indent=2))
            
            # Decide on revision (check on every validation to support multiple iterations)
            if iteration_recommended:
                # Increment revision counter before proceeding
                state.revision_iteration_count += 1
                _graph_status(
                    f"setting 4: revision round {state.revision_iteration_count}/"
                    f"{settings.max_revision_iterations}"
                )
                return GenerateBpmnRevisionNodeSetting4()
            else:
                _graph_status("setting 4: validation ok, finish")
                return End(state.bpmn)
            
        except Exception as e:
            # Create error validation result
            validation_result = ValidationResultSetting4(
                missing_elements=[],
                hallucinated_elements=[],
                structural_issues=[],
                overall_assessment={"iteration_recommended": False},
                assessment_statement=f"Validation error: {str(e)}"
            )
            state.validation_setting4 = validation_result
            _graph_debug_exc("validation (setting 4) failed", e)
            
            # On error, end workflow
            return End(state.bpmn if state.bpmn else BPMNModelJsonNested(pools=[]))


class GenerateBpmnRevisionNodeSetting4(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Generate revised BPMN model based on validation feedback (Setting 4).
    Always revises from current state.bpmn (iterative improvement).
    """

    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'ValidateBpmnNodeSetting4':
        """Execute BPMN revision step based on validation feedback."""
        _graph_status("setting 4: BPMN revision")

        state = ctx.state
        
        if not state.validation_setting4:
            # No validation feedback available, skip revision
            _graph_debug("revision skipped: no validation feedback")
            return ValidateBpmnNodeSetting4()
        
        if not state.bpmn:
            _graph_debug("revision skipped: no BPMN model")
            return ValidateBpmnNodeSetting4()
        
        try:
            # Always use current BPMN state for revision (iterative improvement)
            bpmn_dict = state.bpmn.model_dump()
            import json
            current_bpmn_json = json.dumps(bpmn_dict, indent=2, ensure_ascii=False)
            
            # Run revision agent (use draft as source for missing elements)
            draft_text = state.draft.text_description if state.draft else None
            revised_bpmn = await run_bpmn_agent_revision(
                original_bpmn_json=current_bpmn_json,
                validation_feedback=state.validation_setting4,
                query_structure=state.query_structure,
                draft_text=draft_text,
                retrieved_documents=state.retrieved_documents if not draft_text else None,
                retrieved_metadatas=state.retrieved_metadatas if not draft_text else None,
            )
            
            state.bpmn = revised_bpmn
            
            if settings.debug and _graph_verbose_enabled():
                pn = revised_bpmn.process_name or "unnamed"
                _graph_debug(f"revised BPMN: {len(revised_bpmn.pools)} pool(s), process={pn!r}")
                print("[graph|debug] —— revised BPMN JSON ——")
                print(revised_bpmn.model_dump_json(indent=2))
            
        except Exception as e:
            # On error, keep current BPMN (state.bpmn unchanged)
            _graph_debug_exc("BPMN revision (setting 4) failed", e)
        
        # Continue to validation (may loop again if further revision needed)
        return ValidateBpmnNodeSetting4()


# ============================================================================
# Setting 5: Three Specialized Validators
# ============================================================================

class EnhancedRetrievalNodeSetting5(BaseNode[ProcessState, ProcessDependencies]):
    """Enhanced retrieval with structured query extraction, BM25 pre-filtering, and hybrid search (for Setting 5)."""
    
    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'RelevanceEvaluationNodeSetting5':
        """Execute enhanced retrieval with query extraction, BM25 pre-filtering, and hybrid search."""
        _graph_status("setting 5: enhanced retrieval (query + BM25 + hybrid)")

        state = ctx.state
        deps = ctx.deps

        try:
            # Execute retrieval logic with Setting 5 parameters
            await _execute_enhanced_retrieval_setting2(
                state, 
                deps,
                top_k=settings.retrieval_top_k,
                n_results=settings.retrieval_n_results
            )
        except Exception as e:
            # On complete failure, continue with empty results
            state.retrieved_documents = []
            state.retrieved_metadatas = []
            state.retrieved_ids = []
            state.relevance_scores = []
            _graph_debug_exc("enhanced retrieval (setting 5) failed", e)
        
        return RelevanceEvaluationNodeSetting5()


class RelevanceEvaluationNodeSetting5(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Evaluate relevance of retrieved chunks and filter to top-N (Setting 5)."""

    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'DraftProcessNodeSetting5':
        _graph_status("setting 5: relevance filtering")
        state = ctx.state
        deps = ctx.deps
        await _execute_relevance_evaluation(state, deps)
        return DraftProcessNodeSetting5()


class DraftProcessNodeSetting5(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Create a ProcessDraft from user request with query_structure (for Setting 5)."""
    
    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'GenerateBpmnNodeSetting5':
        """Execute draft creation step with explicit query_structure orientation."""
        _graph_status("setting 5: process draft")

        state = ctx.state
        
        # Use helper function
        await _create_draft_helper(state)
        
        return GenerateBpmnNodeSetting5()


class GenerateBpmnNodeSetting5(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Generate BPMN model from draft, then validate with three specialized validators (Setting 5)."""
    
    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'ValidateBpmnNodeSetting5':
        """Execute BPMN generation step, then proceed to BPMN validation."""
        _graph_status("setting 5: BPMN from draft")

        state = ctx.state
        
        await _generate_bpmn_from_draft_helper(state, save_original=False)
        
        return ValidateBpmnNodeSetting5()


class ValidateBpmnNodeSetting5(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Validate BPMN model for Setting 5 using three specialized validators (against chunks)."""

    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> Union['GenerateBpmnRevisionNodeSetting5', End[BPMNModelJsonNested]]:
        """Execute BPMN validation with three validators. Loop to revision if iteration_recommended."""
        state = ctx.state

        if state.revision_iteration_count >= settings.max_revision_iterations:
            logger.warning(
                "Max revision iterations (%s) reached; stopping.",
                settings.max_revision_iterations,
            )
            _graph_status(f"setting 5: max revisions ({settings.max_revision_iterations}) reached, stop")
            return End(state.bpmn if state.bpmn else BPMNModelJsonNested(pools=[]))

        validation_number = state.revision_iteration_count + 1
        _graph_status(f"setting 5: triple validation (#{validation_number})")

        if not state.bpmn:
            state.validation_setting4 = ValidationResultSetting4(
                missing_elements=[],
                hallucinated_elements=[],
                structural_issues=[],
                overall_assessment={"iteration_recommended": False},
                assessment_statement="No BPMN model to validate"
            )
            return End(state.bpmn if state.bpmn else BPMNModelJsonNested(pools=[]))

        try:
            import json
            nested_json = json.dumps(
                state.bpmn.model_dump(mode='json', exclude_none=True),
                indent=2,
                ensure_ascii=False
            )

            _graph_status("setting 5: validator 1/3 (scope & completeness)")
            validator1_result = await run_scope_completeness_validator_setting5(
                bpmn_json=nested_json,
                retrieved_documents=state.retrieved_documents or [],
                user_request=state.user_request,
                query_structure=state.query_structure,
                retrieved_metadatas=state.retrieved_metadatas
            )

            _graph_status("setting 5: validator 2/3 (factual fidelity)")
            validator2_result = await run_factual_fidelity_validator_setting5(
                bpmn_json=nested_json,
                retrieved_documents=state.retrieved_documents or [],
                user_request=state.user_request,
                query_structure=state.query_structure,
                retrieved_metadatas=state.retrieved_metadatas
            )

            _graph_status("setting 5: validator 3/3 (process logic)")
            validator3_result = await run_process_logic_validator_setting5(
                bpmn_json=nested_json,
                retrieved_documents=state.retrieved_documents or [],
                user_request=state.user_request,
                query_structure=state.query_structure,
                retrieved_metadatas=state.retrieved_metadatas
            )

            if settings.debug and _graph_verbose_enabled():
                _graph_debug(
                    f"validator 1: missing={len(validator1_result.missing_elements)}  "
                    f"revise={validator1_result.overall_assessment.get('iteration_recommended', False)}"
                )
                if validator1_result.missing_elements:
                    print("[graph|debug] —— validator 1 full JSON ——")
                    print(validator1_result.model_dump_json(indent=2))
                _graph_debug(
                    f"validator 2: hallucinated={len(validator2_result.hallucinated_elements)}  "
                    f"revise={validator2_result.overall_assessment.get('iteration_recommended', False)}"
                )
                if validator2_result.hallucinated_elements:
                    print("[graph|debug] —— validator 2 full JSON ——")
                    print(validator2_result.model_dump_json(indent=2))
                _graph_debug(
                    f"validator 3: structural={len(validator3_result.structural_issues)}  "
                    f"revise={validator3_result.overall_assessment.get('iteration_recommended', False)}"
                )
                if validator3_result.structural_issues:
                    print("[graph|debug] —— validator 3 full JSON ——")
                    print(validator3_result.model_dump_json(indent=2))

            validation_result = aggregate_validation_results_setting5([
                validator1_result,
                validator2_result,
                validator3_result
            ])
            state.validation_setting4 = validation_result

            iteration_recommended = validation_result.overall_assessment.get("iteration_recommended", False)
            _graph_status(
                f"setting 5: validation #{validation_number} — "
                f"missing={len(validation_result.missing_elements)}  "
                f"hallucinated={len(validation_result.hallucinated_elements)}  "
                f"structural={len(validation_result.structural_issues)}  "
                f"revise={iteration_recommended}"
            )
            if settings.debug and _graph_verbose_enabled():
                print("[graph|debug] —— aggregated validation JSON ——")
                print(validation_result.model_dump_json(indent=2))

            if iteration_recommended:
                state.revision_iteration_count += 1
                _graph_status(
                    f"setting 5: revision round {state.revision_iteration_count}/"
                    f"{settings.max_revision_iterations}"
                )
                return GenerateBpmnRevisionNodeSetting5()
            else:
                _graph_status("setting 5: validation ok, finish")
                return End(state.bpmn)

        except Exception as e:
            state.validation_setting4 = ValidationResultSetting4(
                missing_elements=[],
                hallucinated_elements=[],
                structural_issues=[],
                overall_assessment={"iteration_recommended": False},
                assessment_statement=f"Validation error: {str(e)}"
            )
            _graph_debug_exc("validation (setting 5) failed", e)
            return End(state.bpmn if state.bpmn else BPMNModelJsonNested(pools=[]))


class GenerateBpmnRevisionNodeSetting5(BaseNode[ProcessState, ProcessDependencies]):
    """Node: Revise BPMN model based on validation feedback (Setting 5)."""

    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'ValidateBpmnNodeSetting5':
        """Execute BPMN revision step based on validation feedback."""
        _graph_status("setting 5: BPMN revision")

        state = ctx.state

        if not state.validation_setting4:
            _graph_debug("revision skipped: no validation feedback")
            return ValidateBpmnNodeSetting5()

        if not state.bpmn:
            _graph_debug("revision skipped: no BPMN model")
            return ValidateBpmnNodeSetting5()

        try:
            import json
            current_bpmn_json = json.dumps(state.bpmn.model_dump(), indent=2, ensure_ascii=False)
            draft_text = state.draft.text_description if state.draft else None
            revised_bpmn = await run_bpmn_agent_revision(
                original_bpmn_json=current_bpmn_json,
                validation_feedback=state.validation_setting4,
                query_structure=state.query_structure,
                draft_text=draft_text,
                retrieved_documents=state.retrieved_documents if not draft_text else None,
                retrieved_metadatas=state.retrieved_metadatas if not draft_text else None,
            )
            state.bpmn = revised_bpmn
            if settings.debug and _graph_verbose_enabled():
                print("[graph|debug] —— revised BPMN JSON (setting 5) ——")
                print(revised_bpmn.model_dump_json(indent=2))
        except Exception as e:
            _graph_debug_exc("BPMN revision (setting 5) failed", e)

        return ValidateBpmnNodeSetting5()


# ============================================================================
# Setting 4: Validation after nested BPMN JSON generation
# ============================================================================

class EnhancedRetrievalNodeSetting4(BaseNode[ProcessState, ProcessDependencies]):
    """Enhanced retrieval for Setting 4."""

    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'RelevanceEvaluationNodeSetting4':
        _graph_status("setting 4: enhanced retrieval (query + BM25 + hybrid)")
        state = ctx.state
        deps = ctx.deps
        try:
            await _execute_enhanced_retrieval_setting2(
                state,
                deps,
                top_k=settings.retrieval_top_k,
                n_results=settings.retrieval_n_results
            )
        except Exception as e:
            state.retrieved_documents = []
            state.retrieved_metadatas = []
            state.retrieved_ids = []
            state.relevance_scores = []
            _graph_debug_exc("enhanced retrieval (setting 4) failed", e)
        return RelevanceEvaluationNodeSetting4()


class RelevanceEvaluationNodeSetting4(BaseNode[ProcessState, ProcessDependencies]):
    """Relevance evaluation for Setting 4."""

    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'DraftProcessNodeSetting4':
        _graph_status("setting 4: relevance filtering")
        state = ctx.state
        deps = ctx.deps
        await _execute_relevance_evaluation(state, deps)
        return DraftProcessNodeSetting4()


class DraftProcessNodeSetting4(BaseNode[ProcessState, ProcessDependencies]):
    """Create ProcessDraft, then go directly to BPMN generation (no draft validation)."""

    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'GenerateBpmnNodeSetting4':
        _graph_status("setting 4: process draft")
        state = ctx.state
        await _create_draft_helper(state)
        return GenerateBpmnNodeSetting4()


class GenerateBpmnNodeSetting4(BaseNode[ProcessState, ProcessDependencies]):
    """Generate nested BPMN JSON from draft, then validate (validation after BPMN generation)."""

    async def run(
        self,
        ctx: GraphRunContext[ProcessState, ProcessDependencies]
    ) -> 'ValidateBpmnNodeSetting4':
        """Execute BPMN generation, then proceed to BPMN validation."""
        _graph_status("setting 4: BPMN from draft")
        state = ctx.state
        await _generate_bpmn_from_draft_helper(state, save_original=False)
        return ValidateBpmnNodeSetting4()


# ============================================================================
# Graph Definitions
# ============================================================================

process_graph_setting_1 = Graph(
    nodes=[
        RetrievalAndBpmnNode,
    ],
    state_type=ProcessState,
)

process_graph_setting_2 = Graph(
    nodes=[
        EnhancedRetrievalNodeSetting2,
        RelevanceEvaluationNodeSetting2,
        GenerateBpmnFromRetrievalNode,
    ],
    state_type=ProcessState,
)

process_graph_setting_3 = Graph(
    nodes=[
        EnhancedRetrievalNodeSetting3,
        RelevanceEvaluationNodeSetting3,
        DraftProcessNodeSetting3,
        GenerateBpmnNodeSetting3,
    ],
    state_type=ProcessState,
)

process_graph_setting_5 = Graph(
    nodes=[
        EnhancedRetrievalNodeSetting5,
        RelevanceEvaluationNodeSetting5,
        DraftProcessNodeSetting5,
        GenerateBpmnNodeSetting5,
        ValidateBpmnNodeSetting5,
        GenerateBpmnRevisionNodeSetting5,
    ],
    state_type=ProcessState,
)

process_graph_setting_4 = Graph(
    nodes=[
        EnhancedRetrievalNodeSetting4,
        RelevanceEvaluationNodeSetting4,
        DraftProcessNodeSetting4,
        GenerateBpmnNodeSetting4,
        ValidateBpmnNodeSetting4,
        GenerateBpmnRevisionNodeSetting4,
    ],
    state_type=ProcessState,
)

# ============================================================================
# Helper Functions
# ============================================================================

def get_graph_for_setting(setting_name: str) -> tuple[Graph, Type[BaseNode[ProcessState]]]:
    """Return the ``Graph`` and its start node class for ``setting_1`` through ``setting_5``."""
    if setting_name == "setting_1":
        return process_graph_setting_1, RetrievalAndBpmnNode
    elif setting_name == "setting_2":
        return process_graph_setting_2, EnhancedRetrievalNodeSetting2
    elif setting_name == "setting_3":
        return process_graph_setting_3, EnhancedRetrievalNodeSetting3
    elif setting_name == "setting_4":
        return process_graph_setting_4, EnhancedRetrievalNodeSetting4
    elif setting_name == "setting_5":
        return process_graph_setting_5, EnhancedRetrievalNodeSetting5
    else:
        raise ValueError(f"Unknown setting name: {setting_name}")

