"""Combined retrieval and BPMN generation agent for Setting 1."""

from __future__ import annotations

import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from pydantic import ValidationError

from ..config import settings
from ..models.bpmn import BPMNModelJsonNested
from ..models.query_structure import QueryStructure
from ..infrastructure.api.ollama_client import call_ollama_json
from ..infrastructure.vector_store.chroma_store import ChromaVectorStore
from ..infrastructure.api.openwebui_client import OpenWebUIClient

logger = logging.getLogger(__name__)


def _retrieval_status(message: str) -> None:
    """Print compact status lines for normal execution."""
    print(f"[retrieval] {message}", flush=True)


def _retrieval_debug(message: str) -> None:
    """Print debug lines only when debug mode is enabled."""
    if settings.debug:
        print(f"[retrieval|debug] {message}", flush=True)


def _retrieval_debug_exc(context: str, exc: BaseException) -> None:
    """Print traceback in debug mode to aid local troubleshooting."""
    if settings.debug:
        print(f"[retrieval|debug] {context}: {exc}", flush=True)
        traceback.print_exc()


def _get_document_name(metadata: Dict[str, Any], index: int) -> str:
    """Extract a display-friendly document name from metadata."""
    if metadata:
        name = (
            metadata.get("file_name")
            or metadata.get("document_title")
            or metadata.get("title")
            or (Path(metadata.get("file_path", "")).name if metadata.get("file_path") else None)
        )
        if name:
            if metadata.get("page_number") is not None:
                return f"{name} (Seite {metadata.get('page_number')})"
            return name
    return f"Dokument {index + 1}"


def _format_chunk_source(metadata: Dict[str, Any], index: int) -> str:
    """Format chunk source with optional chapter/heading information."""
    doc_name = _get_document_name(metadata, index)
    source_parts = [doc_name]

    if metadata and metadata.get("chapter"):
        source_parts.append(f"Kapitel: {metadata.get('chapter')}")
    if metadata and metadata.get("heading"):
        source_parts.append(f"Überschrift: {metadata.get('heading')}")

    return ", ".join(source_parts)


def _iter_documents_with_metadata(
    documents: List[str], metadatas: Optional[List[Dict[str, Any]]]
) -> List[Tuple[int, str, Dict[str, Any]]]:
    """
    Return stable (index, doc, metadata) tuples without truncation.

    Missing metadata entries are replaced with empty dicts.
    """
    if not documents:
        return []
    resolved_metadatas = metadatas or []
    return [
        (i, doc, resolved_metadatas[i] if i < len(resolved_metadatas) else {})
        for i, doc in enumerate(documents)
    ]


async def run_retrieval_bpmn_agent(
    user_request: str,
    vector_store: ChromaVectorStore,
    api_client: OpenWebUIClient,
    file_filter: Optional[Dict[str, Any]] = None
) -> Tuple[BPMNModelJsonNested, List[str], List[Dict[str, Any]], List[float]]:
    """
    Combined retrieval and BPMN generation agent.
    
    Performs basic retrieval (without query expansion) and generates
    BPMN JSON directly from the retrieved documents.
    
    Args:
        user_request: User query specifying the process
        vector_store: ChromaDB vector store instance
        api_client: OpenWebUI API client instance
        file_filter: Optional metadata filter for retrieval
        
    Returns:
        Tuple of (BPMNModelJsonNested, retrieved_documents, retrieved_metadatas, relevance_scores)
    """
    # Step 1: Basic Retrieval (no query expansion)
    try:
        # Direct vector search with user_request (no query expansion)
        embeddings = await api_client.get_embeddings([user_request])
        query_results = vector_store.query(
            query_embeddings=embeddings,
            n_results=settings.retrieval_n_results,
            where=file_filter
        )
        
        docs = query_results.get("documents", [[]])[0] if query_results.get("documents") else []
        metas = query_results.get("metadatas", [[]])[0] if query_results.get("metadatas") else []
        distances = query_results.get("distances", [[]])[0] if query_results.get("distances") else []
        scores = [1 - float(d) if d is not None else 0.0 for d in distances]

        # Filter by score threshold and select top K
        indexed = list(range(len(docs)))
        filtered = [i for i in indexed if scores[i] >= settings.retrieval_score_threshold]
        if not filtered:
            # Fallback: none meet threshold → take top K without filter
            filtered = sorted(indexed, key=lambda i: scores[i], reverse=True)[:settings.retrieval_top_k_setting_1]
        else:
            filtered = sorted(filtered, key=lambda i: scores[i], reverse=True)[:settings.retrieval_top_k_setting_1]

        retrieved_documents = [docs[i] for i in filtered]
        retrieved_metadatas = [metas[i] for i in filtered]
        relevance_scores = [scores[i] for i in filtered]
        
        _retrieval_status(f"retrieved {len(retrieved_documents)} document(s)")
        if settings.debug:
            for i, (doc, score) in enumerate(zip(retrieved_documents, relevance_scores), 1):
                _retrieval_debug(f"doc#{i} score={score:.3f} preview={doc[:200]!r}")
                
    except Exception as e:
        # On retrieval failure, continue with empty results
        retrieved_documents = []
        retrieved_metadatas = []
        relevance_scores = []
        logger.warning("RetrievalBpmnAgent retrieval failed: %s", e)
        _retrieval_debug_exc("retrieval failed", e)
    
    # Step 2: Build prompt with retrieved documents and generate BPMN
    try:
        # Build context from retrieved documents grouped by document with headers
        context_parts = []
        if retrieved_documents:
            grouped_by_document: Dict[str, List[Tuple[int, str, Dict[str, Any], int]]] = {}

            for i, doc, meta in _iter_documents_with_metadata(retrieved_documents, retrieved_metadatas):
                document_name = _get_document_name(meta, i)
                document_key = (meta.get('file_path') if meta else None) or (meta.get('file_name') if meta else None) or document_name
                if document_key not in grouped_by_document:
                    grouped_by_document[document_key] = []

                chunk_index = meta.get('chunk_index', i) if meta else i
                grouped_by_document[document_key].append((chunk_index, doc, meta, i))

            # Keep same ordering behavior as grouped relevance flow (Z -> A by document key)
            for document_key in sorted(grouped_by_document.keys(), reverse=True):
                chunks = grouped_by_document[document_key]
                chunks.sort(key=lambda x: x[0])  # chunk order inside document

                first_meta = chunks[0][2] if chunks else {}
                header_name = (
                    first_meta.get('file_name') or
                    first_meta.get('document_title') or
                    first_meta.get('title') or
                    (Path(document_key).name if document_key else 'Unbekanntes Dokument')
                )

                context_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                context_parts.append(f"📄 DOKUMENT: {header_name}")
                context_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                context_parts.append("")

                for _, doc, meta, original_idx in chunks:
                    source_info = _format_chunk_source(meta, original_idx)
                    context_parts.append(f"[{source_info}]\n{doc}")
                    context_parts.append("")

            context = "\n".join(context_parts).strip()
            prompt = (
                f"AUFGABE: Erstelle ein BPMN-Modell für den folgenden Prozess basierend auf den bereitgestellten Dokumenten.\n\n"
                f"Du darfst ausschließlich die folgenden Informationen verwenden.\n\n"
                f"Der der zu modellierende Prozess ist: {user_request}\n"
                f"Relevante Informationen aus Dokumenten:\n{context}\n\n"
            )

        else:
            prompt = (
                f"AUFGABE: Erstelle ein BPMN-Modell für den folgenden Prozess.\n\n"
                f"Prozessanfrage:\n{user_request}"
            )

        system_prompt= (
                    "ROLLE: Du bist ein Agent zur Prozessidentifikation und zur Generierung eines BPMN verschachtelten JSON Outputs.\n\n"
                    "AUFGABE: Lies die vom User bereitgestellten Textpassagen und erzeuge darauf basierend einen detaillierten Prozessmodell zum gewünschten Prozess, indem du ein gültiges JSON Objekt, das dem unten definierten verschachtelten BPMN JSON Schema und den Modellierungsregeln entspricht.\n\n"
                    
                    "REGELN für den erwarteten Output mit Beispiel:\n"
                    "- Vergesse KEINE vorhandenen Schritte oder Akteure.\n"
                    "- Du darfst KEINE zusätzlichen Inhalte erfinden.\n"
                    "- Du darfst KEINE Informationen verwenden, die nicht explizit enthalten sind.\n"
                    "- Achte darauf, dass die Beschreibung eines Prozessaktivität IMMER nach dem Muster: Nomen + Verb formuliert ist: Beispiel \"Bescheid versenden\" und NICHT \"Versand des Bescheids\"\n\n"
                    "- Scope Regel zur Prozesszugehörigkeit:\n"
                    "- Du darfst ausschließlich Prozesselemente modellieren, die eindeutig zum im User Prompt definierten Prozess gehören. Wenn danach keine Schritte bleiben, gib {\"pools\": []} aus.\n\n"
                    "- Achte aber darauf alles, was zum Prozess gehört und in den Textpassagen enthalten ist, zu modellieren.\n"
                    "- Du darfst KEINE zusätzlichen Informationen dazu erfinden.\n"
                    
                    "REGELN für den erwarteten Output mit Beispiel:\n"
                    "1. Pools repräsentieren separate unabhängige Organisationen oder Institutionen. Lanes sind Abteilungen oder Teilnehmer innerhalb eines Pools. \n"
                    "2. Verwende send und receive message events und message flows nur zwischen Pools, nicht zwischen Lanes.\n"
                    "Beispiel Input: \"Kunde sendet eine Anfrage an eine Firma\".\n"
                    "Beispiel Output: {\"pools\":[{\"name\":\"Kunde\",\"dataObjects\":[{\"name\":\"Anfrage\",\"type\":\"data-file\"}],\"lanes\":[{\"name\":\"Kunde\"}],\"startEvent\":{\"subType\":\"default\",\"name\":\"Anfrage gestartet\",\"laneIndex\":0},\"process\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Anfrage schreiben\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 1)\"},{\"type\":\"event\",\"subType\":\"messageSend\",\"name\":\"Anfrage an Firma senden\",\"receiveEventId\":1,\"laneIndex\":0,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 1)\"}],\"endEvent\":{\"name\":\"Anfrage gesendet\",\"laneIndex\":0}}]}\n\n"
                    "3. Verwende send- und receive-Message-Events eins-zu-eins. Verwende exklusive Gateways (XOR), um basierend auf einer Bedingung einen Pfad zu wählen. Verwende event-basierte Gateways, um basierend auf eingehenden Events einen Pfad zu wählen.\n"
                    "Beispiel Input: \"Firma bereitet ein Angebot vor und sendet es an den Kunden, der entscheidet, ob er es annimmt oder ablehnt. Wenn innerhalb von 10 Tagen keine Antwort eingeht, leitet die Firma eine Nachfrage ein\".\n"
                    "Beispiel Output: {\"pools\":[{\"name\":\"Kunde\",\"dataObjects\":[{\"name\":\"Antwort\",\"type\":\"data-file\"},{\"name\":\"Angebot\",\"type\":\"data-file\"}],\"lanes\":[{\"name\":\"Kunde\"}],\"startEvent\":{\"subType\":\"messageReceive\",\"id\":3,\"name\":\"Angebot von Firma erhalten\",\"laneIndex\":0,\"writeDataObjectRefs\":[1],\"documentation\":\"Dokument: Beispiel.pdf (Seite 2)\"},\"process\":[{\"type\":\"xor\",\"condition\":\"\",\"branches\":[{\"label\":\"\",\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Annahme schreiben\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 3)\"},{\"type\":\"event\",\"subType\":\"messageSend\",\"name\":\"Annahme an Firma senden\",\"receiveEventId\":1,\"laneIndex\":0,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 3)\"}]},{\"label\":\"\",\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Ablehnung schreiben\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 4)\"},{\"type\":\"event\",\"subType\":\"messageSend\",\"name\":\"Ablehnung an Firma senden\",\"receiveEventId\":2,\"laneIndex\":0,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf, Chunk: 3\"}]}],\"laneIndex\":0,\"documentation\":\"Dokument: Beispiel.pdf (Seite 2)\"}],\"endEvent\":{\"name\":\"Antwort gesendet\",\"laneIndex\":0}}]}\n\n"
                    "4. Verwende Datenobjekte, nicht Message-Events, für die Kommunikation zwischen Lanes desselben Pools.\n"
                    "Beispiel Input: \"Die Vertriebsabteilung sendet einen Bericht an die Finanzabteilung.\"\n"
                    "Beispiel Output: {\"pools\":[{\"name\":\"Firma\",\"dataObjects\":[{\"name\":\"Bericht\",\"type\":\"data-file\"}],\"lanes\":[{\"name\":\"Vertriebsabteilung\"},{\"name\":\"Finanzabteilung\"}],\"startEvent\":{\"subType\":\"default\",\"name\":\"Prozess starten\",\"laneIndex\":0},\"process\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Bericht erstellen\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite X)\"},{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Bericht prüfen\",\"laneIndex\":1,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 2)\"}],\"endEvent\":{\"name\":\"Bericht geprüft\",\"laneIndex\":1}}]}\n\n"
                    "5. Verwende data-file nur für digitale oder physische Dateien und data-store für persistente Datenspeicher und Datenbanksysteme.\n"
                    "Beispiel Input: \"Der Händler erhält eine Bestellung und prüft die Lagerdatenbank\".\n"
                    "Beispiel Output: {\"pools\":[{\"name\":\"Händler\",\"dataObjects\":[{\"name\":\"Bestellung\",\"type\":\"data-file\"},{\"name\":\"Lagerdatenbank\",\"type\":\"data-store\"}],\"lanes\":[{\"name\":\"Bestell- und Lagerverwaltung\"}],\"startEvent\":{\"subType\":\"default\",\"name\":\"Bestellprozess starten\",\"laneIndex\":0},\"process\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Bestellung erhalten\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite X)\"},{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Lager prüfen\",\"laneIndex\":0,\"readDataObjectRefs\":[0,1],\"documentation\":\"Dokument: Beispiel.pdf (Seite Y)\"}],\"endEvent\":{\"name\":\"Bestellung bearbeitet\",\"laneIndex\":0}}]}\n\n"
                    "6. Verwende parallele Gateways (type \"parallel\"), wenn mehrere Aktivitäten GLEICHZEITIG ablaufen. Struktur: {\"type\":\"parallel\",\"branches\":[{\"branch\":[<tasks>],\"event\":null},{\"branch\":[<tasks>],\"event\":null}],\"laneIndex\":0,\"documentation\":\"...\"}. KEINE condition, KEINE labels - nur branches mit branch-Arrays.\n"
                    "Example Input: \"Die Behörde ordnet gleichzeitig Abstandsgebot im öffentlichen Raum und Maskenpflicht an.\"\n"
                    "Example Output: {\"type\":\"parallel\",\"branches\":[{\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Abstandsgebot im öffentlichen Raum anordnen\",\"laneIndex\":0,\"readDataObjectRefs\":null,\"writeDataObjectRefs\":null,\"documentation\":\"Verordnung.pdf (Seite 1)\"}],\"event\":null},{\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Maskenpflicht anordnen\",\"laneIndex\":0,\"readDataObjectRefs\":null,\"writeDataObjectRefs\":null,\"documentation\":\"Verordnung.pdf (Seite 1)\"}],\"event\":null}],\"laneIndex\":0,\"documentation\":\"Verordnung.pdf (Seite 1)\"}\n\n"
                    
                    "Abschließende HINWEISE \n\n"
                    "- Gib AUSSCHLIESSLICH ein gültiges JSON zurück.\n"
                    "- Kein erklärender Text außerhalb des JSON.\n"
                    "- Das JSON MUSS BPMN-strukturell korrekt, vollständig verschachtelt und konsistent sein.\n"   
                    )
       
       
        # Generate BPMN JSON
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]


        raw = await call_ollama_json(
            messages=messages,
            model=settings.agentic_bpmn_model,
            temperature=0.0,  # Use deterministic temperature for consistent JSON format
        )
        
        try:
            data = json.loads(raw)
            # Validate and parse as nested model with Pydantic
            bpmn_model = BPMNModelJsonNested(**data)

            _retrieval_status(
                f"generated BPMN (retrieval) pools={len(bpmn_model.pools)} docs={len(retrieved_documents)}"
            )
            if settings.debug:
                _retrieval_debug(
                    f"generated BPMN JSON preview={bpmn_model.model_dump_json(indent=2)[:1000]!r}"
                )
            
            return bpmn_model, retrieved_documents, retrieved_metadatas, relevance_scores
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error in RetrievalBpmnAgent: {e}"
            logger.error(error_msg)
            logger.debug("Raw LLM response (first 500 chars): %s", raw[:500])
            _retrieval_debug(error_msg)
            # Fallback: empty model
            return BPMNModelJsonNested(pools=[]), retrieved_documents, retrieved_metadatas, relevance_scores
            
        except ValidationError as e:
            # Log detailed validation errors
            error_msg = "Pydantic validation error in RetrievalBpmnAgent"
            logger.error("%s: %s", error_msg, str(e))
            
            # Log structured validation errors
            validation_errors = e.errors()
            logger.error("Validation error details (%d errors):", len(validation_errors))
            for i, error in enumerate(validation_errors, 1):
                loc = " -> ".join(str(x) for x in error.get("loc", []))
                error_type = error.get("type", "unknown")
                error_msg_detail = error.get("msg", "No message")
                logger.error(
                    "  Error %d: Field '%s' - Type: %s, Message: %s",
                    i, loc, error_type, error_msg_detail
                )
            
            logger.debug("Raw LLM response (first 500 chars): %s", raw[:500])
            logger.debug("Parsed data (first 1000 chars): %s", json.dumps(data, indent=2)[:1000] if 'data' in locals() else "N/A")
            _retrieval_debug(f"{error_msg}: {len(validation_errors)} error(s)")
            
            # Fallback: empty model
            return BPMNModelJsonNested(pools=[]), retrieved_documents, retrieved_metadatas, relevance_scores
            
    except Exception as e:
        # On generation failure, return empty model with retrieved documents
        error_msg = f"BPMN generation error in RetrievalBpmnAgent: {e}"
        logger.error(error_msg)
        _retrieval_debug_exc("generation failed", e)
        # Fallback: empty model
        return BPMNModelJsonNested(pools=[]), retrieved_documents, retrieved_metadatas, relevance_scores


async def run_retrieval_bpmn_agent_with_structure(
    user_request: str,
    retrieved_documents: List[str],
    retrieved_metadatas: List[Dict[str, Any]],
    query_structure: Optional[QueryStructure],
    api_client: OpenWebUIClient
) -> BPMNModelJsonNested:
    """
    Generate BPMN model from retrieved documents using query structure.
    
    This function uses structured query information to generate more precise BPMN models.
    
    Args:
        user_request: Original user query
        retrieved_documents: List of retrieved document texts
        retrieved_metadatas: List of metadata dicts for retrieved documents
        query_structure: Optional structured query information
        api_client: OpenWebUI API client instance
        
    Returns:
        BPMNModelJsonNested model
    """
    try:
        # Extract query structure information
        process_name = query_structure.process_name if query_structure else "NICHT ANGEGEBEN"
        domain = query_structure.domain if query_structure else "NICHT ANGEGEBEN"
        procedure_type = query_structure.procedure_type if query_structure else "NICHT ANGEGEBEN"
        perspective = query_structure.perspective if (query_structure and query_structure.perspective) else "NICHT ANGEGEBEN"
        
        # Map granularity to readable format
        if query_structure and query_structure.granularity:
            granularity_map = {
                "high_level": "High-Level (Überblick, Hauptschritte)",
                "medium": "Medium (moderate Detaillierung)",
                "detailed": "Detailed (sehr detailliert, kleinschrittig)"
            }
            granularity = granularity_map.get(query_structure.granularity, query_structure.granularity)
        else:
            granularity = "NICHT ANGEGEBEN"
        
        start_hint = query_structure.scope_start if (query_structure and query_structure.scope_start) else "NICHT ANGEGEBEN"
        end_hint = query_structure.scope_end if (query_structure and query_structure.scope_end) else "NICHT ANGEGEBEN"
        
        # Build context from retrieved documents
        context_parts = []
        if retrieved_documents:
            for i, doc, meta in _iter_documents_with_metadata(retrieved_documents, retrieved_metadatas):
                # Document headers: include as separator only (same as draft agent in Setting 3)
                if meta and meta.get('is_document_header', False):
                    context_parts.append(doc)
                else:
                    source_info = _format_chunk_source(meta, i)
                    context_parts.append(f"[{source_info}]\n{doc}")
            
            context = "\n\n".join(context_parts)
            
            # Build query structure information section
            query_info_parts = []
            if process_name != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Prozessname: {process_name}")
            if domain != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Domäne: {domain}")
            if procedure_type != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Verfahrenstyp: {procedure_type}")
            if perspective != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Perspektive: {perspective}")
            if granularity != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Granularität: {granularity}")
            if start_hint != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Prozessstart: {start_hint}")
            if end_hint != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Prozessende: {end_hint}")
            
            
            query_info_section = ""
            if query_info_parts:
                query_info_section = f"\n\nZusätzliche Prozessinformationen:\n" + "\n".join(f"- {info}" for info in query_info_parts) + "\n"
            
            prompt = (
                f"AUFGABE: Erstelle ein BPMN-Modell für den folgenden Prozess basierend auf den bereitgestellten Dokumenten.\n\n"
                f"Du darfst ausschließlich die folgenden Informationen verwenden.\n\n"
                f"Der der zu modellierende Prozess ist: {user_request}\n"

                f"Relevante Informationen aus Dokumenten:\n{context}\n\n"
                f"Prozessanfrage:\n{user_request}{query_info_section}"
            )
        else:
            # Build query structure information section for empty documents case
            query_info_parts = []
            if process_name != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Prozessname: {process_name}")
            if domain != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Domäne: {domain}")
            if procedure_type != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Verfahrenstyp: {procedure_type}")
            if perspective != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Perspektive: {perspective}")
            if granularity != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Granularität: {granularity}")
            if start_hint != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Prozessstart: {start_hint}")
            if end_hint != "NICHT ANGEGEBEN":
                query_info_parts.append(f"Prozessende: {end_hint}")
           
            query_info_section = ""
            if query_info_parts:
                query_info_section = f"\n\nZusätzliche Prozessinformationen:\n" + "\n".join(f"- {info}" for info in query_info_parts)
            
            prompt = (
                f"AUFGABE: Erstelle ein BPMN-Modell für den folgenden Prozess.\n\n"
                f"Prozessanfrage:\n{user_request}{query_info_section}"
            )
    

        system_prompt= (
                    "ROLLE: Du bist ein Agent zur Prozessidentifikation und zur Generierung eines BPMN verschachtelten JSON Outputs.\n\n"
                    "AUFGABE: Lies die vom User bereitgestellten Textpassagen und erzeuge darauf basierend ein detailliertes Prozessmodell zum gewünschten Prozess, indem du ein gültiges JSON Objekt, das dem unten definierten verschachtelten BPMN JSON Schema und den Modellierungsregeln entspricht.\n\n"
                    
                    "REGELN für die BPMN-Modellierung:\n"
                    "- Vergesse KEINE vorhandenen Schritte oder Akteure.\n"
                    "- Du darfst KEINE zusätzlichen Inhalte erfinden.\n"
                    "- Du darfst KEINE Informationen verwenden, die nicht explizit enthalten sind.\n"
                    "- Achte darauf, dass die Beschreibung eines Prozessaktivität IMMER nach dem Muster: Nomen + Verb formuliert ist: Beispiel \"Bescheid versenden\" und NICHT \"Versand des Bescheids\"\n\n"
                    "- Scope Regel zur Prozesszugehörigkeit:\n"
                    "- Du darfst ausschließlich Prozesselemente modellieren, die eindeutig zum im User Prompt definierten Prozess gehören. Wenn danach keine Schritte bleiben, gib {\"pools\": []} aus.\n\n"
                    "- Achte aber darauf alles, was zum Prozess gehört und in den Textpassagen enthalten ist, zu modellieren."
                    "- Berücksichtige bei der Modellierung auch die zusätzlichen Informationen, die im User Prompt enthalten sind. Das sind beispielsweise die Hinweise zum Scope, die Granularität, die Perspektive, die Start- und End-Events."
                    "- Du darfst KEINE zusätzlichen Informationen dazu erfinden.\n"
                

                    "REGELN für den erwarteten Output mit Beispiel:\n"
                    "1. Pools repräsentieren separate unabhängige Organisationen oder Institutionen. Lanes sind Abteilungen oder Teilnehmer innerhalb eines Pools. \n"
                    "2. Verwende send und receive message events und message flows nur zwischen Pools, nicht zwischen Lanes.\n"
                    "Beispiel Input: \"Kunde sendet eine Anfrage an eine Firma\".\n"
                    "Beispiel Output: {\"pools\":[{\"name\":\"Kunde\",\"dataObjects\":[{\"name\":\"Anfrage\",\"type\":\"data-file\"}],\"lanes\":[{\"name\":\"Kunde\"}],\"startEvent\":{\"subType\":\"default\",\"name\":\"Anfrage gestartet\",\"laneIndex\":0},\"process\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Anfrage schreiben\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 1)\"},{\"type\":\"event\",\"subType\":\"messageSend\",\"name\":\"Anfrage an Firma senden\",\"receiveEventId\":1,\"laneIndex\":0,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 1)\"}],\"endEvent\":{\"name\":\"Anfrage gesendet\",\"laneIndex\":0}}]}\n\n"
                    "3. Verwende send- und receive-Message-Events eins-zu-eins. Verwende exklusive Gateways (XOR), um basierend auf einer Bedingung einen Pfad zu wählen. Verwende event-basierte Gateways, um basierend auf eingehenden Events einen Pfad zu wählen.\n"
                    "Beispiel Input: \"Firma bereitet ein Angebot vor und sendet es an den Kunden, der entscheidet, ob er es annimmt oder ablehnt. Wenn innerhalb von 10 Tagen keine Antwort eingeht, leitet die Firma eine Nachfrage ein\".\n"
                    "Beispiel Output: {\"pools\":[{\"name\":\"Kunde\",\"dataObjects\":[{\"name\":\"Antwort\",\"type\":\"data-file\"},{\"name\":\"Angebot\",\"type\":\"data-file\"}],\"lanes\":[{\"name\":\"Kunde\"}],\"startEvent\":{\"subType\":\"messageReceive\",\"id\":3,\"name\":\"Angebot von Firma erhalten\",\"laneIndex\":0,\"writeDataObjectRefs\":[1],\"documentation\":\"Dokument: Beispiel.pdf (Seite 2)\"},\"process\":[{\"type\":\"xor\",\"condition\":\"\",\"branches\":[{\"label\":\"\",\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Annahme schreiben\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 3)\"},{\"type\":\"event\",\"subType\":\"messageSend\",\"name\":\"Annahme an Firma senden\",\"receiveEventId\":1,\"laneIndex\":0,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 3)\"}]},{\"label\":\"\",\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Ablehnung schreiben\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 4)\"},{\"type\":\"event\",\"subType\":\"messageSend\",\"name\":\"Ablehnung an Firma senden\",\"receiveEventId\":2,\"laneIndex\":0,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf, Chunk: 3\"}]}],\"laneIndex\":0,\"documentation\":\"Dokument: Beispiel.pdf (Seite 2)\"}],\"endEvent\":{\"name\":\"Antwort gesendet\",\"laneIndex\":0}}]}\n\n"
                    "4. Verwende Datenobjekte, nicht Message-Events, für die Kommunikation zwischen Lanes desselben Pools.\n"
                    "Beispiel Input: \"Die Vertriebsabteilung sendet einen Bericht an die Finanzabteilung.\"\n"
                    "Beispiel Output: {\"pools\":[{\"name\":\"Firma\",\"dataObjects\":[{\"name\":\"Bericht\",\"type\":\"data-file\"}],\"lanes\":[{\"name\":\"Vertriebsabteilung\"},{\"name\":\"Finanzabteilung\"}],\"startEvent\":{\"subType\":\"default\",\"name\":\"Prozess starten\",\"laneIndex\":0},\"process\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Bericht erstellen\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite X)\"},{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Bericht prüfen\",\"laneIndex\":1,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 2)\"}],\"endEvent\":{\"name\":\"Bericht geprüft\",\"laneIndex\":1}}]}\n\n"
                    "5. Verwende data-file nur für digitale oder physische Dateien und data-store für persistente Datenspeicher und Datenbanksysteme.\n"
                    "Beispiel Input: \"Der Händler erhält eine Bestellung und prüft die Lagerdatenbank\".\n"
                    "Beispiel Output: {\"pools\":[{\"name\":\"Händler\",\"dataObjects\":[{\"name\":\"Bestellung\",\"type\":\"data-file\"},{\"name\":\"Lagerdatenbank\",\"type\":\"data-store\"}],\"lanes\":[{\"name\":\"Bestell- und Lagerverwaltung\"}],\"startEvent\":{\"subType\":\"default\",\"name\":\"Bestellprozess starten\",\"laneIndex\":0},\"process\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Bestellung erhalten\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite X)\"},{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Lager prüfen\",\"laneIndex\":0,\"readDataObjectRefs\":[0,1],\"documentation\":\"Dokument: Beispiel.pdf (Seite Y)\"}],\"endEvent\":{\"name\":\"Bestellung bearbeitet\",\"laneIndex\":0}}]}\n\n"
                    "6. Verwende parallele Gateways (type \"parallel\"), wenn mehrere Aktivitäten GLEICHZEITIG ablaufen. Struktur: {\"type\":\"parallel\",\"branches\":[{\"branch\":[<tasks>],\"event\":null},{\"branch\":[<tasks>],\"event\":null}],\"laneIndex\":0,\"documentation\":\"...\"}. KEINE condition, KEINE labels - nur branches mit branch-Arrays.\n"
                    "Example Input: \"Die Behörde ordnet gleichzeitig Abstandsgebot im öffentlichen Raum und Maskenpflicht an.\"\n"
                    "Example Output: {\"type\":\"parallel\",\"branches\":[{\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Abstandsgebot im öffentlichen Raum anordnen\",\"laneIndex\":0,\"readDataObjectRefs\":null,\"writeDataObjectRefs\":null,\"documentation\":\"Verordnung.pdf (Seite 1)\"}],\"event\":null},{\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Maskenpflicht anordnen\",\"laneIndex\":0,\"readDataObjectRefs\":null,\"writeDataObjectRefs\":null,\"documentation\":\"Verordnung.pdf (Seite 1)\"}],\"event\":null}],\"laneIndex\":0,\"documentation\":\"Verordnung.pdf (Seite 1)\"}\n\n"
                    
                    "Abschließende HINWEISE \n\n"
                    "- Gib AUSSCHLIESSLICH ein gültiges JSON zurück.\n"
                    "- Kein erklärender Text außerhalb des JSON.\n"
                    "- Das JSON MUSS BPMN-strukturell korrekt, vollständig verschachtelt und konsistent sein.\n"   
                    )

      

        

        # Generate BPMN JSON
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        raw = await call_ollama_json(
            messages=messages,
            model=settings.agentic_bpmn_model,
            temperature=0.0,  # Use deterministic temperature for consistent JSON format
        )
        
        try:
            data = json.loads(raw)
            # Validate and parse as nested model with Pydantic
            bpmn_model = BPMNModelJsonNested(**data)

            _retrieval_status(
                f"generated BPMN (retrieval+structure) pools={len(bpmn_model.pools)} docs={len(retrieved_documents)}"
            )
            if settings.debug:
                _retrieval_debug(
                    f"generated BPMN JSON preview={bpmn_model.model_dump_json(indent=2)[:1000]!r}"
                )
            
            return bpmn_model
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error in RetrievalBpmnAgentWithStructure: {e}"
            logger.error(error_msg)
            logger.debug("Raw LLM response (first 500 chars): %s", raw[:500])
            _retrieval_debug(error_msg)
            # Fallback: empty model
            return BPMNModelJsonNested(pools=[])
            
        except ValidationError as e:
            # Log detailed validation errors
            error_msg = "Pydantic validation error in RetrievalBpmnAgentWithStructure"
            logger.error("%s: %s", error_msg, str(e))
            
            # Log structured validation errors
            validation_errors = e.errors()
            logger.error("Validation error details (%d errors):", len(validation_errors))
            for i, error in enumerate(validation_errors, 1):
                loc = " -> ".join(str(x) for x in error.get("loc", []))
                error_type = error.get("type", "unknown")
                error_msg_detail = error.get("msg", "No message")
                logger.error(
                    "  Error %d: Field '%s' - Type: %s, Message: %s",
                    i, loc, error_type, error_msg_detail
                )
            
            logger.debug("Raw LLM response (first 500 chars): %s", raw[:500])
            logger.debug("Parsed data (first 1000 chars): %s", json.dumps(data, indent=2)[:1000] if 'data' in locals() else "N/A")
            _retrieval_debug(f"{error_msg}: {len(validation_errors)} error(s)")
            
            # Fallback: empty model
            return BPMNModelJsonNested(pools=[])
            
    except Exception as e:
        # On generation failure, return empty model
        error_msg = f"BPMN generation error in RetrievalBpmnAgentWithStructure: {e}"
        logger.error(error_msg)
        _retrieval_debug_exc("generation with structure failed", e)
        # Fallback: empty model
        return BPMNModelJsonNested(pools=[])

