"""Relevance evaluation agent for filtering retrieved chunks."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import settings
from ..infrastructure.api.ollama_client import call_ollama_json
from ..models.query_structure import QueryStructure
from ..models.relevance import (
    ChunkAssessment,
    ChunkInput,
    RelevanceEvaluationResult,
)

logger = logging.getLogger(__name__)

RELEVANCE_PRIORITY = {"high": 0, "medium": 1, "low": 2, "none": 3}


def _relevance_debug(msg: str) -> None:
    """Print one debug line when ``settings.debug`` is true (``[relevance|debug]`` prefix)."""
    if settings.debug:
        print(f"[relevance|debug] {msg}", flush=True)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _build_chunk_inputs(
    chunks: List[str],
    metadatas: List[Dict[str, Any]],
    ids: Optional[List[str]] = None,
) -> List[ChunkInput]:
    """Build ``ChunkInput`` rows aligned with ``chunks``, using Chroma IDs when present."""
    inputs: List[ChunkInput] = []
    for idx, chunk_text in enumerate(chunks):
        meta = metadatas[idx] if idx < len(metadatas) else {}

        chunk_id = None
        if ids and idx < len(ids):
            chunk_id = ids[idx]
        elif meta.get("id"):
            chunk_id = meta.get("id")
        else:
            file_name = meta.get("file_name")
            if not file_name and meta.get("file_path"):
                file_name = Path(str(meta.get("file_path"))).stem

            chunk_index = meta.get("chunk_index")
            if file_name and chunk_index is not None:
                chunk_id = f"{file_name}_chunk_{chunk_index}"
            else:
                chunk_id = f"chunk_{idx}"

        chunk_nr = meta.get("chunk_nr", idx + 1)

        inputs.append(
            ChunkInput(
                chunk_nr=chunk_nr,
                chunk_id=chunk_id,
                text=chunk_text,
            )
        )
    return inputs


def _normalize_evidence_spans(data: Dict[str, Any]) -> None:
    """Coerce ``evidence_spans`` list entries from bare strings to dicts before Pydantic validation."""
    if "chunk_assessments" not in data:
        return

    normalized_count = 0
    for assessment in data["chunk_assessments"]:
        if "evidence_spans" not in assessment:
            continue
        spans = assessment["evidence_spans"]
        if not spans or not isinstance(spans, list):
            continue
        normalized_spans: List[Any] = []
        for span in spans:
            if isinstance(span, str):
                normalized_spans.append({"text": span})
                normalized_count += 1
            elif isinstance(span, dict):
                normalized_spans.append(span)
        assessment["evidence_spans"] = normalized_spans

    if normalized_count > 0:
        _relevance_debug(f"normalized {normalized_count} evidence span(s) from string to object format")


def _attach_missing_assessments(
    result: RelevanceEvaluationResult, chunk_inputs: List[ChunkInput]
) -> None:
    """Append a ``none`` placeholder for any input chunk missing from the model output."""
    assessed_nrs = {a.chunk_nr for a in result.chunk_assessments}

    for chunk in chunk_inputs:
        if chunk.chunk_nr not in assessed_nrs:
            result.chunk_assessments.append(
                ChunkAssessment(
                    chunk_nr=chunk.chunk_nr,
                    relevance="none",
                    confidence=0.0,
                    evidence_spans=[],
                    why_not_relevant="Chunk not assessed by judge",
                )
            )


async def _evaluate_batch(
    query: str,
    query_structure: Optional[QueryStructure],
    batch_chunk_inputs: List[ChunkInput],
    batch_num: int,
    retry: bool = False,
) -> Optional[RelevanceEvaluationResult]:
    """One judge LLM call for up to 10 chunks; one automatic retry on parse/validation errors."""
    system_prompt = (
        "ROLLE: Du bist ein Relevanz-Judge für BPMN-Prozessmodellierung.\n\n"
        "AUFGABE:\n"
        "Schaue dir jeden Chunk genau an und entscheide, ob der Inhalt relevant ist, also Informationen zu Prozesselementen und Prozessablauf beinhaltet, um den in der User Query beschriebenen Prozess zu modellieren.\n\n"
        "Input Informationen:\n"
        "- query: Original user query\n"
        "- query_structure: Strukturierte Prozess-Informationen (process_name, domain, procedure_type, etc.)\n"
        "- query_structure.key_terms: Extrahierte Schlüsselbegriffe (falls vorhanden)\n"
        "- query_structure.synonyms: Synonyme pro Schlüsselbegriff (falls vorhanden)\n"
        "- chunks: Liste von Chunk-Objekten mit chunk_nr (Aufzählung), eindeutiger chunk_id und text\n\n"

        "RELEVANZ LABELS (bezogen auf User Query und extrahierte Prozess-Infos):\n"
        "- high: direkt nutzbar für BPMN-Extraktion, enthält relevante Informationen für den gesuchten Prozess\n"
        "- medium: relevant aber unvollständig oder eher Kontext\n"
        "- low: schwacher Bezug\n"
        "- none: kein Bezug oder falscher Prozess/Domäne\n\n"

        "HINWEIS:\n"
        "Nutze die key_terms und synonyms aus query_structure, um die Relevanz des chunks für den gewünschten Prozess besser einzuschätzen. "
        "Nutze die Namen der Dokumente aus welchem ein Chunk kommt, um die Relevanz besser einzuschätzen.\n\n"
        "REGELN:\n"
        "1) Nutze nur die gelieferten Chunks. Kein externes Wissen. Erfinde nichts.\n"
        "2) Bewerte JEDEN Chunk aus dem chunks-Array - keine Auslassungen.\n"
        "3) Verwende GENAU die chunk_nr aus dem Input - keine eigenen Nummern erfinden.\n"
        "4) Output ausschließlich valides JSON gemäß Schema.\n\n"

        "ERWARTETES OUTPUT JSON SCHEMA:\n"
        "{\n"
        '  "query": "string",\n'
        '  "decision": "proceed",\n'
        '  "chunk_assessments": [\n'
        "    {\n"
        '      "chunk_nr": 1,  // GENAU wie im chunks-Array (1, 2, 3, ...)\n'
        '      "relevance": "high|medium|low|none",\n'
        '      "confidence": 0.0,\n'
        '      "why_not_relevant": "string"      // Nur bei low/none relevant\n'
        "    }\n"
        "  ]\n"
        "}\n\n"

        "Abschließende HINWEISE:\n"
        "- Gib ausschließlich das JSON-Objekt ohne zusätzlichen Text."
    )

    user_prompt = json.dumps(
        {
            "query": query,
            "query_structure": query_structure.model_dump() if query_structure else None,
            "chunks": [c.model_dump() for c in batch_chunk_inputs],
        },
        ensure_ascii=False,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_response = await call_ollama_json(
            messages=messages,
            model=settings.agentic_judge_model,
            temperature=0.0,
            timeout=settings.api_timeout,
        )

        data = json.loads(raw_response)
        _normalize_evidence_spans(data)
        result = RelevanceEvaluationResult(**data)

        _attach_missing_assessments(result, batch_chunk_inputs)

        return result

    except (json.JSONDecodeError, Exception) as exc:
        if not retry:
            _relevance_debug(f"batch {batch_num} failed ({type(exc).__name__}), retrying once")
            return await _evaluate_batch(query, query_structure, batch_chunk_inputs, batch_num, retry=True)

        err_name = "JSONDecodeError" if isinstance(exc, json.JSONDecodeError) else type(exc).__name__
        logger.warning("Relevance batch %s failed after retry: %s: %s", batch_num, err_name, exc)
        _relevance_debug(f"batch {batch_num} failed after retry: {err_name}")
        return None


async def evaluate_retrieval(
    query: str,
    query_structure: Optional[QueryStructure],
    chunks: List[str],
    metadatas: List[Dict[str, Any]],
    ids: Optional[List[str]] = None,
    top_n: int = 10,
) -> RelevanceEvaluationResult:
    """Score chunks in batches of up to 10, then return up to ``top_n`` assessments.

    Keeps ``high`` and ``medium`` chunks first (sorted by relevance then confidence).
    If a batch fails after retry, those chunks are marked ``none``; when fewer than
    ``top_n`` high/medium rows exist, ``none`` rows from failed batches may be appended
    until ``top_n`` is reached, then the list is capped to ``top_n``.
    """
    chunk_inputs = _build_chunk_inputs(chunks, metadatas, ids)

    if not chunk_inputs:
        return RelevanceEvaluationResult(
            query=query,
            decision="proceed",
            chunk_assessments=[],
        )

    batch_size = 10
    batches = [chunk_inputs[i : i + batch_size] for i in range(0, len(chunk_inputs), batch_size)]

    _relevance_debug(
        f"processing {len(chunk_inputs)} chunks in {len(batches)} batch(es) (max {batch_size} per batch)"
    )

    all_assessments: Dict[str, ChunkAssessment] = {}
    failed_batch_chunk_ids: List[str] = []

    for batch_num, batch in enumerate(batches, 1):
        batch_chunk_ids = [c.chunk_id for c in batch]

        _relevance_debug(
            f"batch {batch_num}/{len(batches)}: {len(batch)} chunks, "
            f"ids={', '.join(batch_chunk_ids[:5])}{'...' if len(batch_chunk_ids) > 5 else ''}"
        )

        result = await _evaluate_batch(query, query_structure, batch, batch_num)

        if result:
            batch_nr_to_id = {c.chunk_nr: c.chunk_id for c in batch}

            for assessment in result.chunk_assessments:
                chunk_id = batch_nr_to_id.get(assessment.chunk_nr)
                if chunk_id:
                    all_assessments[chunk_id] = assessment

            high_count = sum(1 for a in result.chunk_assessments if a.relevance == "high")
            medium_count = sum(1 for a in result.chunk_assessments if a.relevance == "medium")
            low_count = sum(1 for a in result.chunk_assessments if a.relevance == "low")
            none_count = sum(1 for a in result.chunk_assessments if a.relevance == "none")

            _relevance_debug(
                f"batch {batch_num} ok: H/M/L/none={high_count}/{medium_count}/{low_count}/{none_count}"
            )
        else:
            _relevance_debug(
                f"batch {batch_num} failed: marking {len(batch)} chunks as none and passing through"
            )
            failed_batch_chunk_ids.extend(batch_chunk_ids)

            for chunk_input in batch:
                all_assessments[chunk_input.chunk_id] = ChunkAssessment(
                    chunk_nr=chunk_input.chunk_nr,
                    relevance="none",
                    confidence=0.0,
                    evidence_spans=[],
                    why_not_relevant="Batch evaluation failed after retry",
                )

    all_final_assessments: List[ChunkAssessment] = []
    for chunk_input in chunk_inputs:
        if chunk_input.chunk_id in all_assessments:
            assessment = all_assessments[chunk_input.chunk_id]
            assessment.chunk_nr = chunk_input.chunk_nr
            all_final_assessments.append(assessment)
        else:
            all_final_assessments.append(
                ChunkAssessment(
                    chunk_nr=chunk_input.chunk_nr,
                    relevance="none",
                    confidence=0.0,
                    evidence_spans=[],
                    why_not_relevant="No assessment received",
                )
            )

    for assessment in all_final_assessments:
        if assessment.relevance not in ("high", "medium", "low", "none"):
            assessment.relevance = "none"
        if assessment.confidence < 0.0:
            assessment.confidence = 0.0
        if assessment.confidence > 1.0:
            assessment.confidence = 1.0
        assessment.why_not_relevant = (
            _normalize_whitespace(assessment.why_not_relevant) if assessment.why_not_relevant else None
        )

    all_final_assessments.sort(
        key=lambda a: (RELEVANCE_PRIORITY.get(a.relevance, 999), -a.confidence)
    )

    high_medium = [a for a in all_final_assessments if a.relevance in ("high", "medium")]

    if failed_batch_chunk_ids and len(high_medium) < top_n:
        failed_set = set(failed_batch_chunk_ids)
        for chunk_input in chunk_inputs:
            if len(high_medium) >= top_n:
                break
            if chunk_input.chunk_id not in failed_set:
                continue
            assessment = all_assessments.get(chunk_input.chunk_id)
            if assessment and assessment not in high_medium:
                high_medium.append(assessment)

    if len(high_medium) > top_n:
        high_medium = high_medium[:top_n]

    result = RelevanceEvaluationResult(
        query=query,
        decision="proceed",
        chunk_assessments=high_medium,
    )

    high_count = sum(1 for a in result.chunk_assessments if a.relevance == "high")
    medium_count = sum(1 for a in result.chunk_assessments if a.relevance == "medium")
    low_count = sum(1 for a in result.chunk_assessments if a.relevance == "low")
    none_count = sum(1 for a in result.chunk_assessments if a.relevance == "none")

    _relevance_debug(
        f"final: returned={len(result.chunk_assessments)} (cap top_n={top_n}) "
        f"from {len(chunk_inputs)} inputs; H/M/L/none={high_count}/{medium_count}/{low_count}/{none_count}"
    )
    if failed_batch_chunk_ids:
        _relevance_debug(f"failed-batch chunk ids counted: {len(failed_batch_chunk_ids)}")

    return result
