"""Query-structure extraction and synonym expansion for agentic retrieval.

Used by enhanced retrieval paths: first pass via ``extract_query_structure_and_expand``,
optional second pass via ``generate_additional_synonyms`` when the graph retries retrieval
after low relevance scores (see ``_execute_relevance_evaluation`` in ``src.graphs.pipeline_graphs``).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from ..config import settings
from ..infrastructure.api.ollama_client import call_ollama_json
from ..models.query_structure import QueryStructure

logger = logging.getLogger(__name__)


def _retrieval_debug(msg: str) -> None:
    """Print one debug line when ``settings.debug`` is true (console, ``[retrieval|debug]`` prefix)."""
    if settings.debug:
        print(f"[retrieval|debug] {msg}", flush=True)


def _parse_llm_retrieval_payload(data: Dict[str, Any], user_query: str) -> Tuple[QueryStructure, List[str], List[str], Dict[str, List[str]]]:
    """Parse the JSON object returned by the extraction / re-synonym LLM calls.

    Normalizes ``notes`` (list to comma-separated string), merges ``keyTerms`` and
    ``synonyms`` into the payload for ``QueryStructure``, then builds ``expanded_queries``:
    original query first, then key terms and synonyms that differ from the original
    (case-insensitive), preserving order and skipping duplicate strings.
    """
    query_structure_data = dict(data.get("query_structure") or {})

    if "notes" in query_structure_data and isinstance(query_structure_data["notes"], list):
        query_structure_data["notes"] = ", ".join(str(item) for item in query_structure_data["notes"])
    elif "notes" in query_structure_data and query_structure_data["notes"] is None:
        query_structure_data["notes"] = None

    key_terms: List[str] = list(data.get("keyTerms") or [])
    synonyms_dict: Dict[str, List[str]] = dict(data.get("synonyms") or {})

    if key_terms:
        query_structure_data["key_terms"] = key_terms
    if synonyms_dict:
        query_structure_data["synonyms"] = synonyms_dict

    query_structure = QueryStructure(**query_structure_data)

    expanded_queries: List[str] = [user_query]
    uq_lower = user_query.lower()
    for term in key_terms:
        if term and term.strip() and term.strip().lower() != uq_lower:
            expanded_queries.append(term.strip())
    for _term, synonym_list in synonyms_dict.items():
        if not synonym_list:
            continue
        for synonym in synonym_list:
            if synonym and synonym.strip() and synonym.strip().lower() != uq_lower:
                s = synonym.strip()
                if s not in expanded_queries:
                    expanded_queries.append(s)

    return query_structure, expanded_queries, key_terms, synonyms_dict


def _debug_dump_expansion(label: str, query_structure: QueryStructure, key_terms: List[str], synonyms_dict: Dict[str, List[str]], expanded_queries: List[str]) -> None:
    """Log structured expansion details under debug (``label`` prefixes each line)."""
    if not settings.debug:
        return
    _retrieval_debug(f"{label} process={query_structure.process_name!r} domain={query_structure.domain!r}")
    if query_structure.perspective:
        _retrieval_debug(f"{label} perspective={query_structure.perspective!r}")
    _retrieval_debug(f"{label} key_terms={key_terms}")
    _retrieval_debug(f"{label} synonyms={synonyms_dict}")
    _retrieval_debug(f"{label} expanded_queries ({len(expanded_queries)}):")
    for i, q in enumerate(expanded_queries):
        _retrieval_debug(f"  [{i + 1}] {q}")


async def extract_query_structure_and_expand(user_query: str) -> Tuple[QueryStructure, List[str]]:
    """Run the first extraction LLM call (low temperature).

    Returns a ``QueryStructure`` and a list of search strings (original plus expansions).
    On parse/validation/network failure, logs a warning and returns a minimal
    ``QueryStructure`` with ``domain`` and ``procedure_type`` set to ``Unbekannt`` and
    a single-element list containing only ``user_query``.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "ROLLE: Du bist Experte für Prozessmanagement in der öffentlichen Verwaltung. "
                "AUFGABE: Analysiere die Suchanfrage und extrahiere strukturierte Informationen sowie Synonyme.\n\n"
                "EXTRAHIERE:\n"
                "- **process_name** (PFLICHT): Sprechbarer Prozessname, \n"
                "- **domain** (PFLICHT): Verwaltungsbereich,\n"
                "- **procedure_type** (PFLICHT): Verfahrensart\n"
                "- **perspective** (OPTIONAL): Nur wenn klar in Query erkennbar,\n"
                "- **granularity** (OPTIONAL): Nur wenn explizit genannt: high_level, medium, detailed\n"
                "- **scope_start/scope_end** (OPTIONAL): Nur wenn klar definiert\n"
                "- **notes** (OPTIONAL): Wichtige Begriffe, Synonyme\n\n"
                "ERWEITERE um Schlüsselbegriffe und Synonyme:\n"
                "1. Extrahiere genau 3 Schlüsselbegriffe (keyTerms), falls möglich. Wenn weniger als 3 sinnvolle Begriffe existieren, gib nur diese aus.\n"
                "2. Gib pro Schlüsselbegriff maximal 1 Synonym oder sehr nahe Paraphrase aus.\n"
                "3. Synonyme müssen semantisch sehr nahe sein. Wenn es keine gibt, ist das Array [].\n"
                "4. Vermeide Duplikate und triviale Varianten wie reine Pluralformen, Bindestriche oder Groß Kleinschreibung.\n\n"
                "ERWARTETES OUTPUT JSON SCHEMA:\n"
                "{\n"
                "  \"query_structure\": {\n"
                "    \"original_query\": \"...\",\n"
                "    \"process_name\": \"...\",\n"
                "    \"domain\": \"...\" oder null,\n"
                "    \"procedure_type\": \"...\" oder null,\n"
                "    \"perspective\": \"...\" oder null,\n"
                "    \"granularity\": \"...\" oder null,\n"
                "    \"scope_start\": \"...\" oder null,\n"
                "    \"scope_end\": \"...\" oder null,\n"
                "    \"notes\": \"...\" oder null\n"
                "  },\n"
                "  \"keyTerms\": [ string ],\n"
                "  \"synonyms\": {\n"
                "    \"term\": [ string ]\n"
                "  }\n"
                "}"
            ),
        },
        {"role": "user", "content": f"Analysiere diese Suchanfrage: {user_query}"},
    ]

    if settings.debug:
        preview = user_query if len(user_query) <= 160 else user_query[:160] + "…"
        _retrieval_debug(f"extract+expand LLM call preview={preview!r}")
        logger.debug("Retrieval extract messages: %s", messages)

    try:
        raw = await call_ollama_json(
            messages,
            model=settings.agentic_retrieval_model,
            temperature=0.001,
        )
        data = json.loads(raw)
        query_structure, expanded_queries, key_terms, synonyms_dict = _parse_llm_retrieval_payload(data, user_query)
        _debug_dump_expansion("extract", query_structure, key_terms, synonyms_dict, expanded_queries)
        return query_structure, expanded_queries

    except Exception as exc:
        logger.warning("Query structure extraction failed: %s", exc)
        _retrieval_debug(f"extract+expand fallback after error: {exc}")
        fallback_structure = QueryStructure(
            original_query=user_query,
            process_name=user_query,
            domain="Unbekannt",
            procedure_type="Unbekannt",
        )
        return fallback_structure, [user_query]


async def generate_additional_synonyms(
    user_query: str,
    existing_query_structure: QueryStructure,
) -> Tuple[QueryStructure, List[str]]:
    """Run the follow-up synonym LLM call (higher temperature for more variation).

    Intended for the relevance retry loop: when too few chunks score high/medium,
    the graph may call this with the current ``QueryStructure``, then re-run retrieval
    with the returned ``expanded_queries``. On failure, logs a warning and returns
    ``existing_query_structure`` unchanged with ``[user_query]`` as the only expansion.
    """
    existing_terms: List[str] = []
    if existing_query_structure.key_terms:
        existing_terms.extend(existing_query_structure.key_terms)
    if existing_query_structure.synonyms:
        for term, synonyms in existing_query_structure.synonyms.items():
            existing_terms.append(term)
            existing_terms.extend(synonyms)

    existing_terms_str = ", ".join(existing_terms) if existing_terms else "keine"

    messages = [
        {
            "role": "system",
            "content": (
                "Du bist Experte für Prozessmanagement in der öffentlichen Verwaltung. "
                "Die erste Retrieval-Runde hatte zu wenige relevante Ergebnisse. "
                "Generiere JETZT alternative oder zusätzliche Synonyme und Schlüsselbegriffe.\n\n"
                "WICHTIG:\n"
                "- Verwende ANDERE Begriffe als beim ersten Mal\n"
                "- Fokussiere auf alternative Formulierungen, Fachbegriffe, Abkürzungen\n"
                "- Pro Schlüsselbegriff: 2-3 alternative Synonyme (nicht nur 1)\n"
                "- Denke an verwaltungsspezifische Terminologie, Gesetze, Verordnungen\n"
                "- Verwende auch englische Fachbegriffe, wenn relevant\n"
                "- Vermeide die bereits verwendeten Begriffe\n\n"
                "EXTRAHIERE (wie beim ersten Mal):\n"
                "- **process_name** (PFLICHT): Sprechbarer Prozessname\n"
                "- **domain** (PFLICHT): Verwaltungsbereich\n"
                "- **procedure_type** (PFLICHT): Verfahrensart\n"
                "- **perspective** (OPTIONAL): Nur wenn klar erkennbar\n"
                "- **granularity** (OPTIONAL): Nur wenn explizit genannt\n"
                "- **scope_start/scope_end** (OPTIONAL): Nur wenn klar definiert\n"
                "- **notes** (OPTIONAL): Wichtige Begriffe, Synonyme\n\n"
                "ERWEITERE um ALTERNATIVE Schlüsselbegriffe und Synonyme:\n"
                "1. Extrahiere 3-5 ALTERNATIVE Schlüsselbegriffe (keyTerms)\n"
                "2. Gib pro Schlüsselbegriff 2-3 alternative Synonyme aus\n"
                "3. Verwende andere Formulierungen als beim ersten Mal\n"
                "4. Fokussiere auf Fachbegriffe, Abkürzungen, verwaltungsspezifische Terminologie\n\n"
                "{\n"
                "  \"query_structure\": {\n"
                "    \"original_query\": \"...\",\n"
                "    \"process_name\": \"...\",\n"
                "    \"domain\": \"...\" oder null,\n"
                "    \"procedure_type\": \"...\" oder null,\n"
                "    \"perspective\": \"...\" oder null,\n"
                "    \"granularity\": \"...\" oder null,\n"
                "    \"scope_start\": \"...\" oder null,\n"
                "    \"scope_end\": \"...\" oder null,\n"
                "    \"notes\": \"...\" oder null\n"
                "  },\n"
                "  \"keyTerms\": [ string ],\n"
                "  \"synonyms\": {\n"
                "    \"term\": [ string ]\n"
                "  }\n"
                "}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original Query: {user_query}\n\n"
                f"Bereits verwendete Begriffe (vermeide diese):\n"
                f"- Key Terms: {existing_query_structure.key_terms or []}\n"
                f"- Synonyms: {existing_query_structure.synonyms or {}}\n"
                f"- Alle verwendeten Begriffe: {existing_terms_str}\n\n"
                "Generiere jetzt ALTERNATIVE Synonyme und Begriffe für eine zweite Retrieval-Runde. "
                "Verwende andere Formulierungen, Fachbegriffe und Abkürzungen."
            ),
        },
    ]

    if settings.debug:
        _retrieval_debug("re-retrieval synonym LLM call")
        logger.debug("Retrieval re-synonym messages: %s", messages)

    try:
        raw = await call_ollama_json(
            messages,
            model=settings.agentic_retrieval_model,
            temperature=0.3,
        )
        data = json.loads(raw)
        query_structure, expanded_queries, key_terms, synonyms_dict = _parse_llm_retrieval_payload(data, user_query)
        _debug_dump_expansion("re-retrieve", query_structure, key_terms, synonyms_dict, expanded_queries)
        return query_structure, expanded_queries

    except Exception as exc:
        logger.warning("Additional synonym generation failed: %s", exc)
        _retrieval_debug(f"re-synonym fallback: {exc}")
        return existing_query_structure, [user_query]
