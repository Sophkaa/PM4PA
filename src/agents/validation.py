"""Validation helper for validating BPMN models via the Ollama API."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

import json

from ..config import settings
from ..models.artifacts import (
    ValidationResultSetting4,
    MissingElement,
    HallucinatedElement,
    StructuralIssue,
)
from ..models.query_structure import QueryStructure
from ..infrastructure.api.ollama_client import call_ollama_json

logger = logging.getLogger(__name__)


def _format_chunk_source(metadata: Dict[str, Any], index: int) -> str:
    """
    Format chunk source line for prompts: document name plus page (German labels) when available.

    Returns:
        A string such as ``Dokument.pdf (Seite 3)`` or
        ``Dokument, Kapitel: …, Überschrift: …`` (German labels are intentional for LLM context).
    """
    def _get_document_name(meta: Dict[str, Any], idx: int) -> str:
        if meta:
            name = (
                meta.get('file_name') or
                meta.get('document_title') or
                meta.get('title') or
                (Path(meta.get('file_path', '')).name if meta.get('file_path') else None)
            )
            if name:
                if meta.get('page_number') is not None:
                    return f"{name} (Seite {meta.get('page_number')})"
                return name
        return f"Dokument {idx + 1}"
    
    doc_name = _get_document_name(metadata, index)
    source_parts = [doc_name]
    if metadata and metadata.get('chapter'):
        source_parts.append(f"Kapitel: {metadata.get('chapter')}")
    if metadata and metadata.get('heading'):
        source_parts.append(f"Überschrift: {metadata.get('heading')}")
    return ", ".join(source_parts)


async def run_validation_agent_setting4(
    bpmn_json: str,
    user_request: str,
    query_structure: Optional[QueryStructure] = None,
    retrieved_documents: Optional[List[str]] = None,
    retrieved_metadatas: Optional[List[Dict[str, Any]]] = None,
) -> ValidationResultSetting4:
    """
    Validate BPMN-JSON (nested) against User Query/QueryStructure and retrieved chunks (Setting 4).

    Args:
        bpmn_json: BPMN-JSON as string (nested format)
        user_request: User Prompt (for basic process attributes)
        query_structure: Optional structured query information
        retrieved_documents: Retrieved chunks
        retrieved_metadatas: Metadata for chunks

    Returns:
        ValidationResultSetting4 with structured feedback
    """
    # Build query structure information string
    query_info_parts = []
    if query_structure:
        query_info_parts.append(f"Prozessname: {query_structure.process_name}")
        query_info_parts.append(f"Domain: {query_structure.domain}")
        query_info_parts.append(f"Verfahrenstyp: {query_structure.procedure_type}")
        
        if query_structure.perspective:
            query_info_parts.append(f"Perspektive: {query_structure.perspective}")
        
        if query_structure.granularity:
            granularity_map = {
                "high_level": "High-Level (Überblick, Hauptschritte)",
                "medium": "Medium (moderate Detaillierung)",
                "detailed": "Detailed (sehr detailliert, kleinschrittig)"
            }
            granularity_desc = granularity_map.get(query_structure.granularity, query_structure.granularity)
            query_info_parts.append(f"Detaillierungsgrad: {granularity_desc}")
        
        if query_structure.scope_start or query_structure.scope_end:
            scope_parts = []
            if query_structure.scope_start:
                scope_parts.append(f"Start: {query_structure.scope_start}")
            if query_structure.scope_end:
                scope_parts.append(f"Ende: {query_structure.scope_end}")
            query_info_parts.append(f"Scope: {' → '.join(scope_parts)}")
        
        if query_structure.notes:
            query_info_parts.append(f"Zusätzliche Informationen: {query_structure.notes}")
    
    query_info = "\n".join(query_info_parts) if query_info_parts else ""

    retrieved_context = ""
    if retrieved_documents:
        metadatas = retrieved_metadatas if retrieved_metadatas else [{}] * len(retrieved_documents)
        context_parts = []
        for i, doc in enumerate(retrieved_documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            if meta.get('is_document_header', False):
                context_parts.append(doc)
            else:
                source_info = _format_chunk_source(meta, i)
                context_parts.append(f"[{source_info}]\n{doc}")
        retrieved_context = "\n\n".join(context_parts)

    user_prompt_short_parts = [
        "Dir liegen folgende Eingaben vor:\n\n"
        "1) Die User Query (definiert Zielprozess, Scope, Start- und Endpunkt, Perspektive und Flugebene)\n"
        "2) Die abgerufenen Chunks (einzige inhaltliche Quelle; das BPMN wurde daraus generiert)\n"
        "3) Ein generiertes BPMN-Prozessmodell als JSON\n\n"
        "Eingaben:\n\n"
        "--- USER QUERY ---\n"
        f"{user_request}\n\n"
    ]
    if retrieved_context:
        user_prompt_short_parts.append(f"--- ABGERUFENE CHUNKS ---\n{retrieved_context}\n\n")

    user_prompt_short_parts.append(
        f"--- Zu prüfender BPMN-JSON ---\n{bpmn_json}\n"
    )
    
    if query_info:
        user_prompt_short_parts.append(
            f"--- STRUKTURIERTE USERQUERY-INFORMATIONEN ---\n{query_info}\n\n"
        )
    ref_in = "in den Chunks"
    user_prompt_short_parts.append(
        f"Erstelle das JSON mit Inhalt wie im System-Prompt beschrieben. "
        f"Prüfe, ob alle auch kleine aber relevanten Elemente, die {ref_in} vorkommen, auch im BPMN-JSON vorkommen. "
        f"Prüfe, ob keine Elemente im BPMN vorkommen, die nicht {ref_in} stehen. "
        "Du musst IMMER ein assessment_statement zurückgeben.\n"
    )
    
    user_prompt_short = "".join(user_prompt_short_parts)

    format_explanation = (
        "\n WICHTIG - FORMAT (nested JSON):\n"
        "Das BPMN-JSON ist hierarchisch aufgebaut: pools[] → pro Pool: lanes[], startEvent, process[], endEvent.\n"
        "process[] enthält die Ablaufsequenz: tasks (type='task', name, documentation), events (type='event'), "
        "xor-Gateways (type='xor', condition, branches: [{label, branch: [...]}]).\n"
        "Gateways können verschachtelt sein (branch enthält weitere process-Elemente).\n"
        "Verwende das 'name'-Feld (Tasks/Events) bzw. 'condition' (Gateways) für element_label in deinem Feedback.\n"
        "Quellenangaben stehen im 'documentation'-Attribut.\n\n"
    )

    system_prompt_chunks = (
        "ROLLE: Du bist ein Validierungsagent für BPMN-Prozessmodelle.\n\n"
        "AUFGABE: Bei einem generiertes BPMN-Prozessmodell (JSON) zu überprüfen, ob alle Elemente aus den Chunks, die zum Zielprozess gehören, im BPMN enthalten sind.\n"
        f"{format_explanation}"
        "REGELN für die Validierung:\n"
        "- Du darfst das BPMN-Modell NICHT verändern und KEINE neuen Inhalte erfinden.\n\n"
        "- Du validierst das BPMN-JSON anhand von:\n"
        "- 1. Der User Query (definiert Prozessidentität, Scope, Start-/Endpunkt, Perspektive, Detailierungsgrad)\n"
        "- 2. Die Dokumentpassagen also Chunks (einzige inhaltliche Quelle; das BPMN wurde daraus generiert)\n\n"
        "- Die Chunks sind die EINZIGE Quelle für Prozesselemente.\n\n"
        "- Elemente aus den Chunks, die nicht zum Zielprozess gehören, werden ignoriert.\n"
        "- Wenn detailliertere Informationen NICHT in den Chunks vorhanden sind, dürfen sie auch nicht als fehlend gemeldet werden.\n"
        "- Keine impliziten Schritte, keine Verallgemeinerungen.\n\n"
        "--------------------------------------------------\n"
        "Verpflichtende Arbeitsweise\n\n"
        "1. Lies den Zielprozess und falls vorhanden Informationen wie Detailierungsgrad, Start- und Endpunkt, Perspektive, Scope."
        "2. Lies jeden einzelnen Chunk SEHR GENAU durch."
        "3. Schau dir den generierten Prozess-Draft GENAU an und prüfe folgendes.: "
        "Du prüfst ausschließlich:\n\n"
        "1) Vollständigkeit (alle Elemente aus den Chunks müssen im BPMN enthalten sein)\n"
        "   - Gibt es in anderen Dokumenten-Chunks Elemente, die nicht im BPMN enthalten sind?\n"
        "   - Fehlen Akteure, Tasks, Events oder Gateways, Objekte die explizit im Draft beschrieben sind?\n"
        "   - Falls etwas fehlt, beschreibe in der description, WAS fehlt, WO es genau, also in WELCHE LANE und zwischen welchen anderen Prozesselementen es eingefügt werden soll oder ob eine neue Lane oder Pool angelegt werden muss.\n"
        "   - Wenn Informationen nicht in den Chunks enthalten sind, dürfen sie NICHT als fehlend gemeldet werden.\n\n"
        "2) Korrektheit & Konsistenz inhaltlich sowie sequentiell mit den Chunks\n"
        "   - Sind alle BPMN-Elemente durch die Chunks belegbar?\n"
        "   - Keine Halluzinationen (Elemente im BPMN ohne Entsprechung in den Chunks), keine inhaltlichen Abweichungen.\n"
        "3) Logik des Prozessflows\n"
        "   - Ist der Kontrollfluss logisch korrekt?\n"
        "   - Sehr wichtig: Hat JEDES XOR Gateway eine Bedingung unter condition angegeben?\n"
        "   - Sind Entscheidungen sinnvoll als Gateways modelliert?\n\n"
        "   - Gibt BPMN Standard Verstöße bei der Bezeichnung von Elementen?\n\n"
        "--------------------------------------------------\n"
        "AUSGABEFORMAT\n\n"
        "- Gib AUSSCHLIESSLICH ein gültiges JSON zurück.\n"
        "- Jede Feststellung muss sich auf ein konkretes BPMN-Element beziehen\n"
        "  (element_type: z.B. 'task', 'gateway', 'event', 'pool', 'lane', element_label: Name aus dem BPMN-JSON, kurze Begründung, ggf. Draft-Referenz).\n\n"
        "Das JSON muss GENAU folgendem Schema entsprechen (leere Listen sind erlaubt, nur assessment_statement ist obligatorisch):\n"
        "{\n"
        "  \"assessment_statement\": \"<kurze Gesamteinschätzung (2-4 Sätze)>\",\n"
        "  \"missing_elements\": [\n"
        "    {\"element_type\": \"string\", \"element_label\": \"string\", \"source_chunk_reference\": \"string\", \"description\": \"string\"}\n"
        "  ],\n"
        "  \"hallucinated_elements\": [\n"
        "    {\"element_type\": \"string\", \"element_label\": \"string\", \"description\": \"string\"}\n"
        "  ],\n"
        "  \"structural_issues\": [\n"
        "    {\"issue_type\": \"string\", \"element_label\": \"string\", \"description\": \"string\"}\n"
        "  ],\n"
        "  \"overall_assessment\": {\"iteration_recommended\": true | false}\n"
        "}\n\n"
        "- structural_issues:\n"
        "   - gateway_missing / gateway_incorrect / gateway_missing_condition: Gateway(s) fehlt oder nicht korrekt modelliert\n"
        "   - flow_logic / sequence_error: Reihenfolge oder Kontrollfluss ist falsch\n"
        "   - naming_violation: Bezeichnungen sind nicht BPMN-konform\n\n"
        "- source_chunk_reference: Verwende das Quellenformat wie in den Chunks angezeigt, z.B. \"Dokument.pdf (Seite 3)\".\n"
        
        "Abschließende HINWEISE:\n"
        "Iteration: iteration_recommended = true, wenn mindestens ein Problem vorliegt.\n"
        "Falls nach sorgfältiger Prüfung keine Issues gefunden wurden, ist das auch okay."
        "Spiegle das im Assessment_statement und iteration_recommended = false."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt_chunks},
        {"role": "user", "content": user_prompt_short},
    ]



    
    raw = await call_ollama_json(
        messages=messages,
        model=settings.agentic_validation_model,
    )
    
    # Parse JSON response
    try:
        data = json.loads(raw)
        
        # Extract fields with defaults
        missing_elements = data.get("missing_elements", [])
        hallucinated_elements = data.get("hallucinated_elements", [])
        structural_issues = data.get("structural_issues", [])
        overall_assessment = data.get("overall_assessment", {})
        assessment_statement = data.get("assessment_statement", "")
        
        # Ensure assessment_statement is a string (handle case where LLM returns dict)
        if isinstance(assessment_statement, dict):
            # Try to extract meaningful text from dict, or convert to string
            if "text" in assessment_statement:
                assessment_statement = assessment_statement["text"]
            elif "statement" in assessment_statement:
                assessment_statement = assessment_statement["statement"]
            elif "description" in assessment_statement:
                assessment_statement = assessment_statement["description"]
            else:
                # Fallback: convert dict to string representation
                assessment_statement = str(assessment_statement)
        elif not isinstance(assessment_statement, str):
            # Convert any other type to string
            assessment_statement = str(assessment_statement) if assessment_statement else ""
        
        # Parse structured objects
        parsed_missing = []
        for item in missing_elements if isinstance(missing_elements, list) else []:
            if isinstance(item, dict):
                try:
                    parsed_missing.append(MissingElement(**item))
                except Exception:
                    parsed_missing.append(MissingElement(
                        element_type=item.get("element_type", "unknown"),
                        element_label=item.get("element_label", ""),
                        source_chunk_reference=item.get("source_chunk_reference", ""),
                        description=item.get("description", str(item))
                    ))
        
        parsed_hallucinated = []
        for item in hallucinated_elements if isinstance(hallucinated_elements, list) else []:
            if isinstance(item, dict):
                try:
                    parsed_hallucinated.append(HallucinatedElement(**item))
                except Exception:
                    parsed_hallucinated.append(HallucinatedElement(
                        element_type=item.get("element_type", "unknown"),
                        element_label=item.get("element_label", ""),
                        description=item.get("description", str(item))
                    ))
        
        parsed_structural_issues = []
        for item in structural_issues if isinstance(structural_issues, list) else []:
            if isinstance(item, dict):
                try:
                    parsed_structural_issues.append(StructuralIssue(**item))
                except Exception:
                    parsed_structural_issues.append(StructuralIssue(
                        issue_type=item.get("issue_type", "unknown"),
                        element_label=item.get("element_label", ""),
                        description=item.get("description", str(item))
                    ))
        
        # Ensure overall_assessment is a dict with iteration_recommended
        if not isinstance(overall_assessment, dict):
            overall_assessment = {}
        if "iteration_recommended" not in overall_assessment:
            # Auto-determine based on critical issues
            iteration_recommended = (
                len(parsed_missing) > 0 or
                len(parsed_hallucinated) > 0 or
                len(parsed_structural_issues) > 0
            )
            overall_assessment = {"iteration_recommended": iteration_recommended}
        
        return ValidationResultSetting4(
            missing_elements=parsed_missing,
            hallucinated_elements=parsed_hallucinated,
            structural_issues=parsed_structural_issues,
            overall_assessment=overall_assessment,
            assessment_statement=assessment_statement
        )
        
    except json.JSONDecodeError as e:
        logger.error("JSON parsing error in validation agent setting4: %s", e)
        logger.debug("Raw response: %s...", raw[:500])
        # Return default result with error
        return ValidationResultSetting4(
            missing_elements=[],
            hallucinated_elements=[],
            structural_issues=[],
            overall_assessment={"iteration_recommended": False},
            assessment_statement=f"Validation response could not be parsed as JSON: {str(e)}"
        )
    except Exception as e:
        logger.error("Error in validation agent setting4: %s", e)
        logger.debug(
            "Raw response: %s",
            raw[:500] if "raw" in locals() else "N/A",
        )
        # Return default result with error
        return ValidationResultSetting4(
            missing_elements=[],
            hallucinated_elements=[],
            structural_issues=[],
            overall_assessment={"iteration_recommended": False},
            assessment_statement=f"Validation error: {str(e)}"
        )


# ============================================================================
# Setting 5: Three Specialized Validators
# ============================================================================

BPMN_JSON_READING_HELP = (
    "--------------------------------------------------\n"
    "KURZE LESEHILFE FÜR DAS NESTED BPMN-JSON FORMAT\n\n"
    "- pools[] enthält die Swimlanes. Pro Pool: lanes[], startEvent, process[], endEvent.\n"
    "- process[] enthält die Ablaufsequenz: tasks (type='task', name, documentation), "
    "events (type='event'), xor-Gateways (type='xor', condition, branches: [{label, branch: [...]}]).\n"
    "- Gateways können verschachtelt sein (branch enthält weitere process-Elemente).\n"
    "- Verwende 'name' (Tasks/Events) bzw. 'condition' (Gateways) für element_label.\n"
    "- Quellenangaben im 'documentation'-Attribut.\n\n"
)


async def run_scope_completeness_validator_setting5(
    bpmn_json: str,
    retrieved_documents: List[str],
    user_request: str,
    query_structure: Optional[QueryStructure] = None,
    retrieved_metadatas: Optional[List[Dict[str, Any]]] = None
) -> ValidationResultSetting4:
    """
    Validator 1: scope and completeness.

    Checks whether the BPMN model matches the target process from the user query and whether
    relevant, explicitly mentioned chunk elements are missing from the model.

    Covers:
    - Scope violations: BPMN elements outside the target process
    - Missing elements: items explicitly in chunks and in scope but absent from BPMN

    Rules:
    - Report missing only when explicitly supported by chunks
    - No extra detail demands beyond chunk content
    - No logic scoring, hallucination checks, or source plausibility
    """
    # Build query structure information string
    query_info_parts = []
    if query_structure:
        query_info_parts.append(f"Prozessname: {query_structure.process_name}")
        query_info_parts.append(f"Domain: {query_structure.domain}")
        query_info_parts.append(f"Verfahrenstyp: {query_structure.procedure_type}")
        
        if query_structure.perspective:
            query_info_parts.append(f"Perspektive: {query_structure.perspective}")
        
        if query_structure.granularity:
            granularity_map = {
                "high_level": "High-Level (Überblick, Hauptschritte)",
                "medium": "Medium (moderate Detaillierung)",
                "detailed": "Detailed (sehr detailliert, kleinschrittig)"
            }
            granularity_desc = granularity_map.get(query_structure.granularity, query_structure.granularity)
            query_info_parts.append(f"Detaillierungsgrad: {granularity_desc}")
        
        if query_structure.scope_start or query_structure.scope_end:
            scope_parts = []
            if query_structure.scope_start:
                scope_parts.append(f"Start: {query_structure.scope_start}")
            if query_structure.scope_end:
                scope_parts.append(f"Ende: {query_structure.scope_end}")
            query_info_parts.append(f"Scope: {' → '.join(scope_parts)}")
        
        if query_structure.notes:
            query_info_parts.append(f"Zusätzliche Informationen: {query_structure.notes}")
    
    query_info = "\n".join(query_info_parts) if query_info_parts else ""
    
    # Build context from retrieved documents
    # Document headers: separators only; content chunks: [Document (page X)] format
    context_parts = []
    metadatas = retrieved_metadatas if retrieved_metadatas else [{}] * len(retrieved_documents)
    for i, doc in enumerate(retrieved_documents):
        meta = metadatas[i] if i < len(metadatas) else {}
        if meta.get('is_document_header', False):
            context_parts.append(doc)
        else:
            source_info = _format_chunk_source(meta, i)
            context_parts.append(f"[{source_info}]\n{doc}")
    
    retrieved_context = "\n\n".join(context_parts) if retrieved_documents else ""
    
    # Build user prompt
    user_prompt_parts = [
        "Dir liegt folgende Prozessbeschreibung zur Validierung auf Scope und Vollständigkeit vor:\n\n"
        "--- USER QUERY ---\n"
        f"{user_request}\n\n"
    ]
    
    if query_info:
        user_prompt_parts.append(
            f"--- STRUKTURIERTE USERQUERY-INFORMATIONEN ---\n{query_info}\n\n"
        )
    
    if retrieved_context:
        user_prompt_parts.append(
            f"--- ABGERUFENE CHUNKS ---\n{retrieved_context}\n\n"
        )
    
    user_prompt_parts.append(
        f"--- ZU PRÜFENDES BPMN-JSON MODELL ---\n{bpmn_json}\n\n"
        "DEINE AUFGABE:\n"
        "Prüfe AUSSCHLIESSLICH:\n"
        "1. Scope Violations: Gibt es Elemente im BPMN-Modell, die nicht zum Zielprozess aus der User Query gehören?\n"
        "2. Missing Elements: Fehlen Elemente, die explizit in den Chunks stehen und zum Zielprozess gehören?\n\n"
        "WICHTIG:\n"
        "- Nur Elemente als fehlend melden, die EXPLIZIT in den Chunks erwähnt werden und zum Ziel Prozess passen\n"
        "- Du musst IMMER ein assessment_statement zurückgeben.\n"
    )
    
    user_prompt = "".join(user_prompt_parts)
    
    system_prompt = (
        "Du bist ein Prozessexperte, der ein BPMN-JSON Modell "
        "auf Scope und Vollständigkeit prüft.\n\n"
        "--------------------------------------------------\n"
        "DEIN FOKUS\n\n"
        "Du prüfst NUR zwei Dinge:\n\n"
        "1. Scope Violations:\n"
        "   - Elemente im BPMN-Modell, die NICHT zum Zielprozess aus der User Query gehören\n\n"
        "2. Missing Elements:\n"
        "   - Prozesselemente oder Rollen, die EXPLIZIT in den Chunks erwähnt werden\n"
        "   - UND klar zum Zielprozess gehören\n"
        "   - ABER im BPMN-Modell fehlen\n\n"
        "--------------------------------------------------\n"
        "WAS DU NICHT PRÜFST\n\n"
        "- Keine Halluzinationen\n"
        "- Keine Quellenplausibilität\n"
        "- Keine Prozesslogik\n"
        "- Keine Flow-Korrektheit\n"
        "- Keine Gateway-Modellierung\n\n"
        "--------------------------------------------------\n"
        "🔧 VERPFLICHTENDE ARBEITSREIHENFOLGE\n\n"
        "1. Lese den Zielprozess und die vorhandenen Informationen an."
        "2. Lies die Chunks durch."
        "3. Schau dir das generierte BPMN-JSON Modell GENAU an."
        "4. Identifiziere ALLE Scope Violations.\n"
        "5. Identifiziere ALLE Missing Elements auf Basis der Chunks.\n"
        "6. Trage JEDES identifizierte Problem strukturiert ein:\n"
        "   - Fehlende Elemente → missing_elements\n"
        "7. Schreibe ERST DANACH das assessment_statement\n"
        "   als Zusammenfassung dieser Issues.\n\n"
        "Es ist VERBOTEN, fehlende oder falsche Elemente\n"
        "nur im assessment_statement zu erwähnen,\n"
        "ohne sie strukturiert zu melden.\n\n"
        + BPMN_JSON_READING_HELP +
        "--------------------------------------------------\n"
        "REGELN\n\n"
        "- Fehlende Elemente NUR melden,\n"
        "  wenn sie EXPLIZIT in den Chunks genannt sind\n"
        "- Fehlende Rollen NUR melden,\n"
        "  wenn sie EXPLIZIT in den Chunks genannt sind\n"
        "- KEINE Ableitungen, KEINE Annahmen\n"
        "- Keine Detailforderungen, wenn nicht in den Chunks vorhanden\n"
        "- Elemente aus Chunks, die nicht zum Zielprozess gehören,\n"
        "  werden ignoriert\n"
        "  Wenn Elemente fehlen, NENNE auch EXPLIZIT in der description, in welcher Lane und an welcher Stelle das Element hinzugefügt werden sollte."
        "- Im Zweifel MELDEN statt zu schweigen\n\n"
        "--------------------------------------------------\n"
        "AUSGABEFORMAT\n\n"
        "Gib AUSSCHLIESSLICH ein gültiges JSON zurück:\n\n"
        "{\n"
        "  \"assessment_statement\": \"<kurze Gesamteinschätzung (2–4 Sätze)>\",\n"
        "  \"missing_elements\": [\n"
        "    {\n"
        "      \"element_type\": \"string\",\n"
        "      \"element_label\": \"string\",\n"
        "      \"source_chunk_reference\": \"string\",\n"
        "      \"description\": \"string\"\n"
        "    }\n"
        "  ],\n"
        "  \"hallucinated_elements\": [],\n"
        "  \"structural_issues\": [],\n"
        "  \"overall_assessment\": { \"iteration_recommended\": true | false }\n"
        "}\n\n"
        "--------------------------------------------------\n"
        "🔧 KOPPLUNGSREGELN\n\n"
        "- hallucinated_elements und structural_issues MÜSSEN leer sein.\n"
        "- missing_elements darf NUR Probleme aus diesem Fokus enthalten.\n"
        "- Wenn du im assessment_statement ein Problem erwähnst,\n"
        "  MUSS dafür ein Eintrag in missing_elements existieren.\n"
        "- iteration_recommended = true,\n"
        "  sobald mindestens eine der beiden Listen nicht leer ist.\n"
        "- Gib IMMER ein assessment_statement zurück.\n"
    )

    system_prompt_bpmn_text_vorherige_version = (
        "Du bist ein Prozessexperte, der ein BPMN-JSON Modell "
        "auf Scope und Vollständigkeit prüft.\n\n"
        "--------------------------------------------------\n"
        "DEIN FOKUS\n\n"
        "Du prüfst NUR zwei Dinge:\n\n"
        "1. Scope Violations:\n"
        "   - Elemente im BPMN-Modell, die NICHT zum Zielprozess aus der User Query gehören\n\n"
        "2. Missing Elements:\n"
        "   - Prozesselemente oder Rollen, die EXPLIZIT in den Chunks erwähnt werden\n"
        "   - UND klar zum Zielprozess gehören\n"
        "   - ABER im BPMN-Modell fehlen\n\n"
        "--------------------------------------------------\n"
        "WAS DU NICHT PRÜFST\n\n"
        "- Keine Halluzinationen\n"
        "- Keine Quellenplausibilität\n"
        "- Keine Prozesslogik\n"
        "- Keine Flow-Korrektheit\n"
        "- Keine Gateway-Modellierung\n\n"
        "--------------------------------------------------\n"
        "🔧 VERPFLICHTENDE ARBEITSREIHENFOLGE\n\n"
        "1. Lese den Zielprozess und die vorhandenen Informationen.\n"
        "2. Lies die Chunks durch.\n"
        "3. Schau dir das generierte BPMN-JSON Modell GENAU an.\n"
        "4. Identifiziere ALLE Scope Violations.\n"
        "5. Identifiziere ALLE Missing Rollen und Elements auf Basis der Chunks.\n"
        "6. Trage JEDES identifizierte Problem strukturiert ein:\n"
        "   - Fehlende Elemente → missing_elements\n"
        "7. Schreibe ERST DANACH das assessment_statement\n"
        "   als Zusammenfassung dieser Issues.\n\n"
        "Es ist VERBOTEN, fehlende oder falsche Elemente\n"
        "nur im assessment_statement zu erwähnen,\n"
        "ohne sie strukturiert zu melden.\n\n"
        + BPMN_JSON_READING_HELP +
        "--------------------------------------------------\n"
        "REGELN\n\n"
        "- Fehlende Elemente NUR melden,\n"
        "  wenn sie EXPLIZIT in den Chunks genannt sind\n"
        "- Fehlende Rollen NUR melden,\n"
        "  wenn sie EXPLIZIT in den Chunks genannt sind\n"
        "- KEINE Ableitungen, KEINE Annahmen\n"
        "- Keine Detailforderungen, wenn nicht in den Chunks vorhanden\n"
        "- Elemente aus Chunks, die nicht zum Zielprozess gehören,\n"
        "  werden ignoriert\n"
        "  Wenn Elemente fehlen, NENNE auch EXPLIZIT in der description, in welcher Lane und an welcher Stelle das Element hinzugefügt werden sollte."
        "- Im Zweifel MELDEN statt zu schweigen\n\n"
        "--------------------------------------------------\n"
        "AUSGABEFORMAT\n\n"
        "Gib AUSSCHLIESSLICH ein gültiges JSON zurück:\n\n"
        "{\n"
        "  \"assessment_statement\": \"<kurze Gesamteinschätzung (2–4 Sätze)>\",\n"
        "  \"missing_elements\": [\n"
        "    {\n"
        "      \"element_type\": \"string\",\n"
        "      \"element_label\": \"string\",\n"
        "      \"source_chunk_reference\": \"string\",\n"
        "      \"description\": \"string\"\n"
        "    }\n"
        "  ],\n"
        "  \"hallucinated_elements\": [],\n"
        "  \"structural_issues\": [],\n"
        "  \"overall_assessment\": { \"iteration_recommended\": true | false }\n"
        "}\n\n"
        "--------------------------------------------------\n"
        "🔧 KOPPLUNGSREGELN\n\n"
        "- hallucinated_elements und structural_issues MÜSSEN leer sein.\n"
        "- missing_elements darf NUR Probleme aus diesem Fokus enthalten.\n"
        "- Wenn du im assessment_statement ein Problem erwähnst,\n"
        "  MUSS dafür ein Eintrag in missing_elements existieren.\n"
        "- iteration_recommended = true,\n"
        "  sobald mindestens eine der beiden Listen nicht leer ist.\n"
        "- Gib IMMER ein assessment_statement zurück.\n"
    )

    system_prompt_bpmn_text = (
        "ROLLE: Du bist ein Prozessexperte, der ein BPMN-JSON Modell auf Scope und Vollständigkeit prüft.\n\n"
        "AUFGABE: Bei einem generiertes BPMN-Prozessmodell (JSON) zu überprüfen, ob alle Elemente aus den Chunks, die zum Zielprozess gehören, im BPMN enthalten sind und ob es Elemente im BPMN gibt, die nicht zum Zielprozess gehören.\n"
        + BPMN_JSON_READING_HELP +
        "REGELN für die Validierung:\n"
        "- Du darfst das BPMN-Modell NICHT verändern und KEINE neuen Inhalte erfinden.\n\n"
        "- Du validierst das BPMN-JSON anhand von:\n"
        "- 1. Der User Query (definiert Prozessidentität, Scope, Start-/Endpunkt, Perspektive, Detailierungsgrad)\n"
        "- 2. Die Dokumentpassagen also Chunks (einzige inhaltliche Quelle; das BPMN wurde daraus generiert)\n\n"
        "- Die Chunks sind die EINZIGE Quelle für Prozesselemente.\n\n"
        "- Elemente aus den Chunks, die nicht zum Zielprozess gehören, werden ignoriert.\n"
        "- Wenn detailliertere Informationen NICHT in den Chunks vorhanden sind, dürfen sie auch nicht als fehlend gemeldet werden.\n"
        "- Keine impliziten Schritte, keine Verallgemeinerungen.\n\n"
        "- Fehlende Elemente NUR melden, wenn sie EXPLIZIT in den Chunks genannt sind\n"
        "- Fehlende Rollen NUR melden, wenn sie EXPLIZIT in den Chunks genannt sind\n"
        "- KEINE Ableitungen, KEINE Annahmen\n"
        "  Wenn Elemente fehlen, NENNE auch EXPLIZIT in der description, in welcher Lane und an welcher Stelle das Element hinzugefügt werden sollte."
        "--------------------------------------------------\n"
        "Verpflichtende Arbeitsweise\n\n"
        "1. Lies den Zielprozess und falls vorhanden Informationen wie Detailierungsgrad, Start- und Endpunkt, Perspektive, Scope."
        "2. Lies jeden einzelnen Chunk SEHR GENAU durch."
        "3. Schau dir den generierten Prozess-Draft GENAU an und prüfe folgendes.: "
        "Du prüfst ausschließlich:\n\n"
        "Vollständigkeit (alle Elemente aus den Chunks müssen im BPMN enthalten sein)\n"
        "   - Gibt es in anderen Dokumenten-Chunks Elemente, die nicht im BPMN enthalten sind?\n"
        "   - Fehlen Akteure, Tasks, Events oder Gateways, Objekte die explizit im Draft beschrieben sind?\n"
        "   - Falls etwas fehlt, beschreibe in der description, WAS fehlt, WO es genau, also in WELCHE LANE und zwischen welchen anderen Prozesselementen es eingefügt werden soll oder ob eine neue Lane oder Pool angelegt werden muss.\n"
        "   - Wenn Informationen nicht in den Chunks enthalten sind, dürfen sie NICHT als fehlend gemeldet werden.\n\n"
       "--------------------------------------------------\n"
        "AUSGABEFORMAT\n\n"
        "- Gib AUSSCHLIESSLICH ein gültiges JSON zurück.\n"
        "- Jede Feststellung muss sich auf ein konkretes BPMN-Element beziehen\n"
        "  (element_type: z.B. 'task', 'gateway', 'event', 'pool', 'lane', element_label: Name aus dem BPMN-JSON, kurze Begründung, ggf. Draft-Referenz).\n\n"
        "Das JSON muss GENAU folgendem Schema entsprechen (leere Listen sind erlaubt, nur assessment_statement ist obligatorisch):\n"
        "{\n"
        "  \"assessment_statement\": \"<kurze Gesamteinschätzung (2–4 Sätze)>\",\n"
        "  \"missing_elements\": [\n"
        "    {\n"
        "      \"element_type\": \"string\",\n"
        "      \"element_label\": \"string\",\n"
        "      \"source_chunk_reference\": \"string\",\n"
        "      \"description\": \"string\"\n"
        "    }\n"
        "  ],\n"
        "  \"hallucinated_elements\": [],\n"
        "  \"structural_issues\": [],\n"
        "  \"overall_assessment\": { \"iteration_recommended\": true | false }\n"
        "}\n\n"
        "Abschließende HINWEISE:\n"
        "- hallucinated_elements und structural_issues MÜSSEN leer sein.\n"
        "- missing_elements darf NUR Probleme aus diesem Fokus enthalten.\n"
        "Iteration: iteration_recommended = true, wenn mindestens ein Problem vorliegt.\n"
        "Falls nach sorgfältiger Prüfung keine Issues gefunden wurden, ist das auch okay."
        "Spiegle das im Assessment_statement und iteration_recommended = false."
    )
    
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt_bpmn_text},
        {"role": "user", "content": user_prompt},
    ]

   
    
    raw = await call_ollama_json(
        messages=messages,
        model=settings.agentic_validation_model,
    )
    
    # Parse JSON response
    try:
        data = json.loads(raw)
        
        # Extract fields with defaults
        missing_elements = data.get("missing_elements", [])
        overall_assessment = data.get("overall_assessment", {})
        assessment_statement = data.get("assessment_statement", "")
        
        # Ensure assessment_statement is a string
        if isinstance(assessment_statement, dict):
            assessment_statement = assessment_statement.get("text", assessment_statement.get("statement", str(assessment_statement)))
        elif not isinstance(assessment_statement, str):
            assessment_statement = str(assessment_statement) if assessment_statement else ""
        
        # Parse structured objects
        parsed_missing = []
        for item in missing_elements if isinstance(missing_elements, list) else []:
            if isinstance(item, dict):
                try:
                    parsed_missing.append(MissingElement(**item))
                except Exception:
                    parsed_missing.append(MissingElement(
                        element_type=item.get("element_type", "unknown"),
                        element_label=item.get("element_label", ""),
                        source_chunk_reference=item.get("source_chunk_reference", ""),
                        description=item.get("description", str(item))
                    ))
        
        # Ensure overall_assessment
        if not isinstance(overall_assessment, dict):
            overall_assessment = {}
        if "iteration_recommended" not in overall_assessment:
            iteration_recommended = len(parsed_missing) > 0
            overall_assessment = {"iteration_recommended": iteration_recommended}
        
        return ValidationResultSetting4(
            missing_elements=parsed_missing,
            hallucinated_elements=[],
            structural_issues=[],
            overall_assessment=overall_assessment,
            assessment_statement=assessment_statement
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in scope/completeness validator: {e}")
        return ValidationResultSetting4(
            missing_elements=[],
            hallucinated_elements=[],
            structural_issues=[],
            overall_assessment={"iteration_recommended": False},
            assessment_statement=f"Validation response could not be parsed as JSON: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in scope/completeness validator: {e}")
        return ValidationResultSetting4(
            missing_elements=[],
            hallucinated_elements=[],
            structural_issues=[],
            overall_assessment={"iteration_recommended": False},
            assessment_statement=f"Validation error: {str(e)}"
        )


async def run_factual_fidelity_validator_setting5(
    bpmn_json: str,
    retrieved_documents: List[str],
    user_request: str,
    query_structure: Optional[QueryStructure] = None,
    retrieved_metadatas: Optional[List[Dict[str, Any]]] = None
) -> ValidationResultSetting4:
    """
    Validator 2: factual fidelity.

    Checks for invented elements or content not supported by the retrieved chunks.

    Covers:
    - Hallucinated elements: BPMN items that cannot be justified or diverge strongly from chunks

    Rules:
    - No missing-element checks
    - No flow/logic checks
    """
    # Build query structure information string
    query_info_parts = []
    if query_structure:
        query_info_parts.append(f"Prozessname: {query_structure.process_name}")
        query_info_parts.append(f"Domain: {query_structure.domain}")
        query_info_parts.append(f"Verfahrenstyp: {query_structure.procedure_type}")
        
        if query_structure.perspective:
            query_info_parts.append(f"Perspektive: {query_structure.perspective}")
        
        if query_structure.granularity:
            granularity_map = {
                "high_level": "High-Level (Überblick, Hauptschritte)",
                "medium": "Medium (moderate Detaillierung)",
                "detailed": "Detailed (sehr detailliert, kleinschrittig)"
            }
            granularity_desc = granularity_map.get(query_structure.granularity, query_structure.granularity)
            query_info_parts.append(f"Detaillierungsgrad: {granularity_desc}")
        
        if query_structure.scope_start or query_structure.scope_end:
            scope_parts = []
            if query_structure.scope_start:
                scope_parts.append(f"Start: {query_structure.scope_start}")
            if query_structure.scope_end:
                scope_parts.append(f"Ende: {query_structure.scope_end}")
            query_info_parts.append(f"Scope: {' → '.join(scope_parts)}")
        
        if query_structure.notes:
            query_info_parts.append(f"Zusätzliche Informationen: {query_structure.notes}")
    
    query_info = "\n".join(query_info_parts) if query_info_parts else ""
    
    # Build context from retrieved documents
    # Document headers: separators only; content chunks: [Document (page X)] format
    context_parts = []
    metadatas = retrieved_metadatas if retrieved_metadatas else [{}] * len(retrieved_documents)
    for i, doc in enumerate(retrieved_documents):
        meta = metadatas[i] if i < len(metadatas) else {}
        if meta.get('is_document_header', False):
            context_parts.append(doc)
        else:
            source_info = _format_chunk_source(meta, i)
            context_parts.append(f"[{source_info}]\n{doc}")
    
    retrieved_context = "\n\n".join(context_parts) if retrieved_documents else ""
    
    # Build user prompt
    user_prompt_parts = [
  
        "Dir liegt folgende Prozessbeschreibung zur Validierung auf Faktische Treue vor:\n\n"
        "--- USER QUERY ---\n"
        f"{user_request}\n\n"
    ]
    
    if query_info:
        user_prompt_parts.append(
            f"--- STRUKTURIERTE USERQUERY-INFORMATIONEN ---\n{query_info}\n\n"
        )
    
    if retrieved_context:
        user_prompt_parts.append(
            f"--- ABGERUFENE CHUNKS ---\n{retrieved_context}\n\n"
        )
    
    user_prompt_parts.append(
        f"--- ZU PRÜFENDES BPMN-JSON MODELL ---\n{bpmn_json}\n\n"
        "DEINE AUFGABE:\n"
        "Prüfe AUSSCHLIESSLICH:\n"
        "1. Hallucinated Elements: Gibt es Elemente im BPMN-Modell, die durch die Chunks nicht belegbar sind oder semantisch stark von den Chunks abweichen?\n"
        "   Es muss nicht wörtlich im Chunk stehen, sondern es reicht wenn es semantisch und logisch übereinstimmt.\n"
        "- Du musst zudem IMMER ein assessment_statement zurückgeben.\n"
    )
    
    user_prompt = "".join(user_prompt_parts)
    
    system_prompt = (
        "Du bist ein Prozessexperte, der ein BPMN-JSON Modell auf faktische Treue prüft.\n\n"
        "--------------------------------------------------\n"
        "DEIN FOKUS\n\n"
        "Du prüfst NUR Hallucinated Elements:\n"
        "- Prozesselemente im BPMN-Modell, die nicht durch die Chunks belegbar sind\n"
        "- oder semantisch DEUTLICH von den Chunks abweichen\n\n"
        "HINWEIS: Ein Element gilt als durch einen Chunk gedeckt, wenn die beschriebene Handlung oder Information "
        "inhaltlich äquivalent oder klar implizit im Chunk enthalten ist, auch wenn die Formulierung nicht wortgleich ist.\n\n"
        "--------------------------------------------------\n"
        "WAS DU NICHT PRÜFST\n\n"
        "- Keine Missing Elements\n"
        "- Keine Scope Violations\n"
        "- Keine Flow-Logik\n"
        "- Keine Gateway-Modellierung\n\n"
        "--------------------------------------------------\n"
        " VERPFLICHTENDE ARBEITSREIHENFOLGE\n\n"
        "1. Lies den Zielprozess und die vorhandenen Informationen an."
        "2. Lies die Chunks durch."
        "3. Schau dir das generierte BPMN-JSON Modell GENAU an."
        "4. Überprüfe für jedes Element (Tasks, Events), ob die Dokumentation/Quelle inhaltlich zum Element passt. "
        "Quellenangaben stehen im 'documentation'-Attribut. Ein Element gilt als gedeckt, wenn die Handlung "
        "inhaltlich äquivalent oder implizit im Chunk enthalten ist."
        "5. Falls das Prozesselement durch keinen Chunk belegbar ist, melde es als hallucinated_element."
        "6. Schreibe ERST DANACH das assessment_statement als Zusammenfassung dieser Issues.\n\n"
        "Es ist VERBOTEN, Probleme im assessment_statement zu erwähnen,\n"
        "ohne sie als hallucinated_element oder structural_issue zu melden.\n\n"
        + BPMN_JSON_READING_HELP +
        "--------------------------------------------------\n"
        "REGELN\n\n"
        "- Im Zweifel MELDEN statt zu schweigen\n\n"
        "--------------------------------------------------\n"
        "AUSGABEFORMAT\n\n"
        "Gib AUSSCHLIESSLICH ein gültiges JSON zurück:\n\n"
        "{\n"
        "  \"assessment_statement\": \"<kurze Gesamteinschätzung (2–4 Sätze)>\",\n"
        "  \"missing_elements\": [],\n"
        "  \"hallucinated_elements\": [\n"
        "    {\n"
        "      \"element_type\": \"string\",\n"
        "      \"element_label\": \"string\",\n"
        "      \"description\": \"string\"\n"
        "    }\n"
        "  ],\n"
        "  \"overall_assessment\": { \"iteration_recommended\": true | false }\n"
        "}\n\n"
        "- iteration_recommended = true, sobald hallucinated_elements nicht leer ist.\n"
        "- Gib IMMER ein assessment_statement zurück.\n"
    )
    
    system_prompt_bpmn_text_vorherige_version = (
        "Du bist ein Prozessexperte, der ein BPMN-JSON Modell auf faktische Treue prüft.\n\n"
        "--------------------------------------------------\n"
        "DEIN FOKUS\n\n"
        "Du prüfst NUR Hallucinated Elements:\n"
        "- Prozesselemente im BPMN-Modell, die nicht durch die Chunks belegbar sind\n"
        "- oder semantisch DEUTLICH von den Chunks abweichen\n\n"
        "HINWEIS: Ein Element gilt als durch einen Chunk gedeckt, wenn die beschriebene Handlung oder Information "
        "inhaltlich äquivalent oder klar implizit im Chunk enthalten ist, auch wenn die Formulierung nicht wortgleich ist.\n\n"
        "--------------------------------------------------\n"
        "WAS DU NICHT PRÜFST\n\n"
        "- Keine Missing Elements\n"
        "- Keine Scope Violations\n"
        "- Keine Flow-Logik\n"
        "- Keine Gateway-Modellierung\n\n"
        "--------------------------------------------------\n"
        " VERPFLICHTENDE ARBEITSREIHENFOLGE\n\n"
        "1. Lies den Zielprozess und die vorhandenen Informationen."
        "2. Lies die Chunks durch."
        "3. Schau dir das generierte BPMN-JSON Modell GENAU an."
        "4. Überprüfe für jedes Element, ob die Dokumentation (documentation-Attribut) inhaltlich zum Element passt."
        "5. Falls das Prozesselement durch keinen Chunk belegbar ist, melde es als hallucinated_element."
        "6. Schreibe ERST DANACH das assessment_statement als Zusammenfassung dieser Issues.\n\n"
        "Es ist VERBOTEN, Probleme im assessment_statement zu erwähnen,\n"
        "ohne sie als hallucinated_element oder structural_issue zu melden.\n\n"
        + BPMN_JSON_READING_HELP +
        "--------------------------------------------------\n"
        "REGELN\n\n"
        "- Quellen werden NICHT formal, sondern INHALTLICH geprüft\n"
        "- Im Zweifel MELDEN statt zu schweigen\n\n"
        "--------------------------------------------------\n"
        "AUSGABEFORMAT\n\n"
        "Gib AUSSCHLIESSLICH ein gültiges JSON zurück:\n\n"
        "{\n"
        "  \"assessment_statement\": \"<kurze Gesamteinschätzung (2–4 Sätze)>\",\n"
        "  \"missing_elements\": [],\n"
        "  \"hallucinated_elements\": [\n"
        "    {\n"
        "      \"element_type\": \"string\",\n"
        "      \"element_label\": \"string\",\n"
        "      \"description\": \"string\"\n"
        "    }\n"
        "  ],\n"
        "  \"overall_assessment\": { \"iteration_recommended\": true | false }\n"
        "}\n\n"
        "- iteration_recommended = true, sobald hallucinated_elements nicht leer ist.\n"
        "- Gib IMMER ein assessment_statement zurück.\n"
    )
    

    system_prompt_bpmn_text = (
        "ROLLE: Du bist ein Prozessexperte, der ein BPMN-JSON Modell auf faktische Treue prüft.\n\n"
        "AUFGABE: Bei einem generiertes BPMN-Prozessmodell (JSON) zu überprüfen, ob es Elemente im BPMN gibt, die nicht in den Chunks vorhanden sind oder semantisch stark von den Chunks abweichen.\n"
        + BPMN_JSON_READING_HELP +
        "REGELN für die Validierung:\n"
        "- Du darfst das BPMN-Modell NICHT verändern und KEINE neuen Inhalte erfinden.\n\n"
        "- Du validierst das BPMN-JSON anhand von:\n"
        "- 1. Der User Query (definiert Prozessidentität, Scope, Start-/Endpunkt, Perspektive, Detailierungsgrad)\n"
        "- 2. Die Dokumentpassagen also Chunks (einzige inhaltliche Quelle; das BPMN wurde daraus generiert)\n\n"
        "- Die Chunks sind die EINZIGE Quelle für Prozesselemente.\n\n"
        "- Quellen werden NICHT formal, sondern INHALTLICH geprüft\n"
        "- Im Zweifel MELDEN statt zu schweigen\n\n"
        "- KEINE Ableitungen, KEINE Annahmen\n"
        "- Wenn Elemente halluziniert sind, NENNE auch EXPLIZIT in der description, in welcher Lane und an welcher Stelle das Element entfernt werden sollte."
        "--------------------------------------------------\n"
        "Verpflichtende Arbeitsweise\n\n"
        "1. Lies den Zielprozess und falls vorhanden Informationen wie Detailierungsgrad, Start- und Endpunkt, Perspektive, Scope."
        "2. Lies jeden einzelnen Chunk SEHR GENAU durch."
        "3. Schau dir den generierten das generierte BPMN-JSON Modell GENAU an und prüfe folgendes: "
        "1) Überprüfe für jedes Element, ob die Dokumentation (documentation-Attribut) inhaltlich zum Element passt."
        "2) Überprüfe Korrektheit & Konsistenz inhaltlich sowie sequentiell mit den Chunks\n"
        "   - Sind alle BPMN-Elemente durch die Chunks belegbar?\n"
        "   - Keine Halluzinationen (Elemente im BPMN ohne Entsprechung in den Chunks), keine inhaltlichen Abweichungen.\n"
        "   - Gibt es Prozesselemente im BPMN-Modell, die nicht durch die Chunks belegbar sind oder semantisch DEUTLICH von den Chunks abweichen\n\n"
        "HINWEIS: Ein Element gilt als durch einen Chunk gedeckt, wenn die beschriebene Handlung oder Information "
        "inhaltlich äquivalent oder klar implizit im Chunk enthalten ist, auch wenn die Formulierung nicht wortgleich ist.\n\n"
       
       "--------------------------------------------------\n"
        "AUSGABEFORMAT\n\n"
        "- Gib AUSSCHLIESSLICH ein gültiges JSON zurück.\n"
        "- Jede Feststellung muss sich auf ein konkretes BPMN-Element beziehen\n"
        "  (element_type: z.B. 'task', 'gateway', 'event', 'pool', 'lane', element_label: Name aus dem BPMN-JSON, kurze Begründung, ggf. Draft-Referenz).\n\n"
        "Das JSON muss GENAU folgendem Schema entsprechen (leere Listen sind erlaubt, nur assessment_statement ist obligatorisch):\n"
        "{\n"
        "  \"assessment_statement\": \"<kurze Gesamteinschätzung (2–4 Sätze)>\",\n"
        "  \"missing_elements\": [],\n"
        "  \"hallucinated_elements\": [\n"
        "    {\n"
        "      \"element_type\": \"string\",\n"
        "      \"element_label\": \"string\",\n"
        "      \"description\": \"string\"\n"
        "    }\n"
        "  ],\n"
        "  \"overall_assessment\": { \"iteration_recommended\": true | false }\n"
        "}\n\n"
        
        "Abschließende HINWEISE:\n"
        "- missing_elements und structural_issues MÜSSEN leer sein.\n"
        "- hallucinated_elements darf NUR Probleme aus diesem Fokus enthalten.\n"
        "Iteration: iteration_recommended = true, wenn mindestens ein Problem vorliegt.\n"
        "Falls nach sorgfältiger Prüfung keine Issues gefunden wurden, ist das auch okay."
        "Spiegle das im Assessment_statement und iteration_recommended = false."
    )
    
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt_bpmn_text},
        {"role": "user", "content": user_prompt},
    ]

    

    raw = await call_ollama_json(
        messages=messages,
        model=settings.agentic_validation_model,
    )
    
    # Parse JSON response
    try:
        data = json.loads(raw)
        
        # Extract fields with defaults (structural_issues not parsed - source annotation not validated)
        hallucinated_elements = data.get("hallucinated_elements", [])
        overall_assessment = data.get("overall_assessment", {})
        assessment_statement = data.get("assessment_statement", "")
        
        # Ensure assessment_statement is a string
        if isinstance(assessment_statement, dict):
            assessment_statement = assessment_statement.get("text", assessment_statement.get("statement", str(assessment_statement)))
        elif not isinstance(assessment_statement, str):
            assessment_statement = str(assessment_statement) if assessment_statement else ""
        
        # Parse hallucinated elements only
        parsed_hallucinated = []
        for item in hallucinated_elements if isinstance(hallucinated_elements, list) else []:
            if isinstance(item, dict):
                try:
                    parsed_hallucinated.append(HallucinatedElement(**item))
                except Exception:
                    parsed_hallucinated.append(HallucinatedElement(
                        element_type=item.get("element_type", "unknown"),
                        element_label=item.get("element_label", ""),
                        description=item.get("description", str(item))
                    ))
        
        # Ensure overall_assessment (iteration only based on hallucinated_elements)
        if not isinstance(overall_assessment, dict):
            overall_assessment = {}
        if "iteration_recommended" not in overall_assessment:
            overall_assessment = {"iteration_recommended": len(parsed_hallucinated) > 0}
        
        return ValidationResultSetting4(
            missing_elements=[],
            hallucinated_elements=parsed_hallucinated,
            structural_issues=[],  # Validator 2 does not check source annotation
            overall_assessment=overall_assessment,
            assessment_statement=assessment_statement
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in factual fidelity validator: {e}")
        return ValidationResultSetting4(
            missing_elements=[],
            hallucinated_elements=[],
            structural_issues=[],
            overall_assessment={"iteration_recommended": False},
            assessment_statement=f"Validation response could not be parsed as JSON: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in factual fidelity validator: {e}")
        return ValidationResultSetting4(
            missing_elements=[],
            hallucinated_elements=[],
            structural_issues=[],
            overall_assessment={"iteration_recommended": False},
            assessment_statement=f"Validation error: {str(e)}"
        )


async def run_process_logic_validator_setting5(
    bpmn_json: str,
    retrieved_documents: List[str],
    user_request: str,
    query_structure: Optional[QueryStructure] = None,
    retrieved_metadatas: Optional[List[Dict[str, Any]]] = None
) -> ValidationResultSetting4:
    """
    Validator 3: process logic and modeling quality.

    Checks whether the flow is logically consistent and whether decisions and parallelism are
    modeled with appropriate gateways.

    Covers:
    - Structural issues: inconsistent order, incorrect flow, missing gateways when branching is obvious

    Rules:
    - No missing-element findings
    - No hallucination findings
    - No source-plausibility scoring
    """
    # Build query structure information string
    query_info_parts = []
    if query_structure:
        query_info_parts.append(f"Prozessname: {query_structure.process_name}")
        query_info_parts.append(f"Domain: {query_structure.domain}")
        query_info_parts.append(f"Verfahrenstyp: {query_structure.procedure_type}")
        
        if query_structure.perspective:
            query_info_parts.append(f"Perspektive: {query_structure.perspective}")
        
        if query_structure.granularity:
            granularity_map = {
                "high_level": "High-Level (Überblick, Hauptschritte)",
                "medium": "Medium (moderate Detaillierung)",
                "detailed": "Detailed (sehr detailliert, kleinschrittig)"
            }
            granularity_desc = granularity_map.get(query_structure.granularity, query_structure.granularity)
            query_info_parts.append(f"Detaillierungsgrad: {granularity_desc}")
        
        if query_structure.scope_start or query_structure.scope_end:
            scope_parts = []
            if query_structure.scope_start:
                scope_parts.append(f"Start: {query_structure.scope_start}")
            if query_structure.scope_end:
                scope_parts.append(f"Ende: {query_structure.scope_end}")
            query_info_parts.append(f"Scope: {' → '.join(scope_parts)}")
        
        if query_structure.notes:
            query_info_parts.append(f"Zusätzliche Informationen: {query_structure.notes}")
    
    query_info = "\n".join(query_info_parts) if query_info_parts else ""
    
    # Build context from retrieved documents
    # Document headers: separators only; content chunks: [Document (page X)] format
    context_parts = []
    metadatas = retrieved_metadatas if retrieved_metadatas else [{}] * len(retrieved_documents)
    for i, doc in enumerate(retrieved_documents):
        meta = metadatas[i] if i < len(metadatas) else {}
        if meta.get('is_document_header', False):
            context_parts.append(doc)
        else:
            source_info = _format_chunk_source(meta, i)
            context_parts.append(f"[{source_info}]\n{doc}")
    
    retrieved_context = "\n\n".join(context_parts) if retrieved_documents else ""
    
    # Build user prompt
    user_prompt_parts = [
    
        "Dir liegt folgende Prozessbeschreibung zur Validierung auf Prozesslogik und Modellierungsqualität vor:\n\n"
        "--- USER QUERY ---\n"
        f"{user_request}\n\n"
    ]
    
    if query_info:
        user_prompt_parts.append(
            f"--- STRUKTURIERTE USERQUERY-INFORMATIONEN ---\n{query_info}\n\n"
        )
    
    if retrieved_context:
        user_prompt_parts.append(
            f"--- ABGERUFENE CHUNKS ---\n{retrieved_context}\n\n"
        )
    
    user_prompt_parts.append(
        f"--- ZU PRÜFENDES BPMN-JSON MODELL ---\n{bpmn_json}\n\n"
        "DEINE AUFGABE:\n"
        "Prüfe AUSSCHLIESSLICH:\n"
        "1. Prozesslogik: Ist der Ablauf logisch konsistent?\n"
        "2. Gateway-Modellierung: Sind Entscheidungen und Parallelität sinnvoll als Gateways modelliert?\n"
        "3. BPMN-konforme Bezeichnungen: Sind die Bezeichnungen BPMN-konform?\n"
        "4. Aktivitäten: Sind die Aktivitäten ATOMAR?\n"
        "5. Flow-Korrektheit: Gibt es falsche oder widersprüchliche Reihenfolgen?\n\n"
        "- Du musst IMMER ein assessment_statement zurückgeben.\n"
    )
    
    user_prompt = "".join(user_prompt_parts)
    
    system_prompt = (
        "Du bist ein Prozessexperte, der ein BPMN-JSON Modell auf Prozesslogik und Modellierungsqualität prüft.\n\n"
        "--------------------------------------------------\n"
        "issue_type REGEL (VERPFLICHTEND)\n\n"
        "Für structural_issues MUSS issue_type einer von sein:\n"
        "flow_logic | gateway_missing | gateway_incorrect | sequence_error | naming_violation\n"
        "Verwende NICHT source_plausibility (das ist für den Quellen-Validator).\n\n"
        "--------------------------------------------------\n"
        "DEIN FOKUS\n\n"
        "Du prüfst NUR Prozesslogik und Modellierungsqualität:\n\n"
        "1. Structural Issues:\n"
        "   - falsche oder widersprüchliche Reihenfolge\n"
        "   - inkonsistenter oder unlogischer Flow\n"
        "   - Eine Aktivität darf nicht aus mehreren Teilaktivitäten bestehen, sondern muss ATOMAR sein:\n"
        "     Atomar bedeutet:\n"
        "     - genau EINE fachliche Handlung\n"
        "     - KEINE explizite Logik (kein \"und\", \"oder\", \"falls\", \"ggf.\")\n"
        "     - KEINE Kombination mehrerer Schritte\n"
        "     Wenn ein Satz mehrere Handlungen enthält, müssen daraus mehrere Aktivitäten modelliert werden.\n"
        "   - fehlende oder falsch platzierte Gateways\n\n"
        "2. Fehlende Gateways:\n"
        "   - wenn im Chunk eindeutig Bedingungen, Alternativen oder Parallelität vorkommen,\n"
        "     diese aber im BPMN-Modell ohne Gateway dargestellt sind\n\n"
        "3. BPMN-konforme Bezeichnungen:\n"
        "   - Aktivitäten: NOMEN + VERB (Infinitiv)\n"
        "   - Start- und End-Events: NOMEN + PARTIZIP\n"
        "   - Rollen: NOMEN\n\n"
        "--------------------------------------------------\n"
        "WAS DU NICHT PRÜFST\n\n"
        "- Keine Missing Elements\n"
        "- Keine Scope Violations\n"
        "- Keine Halluzinationen\n"
        "- Keine Quellenplausibilität\n\n"
        "--------------------------------------------------\n"
        "VERPFLICHTENDE ARBEITSREIHENFOLGE\n\n"
        "1. Lies den Zielprozess und die vorhandenen Informationen an."
        "2. Lies die Chunks durch."
        "3. Schau dir das generierte BPMN-JSON Modell GENAU an."
        "4. Identifiziere ALLE Probleme in Prozesslogik oder Modellierungsqualität.\n"
        "5. Trage JEDES identifizierte Problem als Eintrag in `structural_issues` ein.\n"
        "6. Schreibe ERST DANACH das assessment_statement als Zusammenfassung dieser Issues.\n\n"
        "Es ist VERBOTEN, Probleme oder Unklarheiten nur im assessment_statement zu erwähnen,\n"
        "ohne sie als structural_issue zu melden.\n\n"
        + BPMN_JSON_READING_HELP +
        "--------------------------------------------------\n"
        "REGELN\n\n"
        "- Prüfe, ob der Ablauf logisch konsistent mit den Chunks ist\n"
        "- Prüfe, ob Entscheidungen korrekt als Gateways modelliert sind\n"
        "- Wenn Entscheidungslogik erkennbar ist, aber kein Gateway modelliert wurde,\n"
        "  MUSS ein structural_issue mit issue_type = \"gateway_missing\" erstellt werden\n"
        "  NENNE auch EXPLIZIT in der description, in welcher Lane und an welcher Stelle das Gateway hinzugefügt werden sollte.\n"
        "- Prüfe, ob Bezeichnungen BPMN-KONFORM sind, also Aktivitäten: NOMEN + VERB (Infinitiv), Events: NOMEN + PARTIZIP \n"
        "- BPMN-Naming-Verstöße gelten als structural_issues\n"
        "- Im Zweifel MELDEN statt zu schweigen\n\n"
        "--------------------------------------------------\n"
        "AUSGABEFORMAT\n\n"
        "Gib AUSSCHLIESSLICH ein gültiges JSON zurück:\n\n"
        "{\n"
        "  \"assessment_statement\": \"<kurze Gesamteinschätzung (2–4 Sätze)>\",\n"
        "  \"missing_elements\": [],\n"
        "  \"hallucinated_elements\": [],\n"
        "  \"structural_issues\": [\n"
        "    {\n"
        "      \"issue_type\": \"flow_logic | gateway_missing | gateway_incorrect | sequence_error | naming_violation\",\n"
        "      \"element_label\": \"string\",\n"
        "      \"description\": \"string\"\n"
        "    }\n"
        "  ],\n"
        "  \"overall_assessment\": { \"iteration_recommended\": true | false }\n"
        "}\n\n"
        "--------------------------------------------------\n"
        "🔧 KOPPLUNGSREGELN\n\n"
        "- missing_elements und hallucinated_elements\n"
        "  MÜSSEN leer sein.\n"
        "- Wenn du im assessment_statement ein Problem erwähnst,\n"
        "  MUSS dafür ein Eintrag in structural_issues existieren.\n"
        "- iteration_recommended = true,\n"
        "  sobald mindestens ein structural_issue existiert.\n"
        "- Gib IMMER ein assessment_statement zurück.\n"
    )

    system_prompt_bpmn_text_vorherige_version = (
        "Du bist ein Prozessexperte, der ein BPMN-JSON Modell auf Prozesslogik und Modellierungsqualität prüft.\n\n"
        "--------------------------------------------------\n"
        "🔧 issue_type REGEL (VERPFLICHTEND)\n\n"
        "Für structural_issues MUSS issue_type einer von sein:\n"
        "flow_logic | gateway_missing | gateway_incorrect | gateway_condition_missing | sequence_error | naming_violation\n\n"
        "--------------------------------------------------\n"
        "DEIN FOKUS\n\n"
        "Du prüfst NUR Prozesslogik und Modellierungsqualität:\n\n"
        "1. Structural Issues:\n"
        "   - falsche oder widersprüchliche Reihenfolge\n"
        "   - inkonsistenter oder unlogischer Flow\n"
        "   - Eine Aktivität darf nicht aus mehreren Teilaktivitäten bestehen, sondern muss ATOMAR sein:\n"
        "     Atomar bedeutet:\n"
        "     - genau EINE fachliche Handlung\n"
        "     - KEINE explizite Logik (kein \"und\", \"oder\", \"falls\", \"ggf.\")\n"
        "     - KEINE Kombination mehrerer Schritte\n"
        "     Wenn ein Satz mehrere Handlungen enthält, müssen daraus mehrere Aktivitäten modelliert werden.\n"
        "   - fehlende oder falsch platzierte Gateways\n\n"
        "2. Fehlende Gateways:\n"
        "   - wenn im Chunk eindeutig Bedingungen, Alternativen oder Parallelität vorkommen,\n"
        "     diese aber im BPMN-Modell ohne Gateway dargestellt sind\n\n"
        "3. BPMN-konforme Bezeichnungen:\n"
        "   - Aktivitäten: NOMEN + VERB (Infinitiv)\n"
        "   - Start- und End-Events: NOMEN + PARTIZIP\n"
        "   - Rollen: NOMEN\n\n"
        "--------------------------------------------------\n"
        "WAS DU NICHT PRÜFST\n\n"
        "- Keine Missing Elements\n"
        "- Keine Scope Violations\n"
        "- Keine Halluzinationen\n"
        "- Keine Quellenplausibilität\n\n"
        + BPMN_JSON_READING_HELP +
        "--------------------------------------------------\n"
        "VERPFLICHTENDE ARBEITSREIHENFOLGE\n\n"
        "1. Lies den Zielprozess und die vorhandenen Informationen."
        "2. Lies die Chunks durch."
        "3. Schau dir das generierte BPMN-JSON Modell GENAU an."
        "4. Identifiziere ALLE Probleme in Prozesslogik oder Modellierungsqualität.\n"
        "5. Trage JEDES identifizierte Problem als Eintrag in `structural_issues` ein.\n"
        "6. Schreibe ERST DANACH das assessment_statement als Zusammenfassung dieser Issues.\n\n"
        "Es ist VERBOTEN, Probleme oder Unklarheiten nur im assessment_statement zu erwähnen,\n"
        "ohne sie als structural_issue zu melden.\n\n"
        "--------------------------------------------------\n"
        "REGELN\n\n"
        "- Prüfe, ob der Ablauf logisch konsistent mit den Chunks ist\n"
        "- Prüfe, ob Entscheidungen korrekt als Gateways modelliert sind\n"
        "- Prüfe, ob jedes XOR-Gateway eine Condition angegeben hat und diese korrekt ist\n"
        "- Wenn Entscheidungslogik erkennbar ist, aber kein Gateway modelliert wurde,\n"
        "  MUSS ein structural_issue mit issue_type = \"gateway_missing\" erstellt werden\n"
        "  NENNE auch EXPLIZIT in der description, in welcher Lane und an welcher Stelle das Gateway hinzugefügt werden sollte.\n"
        "- Prüfe, ob alle Bezeichnungen BPMN-KONFORM sind, also Aktivitäten: NOMEN + VERB (Infinitiv), Events: NOMEN + PARTIZIP \n"
        "- BPMN-Naming-Verstöße gelten als structural_issues\n"
        "- Im Zweifel MELDEN statt zu schweigen\n\n"
        "--------------------------------------------------\n"
        "AUSGABEFORMAT\n\n"
        "Gib AUSSCHLIESSLICH ein gültiges JSON zurück:\n\n"
        "{\n"
        "  \"assessment_statement\": \"<kurze Gesamteinschätzung (2–4 Sätze)>\",\n"
        "  \"missing_elements\": [],\n"
        "  \"hallucinated_elements\": [],\n"
        "  \"structural_issues\": [\n"
        "    {\n"
        "      \"issue_type\": \"flow_logic | gateway_missing | gateway_incorrect | gateway_condition_missing | sequence_error | naming_violation\",\n"
        "      \"element_label\": \"string\",\n"
        "      \"description\": \"string\"\n"
        "    }\n"
        "  ],\n"
        "  \"overall_assessment\": { \"iteration_recommended\": true | false }\n"
        "}\n\n"
        "--------------------------------------------------\n"
        "🔧 KOPPLUNGSREGELN\n\n"
        "- missing_elements und hallucinated_elements\n"
        "  MÜSSEN leer sein.\n"
        "- Wenn du im assessment_statement ein Problem erwähnst,\n"
        "  MUSS dafür ein Eintrag in structural_issues existieren.\n"
        "- iteration_recommended = true,\n"
        "  sobald mindestens ein structural_issue existiert.\n"
        "- Gib IMMER ein assessment_statement zurück.\n"
    )


    system_prompt_bpmn_text = (
        "ROLLE: Du bist ein Prozessexperte, der ein BPMN-JSON Modell auf Prozesslogik und Modellierungsqualität prüft.\n\n"
        "AUFGABE: Bei einem generiertes BPMN-Prozessmodell (JSON) zu überprüfen, ob es Elemente im BPMN gibt, die nicht in den Chunks vorhanden sind oder semantisch stark von den Chunks abweichen.\n"
        + BPMN_JSON_READING_HELP +
        "REGELN für die Validierung:\n"
        "- Du darfst das BPMN-Modell NICHT verändern und KEINE neuen Inhalte erfinden.\n\n"
        "- Du validierst das BPMN-JSON anhand von:\n"
        "- 1. Der User Query (definiert Prozessidentität, Scope, Start-/Endpunkt, Perspektive, Detailierungsgrad)\n"
        "- 2. Die Dokumentpassagen also Chunks (einzige inhaltliche Quelle; das BPMN wurde daraus generiert)\n\n"
        "- Die Chunks sind die EINZIGE Quelle für Prozesselemente.\n\n"
        "- Im Zweifel MELDEN statt zu schweigen\n\n"
        "- KEINE Ableitungen, KEINE Annahmen\n"
        "- Für structural_issues MUSS issue_type einer von sein:\n"
        "flow_logic | gateway_missing | gateway_incorrect | gateway_condition_missing | sequence_error | naming_violation\n\n"
        "- Prüfe, ob der Ablauf logisch konsistent mit den Chunks ist\n"
        "- Prüfe, ob jede Entscheidungen korrekt als Gateways modelliert sind\n"
        "- Prüfe, ob jedes XOR-Gateway eine Condition angegeben hat und diese korrekt ist\n"
        "- Wenn Entscheidungslogik erkennbar ist, aber kein Gateway modelliert wurde,\n"
        "  MUSS ein structural_issue mit issue_type = \"gateway_missing\" erstellt werden\n"
        "  NENNE auch EXPLIZIT in der description, in welcher Lane und an welcher Stelle das Gateway hinzugefügt werden sollte.\n"
        "- Prüfe, ob alle Bezeichnungen BPMN-KONFORM sind, also Aktivitäten: NOMEN + VERB (Infinitiv), Events: NOMEN + PARTIZIP \n"
        "- BPMN-Naming-Verstöße gelten als structural_issues\n"
        "--------------------------------------------------\n"
        "Verpflichtende Arbeitsweise\n\n"
        "1. Lies den Zielprozess und falls vorhanden Informationen wie Detailierungsgrad, Start- und Endpunkt, Perspektive, Scope."
        "2. Lies jeden einzelnen Chunk SEHR GENAU durch."
        "3. Schau dir den generierten das generierte BPMN-JSON Modell GENAU an und prüfe das BPMN Modell auf Prozesslogik und Modellierungsqualität basierend auf den CHunks nach folgendem Muster:\n\n"
        "   1) Structural Issues:\n"
        "   - falsche oder widersprüchliche Reihenfolge\n"
        "   - inkonsistenter oder unlogischer Flow\n"
        "   - Eine Aktivität darf nicht aus mehreren Teilaktivitäten bestehen, sondern muss ATOMAR sein:\n"
        "     Atomar bedeutet:\n"
        "     - genau EINE fachliche Handlung\n"
        "     - KEINE explizite Logik (kein \"und\", \"oder\", \"falls\", \"ggf.\")\n"
        "     - KEINE Kombination mehrerer Schritte\n"
        "     Wenn ein Satz mehrere Handlungen enthält, müssen daraus mehrere Aktivitäten modelliert werden.\n"
        "   - fehlende oder falsch platzierte Gateways\n\n"
        "   2) Fehlende Gateways:\n"
        "   - wenn im Chunk eindeutig Bedingungen, Alternativen oder Parallelität vorkommen, diese aber im BPMN-Modell ohne Gateway dargestellt sind\n\n"
        "   3) BPMN-konforme Bezeichnungen:\n"
        "   - Aktivitäten: NOMEN + VERB (Infinitiv)\n"
        "   - Start- und End-Events: NOMEN + PARTIZIP\n"
        "   - Rollen: NOMEN\n\n"
        "4. Trage JEDES identifizierte Problem als Eintrag in `structural_issues` ein.\n"
       "--------------------------------------------------\n"
        "AUSGABEFORMAT\n\n"
        "- Gib AUSSCHLIESSLICH ein gültiges JSON zurück.\n"
        "- Jede Feststellung muss sich auf ein konkretes BPMN-Element beziehen\n"
        "  (element_type: z.B. 'task', 'gateway', 'event', 'pool', 'lane', element_label: Name aus dem BPMN-JSON, kurze Begründung, ggf. Draft-Referenz).\n\n"
        "Das JSON muss GENAU folgendem Schema entsprechen (leere Listen sind erlaubt, nur assessment_statement ist obligatorisch):\n"
        "{\n"
        "  \"assessment_statement\": \"<kurze Gesamteinschätzung (2–4 Sätze)>\",\n"
        "  \"missing_elements\": [],\n"
        "  \"hallucinated_elements\": [],\n"
        "  \"structural_issues\": [\n"
        "    {\n"
        "      \"issue_type\": \"flow_logic | gateway_missing | gateway_incorrect | gateway_condition_missing | sequence_error | naming_violation\",\n"
        "      \"element_label\": \"string\",\n"
        "      \"description\": \"string\"\n"
        "    }\n"
        "  ],\n"
        "  \"overall_assessment\": { \"iteration_recommended\": true | false }\n"
        "}\n\n"
        
        "Abschließende HINWEISE:\n"
        "- missing_elements und hallucinated_elements MÜSSEN leer sein.\n"
        "- structural_issues darf NUR Probleme aus diesem Fokus enthalten.\n"
        "Iteration: iteration_recommended = true, wenn mindestens ein Problem vorliegt.\n"
        "Falls nach sorgfältiger Prüfung keine Issues gefunden wurden, ist das auch okay."
        "Spiegle das im Assessment_statement und iteration_recommended = false."
    )

    
   
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt_bpmn_text},
        {"role": "user", "content": user_prompt},
    ]

    
    
    raw = await call_ollama_json(
        messages=messages,
        model=settings.agentic_validation_model,
    )
    
    # Parse JSON response
    try:
        data = json.loads(raw)
        
        # Extract fields with defaults
        structural_issues = data.get("structural_issues", [])
        overall_assessment = data.get("overall_assessment", {})
        assessment_statement = data.get("assessment_statement", "")
        
        # Ensure assessment_statement is a string
        if isinstance(assessment_statement, dict):
            assessment_statement = assessment_statement.get("text", assessment_statement.get("statement", str(assessment_statement)))
        elif not isinstance(assessment_statement, str):
            assessment_statement = str(assessment_statement) if assessment_statement else ""
        
        # Parse structured objects
        parsed_structural_issues = []
        for item in structural_issues if isinstance(structural_issues, list) else []:
            if isinstance(item, dict):
                try:
                    parsed_structural_issues.append(StructuralIssue(**item))
                except Exception:
                    parsed_structural_issues.append(StructuralIssue(
                        issue_type=item.get("issue_type", "unknown"),
                        element_label=item.get("element_label", ""),
                        description=item.get("description", str(item))
                    ))
        
        # Ensure overall_assessment
        if not isinstance(overall_assessment, dict):
            overall_assessment = {}
        if "iteration_recommended" not in overall_assessment:
            iteration_recommended = len(parsed_structural_issues) > 0
            overall_assessment = {"iteration_recommended": iteration_recommended}
        
        return ValidationResultSetting4(
            missing_elements=[],
            hallucinated_elements=[],
            structural_issues=parsed_structural_issues,
            overall_assessment=overall_assessment,
            assessment_statement=assessment_statement
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in process logic validator: {e}")
        return ValidationResultSetting4(
            missing_elements=[],
            hallucinated_elements=[],
            structural_issues=[],
            overall_assessment={"iteration_recommended": False},
            assessment_statement=f"Validation response could not be parsed as JSON: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in process logic validator: {e}")
        return ValidationResultSetting4(
            missing_elements=[],
            hallucinated_elements=[],
            structural_issues=[],
            overall_assessment={"iteration_recommended": False},
            assessment_statement=f"Validation error: {str(e)}"
        )


def aggregate_validation_results_setting5(
    results: List[ValidationResultSetting4]
) -> ValidationResultSetting4:
    """
    Aggregate results from three specialized validators.

    Issue priority (conceptual):
    1. Scope violations
    2. Hallucinated elements
    3. Missing elements
    4. Flow and gateway issues

    (Source annotation is not checked; validator 2 no longer emits structural_issues.)

    Args:
        results: Up to three ``ValidationResultSetting4`` instances (one per validator).

    Returns:
        Single merged ``ValidationResultSetting4``.
    """
    if len(results) != 3:
        logger.warning(f"Expected 3 validation results, got {len(results)}")
    
    # Extract results from each validator
    # Validator 1: missing_elements
    # Validator 2: hallucinated_elements only 
    # Validator 3: structural_issues (flow/gateway)
    
    validator1_result = results[0] if len(results) > 0 else ValidationResultSetting4()
    validator2_result = results[1] if len(results) > 1 else ValidationResultSetting4()
    validator3_result = results[2] if len(results) > 2 else ValidationResultSetting4()
    
    # Aggregate missing_elements (from Validator 1)
    missing_elements = validator1_result.missing_elements.copy()
    
    # Aggregate hallucinated_elements (from Validator 2)
    hallucinated_elements = validator2_result.hallucinated_elements.copy()
    
    # Aggregate structural_issues (only from Validator 3 - flow/gateway)
    structural_issues = validator3_result.structural_issues.copy()
    
    # Combine assessment statements
    assessment_parts = []
    if validator1_result.assessment_statement:
        assessment_parts.append(f"[Validator 1 - Scope & completeness]: {validator1_result.assessment_statement}")
    if validator2_result.assessment_statement:
        assessment_parts.append(f"[Validator 2 - Factual fidelity]: {validator2_result.assessment_statement}")
    if validator3_result.assessment_statement:
        assessment_parts.append(f"[Validator 3 - Process logic & modeling]: {validator3_result.assessment_statement}")
    
    combined_assessment = " ".join(assessment_parts) if assessment_parts else "No validation issues detected."
    
    # Determine iteration_recommended
    # True if at least one agent recommends iteration OR if any list is not empty
    iteration_recommended = (
        validator1_result.overall_assessment.get("iteration_recommended", False) or
        validator2_result.overall_assessment.get("iteration_recommended", False) or
        validator3_result.overall_assessment.get("iteration_recommended", False) or
        len(missing_elements) > 0 or
        len(hallucinated_elements) > 0 or
        len(structural_issues) > 0
    )
    
    return ValidationResultSetting4(
        missing_elements=missing_elements,
        hallucinated_elements=hallucinated_elements,
        structural_issues=structural_issues,
        overall_assessment={"iteration_recommended": iteration_recommended},
        assessment_statement=combined_assessment
    )

