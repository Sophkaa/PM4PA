"""LLM-as-a-judge helper to compare generated BPMN models against a gold standard."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union
from xml.etree import ElementTree as ET

from ..infrastructure.api.ollama_client import call_ollama_json
from ..config import settings
from ..models.artifacts import LLMJudgeResult


logger = logging.getLogger(__name__)

# BPMN namespace
BPMN_NS = "{http://www.omg.org/spec/BPMN/20100524/MODEL}"


def _bpmn_local_name(elem: ET.Element) -> str:
    tag = elem.tag
    if not isinstance(tag, str):
        return ""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _findall_bpmn_processes(root: ET.Element) -> List[ET.Element]:
    """Return bpmn:process elements; fall back to any element named process in the BPMN 2.0 model namespace."""
    found = root.findall(f".//{BPMN_NS}process")
    if found:
        return found
    return [
        e
        for e in root.iter()
        if _bpmn_local_name(e) == "process" and isinstance(e.tag, str) and "BPMN/20100524/MODEL" in e.tag
    ]


def _get_event_subtype(event: ET.Element) -> str:
    """Extract event subtype from event element."""
    if event.find(f".//{BPMN_NS}messageEventDefinition") is not None:
        return "message" + ("Receive" if "Catch" in event.tag else "Send")
    elif event.find(f".//{BPMN_NS}timerEventDefinition") is not None:
        return "timer"
    elif event.find(f".//{BPMN_NS}signalEventDefinition") is not None:
        return "signal"
    return "default"


def _get_gateway_type(gw_type: str) -> str:
    """Convert gateway XML type to BPMN type."""
    if "exclusive" in gw_type:
        return "XOR"
    elif "parallel" in gw_type:
        return "AND"
    elif "inclusive" in gw_type:
        return "OR"
    elif "eventBased" in gw_type:
        return "Event-Based"
    return "XOR"


def format_bpmn_xml_for_judge(xml_source: Union[str, Path, bytes]) -> str:
    """
    Parse BPMN XML (from file path, XML string, or bytes) and create structured text representation for LLM judge.
    
    This function extracts:
    - Pools and Lanes
    - Start/End Events
    - Tasks/Activities
    - Gateways with conditions
    - Sequence Flows (explicit process flow logic)
    
    Args:
        xml_source: Path to XML file, XML string, or XML bytes
        
    Returns:
        Structured text representation optimized for LLM understanding
    """
    # Parse XML: avoid treating arbitrary strings as paths when a same-named file exists.
    if isinstance(xml_source, Path):
        if not xml_source.is_file():
            raise FileNotFoundError(f"BPMN XML file not found or not a file: {xml_source}")
        root = ET.parse(xml_source).getroot()
    elif isinstance(xml_source, str):
        stripped = xml_source.lstrip(" \t\n\r")
        if stripped.startswith("\ufeff"):
            stripped = stripped[1:].lstrip(" \t\n\r")
        if stripped.startswith("<"):
            root = ET.fromstring(xml_source)
        else:
            path = Path(xml_source)
            if path.is_file():
                root = ET.parse(path).getroot()
            else:
                root = ET.fromstring(xml_source)
    elif isinstance(xml_source, bytes):
        root = ET.fromstring(xml_source)
    else:
        raise ValueError(f"Unsupported XML source type: {type(xml_source)}")
    
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("BPMN PROCESS MODEL")
    lines.append("=" * 80)
    lines.append("")
    
    # Extract collaboration (pools)
    collaboration = root.find(f".//{BPMN_NS}collaboration")
    pools = []
    if collaboration is not None:
        participants = collaboration.findall(f".//{BPMN_NS}participant")
        for participant in participants:
            pools.append({
                "id": participant.get("id", ""),
                "name": participant.get("name", "Unnamed Pool"),
                "processRef": participant.get("processRef", "")
            })
    
    # Extract processes (standard path + tolerant fallback for unusual exports)
    processes = _findall_bpmn_processes(root)
    
    # If no collaboration, create default pool from first process
    if not pools and processes:
        for process in processes:
            pools.append({
                "id": f"Pool_{process.get('id', 'default')}",
                "name": process.get("name", "Process"),
                "processRef": process.get("id", "")
            })
    
    # Validate that we found at least one process
    if not processes:
        error_msg = (
            "WARNUNG: Keine BPMN-Prozesse in der XML-Datei gefunden. "
            "Die Datei könnte leer, ungültig oder in einem anderen Format sein."
        )
        lines.append(error_msg)
        lines.append("")
        root_tag = root.tag if isinstance(root.tag, str) else str(root.tag)
        logger.warning(
            "No BPMN processes found in XML (root element: %s). "
            "Expected BPMN 2.0 elements in namespace http://www.omg.org/spec/BPMN/20100524/MODEL",
            root_tag,
        )
        return "\n".join(lines).strip()
    
    if not pools:
        error_msg = (
            "WARNUNG: Keine Pools oder Prozesse in der XML-Datei gefunden. "
            "Die Datei könnte leer, ungültig oder in einem anderen Format sein."
        )
        lines.append(error_msg)
        lines.append("")
        logger.warning("No pools or processes found in XML file")
        return "\n".join(lines).strip()
    
    # Process each pool/process
    for pool_idx, pool in enumerate(pools):
        lines.append(f"POOL {pool_idx + 1}: {pool['name']}")
        lines.append("")
        
        # Find corresponding process
        process = None
        for p in processes:
            if p.get("id") == pool.get("processRef"):
                process = p
                break
        
        if not process and processes:
            process = processes[pool_idx] if pool_idx < len(processes) else processes[0]
        
        if not process:
            continue
        
        # Extract lanes
        lanes = []
        lane_sets = process.findall(f".//{BPMN_NS}laneSet")
        for lane_set in lane_sets:
            for lane in lane_set.findall(f".//{BPMN_NS}lane"):
                lanes.append({
                    "id": lane.get("id", ""),
                    "name": lane.get("name", "Unnamed Lane")
                })
        
        if lanes:
            lines.append("LANES:")
            for lane in lanes:
                lines.append(f"  - {lane['name']}")
            lines.append("")
        
        # Extract all elements with their IDs for flow mapping
        elements = {}  # id -> {"type": "task|event|gateway", "name": "...", ...}
        
        # Extract start events
        start_events = process.findall(f".//{BPMN_NS}startEvent")
        for event in start_events:
            event_id = event.get("id", "")
            event_name = event.get("name", "Start")
            elements[event_id] = {
                "type": "startEvent",
                "name": event_name,
                "subType": _get_event_subtype(event)
            }
        
        # Extract tasks
        for task_type in ["userTask", "serviceTask", "scriptTask", "businessRuleTask", "manualTask", "task"]:
            tasks = process.findall(f".//{BPMN_NS}{task_type}")
            for task in tasks:
                task_id = task.get("id", "")
                if task_id not in elements:  # Avoid duplicates
                    elements[task_id] = {
                        "type": "task",
                        "name": task.get("name", "Unnamed Task"),
                        "taskType": task_type
                    }
        
        # Extract intermediate events
        for event_type in ["intermediateCatchEvent", "intermediateThrowEvent"]:
            events = process.findall(f".//{BPMN_NS}{event_type}")
            for event in events:
                event_id = event.get("id", "")
                if event_id not in elements:
                    elements[event_id] = {
                        "type": "intermediateEvent",
                        "name": event.get("name", "Event"),
                        "subType": _get_event_subtype(event),
                        "direction": "catch" if "Catch" in event_type else "throw"
                    }
        
        # Extract end events
        end_events = process.findall(f".//{BPMN_NS}endEvent")
        for event in end_events:
            event_id = event.get("id", "")
            elements[event_id] = {
                "type": "endEvent",
                "name": event.get("name", "End"),
                "subType": _get_event_subtype(event)
            }
        
        # Extract gateways
        for gw_type in ["exclusiveGateway", "inclusiveGateway", "parallelGateway", "eventBasedGateway"]:
            gateways = process.findall(f".//{BPMN_NS}{gw_type}")
            for gateway in gateways:
                gw_id = gateway.get("id", "")
                gw_name = gateway.get("name", "")
                gw_bpmn_type = _get_gateway_type(gw_type)
                
                # Extract conditions from all outgoing flows
                conditions = []
                outgoing_flows = gateway.findall(f".//{BPMN_NS}outgoing")
                # Get all sequence flows to search through
                all_sequence_flows = process.findall(f".//{BPMN_NS}sequenceFlow")
                flow_dict = {flow.get("id", ""): flow for flow in all_sequence_flows}
                
                for flow_ref in outgoing_flows:
                    flow_id = flow_ref.text
                    if flow_id and flow_id in flow_dict:
                        flow_elem = flow_dict[flow_id]
                        flow_name = flow_elem.get("name", "")
                        if flow_name:
                            conditions.append(flow_name)
                
                # Use gateway name as condition if no flow labels found
                condition = gw_name if not conditions and gw_name else (", ".join(conditions) if conditions else None)
                
                elements[gw_id] = {
                    "type": "gateway",
                    "name": gw_name or f"Gateway {gw_id}",
                    "gatewayType": gw_bpmn_type,
                    "condition": condition
                }
        
        # Display elements by type
        if start_events:
            lines.append("START EVENTS:")
            for event in start_events:
                event_id = event.get("id", "")
                elem = elements.get(event_id, {})
                lines.append(f"  - {elem.get('name', 'Start')} (id: {event_id})")
            lines.append("")
        
        tasks_list = [e for e in elements.values() if e.get("type") == "task"]
        if tasks_list:
            lines.append("TASKS/ACTIVITIES:")
            for task in tasks_list:
                task_id = next((k for k, v in elements.items() if v == task), "unknown")
                lines.append(f"  - {task['name']} (id: {task_id})")
            lines.append("")
        
        intermediate_events_list = [e for e in elements.values() if e.get("type") == "intermediateEvent"]
        if intermediate_events_list:
            lines.append("INTERMEDIATE EVENTS:")
            for event in intermediate_events_list:
                sub_type = event.get("subType", "")
                direction = event.get("direction", "")
                lines.append(f"  - {event['name']} ({sub_type}, {direction})")
            lines.append("")
        
        gateways_list = [e for e in elements.values() if e.get("type") == "gateway"]
        if gateways_list:
            lines.append("GATEWAYS:")
            for gw in gateways_list:
                gw_type = gw.get("gatewayType", "")
                condition = gw.get("condition", "")
                condition_str = f" - condition: {condition}" if condition else ""
                lines.append(f"  - {gw['name']} (type: {gw_type}{condition_str})")
            lines.append("")
        
        end_events_list = [e for e in elements.values() if e.get("type") == "endEvent"]
        if end_events_list:
            lines.append("END EVENTS:")
            for event in end_events_list:
                lines.append(f"  - {event.get('name', 'End')}")
            lines.append("")
        
        # Extract and display sequence flows (THE KEY PART!)
        sequence_flows = process.findall(f".//{BPMN_NS}sequenceFlow")
        if sequence_flows:
            lines.append("=" * 80)
            lines.append("PROCESS FLOW (Sequence Flows)")
            lines.append("=" * 80)
            lines.append("")
            
            # Build flow map for better visualization
            flow_map = {}  # source -> [(target, label, flow_id), ...]
            flow_dict = {flow.get("id", ""): flow for flow in sequence_flows}
            
            for flow in sequence_flows:
                source = flow.get("sourceRef", "")
                target = flow.get("targetRef", "")
                label = flow.get("name", "")
                flow_id = flow.get("id", "")
                
                if source not in flow_map:
                    flow_map[source] = []
                flow_map[source].append((target, label, flow_id))
            
            # Helper function to trace a path from a gateway until next gateway or end
            def trace_path(start_id: str, visited: set) -> List[str]:
                """Trace path from start_id until gateway or end event, return list of element names."""
                path = []
                current = start_id
                max_depth = 50  # Prevent infinite loops
                depth = 0
                
                while current and depth < max_depth:
                    if current in visited:
                        break
                    visited.add(current)
                    
                    elem = elements.get(current, {})
                    elem_type = elem.get("type", "")
                    elem_name = elem.get("name", current)
                    
                    # Add current element to path (always include first element)
                    if elem_type == "task":
                        path.append(f"{elem_name} (task)")
                    elif elem_type == "intermediateEvent":
                        path.append(f"{elem_name} (intermediate event)")
                    elif elem_type == "endEvent":
                        path.append(f"{elem_name} (end event)")
                        break  # Stop at end events
                    elif elem_type == "gateway":
                        # Stop at gateways (but don't add to path, as it's the endpoint)
                        break
                    
                    # Follow to next element
                    if current in flow_map and flow_map[current]:
                        # Take first outgoing flow (for path tracing)
                        next_id, _, _ = flow_map[current][0]
                        current = next_id
                    else:
                        break
                    depth += 1
                
                return path
            
            # Group flows by gateways to show paths explicitly
            gateway_ids = [gw_id for gw_id, elem in elements.items() if elem.get("type") == "gateway"]
            
            if gateway_ids:
                lines.append("GATEWAY PATHS (Aktivitäten auf Pfaden von Gateways):")
                lines.append("")
                
                for gw_id in gateway_ids:
                    gw_elem = elements.get(gw_id, {})
                    gw_name = gw_elem.get("name", gw_id)
                    gw_type = gw_elem.get("gatewayType", "XOR")
                    
                    if gw_id in flow_map:
                        outgoing = flow_map[gw_id]
                        lines.append(f"  Gateway: {gw_name} (type: {gw_type})")
                        
                        for path_idx, (target_id, label, flow_id) in enumerate(outgoing, 1):
                            target_elem = elements.get(target_id, {})
                            target_name = target_elem.get("name", target_id)
                            target_type = target_elem.get("type", "element")
                            
                            # Build condition/label string
                            condition_str = f" [condition: {label}]" if label else " [no condition]"
                            
                            # Trace the path from this gateway branch
                            visited = {gw_id}
                            path_elements = trace_path(target_id, visited)
                            
                            # Build complete path string
                            if path_elements:
                                # path_elements already includes the target and all following elements
                                path_str = " --> ".join(path_elements)
                                lines.append(f"    Pfad {path_idx}{condition_str}: {path_str}")
                            else:
                                # No path found beyond target, just show target
                                lines.append(f"    Pfad {path_idx}{condition_str}: {target_name} ({target_type})")
                        
                        lines.append("")
            
            # Also show all flows in a flat list for completeness
            lines.append("ALL SEQUENCE FLOWS (complete flow graph):")
            lines.append("")
            for source_id, targets in flow_map.items():
                source_elem = elements.get(source_id, {})
                source_name = source_elem.get("name", source_id)
                source_type = source_elem.get("type", "element")
                
                for target_id, label, _ in targets:
                    target_elem = elements.get(target_id, {})
                    target_name = target_elem.get("name", target_id)
                    target_type = target_elem.get("type", "element")
                    
                    label_str = f" [{label}]" if label else ""
                    lines.append(f"  {source_name} ({source_type}) --{label_str}--> {target_name} ({target_type})")
            
            lines.append("")
    
    result = "\n".join(lines).strip()
    
    # Validate that we actually extracted some content
    # If result is too short (only header), something went wrong
    if len(result) < 100:
        logger.warning(
            "format_bpmn_xml_for_judge produced very short output (%d chars). "
            "This might indicate parsing issues.",
            len(result)
        )
        result += "\n\nWARNUNG: Die XML-Datei scheint keine BPMN-Elemente zu enthalten oder konnte nicht korrekt geparst werden."
    
    return result


def _clamp_score(value: Any) -> int:
    """Clamp score value to 0-100 range."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


async def run_llm_judge_agent(
    task_description: str,
    gold_bpmn_xml: Union[str, Path, bytes],
    predicted_bpmn_xml: Union[str, Path, bytes],
    model_override: Optional[str] = None,
) -> LLMJudgeResult:
    """
    Compare generated BPMN XML against gold standard XML via LLM.
    
    Args:
        task_description: Description of the task/query
        gold_bpmn_xml: Gold standard BPMN XML (file path, XML string, or bytes)
        predicted_bpmn_xml: Predicted/generated BPMN XML (file path, XML string, or bytes)
        model_override: Optional model name override
        
    Returns:
        LLMJudgeResult with semantic alignment score and justification
    """
    description = task_description.strip() if task_description else "(keine Beschreibung)"

    # Format both models from XML
    # IMPORTANT: Ensure correct mapping:
    # - gold_bpmn_xml -> gold_summary -> "GOLDSTANDARD BPMN MODELL"
    # - predicted_bpmn_xml -> predicted_summary -> "GENERIERTES BPMN MODELL"
    try:
        gold_summary = format_bpmn_xml_for_judge(gold_bpmn_xml)
        if len(gold_summary) < 100:
            logger.warning("Gold standard XML produced very short summary (%d chars)", len(gold_summary))
        else:
            logger.debug("Gold standard XML summary length: %d chars", len(gold_summary))
    except Exception as e:
        logger.error("Failed to format gold standard XML: %s", e, exc_info=True)
        gold_summary = f"FEHLER beim Parsen der Gold Standard XML: {str(e)}"
    
    try:
        predicted_summary = format_bpmn_xml_for_judge(predicted_bpmn_xml)
        if len(predicted_summary) < 100:
            logger.warning("Predicted XML produced very short summary (%d chars)", len(predicted_summary))
        else:
            logger.debug("Predicted XML summary length: %d chars", len(predicted_summary))
    except Exception as e:
        logger.error("Failed to format predicted XML: %s", e, exc_info=True)
        predicted_summary = f"FEHLER beim Parsen der Predicted XML: {str(e)}"

    system_content = (
        "ROLLE: Du bist ein BPMN-Prozessexperte. "
        "Deine Aufgabe ist es, ein generiertes BPMN-Modell mit einem Goldstandard SEHR DETAILLIERT zu vergleichen und einen DETAILLIERTEN und STRUKTURIERTEN Score zu vergeben.\n\n"
        "REGELN für die Bewertung:\n"
        "1. MODELL-IDENTIFIKATION\n"
        "- Das GOLDSTANDARD BPMN MODELL ist die Referenz (korrektes Modell)\n"
        "- Das GENERIERTES BPMN MODELL ist das zu bewertende Modell\n"
        "- ACHTE GENAU darauf, welche Elemente in welchem Modell vorkommen!\n"
        "- Verwechsle NICHT die beiden Modelle in deiner Analyse!\n\n"
        "2. Analysiere BEIDE Modelle SYSTEMATISCH:\n"
        "   a) Zähle und liste ALLE Elemente pro Modell:\n"
        "      - Pools und Lanes (Anzahl, Namen)\n"
        "      - Start Events (Anzahl, Namen)\n"
        "      - Tasks/Activities (Anzahl, Namen)\n"
        "      - Gateways (Anzahl, Typen, Namen, Conditions)\n"
        "      - End Events (Anzahl, Namen)\n"
        "   b) Analysiere die Sequence Flows:\n"
        "      - Welche Flows gibt es im Goldstandard?\n"
        "      - Welche Flows gibt es im generierten Modell?\n"
        "      - Vergleiche sie und prüfe: Welche Flows fehlen oder sind zusätzlich?\n\n"
        "3. VERGLEICHE ELEMENT-FÜR-ELEMENT:\n"
        "   - Welche Tasks/Activities aus dem Goldstandard fehlen im generierten Modell?\n"
        "   - Welche Tasks/Activities sind zusätzlich im generierten Modell?\n"
        "   - Welche Gateways fehlen oder sind zusätzlich?\n"
        "   - Stimmt die Ablauflogik (Sequence Flows) überein?\n"
        "   - Sind die Entscheidungspunkte (Gateways) korrekt modelliert?\n\n"
        "4. BEWERTE die semantische Übereinstimmung objektiv:\n"
        "   - semantic_alignment_score (0-100) gibt den Prozentsatz der Übereinstimmung der beiden Modelle an:\n"
        "   - Berücksichtige: Es muss nicht 1:1 übereinstimmen, aber die KERNLOGIK muss korrekt sein\n"
        "5. DOKUMENTIERE deinen Denkprozess:\n"
        "   - Liste konkret und DETAILLIERT auf, welche Elemente wo vorkommen\n"
        "   - Erkläre PRÄZISE, welche Unterschiede du feststellst\n"
        "   - Begründe deine Score-Vergabe mit konkreten Beispielen\n\n"
        "# REGELN für das Output-Format\n"
        "Antworte ausschließlich mit einem JSON-Objekt mit genau diesen Feldern:\n"
        "{\n"
        '  "semantic_alignment_score": <int 0-100>,\n'
        '  "chain_of_thought": "<Strukturierter Denkprozess: 1) Element-Analyse beider Modelle, 2) Konkreter Vergleich, 3) Score-Begründung. Max 20 Sätze, präzise und konkret>",\n'
        '  "justification": "<Präzise Begründung mit konkreten Beispielen: Welche Elemente fehlen/sind zusätzlich, wie wirkt sich das auf die Logik aus. Max 8 Sätze>"\n'
        "}\n\n"
        "Abschließende HINWEISE:\n\n"
        "Sei präzise und detailiert! Nenne konkret Element-Namen und erkläre genau, was unterschiedlich ist."
    )

    user_content = (
        f"QUERY (ursprüngliche Aufgabe, basierend auf der das Modell generiert wurde):\n{description}\n\n"
        "GOLDSTANDARD BPMN MODELL (REFERENZ - korrektes Modell):\n"
        f"{gold_summary}\n\n"
        "GENERIERTES BPMN MODELL (ZU BEWERTEN - mit Goldstandard vergleichen):\n"
        f"{predicted_summary}\n\n"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Warn if summaries are very long (might cause timeout)
    total_length = len(gold_summary) + len(predicted_summary) + len(description) + len(system_content) + len(user_content)
    if total_length > 20000:  # ~20k chars
        logger.warning(
            "Total prompt length is very long (%d chars). This might cause timeout issues. "
            "Consider using smaller models or shorter BPMN models.",
            total_length
        )

    try:
        raw = await call_ollama_json(
            messages=messages,
            model=model_override or settings.agentic_judge_model,
            temperature=settings.agentic_temperature,
            max_retries=3,  # Increase retries for judge (large models can be slow)
        )
    except Exception as e:
        error_msg = str(e)
        if "504" in error_msg or "Gateway Timeout" in error_msg:
            logger.error(
                "504 Gateway Timeout beim LLM Judge. "
                "Mögliche Ursachen: "
                "1) Anfrage zu groß (BPMN-Modelle zu komplex) - Summary-Länge: Gold=%d, Predicted=%d, "
                "2) Server überlastet, "
                "3) Modell braucht zu lange zum Verarbeiten. "
                "Versuche: Kleinere Modelle verwenden oder BPMN-Modelle vereinfachen.",
                len(gold_summary),
                len(predicted_summary),
            )
        raise

    try:
        data = json.loads(raw)
        result = LLMJudgeResult(
            semantic_alignment_score=_clamp_score(data.get("semantic_alignment_score")),
            justification=str(data.get("justification") or "").strip(),
            chain_of_thought=str(data.get("chain_of_thought") or "").strip(),
        )
    except Exception as exc:
        logger.warning("Failed to parse LLM judge response: %s", exc)
        logger.debug("LLM judge raw response: %s", raw)
        result = LLMJudgeResult(
            semantic_alignment_score=0,
            justification=f"Judge response could not be parsed as JSON: {raw[:200]}",
            chain_of_thought="",
        )

    return result


__all__ = ["format_bpmn_xml_for_judge", "run_llm_judge_agent"]
