"""Helpers for creating process drafts and BPMN models via the Ollama API."""

from __future__ import annotations

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from pydantic import ValidationError

from ..config import settings
from ..models.artifacts import ProcessDraft, ValidationResultSetting4
from ..models.bpmn import BPMNModelJsonNested
from ..models.query_structure import QueryStructure
from ..infrastructure.api.ollama_client import call_ollama_chat, call_ollama_json

logger = logging.getLogger(__name__)

_GRANULARITY_LABELS: Dict[str, str] = {
    "high_level": "High-Level (Überblick, Hauptschritte)",
    "medium": "Medium (moderate Detaillierung)",
    "detailed": "Detailed (sehr detailliert, kleinschrittig)",
}


def _bpmn_debug(msg: str) -> None:
    if settings.debug:
        print(f"[bpmn|debug] {msg}")


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken if available, otherwise estimate.
    
    For Llama 3.1 models, we use cl100k_base encoding (similar to GPT-4)
    as it's a reasonable approximation. The actual Llama tokenizer might differ
    slightly, but this gives a good estimate.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Estimated number of tokens
    """
    try:
        import tiktoken
        # Use cl100k_base encoding (GPT-4 style) as approximation for Llama 3.1
        # This is close enough for estimation purposes
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        # This is a conservative estimate
        return len(text) // 4
    except Exception as e:
        # If tiktoken fails for any reason, fall back to estimation
        logger.warning(f"Token counting with tiktoken failed: {e}, using estimation")
        return len(text) // 4


async def run_draft_agent_with_structure(
    user_request: str,
    retrieved_documents: List[str],
    retrieved_metadatas: List[Dict[str, Any]],
    expanded_queries: List[str],
    query_structure: Optional[QueryStructure]
) -> tuple[ProcessDraft, str]:
    """
    Create a textual process draft via Ollama with query_structure orientation.
    
    Args:
        user_request: Original user query
        retrieved_documents: List of retrieved document texts
        retrieved_metadatas: List of metadata dicts for retrieved documents
        expanded_queries: List of expanded query variants
        query_structure: Optional structured query information
        
    Returns:
        Tuple of (ProcessDraft, user_prompt)
    """
    # Helper function to extract document name from metadata
    def get_document_name(metadata: Dict[str, Any], index: int) -> str:
        """Extract document name from metadata, fallback to index if not available."""
        if metadata:
            name = (
                metadata.get('file_name') or
                metadata.get('document_title') or
                metadata.get('title') or
                (Path(metadata.get('file_path', '')).name if metadata.get('file_path') else None)
            )
            if name:
                if metadata.get('page_number') is not None:
                    return f"{name} (Seite {metadata.get('page_number')})"
                return name
        return f"Dokument {index + 1}"
    
    def format_chunk_source(metadata: Dict[str, Any], index: int) -> str:
        """
        Format chunk source for prompts: document name and page (German labels) when available.
        Relevance evaluation keeps ``chunk_nr`` internally.

        Args:
            metadata: Chunk metadata dictionary
            index: Fallback index if metadata is not available

        Returns:
            A source string such as ``Dokument (Seite 3)`` or with chapter/heading (German labels).
        """
        doc_name = get_document_name(metadata, index)
        source_parts = [doc_name]
        
        # Add chapter if available
        if metadata and metadata.get('chapter'):
            source_parts.append(f"Kapitel: {metadata.get('chapter')}")
        
        # Add heading if available
        if metadata and metadata.get('heading'):
            source_parts.append(f"Überschrift: {metadata.get('heading')}")
        
        # Page: get_document_name already appends "(Seite X)" when page_number is set
        return ", ".join(source_parts)
    
    # Build structured query information string
    query_info_parts = []
    if query_structure:
        query_info_parts.append(f"Prozessname: {query_structure.process_name}")
        query_info_parts.append(f"Domain: {query_structure.domain}")
        query_info_parts.append(f"Verfahrenstyp: {query_structure.procedure_type}")
        
        if query_structure.perspective:
            query_info_parts.append(f"Perspektive: {query_structure.perspective}")
        
        if query_structure.granularity:
            granularity_desc = _GRANULARITY_LABELS.get(
                query_structure.granularity, query_structure.granularity
            )
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
    
    # Build user prompt - simple and maintainable structure
    user_prompt_parts = []
    
    # 1. User request
    user_prompt_parts.append(f"Anfrage (Zielprozess): {user_request}")
    
    # 2. Query structure information (if available)
    if query_info:
        user_prompt_parts.append(f"\nInformationen zum Zielprozess:\n{query_info}")
    
    # 3. Retrieved documents with source information
    if retrieved_documents:
        user_prompt_parts.append("\nRelevante Textpassagen aus Dokumenten:")
        for i, (doc, meta) in enumerate(zip(
            retrieved_documents,
            retrieved_metadatas if retrieved_metadatas else [{}] * len(retrieved_documents)
        )):
            # Check if this is a document header
            if meta and meta.get('is_document_header', False):
                user_prompt_parts.append(f"\n{doc}")
            else:
                # Regular chunk: add with formatted source information
                source_info = format_chunk_source(meta, i)
                user_prompt_parts.append(f"\n[{source_info}]\n{doc}")
    user_prompt_parts.append(
        "Gib ausschließlich den Prozess-Draft im vorgegebenen xml-ähnlichen Format zurück."
        )
    
    user_prompt = "\n".join(user_prompt_parts)

    system_prompt_bpmn_text = (
        "ROLLE: Du bist Prozessexperte in der öffentlichen Verwaltung.\n"
        "AUFGABE: Erstelle ein sehr detailliertes und vollständiges XML-ähnliches Prozessmodell ausschließlich aus den bereitgestellten Textpassagen.\n\n"
        "REGELN für die Prozess Modellierung:\n"
        "- Achte darauf, dass ALLE erwähnten Prozesselemente (wie Akteure, Aktivitäten, Events, Gateways also Prüfungen oder Entscheidungen), die in den Textpassagen stehen, in dem finalen XML-artigen-Prozessmodell vorhanden sein.\n"
        "- Vergesse KEINE vorhandenen Schritte oder Akteure.\n"
        "- Du darfst KEINE zusätzlichen Inhalte erfinden.\n"
        "- Du darfst KEINE Informationen verwenden, die nicht explizit enthalten sind.\n"
        "- Sequenziere die Schritte logisch entsprechend der Textstruktur.\n\n"
        "- Entscheidungen und Prüfungen MÜSSEN mit Gateways modelliert werden, NICHT als Aktivitäten.\n"
        

        "QUELLENANGABEN (VERPFLICHTEND)\n\n"
        "Für JEDES Element MÜSSEN in dem Attribut source_document ALLE Referenzquellen enthalten sein, woraus der Prozessschritt/das Prozesselement abgeleitet wurde.\n"
        "- Darf NICHT leer sein.\n"
        "- Muss Dokumentennamen enthalten.\n"
        "- Falls vorhanden: (Seite X), Kapitel: X, Überschrift: Y übernehmen.\n"
        "- Mehrere Quellen → mit Semikolon trennen.\n"
        "- Keine Quelle weglassen!\n\n"
        
        "--------------------------------------------------\n"

        "REGELN für den erwarteten Output:\n"
        "- Gib nur XML-ähnlichen Inhalt aus.\n"
        "- Kein Markdown, keine Erklärungen, keine Kommentare.\n"
        "- Ausgabe beginnt mit <process> und endet mit </process>.\n\n"

        "PROZESSELEMENTE:\n\n"

        "START EVENT\n"
        "<startEvent id='' description='Antragsstellung gestartet' source_document='' />\n"
        "- description:''\n\n"

        "END EVENT\n"
        "<endEvent id='' description='Antragsstellung abgeschlossen' source_document='' />\n"
        "- description:''\n\n"

        "INTERMEDIATE EVENT\n"
        "<intermediateEvent type='' description='' id='' source_document='' />\n"
        "- type ∈ {timer, message, signal, condition}\n\n"

        "AKTIVITÄT\n"
        "<activity role='' action='' objects='' id='' source_document='' />\n"
        "- action: Nomen + Verb (Infinitiv).\n"
        "- Atomar!\n\n"

        "GATEWAYS\n"
        "- exclusiveGateway für entweder-oder.\n"
        "- parallelGateway für gleichzeitig.\n"
        "- inclusiveGateway für mehrere mögliche Pfade.\n"

        "--------------------------------------------------\n"
        "ID-REGELN\n"
        "- Alle IDs global eindeutig.\n"
        "- Keine Wiederverwendung.\n"
        "- Format z. B. s1, a1, g1, ev1, b1.\n\n"

        "--------------------------------------------------\n"
        "SEQUENZREGELN\n"
        "- Sequentielle Schritte einfach hintereinander.\n"
        "- KEIN Gateway für reine Sequenz.\n"
        "- Gateways nur bei Verzweigung oder Parallelität.\n\n"

        "--------------------------------------------------\n"
        "EVENT-REGELN\n"
        "- startEvent steht immer zuerst.\n"
        "- Jeder Pfad endet mit endEvent.\n"
        "- Events ersetzen KEINE Gateways.\n"
        "- Wenn unklar → als Aktivität modellieren.\n\n"

        "--------------------------------------------------\n"
        "BEISPIEL (zur Orientierung des Formats)\n\n"
        "<process>\n"
        "  <pool name='Antragsteller' source_document='Beispiel_Dokument.pdf:chunk_0'>\n"
        "  <startEvent id='s1' description='Antragsstellung gestartet' source_document='Beispiel_Dokument.pdf:chunk_1'/>\n\n"
        "  <activity role='Antragsteller' action='Antrag einreichen' objects='Antragsformular' id='a1' source_document='Beispiel_Gesetz.pdf:chunk_1'/>\n\n"
        "  <intermediateEvent type='message' description='[msg:m1][send] Antrag an Organisation senden' id='ev_msg_send_1' source_document='Beispiel_Dokument.pdf:chunk_1'/>\n\n"
        "  <exclusiveGateway id='g1' condition='' source_document='Beispiel_Dokument.pdf:chunk_2'>\n"
        "    <branch condition='Antrag vollständig' id='b1' source_document='Beispiel_Dokument.pdf:chunk_2'>\n"
        "      <activity role='Behörde' action='Antrag prüfen' objects='Antrag' id='a2' source_document='Beispiel_Dokument.pdf:chunk_2'/>\n"
        "    </branch>\n"
        "    <branch condition='Antrag nicht vollständig' id='b2' source_document='Beispiel_Dokument.pdf:chunk_2'>\n"
        "      <activity role='Behörde' action='Fehlende Informationen anfordern' objects='Antrag' id='a3' source_document='Beispiel_Gesetz.pdf:chunk_2'/>\n"
        "    </branch>\n"
        "  </exclusiveGateway>\n\n"
        "  <intermediateEvent type='timer' description='14 Tage sind vergangen' id='ev1' source_document='Beispiel_Dokument.pdf:chunk_3'/>\n\n"
        "  <activity role='Behörde' action='Finale Entscheidung treffen' objects='Antrag' id='a4' source_document='Beispiel_Interview.pdf:chunk_3'/>\n\n"
        "  <endEvent id='e1' description='Prozess abgeschlossen' source_document='Beispiel_Dokument.pdf:chunk_3'/>\n"
        "  </pool>\n"
        "  <pool name='Organisation' source_document='Beispiel_Dokument.pdf:chunk_0'>\n"
        "    <startEvent id='s2' description='Prozess starten' source_document='Beispiel_Dokument.pdf:chunk_1'/>\n\n"
        "    <intermediateEvent type='message' description='[msg:m1][receive] Antrag vom Antragsteller empfangen' id='ev_msg_recv_1' source_document='Beispiel_Dokument.pdf:chunk_1'/>\n\n"
        "    <activity role='Personalabteilung' action='Antrag prüfen' objects='Antragsformular' id='a5' source_document='Beispiel_Gesetz.pdf:chunk_1'/>\n\n"
        "    <activity role='IT-Abteilung' action='Konto erstellen' objects='Antragsformular' id='a6' source_document='Beispiel_Gesetz.pdf:chunk_1'/>\n\n"
        "    <endEvent id='e2' description='Prozess abgeschlossen' source_document='Beispiel_Dokument.pdf:chunk_3'/>\n"
        "  </pool>\n"
        "</process>\n\n"
        "ENDE DES BEISPIELS\n"

        "Abschließende HINWEISE:\n"
        "Erzeuge nun das vollständige Prozessmodell und gebe nur die xml-ähnliche Ausgabe aus."
        )

    n = len(user_prompt)
    preview = (
        user_prompt
        if n <= 8000
        else user_prompt[:8000] + f"\n... [{n} chars total]"
    )
    _bpmn_debug(f"user_prompt draft agent with structure:\n{preview}")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt_bpmn_text},
        {"role": "user", "content": user_prompt},
    ]

    # Call LLM
    try:
        text = await call_ollama_chat(
            messages=messages,
            model=settings.agentic_draft_model,
        )
        if not text:
            # Fallback if call_ollama_chat returns None or empty string
            text = user_request
        return ProcessDraft(text_description=text), user_prompt
    except Exception as e:
        logger.error(f"Error in run_draft_agent_with_structure: {e}")
        # Return fallback draft with user request
        return ProcessDraft(text_description=user_request), user_prompt


async def run_bpmn_agent_with_structure(
    draft_text: str,
    query_structure: Optional[QueryStructure]
) -> BPMNModelJsonNested:
    """
    Create a BPMN JSON model via Ollama from draft with query_structure orientation.
    
    Args:
        draft_text: Textual process description (draft)
        query_structure: Optional structured query information
        
    Returns:
        BPMNModelJsonNested model
    """
    # Build structured query information string
    query_info_parts = []
    if query_structure:
        query_info_parts.append(f"Prozessname: {query_structure.process_name}")
        query_info_parts.append(f"Domain: {query_structure.domain}")
        query_info_parts.append(f"Verfahrenstyp: {query_structure.procedure_type}")
        
        if query_structure.perspective:
            query_info_parts.append(f"Perspektive: {query_structure.perspective}")
        
        if query_structure.granularity:
            granularity_desc = _GRANULARITY_LABELS.get(
                query_structure.granularity, query_structure.granularity
            )
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
    
    # Build user prompt
    prompt_parts = [f"Prozessbeschreibung:\n{draft_text}"]
    
    if query_info:
        prompt_parts.append(f"\n\nStrukturierte Prozessinformationen (query_structure):\n{query_info}")
    
    user_prompt = "\n".join(prompt_parts)
    
    system_prompt_bpmn_text= (
        "ROLLE: Du bist ein Prozessmodellierungsexperte.\n\n"
        "AUFGABE: Deine Aufgabe ist es, ein sehr detailliertes BPMN-konformes, verschachteltes JSON-Prozessmodell\n"
        
        "REGELN für die BPMN-Modellierung:\n"
        "- Vergesse KEINE vorhandenen Schritte oder Akteure. ALLE Prozesselemente (wie Akteure, Aktivitäten, Events, Gateways), die in dem Draft zwischen <process> und </process> stehen, MÜSSEN in dem finalen JSON-Prozessmodell vorhanden sein..\n"
        "- Du darfst KEINE zusätzlichen Inhalte erfinden.\n"
        "- Du darfst KEINE Informationen verwenden, die nicht explizit im Draft enthalten sind.\n"
        "- Achte darauf, dass die Beschreibung eines Prozessaktivität IMMER nach dem Muster: Nomen + Verb formuliert ist: Beispiel \"Bescheid versenden\" und NICHT \"Versand des Bescheids\"\n\n"
        "- Jede Teilaktivität muss eine eigene Quellenangabe haben"
        "- Entscheidungen und Prüfungen MÜSSEN mit Gateways modelliert werden, NICHT als Aktivitäten.\n"
        "- Achte darauf, dass die Sequenz der Prozessschritte korrekt, sinnvoll und logisch ist, insbesondere bei Gateways.\n"
        
        "VERPFLICHTENDE QUELLENANGABEN \n\n"
        "- JEDES BPMN-Element (Task, Event, Gateway) MUSS ein `documentation`-Attribut besitzen.\n"
        "- Die Quelle DARF AUSSCHLIESSLICH aus dem Draft übernommen werden.\n"
        "- Die Quelle ist im Draft im Attribut `source_document` enthalten.\n\n"
        "Regeln:\n"
        "- Extrahiere den vollständigen Wert von source_document unverändert.\n"
        "- KEINE neuen Quellen erfinden - NUR Quellen aus dem Draft übernehmen.\n"
        "- KEINE Quellen zusammenfassen oder interpretieren.\n"
        "- KEINE Fallbacks wie \"Quelle nicht angegeben\" - erfinde KEINE Quelle!\n\n"
        "Format für `documentation`:\n"
        "\"<source_document aus dem Draft - inkl. Kapitel und Überschrift wenn angegeben>\"\n\n"
        "Beispiel:\n"
        "Draft:\n"
        "<activity ... source_document='Verwaltungsrichtlinie.pdf (Seite 3), Kapitel: 3. Allgemeines, Überschrift: 3.2 Antragsstellung'/>\n\n"
        
        
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
        
        "Abschließende HINWEISE:\n\n"
        "- Gib AUSSCHLIESSLICH ein gültiges JSON zurück.\n"
        "- Kein erklärender Text außerhalb des JSON.\n"
        "- Das JSON MUSS BPMN-strukturell korrekt, vollständig verschachtelt und konsistent sein.\n"
        
    )
    
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt_bpmn_text},
        {"role": "user", "content": user_prompt},
    ]
    
   
    
    raw = await call_ollama_json(
        messages=messages,
        model=settings.agentic_bpmn_model,
    )
    
    try:
        data = json.loads(raw)
        
        # Normalize event IDs: remove string IDs from startEvent/endEvent (except messageReceive)
        def normalize_event_ids(obj: Any) -> Any:
            """Normalize event IDs: remove string IDs from startEvent/endEvent, keep int IDs for messageReceive events."""
            if isinstance(obj, dict):
                # Fix startEvent: remove string IDs, keep only int IDs for messageReceive
                if 'startEvent' in obj and isinstance(obj['startEvent'], dict):
                    start_event = obj['startEvent']
                    if 'id' in start_event:
                        # Only keep id if it's a messageReceive event (needs int ID)
                        if start_event.get('subType') == 'messageReceive':
                            # Try to convert string to int if possible
                            if isinstance(start_event['id'], str):
                                try:
                                    start_event['id'] = int(start_event['id'])
                                except (ValueError, TypeError):
                                    del start_event['id']
                        else:
                            # Remove id for non-messageReceive events
                            del start_event['id']
                
                # Fix endEvent: remove string IDs (endEvent should not have id)
                if 'endEvent' in obj and isinstance(obj['endEvent'], dict):
                    end_event = obj['endEvent']
                    if 'id' in end_event:
                        # Only keep id if it's a messageReceive event
                        if end_event.get('subType') == 'messageReceive':
                            if isinstance(end_event['id'], str):
                                try:
                                    end_event['id'] = int(end_event['id'])
                                except (ValueError, TypeError):
                                    del end_event['id']
                        else:
                            del end_event['id']
                
                # Recursively process all values
                for key, value in obj.items():
                    obj[key] = normalize_event_ids(value)
            elif isinstance(obj, list):
                return [normalize_event_ids(item) for item in obj]
            return obj
        
        # Normalize event IDs before validation
        data = normalize_event_ids(data)
        
        # Validate and parse as nested model with Pydantic
        bpmn_model = BPMNModelJsonNested(**data)
          
        return bpmn_model
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing error in BPMN agent with structure: {e}"
        logger.error(error_msg)
        logger.debug("Raw LLM response (first 500 chars): %s", raw[:500])
        _bpmn_debug(error_msg)
        _bpmn_debug(f"Raw response (truncated): {raw[:200]}...")
        # Fallback: empty model
        return BPMNModelJsonNested(pools=[])
    except ValidationError as e:
        # Log detailed validation errors
        error_msg = "Pydantic validation error in BPMN agent with structure"
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
        
        _bpmn_debug(error_msg)
        _bpmn_debug(f"Validation failed with {len(validation_errors)} error(s):")
        for i, error in enumerate(validation_errors[:5], 1):
            loc = " -> ".join(str(x) for x in error.get("loc", []))
            _bpmn_debug(f"   {i}. Field '{loc}': {error.get('msg', 'No message')}")
        if len(validation_errors) > 5:
            _bpmn_debug(f"   ... and {len(validation_errors) - 5} more error(s)")
        _bpmn_debug(f"Raw response (truncated): {raw[:200]}...")
        
        # Fallback: empty model
        return BPMNModelJsonNested(pools=[])


async def run_bpmn_agent_revision(
    original_bpmn_json: str,
    validation_feedback: ValidationResultSetting4,
    query_structure: Optional[QueryStructure] = None,
    draft_text: Optional[str] = None,
    retrieved_documents: Optional[List[str]] = None,
    retrieved_metadatas: Optional[List[Dict[str, Any]]] = None
) -> BPMNModelJsonNested:
    """
    Revise BPMN-JSON based on validation feedback (reactive corrections only).

    Strict revision rules:
    - Add only missing elements listed in ``missing_elements``
    - Remove only hallucinated elements listed in ``hallucinated_elements``

    Not allowed:
    - Inventing new elements
    - Restructuring the flow
    - Changing labels without explicit instruction
    - Expanding scope beyond feedback

    Args:
        original_bpmn_json: Original BPMN-JSON as string
        validation_feedback: Validation feedback with issues
        query_structure: Optional structured query information
        draft_text: Optional process draft (preferred source for missing elements; BPMN was generated from draft)
        retrieved_documents: Optional retrieved chunks (fallback source for missing elements)
        retrieved_metadatas: Chunk metadata (German ``[Dokument (Seite X)]``-style source lines in prompts)

    Returns:
        Revised ``BPMNModelJsonNested``
    """
    # Build validation feedback summary
    feedback_parts = []
    
    if validation_feedback.missing_elements:
        feedback_parts.append("FEHLENDE ELEMENTE (müssen ergänzt werden):")
        for elem in validation_feedback.missing_elements:
            feedback_parts.append(
                f"  - [{elem.element_type}] '{elem.element_label}' "
                f"(Quelle: {elem.source_chunk_reference}): {elem.description}"
            )
        feedback_parts.append("")
    
    if validation_feedback.hallucinated_elements:
        feedback_parts.append("HALLUZINIERTE ELEMENTE (müssen entfernt werden):")
        for elem in validation_feedback.hallucinated_elements:
            feedback_parts.append(
                f"  - [{elem.element_type}] '{elem.element_label}': {elem.description}"
            )
        feedback_parts.append("")
    
    if validation_feedback.structural_issues:
        feedback_parts.append("STRUKTURELLE PROBLEME (müssen korrigiert werden - siehe issue_type für Korrekturart):")
        for issue in validation_feedback.structural_issues:
            feedback_parts.append(
                f"  - [{issue.issue_type}] '{issue.element_label}': {issue.description}"
            )
        feedback_parts.append("")
    
    validation_summary = "\n".join(feedback_parts) if feedback_parts else "Keine kritischen Probleme gefunden."
    
    # Build query structure information if available
    query_info_parts = []
    if query_structure:
        query_info_parts.append(f"Prozessname: {query_structure.process_name}")
        if query_structure.perspective:
            query_info_parts.append(f"Perspektive: {query_structure.perspective}")
        if query_structure.granularity:
            granularity_desc = _GRANULARITY_LABELS.get(
                query_structure.granularity, query_structure.granularity
            )
            query_info_parts.append(f"Detaillierungsgrad: {granularity_desc}")
    
    query_info = "\n".join(query_info_parts) if query_info_parts else ""
    
    # Build context for missing elements: prefer draft (BPMN was generated from it), else chunks
    retrieved_context = ""
    if draft_text:
        retrieved_context = f"--- PROZESS-DRAFT (Quelle für fehlende Elemente) ---\n{draft_text}"
    elif retrieved_documents:
        def _format_chunk_source_bpmn_rev(metadata: Dict[str, Any], index: int) -> str:
            def _get_doc_name(meta: Dict[str, Any], idx: int) -> str:
                if meta:
                    name = (
                        meta.get('file_name') or meta.get('document_title') or meta.get('title') or
                        (Path(meta.get('file_path', '')).name if meta.get('file_path') else None)
                    )
                    if name:
                        if meta.get('page_number') is not None:
                            return f"{name} (Seite {meta.get('page_number')})"
                        return name
                return f"Dokument {idx + 1}"
            doc_name = _get_doc_name(metadata, index)
            parts = [doc_name]
            if metadata and metadata.get('chapter'):
                parts.append(f"Kapitel: {metadata.get('chapter')}")
            if metadata and metadata.get('heading'):
                parts.append(f"Überschrift: {metadata.get('heading')}")
            return ", ".join(parts)
        
        context_parts = []
        metadatas = retrieved_metadatas if retrieved_metadatas else [{}] * len(retrieved_documents)
        for i, doc in enumerate(retrieved_documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            if meta.get('is_document_header', False):
                context_parts.append(doc)
            else:
                source_info = _format_chunk_source_bpmn_rev(meta, i)
                context_parts.append(f"[{source_info}]\n{doc}")
        retrieved_context = "\n\n".join(context_parts)
    
    # Build user prompt
    user_prompt_parts = [
        f"Originales BPMN-JSON Modell:\n{original_bpmn_json}\n\n",
        f"Validation-Feedback:\n{validation_summary}\n\n"
        f"Wichtig!:\n"
        "- Änderungen nur mit expliziter Aufforderung im Feedback\n\n"
    ]
    
    if retrieved_context:
        user_prompt_parts.append(f"{retrieved_context}\n\n")
    
    if query_info:
        user_prompt_parts.append(f"Query-Informationen (für Orientierung):\n{query_info}\n\n"
         f"Wichtig!:\n"
        "- Änderungen nur mit expliziter Aufforderung im Feedback\n\n")
    
    user_prompt = "".join(user_prompt_parts)
    
    
    system_prompt = (
        "ROLLE: Du bist ein Überarbeitungs-Agent für BPMN-Prozessmodelle (JSON).\n\n"
        "AUFGABE: Ein bestehendes BPMN-JSON auf Basis des bereitgestellten Validation-Feedbacks und unten stehender Hinweise zu korrigieren.\n\n"
        "Du darfst das BPMN-Modell nur gezielt überarbeiten.\n\n"
        "--------------------------------------------------\n"
        
        "REGELN für die Korrektur\n\n"
        "1. Fehlende BPMN-Elemente an der richtigen Stelle im Ablauf ergänzen, die im Validation-Feedback explizit unter \"missing_elements\" genannt sind.\n"
        "   Dies betrifft Tasks, Events, Gateways oder Branches.\n"
        "   Orientiere dich hierfür an den Chunks und der source_chunk_reference.\n\n"
        "2. Halluzinierte oder nicht passende Elemente entfernen, die im Validation-Feedback explizit unter \"hallucinated_elements\" genannt sind.\n\n"
        "3. Strukturelle Probleme korrigieren basierend auf issue_type und den Chunks:\n"
        "   - gateway_missing / gateway_incorrect / gateway_condition_missing: Gateways einfügen oder korrigieren, Bedingung im CONDITION-Attribut angeben\n"
        "   - flow_logic / sequence_error: Reihenfolge oder Kontrollfluss korrigieren\n"
        "   - naming_violation: Bezeichnungen korrigieren\n\n"
        "4. Folgende MODELLIERUNGSPRINZIPIEN IMMER berücksichtigen:\n\n"

        "1) Explizite Kontrolllogik\n"
        "- Modellierung von Entscheidungen immer mit Gateways.\n"
        "- Wenn Bedingungen, Alternativen oder Parallelität erkennbar sind → Gateway modellieren.\n"
        "- XOR-Gateways MUSS die Bedingung immer im CONDITION-Attribut angegeben werden. Beispiel: {\"type\":\"xor\",\"condition\":\"Annahme oder Ablehnung?\",\"branches\":[...]}\n\n"
        "- Jedes XOR-Gateway DARF genau EINE Entscheidung und Bedingung abbilden.\n"
        "- Lieber ein Gateway zu viel als implizite Logik in Aktivitäten.\n\n"

        "2) Atomare Aktivitäten\n"
        "- Jede Aktivität darf genau EINE fachliche Handlung enthalten.\n"
        "- Keine Kombinationen wie 'prüfen und entscheiden'.\n"
        "- Bei mehreren Handlungen → mehrere separate Aktivitäten modellieren.\n\n"

        "3) Events\n"
        "- Passe bei Bedarf ALLE Startevents an, damit sie zum Zielprozess passen z.B. 'Anstragsstellung gestartet'.\n"
        "- Passe bei Bedarf ALLE Endevents an, damit sie zum Zielprozess passen z.B. 'Anstragsstellung abgeschlossen'.\n"
        "--------------------------------------------------\n"
        "DU DARFST NICHT:\n"
        "- Neue Elemente erfinden, die nicht im Validation-Feedback stehen\n"
        "- Den Flow restrukturieren außer explizit im Feedback gefordert\n"
        "- Elemente hinzufügen, die nicht explizit als 'missing_elements' markiert sind\n\n"
        "--------------------------------------------------\n"
        "VERPFLICHTENDE Arbeitsweise\n"
        "1. Lies das originale BPMN-JSON vollständig.\n"
        "2. Lies das Validation-Feedback vollständig.\n"
        "3. Arbeite das Feedback in dieser Reihenfolge ab:\n"
        "   1) missing_elements → Ergänzen an der logisch korrekten Stelle (Chunks als Quelle)\n"
        "   2) hallucinated_elements → Entfernen\n"
        "   3) structural_issues → Gemäß issue_type korrigieren\n"
        "4. Überprüfe am Ende, dass alles BPMN-konform bezeichnet ist und passe es an, falls notwendig:\n"
        "   - Aktivitäten: '' (Bsp. \"Antrag stellen\", \"Formular ausfüllen\", \"Antrag an Fachbeirat weiterleiten\")\n"
        "   - Start- und End-Events: '' (Bsp. \"Anstragsstellung gestartet\", \"Anstragsstellung abgeschlossen\")\n"
        "   - Rollen: '' (Bsp. \"Antragsteller\", \"Behörde\", \"Sachbearbeiter\")\n\n"
        "5. Berücksichtige die Modellierungsprinzipien.\n"
        "6. Stelle sicher, dass das korrigierte BPMN-JSON strukturell gültig bleibt.\n\n"
        "--------------------------------------------------\n"
       
       "REGELN für die Überarbeitung des BPMN JSONs \n\n"
        "1. Pools repräsentieren separate unabhängige Organisationen. Lanes sind Abteilungen oder Teilnehmer innerhalb eines Pools. \n"
        "2. Verwende send und receive message events und message flows nur zwischen Pools, nicht zwischen Lanes.\n"
        "Beispiel Input: \"Kunde sendet eine Anfrage an eine Firma\".\n"
        "Beispiel Output: {\"pools\":[{\"name\":\"Kunde\",\"dataObjects\":[{\"name\":\"Anfrage\",\"type\":\"data-file\"}],\"lanes\":[{\"name\":\"Kunde\"}],\"startEvent\":{\"subType\":\"default\",\"name\":\"Start Anfrage\",\"laneIndex\":0},\"process\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Anfrage schreiben\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 1)\"},{\"type\":\"event\",\"subType\":\"messageSend\",\"name\":\"Anfrage an Firma senden\",\"receiveEventId\":1,\"laneIndex\":0,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 1)\"}],\"endEvent\":{\"name\":\"Anfrage gesendet\",\"laneIndex\":0}}]}\n\n"
        "3. Verwende send- und receive-Message-Events eins-zu-eins. Verwende exklusive Gateways, um basierend auf einer Bedingung einen Pfad zu wählen. Verwende event-basierte Gateways, um basierend auf eingehenden Events einen Pfad zu wählen.\n"
        "Beispiel Input: \"Firma bereitet ein Angebot vor und sendet es an den Kunden, der entscheidet, ob er es annimmt oder ablehnt. Wenn innerhalb von 10 Tagen keine Antwort eingeht, leitet die Firma eine Nachfrage ein\".\n"
        "Beispiel Output: {\"pools\":[{\"name\":\"Kunde\",\"dataObjects\":[{\"name\":\"Antwort\",\"type\":\"data-file\"},{\"name\":\"Angebot\",\"type\":\"data-file\"}],\"lanes\":[{\"name\":\"Kunde\"}],\"startEvent\":{\"subType\":\"messageReceive\",\"id\":3,\"name\":\"Angebot von Firma erhalten\",\"laneIndex\":0,\"writeDataObjectRefs\":[1],\"documentation\":\"Dokument: Beispiel.pdf (Seite 2)\"},\"process\":[{\"type\":\"xor\",\"condition\":\"Annahme oder Ablehnung?\",\"branches\":[{\"label\":\"Annahme\",\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Annahme schreiben\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 3)\"},{\"type\":\"event\",\"subType\":\"messageSend\",\"name\":\"Annahme an Firma senden\",\"receiveEventId\":1,\"laneIndex\":0,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 3)\"}]},{\"label\":\"Ablehnung\",\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Ablehnung schreiben\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 4)\"},{\"type\":\"event\",\"subType\":\"messageSend\",\"name\":\"Ablehnung an Firma senden\",\"receiveEventId\":2,\"laneIndex\":0,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf, Chunk: 3\"}]}],\"laneIndex\":0,\"documentation\":\"Dokument: Beispiel.pdf (Seite 2)\"}],\"endEvent\":{\"name\":\"Antwort gesendet\",\"laneIndex\":0}}]}\n\n"
        "!Achtung: Bei XOR-Gateways MUSS die Bedingung immer im CONDITION-Attribut angegeben werden. Beispiel: {\"type\":\"xor\",\"condition\":\"Annahme oder Ablehnung?\",\"branches\":[...]}\n\n"
        "4. Verwende Datenobjekte, nicht Message-Events, für die Kommunikation zwischen Lanes desselben Pools.\n"
        "Beispiel Input: \"Die Vertriebsabteilung sendet einen Bericht an die Finanzabteilung\".\n"
        "Beispiel Output: {\"pools\":[{\"name\":\"Firma\",\"dataObjects\":[{\"name\":\"Bericht\",\"type\":\"data-file\"}],\"lanes\":[{\"name\":\"Vertriebsabteilung\"},{\"name\":\"Finanzabteilung\"}],\"startEvent\":{\"subType\":\"default\",\"name\":\"Prozess starten\",\"laneIndex\":0},\"process\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Bericht erstellen\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite X)\"},{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Bericht prüfen\",\"laneIndex\":1,\"readDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite 2)\"}],\"endEvent\":{\"name\":\"Bericht geprüft\",\"laneIndex\":1}}]}\n\n"
        "5. Verwende data-file nur für digitale oder physische Dateien und data-store für persistente Datenspeicher und Datenbanksysteme.\n"
        "Beispiel Input: \"Der Händler erhält eine Bestellung und prüft die Lagerdatenbank\".\n"
        "Beispiel Output: {\"pools\":[{\"name\":\"Händler\",\"dataObjects\":[{\"name\":\"Bestellung\",\"type\":\"data-file\"},{\"name\":\"Lagerdatenbank\",\"type\":\"data-store\"}],\"lanes\":[{\"name\":\"Bestell- und Lagerverwaltung\"}],\"startEvent\":{\"subType\":\"default\",\"name\":\"Bestellprozess starten\",\"laneIndex\":0},\"process\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Bestellung erhalten\",\"laneIndex\":0,\"writeDataObjectRefs\":[0],\"documentation\":\"Dokument: Beispiel.pdf (Seite X)\"},{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Lager prüfen\",\"laneIndex\":0,\"readDataObjectRefs\":[0,1],\"documentation\":\"Dokument: Beispiel.pdf (Seite Y)\"}],\"endEvent\":{\"name\":\"Bestellung bearbeitet\",\"laneIndex\":0}}]}\n\n"
        "6. Verwende parallele Gateways (type \"parallel\"), wenn mehrere Aktivitäten GLEICHZEITIG ablaufen. Struktur: {\"type\":\"parallel\",\"branches\":[{\"branch\":[<tasks>],\"event\":null},{\"branch\":[<tasks>],\"event\":null}],\"laneIndex\":0,\"documentation\":\"...\"}. KEINE condition, KEINE labels - nur branches mit branch-Arrays.\n"
        "Example Input: \"Die Behörde ordnet gleichzeitig Abstandsgebot im öffentlichen Raum und Maskenpflicht an.\"\n"
        "Example Output: {\"type\":\"parallel\",\"branches\":[{\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Abstandsgebot im öffentlichen Raum anordnen\",\"laneIndex\":0,\"readDataObjectRefs\":null,\"writeDataObjectRefs\":null,\"documentation\":\"Verordnung.pdf (Seite 1)\"}],\"event\":null},{\"branch\":[{\"type\":\"task\",\"subType\":\"abstract\",\"name\":\"Maskenpflicht anordnen\",\"laneIndex\":0,\"readDataObjectRefs\":null,\"writeDataObjectRefs\":null,\"documentation\":\"Verordnung.pdf (Seite 1)\"}],\"event\":null}],\"laneIndex\":0,\"documentation\":\"Verordnung.pdf (Seite 1)\"}\n\n"
       
        "--------------------------------------------------\n"
        "Abschließende HINWEISE\n\n"
        "- Gib AUSSCHLIESSLICH das korrigierte BPMN-JSON zurück.\n"
        "- KEIN zusätzlicher Text, KEINE Erklärungen, KEIN Markdown.\n"
        "- Das Format muss identisch zum originalen Schema sein.\n\n"
        "--------------------------------------------------\n"
        "WICHTIGE LEITLINIE\n\n"
        "Führe nur die minimal notwendigen Änderungen durch, die explizit im Validation-Feedback gefordert werden."
    )
    

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    raw = await call_ollama_json(
        messages=messages,
        model=settings.agentic_revision_model,
    )
    
    try:
        data = json.loads(raw)
        # Validate and parse as nested model with Pydantic
        return BPMNModelJsonNested(**data)
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing error in BPMN revision agent: {e}"
        logger.error(error_msg)
        logger.debug("Raw LLM response (first 500 chars): %s", raw[:500])
        _bpmn_debug(error_msg)
        _bpmn_debug(f"Raw response (truncated): {raw[:200]}...")
        # Fallback: try to return original model (parse it first)
        try:
            original_data = json.loads(original_bpmn_json)
            return BPMNModelJsonNested(**original_data)
        except Exception as exc:
            logger.warning("BPMN revision: could not parse original BPMN JSON for fallback: %s", exc)
            return BPMNModelJsonNested(pools=[])
    except ValidationError as e:
        # Log detailed validation errors
        error_msg = "Pydantic validation error in BPMN revision agent"
        logger.error("%s: %s", error_msg, str(e))
        
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
        
        _bpmn_debug(error_msg)
        _bpmn_debug(f"Validation failed with {len(validation_errors)} error(s):")
        for i, error in enumerate(validation_errors[:5], 1):
            loc = " -> ".join(str(x) for x in error.get("loc", []))
            _bpmn_debug(f"   {i}. Field '{loc}': {error.get('msg', 'No message')}")
        if len(validation_errors) > 5:
            _bpmn_debug(f"   ... and {len(validation_errors) - 5} more error(s)")
        _bpmn_debug(f"Raw response (truncated): {raw[:200]}...")
        
        # Fallback: try to return original model
        try:
            original_data = json.loads(original_bpmn_json)
            return BPMNModelJsonNested(**original_data)
        except Exception as exc:
            logger.warning("BPMN revision: could not parse original BPMN JSON for fallback: %s", exc)
            return BPMNModelJsonNested(pools=[])


async def run_draft_agent_revision(
    original_draft_text: str,
    validation_feedback: ValidationResultSetting4,
    query_structure: Optional[QueryStructure] = None,
    retrieved_documents: Optional[List[str]] = None,
    retrieved_metadatas: Optional[List[Dict[str, Any]]] = None
) -> ProcessDraft:
    """
    Revise Draft-Text based on validation feedback (reactive corrections only).
    
    Revision-Regeln (strikt):
    - Nur fehlende Elemente ergänzen (aus missing_elements)
    - Nur halluzinierte Elemente entfernen (aus hallucinated_elements)
    - Scope-Verletzungen korrigieren
    
    NICHT erlaubt:
    - Neue Elemente erfinden
    - Struktur komplett umschreiben
    - Labels ändern ohne expliziten Hinweis
    
    Args:
        original_draft_text: Original draft text description
        validation_feedback: Validation feedback with issues
        query_structure: Optional structured query information
        retrieved_documents: Retrieved Chunks (source for missing elements)
        
    Returns:
        ProcessDraft (revised draft)
    """
    # Build validation feedback summary
    feedback_parts = []
    
    if validation_feedback.missing_elements:
        feedback_parts.append("FEHLENDE ELEMENTE (müssen an der richtigen Stelle im Ablauf ergänzt werden):")
        for elem in validation_feedback.missing_elements:
            feedback_parts.append(
                f"  - [{elem.element_type}] '{elem.element_label}' "
                f"(Quelle: {elem.source_chunk_reference}): {elem.description}"
            )
        feedback_parts.append("")
    
    if validation_feedback.hallucinated_elements:
        feedback_parts.append("HALLUZINIERTE ELEMENTE (müssen entfernt werden):")
        for elem in validation_feedback.hallucinated_elements:
            feedback_parts.append(
                f"  - [{elem.element_type}] '{elem.element_label}': {elem.description}"
            )
        feedback_parts.append("")
    
    if validation_feedback.structural_issues:
        feedback_parts.append("STRUKTURELLE PROBLEME (müssen korrigiert werden - siehe issue_type für Korrekturart):")
        for issue in validation_feedback.structural_issues:
            feedback_parts.append(
                f"  - [{issue.issue_type}] '{issue.element_label}': {issue.description}"
            )
        feedback_parts.append("")
    
    validation_summary = "\n".join(feedback_parts) if feedback_parts else "Keine kritischen Probleme gefunden."
    
    # Build query structure information if available
    query_info_parts = []
    if query_structure:
        query_info_parts.append(f"Prozessname: {query_structure.process_name}")
        if query_structure.perspective:
            query_info_parts.append(f"Perspektive: {query_structure.perspective}")
        if query_structure.granularity:
            granularity_desc = _GRANULARITY_LABELS.get(
                query_structure.granularity, query_structure.granularity
            )
            query_info_parts.append(f"Detaillierungsgrad: {granularity_desc}")
    
    query_info = "\n".join(query_info_parts) if query_info_parts else ""
    
    # Helper function to format chunk source (defined locally to avoid dependency on outer scope)
    def format_chunk_source_revision(metadata: Dict[str, Any], index: int) -> str:
        """Format chunk source information: document name with Seite (page) when available."""
        def get_document_name_revision(metadata: Dict[str, Any], index: int) -> str:
            """Extract document name from metadata, fallback to index if not available."""
            if metadata:
                name = (
                    metadata.get('file_name') or
                    metadata.get('document_title') or
                    metadata.get('title') or
                    (Path(metadata.get('file_path', '')).name if metadata.get('file_path') else None)
                )
                if name:
                    if metadata.get('page_number') is not None:
                        return f"{name} (Seite {metadata.get('page_number')})"
                    return name
            return f"Dokument {index + 1}"
        
        doc_name = get_document_name_revision(metadata, index)
        source_parts = [doc_name]
        
        # Add chapter if available
        if metadata and metadata.get('chapter'):
            source_parts.append(f"Kapitel: {metadata.get('chapter')}")
        
        # Add heading if available
        if metadata and metadata.get('heading'):
            source_parts.append(f"Überschrift: {metadata.get('heading')}")
        
        return ", ".join(source_parts)
    
    # Build context from retrieved documents
    # Document headers (is_document_header) are shown as separators only; content chunks get [source_info]
    retrieved_context = ""
    if retrieved_documents:
        context_parts = []
        metadatas = retrieved_metadatas if retrieved_metadatas else [{}] * len(retrieved_documents)
        for i, doc in enumerate(retrieved_documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            if meta.get('is_document_header', False):
                context_parts.append(doc)
            else:
                source_info = format_chunk_source_revision(meta, i)
                context_parts.append(f"[{source_info}]\n{doc}")
        retrieved_context = "\n\n".join(context_parts)
    
    # Build user prompt
    user_prompt_parts = [
        f"Originaler Prozess-Draft:\n{original_draft_text}\n\n",
        f"Validation-Feedback:\n{validation_summary}\n\n"
    ]
    
    if retrieved_context:
        user_prompt_parts.append(
            f"--- ABGERUFENE CHUNKS (Quelle für fehlende Elemente) ---\n{retrieved_context}\n\n"
        )
    user_prompt_parts.append(
        f"Wichtig!:\n"
        "- Änderungen nur mit expliziter Aufforderung im Feedback und basierend auf dem Feedback und den Chunks\n"
        "- Behalte die BPMN-orientierte XML-artige Struktur bei\n"
        "- Ergänze fehlende Elemente an der passenden Stelle im Ablauf\n"
        "- Entferne halluzinierte Elemente vollständig\n"
        "- Orientiere dich dabei an den Descriptions und den Chunks\n"
    )
    
    if query_info:
        user_prompt_parts.append(f"Query-Informationen (für Orientierung):\n{query_info}\n\n")
    
    user_prompt = "".join(user_prompt_parts)

    system_prompt_bpmn_text = (
        "ROLLE: Du bist ein Überarbeitungs-Agent für Prozess-Drafts im XML-artigen Format.\n\n"
        "AUFGABE: Ein bestehenden Prozess-Draft auf Basis des bereitgestellten Validation-Feedbacks und unten stehender Hinweise zu korrigieren.\n\n"
        "Du darfst den Draft nur gezielt überarbeiten.\n\n"
        "--------------------------------------------------\n"
        
        "REGELN für die Korrektur\n\n"
        "1. Fehlende BPMN-Elemente an der richtigen Stelle im Ablauf ergänzen, die im Validation-Feedback explizit unter \"missing_elements\" genannt sind.\n"
        "   Dies betrifft Tasks, Events, Gateways oder Branches.\n"
        "   Orientiere dich hierfür an den Chunks und der source_chunk_reference.\n\n"
        "2. Halluzinierte oder nicht passende Elemente entfernen, die im Validation-Feedback explizit unter \"hallucinated_elements\" genannt sind.\n\n"
        "3. Strukturelle Probleme korrigieren basierend auf issue_type und den Chunks:\n"
        "   - gateway_missing / gateway_incorrect / gateway_condition_missing: Gateways einfügen oder korrigieren, Bedingung im CONDITION-Attribut angeben\n"
        "   - flow_logic / sequence_error: Reihenfolge oder Kontrollfluss korrigieren\n"
        "   - naming_violation: Bezeichnungen korrigieren\n\n"
        "4. Folgende MODELLIERUNGSPRINZIPIEN IMMER berücksichtigen:\n\n"

        "1) Explizite Kontrolllogik\n"
        "- Modellierung von Entscheidungen immer mit Gateways.\n"
        "- Wenn Bedingungen, Alternativen oder Parallelität erkennbar sind → Gateway modellieren.\n"
        "- XOR-Gateways MUSS die Bedingung immer im CONDITION-Attribut angegeben werden. Beispiel: {\"type\":\"xor\",\"condition\":\"Annahme oder Ablehnung?\",\"branches\":[...]}\n\n"
        "- Jedes XOR-Gateway DARF genau EINE Entscheidung und Bedingung abbilden.\n"
        "- Lieber ein Gateway zu viel als implizite Logik in Aktivitäten.\n\n"

        "2) Atomare Aktivitäten\n"
        "- Jede Aktivität darf genau EINE fachliche Handlung enthalten.\n"
        "- Keine Kombinationen wie 'prüfen und entscheiden'.\n"
        "- Bei mehreren Handlungen → mehrere separate Aktivitäten modellieren.\n\n"

        "3) Events\n"
        "- Passe bei Bedarf ALLE Startevents an, damit sie zum Zielprozess passen z.B. 'Anstragsstellung gestartet'.\n"
        "- Passe bei Bedarf ALLE Endevents an, damit sie zum Zielprozess passen z.B. 'Anstragsstellung abgeschlossen'.\n"
        "--------------------------------------------------\n"
        "DU DARFST NICHT:\n"
        "- Neue Elemente erfinden, die nicht im Validation-Feedback stehen\n"
        "- Den Flow restrukturieren außer explizit im Feedback gefordert\n"
        "- Elemente hinzufügen, die nicht explizit als 'missing_elements' markiert sind\n\n"
        "--------------------------------------------------\n"
        "VERPFLICHTENDE Arbeitsweise\n"
        "1. Lies den originalen Prozess-Draft vollständig.\n"
        "2. Lies das Validation-Feedback vollständig.\n"
        "3. Arbeite das Feedback in dieser Reihenfolge ab:\n"
        "   1) missing_elements → Ergänzen an der logisch korrekten Stelle (Chunks als Quelle)\n"
        "   2) hallucinated_elements → Entfernen\n"
        "   3) structural_issues → Gemäß issue_type korrigieren\n"
        "4. Überprüfe am Ende, dass alles BPMN-konform bezeichnet ist und passe es an, falls notwendig:\n"
        "   - Aktivitäten: '' (Bsp. \"Antrag stellen\", \"Formular ausfüllen\", \"Antrag an Fachbeirat weiterleiten\")\n"
        "   - Start- und End-Events: '' (Bsp. \"Anstragsstellung gestartet\", \"Anstragsstellung abgeschlossen\")\n"
        "   - Rollen: '' (Bsp. \"Antragsteller\", \"Behörde\", \"Sachbearbeiter\")\n\n"
        "5. Berücksichtige die Modellierungsprinzipien.\n"
        "6. Behalte das XML-artige, BPMN-orientierte Format EXAKT bei:\n"
        "   - <process>\n"
        "   - <startEvent>, <intermediateEvent>, <endEvent>\n"
        "   - <activity>\n"
        "   - <exclusiveGateway>, <parallelGateway>, <inclusiveGateway>\n"
        "   - <branch>\n\n"
        "   - </process>\n\n"
        "--------------------------------------------------\n"
        "Abschließende HINWEISE\n\n"
        "- Gib AUSSCHLIESSLICH den korrigierten Prozess-Draft zurück.\n"
        "- KEIN zusätzlicher Text, KEINE Erklärungen, KEIN Markdown.\n"
        "- Das Format muss identisch zum originalen Format sein.\n\n"
        "--------------------------------------------------\n"
        "WICHTIGE LEITLINIE\n\n"
        "Führe nur die minimal notwendigen Änderungen durch, die explizit im Validation-Feedback gefordert werden."
    )

    
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt_bpmn_text},
        {"role": "user", "content": user_prompt},
    ]
    
    
    
    text = await call_ollama_chat(
        messages=messages,
        model=settings.agentic_revision_model,
    )
    
    return ProcessDraft(text_description=text)

