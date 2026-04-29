"""Streamlit MVP GUI for Agentic RAG BPMN System."""

import streamlit as st
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple
import sys
import re
import base64

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try relative imports first, fallback to absolute
try:
    from src.app.pipeline import GraphRAGSystem
    from src.bpmn_service.service_submitter import submit_to_bpmn_service, SubmitToServiceInput
    from src.config import settings
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(project_root))
    from src.app.pipeline import GraphRAGSystem
    from src.bpmn_service.service_submitter import submit_to_bpmn_service, SubmitToServiceInput
    from src.config import settings

# Page config
st.set_page_config(
    page_title="Process Modeller for Public Administration (PM4PA)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Override Streamlit primary/accent color to header blue (buttons, tabs, radio)
st.markdown("""
<style>
    /* Primary buttons - blue instead of red */
    div[data-testid="stButton"] button[kind="primary"],
    .stButton > button[kind="primary"] {
        background-color: #2d5a87 !important;
        background: linear-gradient(180deg, #2d5a87 0%, #1e3a5f 100%) !important;
        color: white !important;
        border: none !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #3d6a97 !important;
        background: linear-gradient(180deg, #3d6a97 0%, #2d5a87 100%) !important;
        border-color: #2d5a87 !important;
    }
    /* Tabs - selected tab */
    .stTabs [data-baseweb="tab-list"] [aria-selected="true"] {
        color: #2d5a87 !important;
        border-color: #2d5a87 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #2d5a87 !important;
    }
    /* Selectbox focus */
    [data-baseweb="select"]:focus-within {
        border-color: #2d5a87 !important;
        box-shadow: 0 0 0 1px #2d5a87 !important;
    }
    /* Pipeline fixed to setting_4: hide sidebar and expand control */
    section[data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'bpmn_xml' not in st.session_state:
    st.session_state.bpmn_xml = None
if 'bpmn_filename' not in st.session_state:
    st.session_state.bpmn_filename = None
if 'bpmn_info' not in st.session_state:
    st.session_state.bpmn_info = None
if 'bpmn_xml_edited' not in st.session_state:
    st.session_state.bpmn_xml_edited = None
# Fixed pipeline: draft validation & revision (former "setting 4")
DEFAULT_SETTING = "setting_4"
_VALID_SETTINGS = frozenset(f"setting_{i}" for i in range(1, 6))


def init_rag_system(setting_name: Optional[str] = None):
    """Initialize the RAG system with default configuration (uploaded documents only)."""
    setting = setting_name or DEFAULT_SETTING
    if setting not in _VALID_SETTINGS:
        setting = DEFAULT_SETTING
    if st.session_state.rag_system is None:
        with st.spinner("Initializing system..."):
            st.session_state.rag_system = GraphRAGSystem(setting_name=setting)
    elif st.session_state.rag_system.setting_name != setting:
        st.session_state.rag_system = GraphRAGSystem(setting_name=setting)


async def process_file_upload(uploaded_file) -> int:
    """Process uploaded file and return number of chunks."""
    if st.session_state.rag_system is None:
        init_rag_system()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)
    
    try:
        # Ingest file
        metadata = {
            "source": "streamlit_upload",
            "file_name": uploaded_file.name,
            "file_path": uploaded_file.name
        }
        
        chunk_ids = await st.session_state.rag_system.ingest_file(
            tmp_path,
            metadata=metadata,
            use_semantic_chunking=True
        )
        
        return len(chunk_ids)
    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


def extract_chunk_references_from_bpmn(bpmn_json: Dict[str, Any]) -> Set[Tuple[str, int]]:
    """
    Extract source references from BPMN JSON documentation fields.
    
    Returns a set of tuples (document_name, page_number) that are referenced.
    page_number is None when format is just document name without page.
    """
    refs = set()
    
    def extract_from_documentation(doc_str: Optional[str]):
        """Extract source references from a documentation string."""
        if not doc_str:
            return
        
        # Pattern to match: "Name (Page X)" or "Name (Seite X)"
        page_pattern = r'([^;,]+?)\s*\((?:Page|Seite)\s*(\d+)\)'
        for match in re.finditer(page_pattern, doc_str):
            doc_part = match.group(1).strip()
            page_num = int(match.group(2))
            # Remove "Document:" / "Dokument:" prefix if present
            doc_name = re.sub(r'^(?:Document|Dokument):\s*', '', doc_part, flags=re.IGNORECASE).strip()
            doc_name = re.sub(r'[,;]\s*$', '', doc_name).strip()
            if doc_name:
                refs.add((doc_name, page_num))
    
    def traverse_dict(obj: Any):
        """Recursively traverse the BPMN JSON structure."""
        if isinstance(obj, dict):
            # Check for documentation field
            if 'documentation' in obj:
                doc_value = obj['documentation']
                if isinstance(doc_value, str):
                    extract_from_documentation(doc_value)
                elif isinstance(doc_value, list):
                    for item in doc_value:
                        if isinstance(item, dict) and 'document' in item:
                            extract_from_documentation(item['document'])
            
            # Recursively traverse all values
            for value in obj.values():
                traverse_dict(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse_dict(item)
    
    traverse_dict(bpmn_json)
    return refs


def filter_used_chunks(
    retrieved_docs_info: List[Dict[str, Any]],
    bpmn_json: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Filter retrieved documents to only include chunks that are referenced
    in the BPMN model's documentation fields (by document name and page).
    """
    refs = extract_chunk_references_from_bpmn(bpmn_json)
    
    if not refs:
        return []
    
    # Match chunks by document name and page number
    used_chunks = []
    for doc_info in retrieved_docs_info:
        doc_name = doc_info.get("file_name", doc_info.get("document_title", ""))
        page_num = doc_info.get("page_number")
        
        for ref_name, ref_page in refs:
            name_match = (
                ref_name.lower() in doc_name.lower() or
                doc_name.lower() in ref_name.lower() or
                ref_name in doc_name or doc_name in ref_name
            )
            page_match = page_num is not None and ref_page == page_num
            if name_match and page_match:
                used_chunks.append(doc_info)
                break
    
    return used_chunks


def create_bpmn_viewer_html(bpmn_xml: str, editable: bool = True, filename: str = "process.bpmn") -> str:
    """
    Create HTML with embedded bpmn-js viewer/modeler (developed by Camunda).
    
    Args:
        bpmn_xml: BPMN XML string
        editable: If True, use modeler (editable), else use viewer (read-only)
        filename: Filename for downloads (e.g. "process.bpmn")
    
    Returns:
        HTML string with embedded bpmn-js
    """
    # Encode XML as base64 for safe embedding
    bpmn_xml_base64 = base64.b64encode(bpmn_xml.encode('utf-8')).decode('utf-8')
    # Escape for JavaScript string
    filename_js = filename.replace("\\", "\\\\").replace("'", "\\'")
    
    # Choose between viewer (read-only) or modeler (editable)
    # Using UMD build that exports classes globally
    if editable:
        script_url = "https://unpkg.com/bpmn-js@14.0.0/dist/bpmn-modeler.development.js"
        component_class = "BpmnModeler"
        mode_text = "Modeler (Editable)"
    else:
        script_url = "https://unpkg.com/bpmn-js@14.0.0/dist/bpmn-viewer.development.js"
        component_class = "BpmnViewer"
        mode_text = "Viewer (Read-only)"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>BPMN {mode_text}</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bpmn-js@14.0.0/dist/assets/diagram-js.css" />
        <style>
            /* Override diagram-js selection/highlight to match header blue (#1e3a5f, #2d5a87) */
            .djs-parent,
            #bpmn-container {{
                --element-selected-outline-stroke-color: #2d5a87 !important;
                --element-selected-outline-secondary-stroke-color: #4a7ba7 !important;
                --element-hover-outline-fill-color: #2d5a87 !important;
                --element-dragger-color: #2d5a87 !important;
                --bendpoint-fill-color: #2d5a87 !important;
                --lasso-stroke-color: #2d5a87 !important;
                --resizer-fill-color: #2d5a87 !important;
                --shape-attach-allowed-stroke-color: #2d5a87 !important;
                --shape-resize-preview-stroke-color: #2d5a87 !important;
            }}
        </style>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bpmn-js@14.0.0/dist/assets/bpmn-font/css/bpmn.css" />
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bpmn-js-properties-panel@1.0.0/dist/assets/properties-panel.css" />
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: white;
            }}
            #bpmn-container-wrapper {{
                display: flex;
                width: 100%;
                height: 800px;
                border: 1px solid #e2e8f0;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                overflow: hidden;
            }}
            #bpmn-container {{
                flex: 1;
                height: 100%;
            }}
            #properties-panel {{
                width: 300px;
                border-left: 1px solid #ddd;
                background: white;
                overflow-y: auto;
            }}
            #properties-panel:empty {{
                display: none;
            }}
            .loading {{
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
                color: #666;
                font-size: 14px;
            }}
            .error {{
                padding: 40px;
                text-align: center;
                color: #d32f2f;
            }}
            .download-section {{
                padding: 16px 0;
                margin-top: 12px;
            }}
            .download-section h3 {{
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 12px;
                color: #1e293b;
            }}
            .download-buttons {{
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }}
            .download-buttons button {{
                padding: 6px 12px;
                color: #475569;
                border: 1px solid rgba(148, 163, 184, 0.4);
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
                font-weight: 500;
                min-height: 28px;
                background: #fafbfc;
                box-shadow: none;
                transition: all 0.15s ease;
            }}
            .download-buttons button:hover {{
                border-color: rgba(148, 163, 184, 0.7);
                background: #f1f5f9;
                box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            }}
            .download-divider {{
                border-top: 1px solid #e2e8f0;
                margin: 28px 0 24px 0;
            }}
            .download-section-last {{
                padding-bottom: 40px;
            }}
        </style>
    </head>
    <body>
        <div id="bpmn-container-wrapper">
            <div id="bpmn-container">
                <div class="loading">Loading BPMN diagram...</div>
            </div>
            <div id="properties-panel"></div>
        </div>
        <div class="download-section">
            <h3>💾 ✏️ Download Edited Model (with current edits)</h3>
            <div class="download-buttons">
                <button type="button" id="btn-edited-xml">⬇ XML</button>
                <button type="button" id="btn-edited-bpmn">⬇ BPMN</button>
                <button type="button" id="btn-edited-png">⬇ PNG</button>
                <button type="button" id="btn-edited-pdf">⬇ PDF</button>
            </div>
        </div>
        <div class="download-divider"></div>
        <div class="download-section download-section-last">
            <h3>💾 Download Original Model (before edits)</h3>
            <div class="download-buttons">
                <button type="button" id="btn-orig-xml">⬇ XML</button>
                <button type="button" id="btn-orig-bpmn">⬇ BPMN</button>
                <button type="button" id="btn-orig-png">⬇ PNG</button>
                <button type="button" id="btn-orig-pdf">⬇ PDF</button>
            </div>
        </div>
        <script>
            (function() {{
                // UTF-8 safe base64 decode function
                function base64ToUtf8(str) {{
                    try {{
                        // Decode base64
                        const binaryString = atob(str);
                        // Convert binary string to UTF-8
                        const bytes = new Uint8Array(binaryString.length);
                        for (let i = 0; i < binaryString.length; i++) {{
                            bytes[i] = binaryString.charCodeAt(i);
                        }}
                        // Decode UTF-8
                        return new TextDecoder('utf-8').decode(bytes);
                    }} catch (e) {{
                        // Fallback for older browsers
                        return decodeURIComponent(escape(atob(str)));
                    }}
                }}
                
                // Decode the BPMN XML from base64 (UTF-8 safe)
                const bpmnXmlBase64 = '{bpmn_xml_base64}';
                let bpmnXml;
                try {{
                    bpmnXml = base64ToUtf8(bpmnXmlBase64);
                }} catch (e) {{
                    document.getElementById('bpmn-container').innerHTML = 
                        '<div class="error"><h3>Error decoding BPMN XML</h3><p>' + e.message + '</p></div>';
                    return;
                }}
                
                // Function to initialize the viewer
                function initViewer() {{
                    try {{
                        // Check if the class is available (try different possible locations)
                        let ViewerClass = null;
                        
                        // The bpmn-js library exports classes via BpmnJS namespace
                        // Try BpmnJS namespace first (most common)
                        if (typeof BpmnJS !== 'undefined' && BpmnJS['{component_class}']) {{
                            ViewerClass = BpmnJS['{component_class}'];
                        }}
                        else if (typeof window !== 'undefined' && typeof window.BpmnJS !== 'undefined' && window.BpmnJS['{component_class}']) {{
                            ViewerClass = window.BpmnJS['{component_class}'];
                        }}
                        // Try direct global access (fallback)
                        else if (typeof window !== 'undefined' && typeof window['{component_class}'] !== 'undefined') {{
                            ViewerClass = window['{component_class}'];
                        }}
                        // Try accessing from module exports (if UMD)
                        else if (typeof module !== 'undefined' && module.exports && module.exports['{component_class}']) {{
                            ViewerClass = module.exports['{component_class}'];
                        }}
                        // If BpmnJS exists but class not found, try to access it differently
                        else if (typeof BpmnJS !== 'undefined') {{
                            // List all properties of BpmnJS for debugging
                            const props = Object.keys(BpmnJS);
                            console.log('BpmnJS properties:', props);
                            console.log('BpmnJS value:', BpmnJS);
                            
                            // Maybe BpmnJS itself IS the class (if it's a function/constructor)
                            if (typeof BpmnJS === 'function') {{
                                ViewerClass = BpmnJS;
                            }}
                            // Try to get the class from BpmnJS - it might be the default export
                            else if (BpmnJS.default) {{
                                if (BpmnJS.default['{component_class}']) {{
                                    ViewerClass = BpmnJS.default['{component_class}'];
                                }} else if (typeof BpmnJS.default === 'function') {{
                                    // Maybe the default IS the class
                                    ViewerClass = BpmnJS.default;
                                }}
                            }}
                            // Try accessing via bracket notation (case-insensitive)
                            else {{
                                const lowerClass = '{component_class}'.toLowerCase();
                                for (let key in BpmnJS) {{
                                    if (key.toLowerCase() === lowerClass || key === '{component_class}') {{
                                        ViewerClass = BpmnJS[key];
                                        break;
                                    }}
                                }}
                            }}
                        }}
                        
                        if (!ViewerClass) {{
                            // Debug: log available globals and BpmnJS properties
                            const available = typeof window !== 'undefined' ? Object.keys(window).filter(k => k.includes('Bpmn') || k.includes('bpmn')).join(', ') : 'window not available';
                            let bpmnProps = 'BpmnJS not available';
                            if (typeof BpmnJS !== 'undefined') {{
                                bpmnProps = Object.keys(BpmnJS).join(', ');
                                // Also log the actual BpmnJS object structure
                                console.log('BpmnJS object:', BpmnJS);
                                console.log('BpmnJS type:', typeof BpmnJS);
                                if (BpmnJS.default) {{
                                    console.log('BpmnJS.default:', BpmnJS.default);
                                    console.log('BpmnJS.default type:', typeof BpmnJS.default);
                                }}
                            }}
                            throw new Error('{component_class} class not found. Available globals: ' + available + '. BpmnJS properties: ' + bpmnProps);
                        }}
                        
                        // Create viewer/modeler instance (developed by Camunda)
                        const viewer = new ViewerClass({{
                            container: '#bpmn-container',
                            keyboard: {{
                                bindTo: window
                            }}
                        }});
                        
                        // Store viewer instance and class globally for export
                        window.bpmnViewer = viewer;
                        window.__bpmnViewerClass = ViewerClass;
                        
                        // Add properties panel functionality to show documentation
                        if ('{component_class}' === 'BpmnModeler') {{
                            // Helper function to get element type label
                            function getElementTypeLabel(type) {{
                                const typeMap = {{
                                    'bpmn:Task': 'Task',
                                    'bpmn:ServiceTask': 'Service Task',
                                    'bpmn:UserTask': 'User Task',
                                    'bpmn:ScriptTask': 'Script Task',
                                    'bpmn:SendTask': 'Send Task',
                                    'bpmn:ReceiveTask': 'Receive Task',
                                    'bpmn:ManualTask': 'Manual Task',
                                    'bpmn:BusinessRuleTask': 'Business Rule Task',
                                    'bpmn:ExclusiveGateway': 'Exclusive Gateway',
                                    'bpmn:InclusiveGateway': 'Inclusive Gateway',
                                    'bpmn:ParallelGateway': 'Parallel Gateway',
                                    'bpmn:EventBasedGateway': 'Event-Based Gateway',
                                    'bpmn:StartEvent': 'Start Event',
                                    'bpmn:EndEvent': 'End Event',
                                    'bpmn:IntermediateThrowEvent': 'Intermediate Throw Event',
                                    'bpmn:IntermediateCatchEvent': 'Intermediate Catch Event',
                                    'bpmn:BoundaryEvent': 'Boundary Event',
                                    'bpmn:SequenceFlow': 'Sequence Flow',
                                    'bpmn:DataObject': 'Data Object',
                                    'bpmn:DataStore': 'Data Store',
                                    'bpmn:SubProcess': 'Sub-Process',
                                    'bpmn:CallActivity': 'Call Activity'
                                }};
                                return typeMap[type] || type.replace('bpmn:', '') || 'Element';
                            }}
                            
                            // Listen to selection changes to show documentation
                            viewer.on('selection.changed', function(e) {{
                                const element = e.newSelection[0];
                                const panel = document.getElementById('properties-panel');
                                
                                if (!panel) return;
                                
                                if (element && element.businessObject) {{
                                    const bo = element.businessObject;
                                    const elementType = getElementTypeLabel(element.type);
                                    const elementName = bo.name || '';
                                    let html = '<div style="padding: 15px;">';
                                    html += '<h2 style="margin: 0 0 15px 0; font-size: 22px; font-weight: 700; color: #1e293b; letter-spacing: -0.02em;">🔍 Detailed information</h2>';
                                    
                                    // Element type and name together as title
                                    let titleText = elementType;
                                    if (elementName) {{
                                        titleText += ': ' + elementName;
                                    }} else {{
                                        titleText += ' (Unnamed)';
                                    }}
                                    html += '<h3 style="margin-bottom: 15px; color: #333; border-bottom: 2px solid #2d5a87; padding-bottom: 8px; font-size: 16px;">' + titleText + '</h3>';
                                    
                                    // Show documentation (now called Reference) if available
                                    if (bo.documentation && bo.documentation.length > 0) {{
                                        function escapeHtml(s) {{
                                            if (!s) return '';
                                            return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
                                        }}
                                        function formatReference(text) {{
                                            if (!text || !text.trim()) return '';
                                            var sources = text.split(/;\\s*/);
                                            var result = [];
                                            for (var s = 0; s < sources.length; s++) {{
                                                var part = sources[s].trim();
                                                if (!part) continue;
                                                var lines = [];
                                                var docPart = part, kapitelPart = '', uberschriftPart = '';
                                                var chapterMarker = ', Chapter: ';
                                                var headingMarker = ', Heading: ';
                                                var kIdx = part.indexOf(chapterMarker);
                                                if (kIdx < 0) kIdx = part.indexOf(', Kapitel: ');
                                                if (kIdx >= 0) {{
                                                    docPart = part.substring(0, kIdx).trim();
                                                    var chapterLen = part.indexOf(chapterMarker) >= 0 ? chapterMarker.length : ', Kapitel: '.length;
                                                    var rest = part.substring(kIdx + chapterLen);
                                                    var uIdx = rest.indexOf(headingMarker);
                                                    if (uIdx < 0) uIdx = rest.indexOf(', Überschrift: ');
                                                    if (uIdx >= 0) {{
                                                        kapitelPart = rest.substring(0, uIdx).trim();
                                                        var headingLen = rest.indexOf(headingMarker) >= 0 ? headingMarker.length : ', Überschrift: '.length;
                                                        uberschriftPart = rest.substring(uIdx + headingLen).trim();
                                                    }} else {{
                                                        kapitelPart = rest.trim();
                                                    }}
                                                }} else {{
                                                    var uIdx = part.indexOf(headingMarker);
                                                    if (uIdx < 0) uIdx = part.indexOf(', Überschrift: ');
                                                    if (uIdx >= 0) {{
                                                        docPart = part.substring(0, uIdx).trim();
                                                        var headingLen = part.indexOf(headingMarker) >= 0 ? headingMarker.length : ', Überschrift: '.length;
                                                        uberschriftPart = part.substring(uIdx + headingLen).trim();
                                                    }}
                                                }}
                                                var docName = docPart, seiteNum = '';
                                                var seiteMatch = docPart.match(/\\((?:Page|Seite)\\s*(\\d+)\\)$/);
                                                if (seiteMatch) {{
                                                    docName = docPart.substring(0, docPart.length - seiteMatch[0].length).trim();
                                                    seiteNum = seiteMatch[1];
                                                }}
                                                if (docName) lines.push('<strong>' + escapeHtml(docName) + '</strong>');
                                                if (seiteNum) lines.push('Page: ' + escapeHtml(seiteNum));
                                                if (kapitelPart) lines.push('Chapter: ' + escapeHtml(kapitelPart));
                                                if (uberschriftPart) lines.push('Heading: ' + escapeHtml(uberschriftPart));
                                                if (lines.length > 0) result.push(lines.join('\\n'));
                                            }}
                                            return result.join('\\n\\n');
                                        }}
                                        html += '<div style="margin-bottom: 15px;">';
                                        html += '<h4 style="margin-bottom: 8px; color: #666; font-size: 14px;">📄 Reference</h4>';
                                        html += '<div style="background: white; padding: 12px; border-radius: 4px; border-left: 3px solid #2d5a87; white-space: pre-wrap; word-wrap: break-word; font-size: 13px; line-height: 1.6;">';
                                        
                                        bo.documentation.forEach(function(doc, index) {{
                                            if (doc.text) {{
                                                html += formatReference(doc.text);
                                                if (index < bo.documentation.length - 1) {{
                                                    html += '\\n\\n';
                                                }}
                                            }}
                                        }});
                                        
                                        html += '</div></div>';
                                    }} else {{
                                        html += '<div style="padding: 10px; background: #f5f5f5; border-radius: 4px; color: #999; font-size: 13px;">No reference available for this element.</div>';
                                    }}
                                    
                                    // Show element ID (technical info)
                                    html += '<div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">';
                                    if (bo.id) {{
                                        html += '<div style="font-size: 12px; color: #999;">ID: <code style="background: #f0f0f0; padding: 2px 6px; border-radius: 3px;">' + bo.id + '</code></div>';
                                    }}
                                    html += '</div>';
                                    
                                    html += '</div>';
                                    panel.innerHTML = html;
                                }} else {{
                                    panel.innerHTML = '<div style="padding: 15px;"><h2 style="margin: 0 0 15px 0; font-size: 22px; font-weight: 700; color: #1e293b; letter-spacing: -0.02em;">🔍 Detailed information</h2><p style="text-align: center; color: #999; margin: 0;">Select an element to view its properties and reference.</p></div>';
                                }}
                            }});
                            
                            // Clear panel when nothing is selected
                            viewer.on('canvas.drag', function() {{
                                const panel = document.getElementById('properties-panel');
                                if (panel && !viewer.get('selection').get()) {{
                                    panel.innerHTML = '<div style="padding: 15px;"><h2 style="margin: 0 0 15px 0; font-size: 22px; font-weight: 700; color: #1e293b; letter-spacing: -0.02em;">🔍 Detailed information</h2><p style="text-align: center; color: #999; margin: 0;">Select an element to view its properties and reference.</p></div>';
                                }}
                            }});
                        }}
                        
                        // Function to export XML (UTF-8 safe)
                        window.exportBpmnXml = function() {{
                            if (!viewer) {{
                                console.error('Viewer not initialized');
                                return Promise.resolve(null);
                            }}
                            return viewer.saveXML({{ format: true }})
                                .then(function(result) {{
                                    // Ensure XML has UTF-8 declaration if not present
                                    let xml = result.xml;
                                    if (!xml.includes('encoding=')) {{
                                        // Add UTF-8 encoding declaration if missing
                                        xml = xml.replace('<?xml version="1.0"?>', '<?xml version="1.0" encoding="UTF-8"?>');
                                        if (!xml.startsWith('<?xml')) {{
                                            xml = '<?xml version="1.0" encoding="UTF-8"?>\\n' + xml;
                                        }}
                                    }}
                                    return xml;
                                }})
                                .catch(function(err) {{
                                    console.error('Error exporting XML:', err);
                                    return null;
                                }});
                        }};
                        
                        // Function to copy XML to clipboard
                        window.copyBpmnXmlToClipboard = function() {{
                            return window.exportBpmnXml().then(function(xml) {{
                                if (xml) {{
                                    // Copy to clipboard
                                    navigator.clipboard.writeText(xml).then(function() {{
                                        alert('BPMN XML copied to clipboard! Paste it in the text area below.');
                                    }}).catch(function(err) {{
                                        console.error('Failed to copy to clipboard:', err);
                                        // Fallback: show in console
                                        console.log('BPMN XML:', xml);
                                        alert('Failed to copy to clipboard. Check browser console (F12) for the XML.');
                                    }});
                                }} else {{
                                    alert('Failed to export XML. Check browser console for errors.');
                                }}
                            }});
                        }};
                        
                        // Function to download edited model as XML
                        window.downloadEditedAsXml = function() {{
                            return window.exportBpmnXml().then(function(xml) {{
                                if (xml) {{
                                    // Create download link with UTF-8 encoding
                                    const blob = new Blob([xml], {{ type: 'application/xml;charset=utf-8' }});
                                    const url = URL.createObjectURL(blob);
                                    const link = document.createElement('a');
                                    link.href = url;
                                    link.download = 'bpmn_edited.xml';
                                    document.body.appendChild(link);
                                    link.click();
                                    document.body.removeChild(link);
                                    URL.revokeObjectURL(url);
                                    
                                    return xml;
                                }} else {{
                                    alert('❌ Failed to export XML. Please try again.');
                                    return null;
                                }}
                            }});
                        }};
                        
                        // Function to download edited model as BPMN
                        window.downloadEditedAsBpmn = function() {{
                            return window.exportBpmnXml().then(function(xml) {{
                                if (xml) {{
                                    // Create download link with UTF-8 encoding
                                    const blob = new Blob([xml], {{ type: 'application/xml;charset=utf-8' }});
                                    const url = URL.createObjectURL(blob);
                                    const link = document.createElement('a');
                                    link.href = url;
                                    link.download = 'bpmn_edited.bpmn';
                                    document.body.appendChild(link);
                                    link.click();
                                    document.body.removeChild(link);
                                    URL.revokeObjectURL(url);
                                    
                                    return xml;
                                }} else {{
                                    alert('❌ Failed to export BPMN. Please try again.');
                                    return null;
                                }}
                            }});
                        }};
                        
                        // Function to export diagram as SVG (from current canvas)
                        window.exportBpmnSvg = function() {{
                            if (!viewer) {{
                                console.error('Viewer not initialized');
                                return Promise.resolve(null);
                            }}
                            return viewer.saveSVG({{ format: true }})
                                .then(function(result) {{
                                    return result.svg;
                                }})
                                .catch(function(err) {{
                                    console.error('Error exporting SVG:', err);
                                    return null;
                                }});
                        }};
                        
                        // Function to export ORIGINAL diagram as SVG (uses stored bpmnXml, not current canvas)
                        function exportOriginalAsSvg() {{
                            const ViewerClass = window.__bpmnViewerClass || (typeof BpmnJS !== 'undefined' && BpmnJS.BpmnModeler) || (window.BpmnJS && window.BpmnJS.BpmnModeler);
                            if (!ViewerClass) {{
                                alert('❌ BPMN viewer not ready. Please wait for the diagram to load and try again.');
                                return Promise.resolve(null);
                            }}
                            const container = document.createElement('div');
                            container.style.cssText = 'position:absolute;left:-9999px;width:1200px;height:800px;';
                            document.body.appendChild(container);
                            const tempViewer = new ViewerClass({{ container: container }});
                            return tempViewer.importXML(bpmnXml)
                                .then(function() {{
                                    return tempViewer.saveSVG({{ format: true }});
                                }})
                                .then(function(result) {{
                                    tempViewer.destroy();
                                    document.body.removeChild(container);
                                    return result.svg;
                                }})
                                .catch(function(err) {{
                                    if (tempViewer && tempViewer.destroy) tempViewer.destroy();
                                    if (container.parentNode) document.body.removeChild(container);
                                    console.error('Error exporting original SVG:', err);
                                    return null;
                                }});
                        }};
                        
                        // Function to convert SVG to PNG
                        function svgToPng(svgString) {{
                            return new Promise(function(resolve, reject) {{
                                const img = new Image();
                                const svgBlob = new Blob([svgString], {{ type: 'image/svg+xml;charset=utf-8' }});
                                const url = URL.createObjectURL(svgBlob);
                                
                                img.onload = function() {{
                                    // Parse SVG to get dimensions
                                    const parser = new DOMParser();
                                    const svgDoc = parser.parseFromString(svgString, 'image/svg+xml');
                                    const svgElement = svgDoc.documentElement;
                                    
                                    // Get dimensions from width/height or viewBox
                                    let width = parseInt(svgElement.getAttribute('width')) || 0;
                                    let height = parseInt(svgElement.getAttribute('height')) || 0;
                                    
                                    if (width === 0 || height === 0) {{
                                        const viewBox = svgElement.getAttribute('viewBox');
                                        if (viewBox) {{
                                            const parts = viewBox.split(/[\\s,]+/);
                                            if (parts.length >= 4) {{
                                                width = parseInt(parts[2]) || 1200;
                                                height = parseInt(parts[3]) || 800;
                                            }}
                                        }}
                                    }}
                                    
                                    // Use natural dimensions if available and better
                                    if (img.naturalWidth && img.naturalWidth > width) {{
                                        width = img.naturalWidth;
                                    }}
                                    if (img.naturalHeight && img.naturalHeight > height) {{
                                        height = img.naturalHeight;
                                    }}
                                    
                                    // Default dimensions if still not found
                                    if (width === 0 || height === 0) {{
                                        width = 1200;
                                        height = 800;
                                    }}
                                    
                                    const canvas = document.createElement('canvas');
                                    canvas.width = width;
                                    canvas.height = height;
                                    const ctx = canvas.getContext('2d');
                                    
                                    // White background
                                    ctx.fillStyle = '#ffffff';
                                    ctx.fillRect(0, 0, width, height);
                                    
                                    // Draw SVG
                                    ctx.drawImage(img, 0, 0, width, height);
                                    
                                    canvas.toBlob(function(blob) {{
                                        URL.revokeObjectURL(url);
                                        resolve(blob);
                                    }}, 'image/png');
                                }};
                                
                                img.onerror = function() {{
                                    URL.revokeObjectURL(url);
                                    reject(new Error('Failed to load SVG'));
                                }};
                                
                                img.src = url;
                            }});
                        }};
                        
                        // Function to download ORIGINAL model as PNG (uses stored bpmnXml)
                        window.downloadOriginalAsPng = function() {{
                            return exportOriginalAsSvg().then(function(svg) {{
                                if (svg) {{
                                    return svgToPng(svg).then(function(pngBlob) {{
                                        const url = URL.createObjectURL(pngBlob);
                                        const link = document.createElement('a');
                                        link.href = url;
                                        link.download = 'bpmn_original.png';
                                        document.body.appendChild(link);
                                        link.click();
                                        document.body.removeChild(link);
                                        URL.revokeObjectURL(url);
                                        return true;
                                    }});
                                }} else {{
                                    alert('❌ Failed to export PNG. Please try again.');
                                    return null;
                                }}
                            }});
                        }};
                        
                        window.downloadEditedAsPng = function() {{
                            return window.exportBpmnSvg().then(function(svg) {{
                                if (svg) {{
                                    return svgToPng(svg).then(function(pngBlob) {{
                                        const url = URL.createObjectURL(pngBlob);
                                        const link = document.createElement('a');
                                        link.href = url;
                                        link.download = 'bpmn_edited.png';
                                        document.body.appendChild(link);
                                        link.click();
                                        document.body.removeChild(link);
                                        URL.revokeObjectURL(url);
                                        return true;
                                    }});
                                }} else {{
                                    alert('❌ Failed to export PNG. Please try again.');
                                    return null;
                                }}
                            }});
                        }};
                        
                        // Function to download model as PDF (from edited canvas)
                        function downloadAsPdf(filename) {{
                            return window.exportBpmnSvg().then(function(svg) {{
                                if (svg) {{
                                    return svgToPng(svg).then(function(pngBlob) {{
                                        return pngBlobToPdf(pngBlob, filename);
                                    }}).then(function() {{ return true; }});
                                }} else {{
                                    alert('❌ Failed to export PDF. Please try again.');
                                    return null;
                                }}
                            }}).catch(function(err) {{
                                alert('❌ Failed to export PDF: ' + (err && err.message ? err.message : 'Unknown error'));
                            }});
                        }};
                        
                        function pngBlobToPdf(pngBlob, filename) {{
                            const jsPDF = (window.jspdf && (window.jspdf.jsPDF || window.jspdf.default));
                            if (jsPDF) {{
                                return new Promise(function(resolve, reject) {{
                                    const img = new Image();
                                    const url = URL.createObjectURL(pngBlob);
                                    img.onload = function() {{
                                        try {{
                                            const pdf = new jsPDF({{
                                                orientation: img.width > img.height ? 'landscape' : 'portrait',
                                                unit: 'px',
                                                format: [img.width, img.height]
                                            }});
                                            pdf.addImage(img, 'PNG', 0, 0, img.width, img.height);
                                            pdf.save(filename);
                                            URL.revokeObjectURL(url);
                                            resolve(true);
                                        }} catch (e) {{
                                            URL.revokeObjectURL(url);
                                            reject(e);
                                        }};
                                    }};
                                    img.onerror = function() {{
                                        URL.revokeObjectURL(url);
                                        reject(new Error('Failed to load image'));
                                    }};
                                    img.src = url;
                                }});
                            }}
                            return new Promise(function(resolve, reject) {{
                                const script = document.createElement('script');
                                script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
                                script.onload = function() {{
                                    const JsPdf = window.jspdf.jsPDF || (window.jspdf && window.jspdf.default);
                                    if (!JsPdf) {{ reject(new Error('jsPDF not loaded')); return; }}
                                    pngBlobToPdf(pngBlob, filename).then(resolve).catch(reject);
                                }};
                                script.onerror = function() {{
                                    reject(new Error('Failed to load jsPDF'));
                                }};
                                document.head.appendChild(script);
                            }});
                        }}
                        
                        window.downloadOriginalAsPdf = function() {{
                            return exportOriginalAsSvg().then(function(svg) {{
                                if (svg) {{
                                    return svgToPng(svg).then(function(pngBlob) {{
                                        return pngBlobToPdf(pngBlob, 'bpmn_original.pdf');
                                    }}).then(function() {{ return true; }});
                                }} else {{
                                    alert('❌ Failed to export PDF. Please try again.');
                                    return null;
                                }}
                            }}).catch(function(err) {{
                                alert('❌ Failed to export PDF: ' + (err && err.message ? err.message : 'Unknown error'));
                            }});
                        }};
                        
                        window.downloadEditedAsPdf = function() {{
                            return downloadAsPdf('bpmn_edited.pdf');
                        }};
                        
                        // Import and display the diagram
                        viewer.importXML(bpmnXml)
                            .then(function() {{
                                console.log('BPMN diagram loaded successfully');
                                // Fit viewport to diagram
                                const canvas = viewer.get('canvas');
                                canvas.zoom('fit-viewport');
                                
                                // Remove loading message
                                const container = document.getElementById('bpmn-container');
                                const loading = container.querySelector('.loading');
                                if (loading) {{
                                    loading.remove();
                                }}
                                
                                // Set initial properties panel content
                                const panel = document.getElementById('properties-panel');
                                if (panel) {{
                                    panel.innerHTML = '<div style="padding: 15px;"><h2 style="margin: 0 0 15px 0; font-size: 22px; font-weight: 700; color: #1e293b; letter-spacing: -0.02em;">🔍 Detailed information</h2><p style="text-align: center; color: #999; margin: 0;">Select an element to view its properties and reference.</p></div>';
                                }}
                                
                                // Wire download buttons (all in same document - no cross-frame needed)
                                const dlFilename = '{filename_js}';
                                const dlFilenameXml = dlFilename.replace(/\\.bpmn$/i, '.xml');
                                
                                function triggerDownload(blob, filename) {{
                                    const url = URL.createObjectURL(blob);
                                    const a = document.createElement('a');
                                    a.href = url;
                                    a.download = filename;
                                    document.body.appendChild(a);
                                    a.click();
                                    document.body.removeChild(a);
                                    URL.revokeObjectURL(url);
                                }}
                                
                                var b1 = document.getElementById('btn-edited-xml');
                                if (b1) b1.onclick = function() {{ window.downloadEditedAsXml(); }};
                                var b2 = document.getElementById('btn-edited-bpmn');
                                if (b2) b2.onclick = function() {{ window.downloadEditedAsBpmn(); }};
                                var b3 = document.getElementById('btn-edited-png');
                                if (b3) b3.onclick = function() {{ window.downloadEditedAsPng(); }};
                                var b4 = document.getElementById('btn-edited-pdf');
                                if (b4) b4.onclick = function() {{ window.downloadEditedAsPdf(); }};
                                var origXml = document.getElementById('btn-orig-xml');
                                if (origXml) origXml.onclick = function() {{
                                    const blob = new Blob([bpmnXml], {{ type: 'application/xml;charset=utf-8' }});
                                    triggerDownload(blob, dlFilenameXml);
                                }};
                                var origBpmn = document.getElementById('btn-orig-bpmn');
                                if (origBpmn) origBpmn.onclick = function() {{
                                    const blob = new Blob([bpmnXml], {{ type: 'application/xml;charset=utf-8' }});
                                    triggerDownload(blob, dlFilename);
                                }};
                                var origPng = document.getElementById('btn-orig-png');
                                if (origPng) origPng.onclick = function() {{ window.downloadOriginalAsPng(); }};
                                var origPdf = document.getElementById('btn-orig-pdf');
                                if (origPdf) origPdf.onclick = function() {{ window.downloadOriginalAsPdf(); }};
                            }})
                            .catch(function(err) {{
                                console.error('Error loading BPMN diagram:', err);
                                const container = document.getElementById('bpmn-container');
                                container.innerHTML = 
                                    '<div class="error">' +
                                    '<h3 style="margin-bottom: 10px;">Error loading BPMN diagram</h3>' +
                                    '<p style="color: #666;">' + (err.message || 'Unknown error') + '</p>' +
                                    '</div>';
                            }});
                    }} catch (e) {{
                        console.error('Error creating viewer:', e);
                        console.error('Available globals:', Object.keys(window).filter(k => k.includes('Bpmn')));
                        document.getElementById('bpmn-container').innerHTML = 
                            '<div class="error">' +
                            '<h3>Error initializing viewer</h3>' +
                            '<p>' + e.message + '</p>' +
                            '<p style="font-size: 12px; margin-top: 10px;">Check browser console for details.</p>' +
                            '</div>';
                    }}
                }}
                
                // Load the bpmn-js library
                const script = document.createElement('script');
                script.src = '{script_url}';
                script.onload = function() {{
                    // Wait a bit to ensure the class is registered
                    // Try multiple times with increasing delays
                    let attempts = 0;
                    const maxAttempts = 10;
                    const checkAndInit = function() {{
                        attempts++;
                        // Check if BpmnJS is available and has the class
                        if (typeof BpmnJS !== 'undefined' && BpmnJS['{component_class}']) {{
                            initViewer();
                        }} else if (typeof window !== 'undefined' && typeof window.BpmnJS !== 'undefined' && window.BpmnJS['{component_class}']) {{
                            initViewer();
                        }} else if (typeof window !== 'undefined' && typeof window['{component_class}'] !== 'undefined') {{
                            initViewer();
                        }} else if (attempts < maxAttempts) {{
                            setTimeout(checkAndInit, 100);
                        }} else {{
                            // Last attempt - try anyway
                            initViewer();
                        }}
                    }};
                    checkAndInit();
                }};
                script.onerror = function() {{
                    document.getElementById('bpmn-container').innerHTML = 
                        '<div class="error">' +
                        '<h3>Error loading bpmn-js library</h3>' +
                        '<p>Failed to load the bpmn-js script from CDN. Please check your internet connection.</p>' +
                        '</div>';
                }};
                document.head.appendChild(script);
            }})();
        </script>
    </body>
    </html>
    """
    return html


async def generate_bpmn(query: str) -> tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Generate BPMN from query using external BPMN service. Uses only uploaded documents."""
    import time
    start_time = time.perf_counter()
    
    # Always use uploaded documents only
    file_filter = {"source": "streamlit_upload"}
    
    init_rag_system()
    
    # Run query with uploaded documents only
    state = await st.session_state.rag_system.run(query, file_filter=file_filter)
    
    if state.bpmn is None or len(state.bpmn.pools) == 0:
        elapsed_time = time.perf_counter() - start_time
        st.warning(f"⏱️ Generation Time: {elapsed_time:.2f} seconds (no BPMN model generated)")
        return None, None, None
    
    # Convert BPMN model to JSON dict
    bpmn_json = state.bpmn.model_dump(mode='json', exclude_none=True)
    
    # Get process name
    process_name = state.bpmn.process_name or "process"
    
    # Submit to external BPMN service
    service_input = SubmitToServiceInput(
        bpmn_json=bpmn_json,
        process_name=process_name,
        user_query=query
    )
    
    result = submit_to_bpmn_service(service_input)
    
    # Calculate total elapsed time (from start to service response)
    elapsed_time = time.perf_counter() - start_time
    
    if not result.success:
        # Return error information
        st.warning(f"⏱️ Generation Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes) - Service error occurred")
        return None, None, {
            "error": result.message,
            "process_name": process_name,
            "pools": len(state.bpmn.pools),
            "retrieved_docs": len(state.retrieved_documents) if state.retrieved_documents else 0,
            "generation_time_seconds": elapsed_time
        }
    
    # Generate filename: first three words of the query plus a short timestamp
    words = query.strip().split()[:3]
    name_part = "_".join(w for w in words if w) if words else "process"
    # Filename-safe: alphanumeric, hyphen, underscore only
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name_part).strip("_") or "process"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{safe_name}_{timestamp}.bpmn"
    
    # Prepare retrieved documents info
    retrieved_docs_info = []
    if state.retrieved_documents and state.retrieved_metadatas:
        for i, (doc, meta) in enumerate(zip(state.retrieved_documents, state.retrieved_metadatas)):
            doc_info = {
                "index": i,
                "chunk_index": meta.get('chunk_index', i) if meta else i,
                "file_name": meta.get('file_name') or meta.get('document_title') or meta.get('title') or f"Document {i+1}",
                "file_path": meta.get('file_path', ''),
                "page_number": meta.get('page_number'),
                "heading": meta.get('heading'),
                "chapter": meta.get('chapter')
            }
            retrieved_docs_info.append(doc_info)
    
    # Filter to only chunks that are actually used in BPMN documentation
    used_chunks_info = filter_used_chunks(retrieved_docs_info, bpmn_json)
    
    # Display timing information in GUI
    st.success(f"⏱️ **Generation Time:** {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return result.bpmn_xml, filename, {
        "process_name": process_name,
        "pools": len(state.bpmn.pools),
        "retrieved_docs": len(state.retrieved_documents) if state.retrieved_documents else 0,
        "retrieved_docs_info": retrieved_docs_info,
        "used_chunks_info": used_chunks_info,  # Only chunks referenced in BPMN documentation
        "service_file_path": result.file_path,
        "generation_time_seconds": elapsed_time
    }


# Main UI - Header with colored background (Agentic RAG BPMN Generator = largest)
st.markdown(
    '<div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); color: white; padding: 1.5rem 2rem; margin-bottom: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'
    '<h1 style="margin: 0; font-weight: 600; font-size: 2.4rem; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;">🤖 Process Modeller for Public Administration (PM4PA)</h1>'
    '<p style="margin: 0.5rem 0 0 0; opacity: 0.95; font-size: 0.95rem;">Upload documents and automatically generate BPMN process models</p>'
    '</div>',
    unsafe_allow_html=True
)

# Main content area
tab1, tab2 = st.tabs(["📄 Upload Documents", "🔍 Generate Process"])

with tab1:
    st.markdown('<h2 style="font-size: 1.5rem; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; color: #262730;">📄 Upload Documents</h2>', unsafe_allow_html=True)
    st.markdown("Upload PDF, DOCX, TXT or other documents containing process information.")
    
    uploaded_files = st.file_uploader(
        "Select documents",
        type=["pdf", "docx", "txt", "md", "html"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.get('file_uploader_key', 0)}",
        help="Supported formats: PDF, DOCX, TXT, MD, HTML"
    )
    
    if uploaded_files:
        st.info(f"📎 {len(uploaded_files)} file(s) selected")
        
        if st.button("⬆ Process Documents", type="primary"):
            init_rag_system()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_chunks = 0
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    chunks = asyncio.run(process_file_upload(uploaded_file))
                    total_chunks += chunks
                    st.session_state.uploaded_files.append({
                        "name": uploaded_file.name,
                        "chunks": chunks,
                        "status": "✅ Success"
                    })
                except Exception as e:
                    st.session_state.uploaded_files.append({
                        "name": uploaded_file.name,
                        "chunks": 0,
                        "status": f"❌ Error: {str(e)}"
                    })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text(f"✅ Done! {total_chunks} chunks processed")
            st.success(f"✅ {len(uploaded_files)} document(s) successfully processed")
            
            # Show uploaded files table
            if st.session_state.uploaded_files:
                st.subheader("Processed Documents")
                for file_info in st.session_state.uploaded_files:
                    st.markdown(f"- **{file_info['name']}**: {file_info['status']} ({file_info['chunks']} chunks)")

    # Button to remove all uploaded documents (visible when there are processed docs)
    if st.session_state.uploaded_files:
        st.markdown("---")
        if st.button("🗑 Remove all documents", type="secondary"):
            try:
                if st.session_state.rag_system is not None:
                    st.session_state.rag_system.vector_store.delete(
                        where={"source": "streamlit_upload"}
                    )
                st.session_state.uploaded_files = []
                st.session_state.bpmn_xml = None
                st.session_state.bpmn_filename = None
                st.session_state.bpmn_info = None
                st.session_state.bpmn_xml_edited = None
                st.session_state.file_uploader_key = st.session_state.get('file_uploader_key', 0) + 1  # Reset file uploader in GUI
                st.success("All documents have been removed.")
                st.rerun()
            except Exception as e:
                st.error(f"Error removing documents: {str(e)}")

with tab2:
    st.markdown('<h2 style="font-size: 1.5rem; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; color: #262730;">🔍 Generate Process</h2>', unsafe_allow_html=True)
    st.markdown("""
**Enter what process you would like to be modelled** based on the documents you just uploaded.

For best results, please be specific:
- **What** should be modelled (scope, start and end of the process)
- **How detailed** (high-level overview vs. step-by-step)
- **What must be included** (key steps, decisions, or outcomes)
- **Use the official process name or designation** as it appears in the documents and guidelines

*Less detailed prompts tend to produce more generic models.*

Typical public administration examples:
- **Building permit application process** (from application submission to final permit decision)
- **Residence registration process** (Anmeldung at the local registration office)
""")
    
    # Model type: Open Source (Ollama) vs. Closed Source (GPT/Azure)
    model_type = st.radio(
        "Model type",
        options=["Open Source (Ollama)", "Closed Source (GPT/Azure)"],
        index=0 if settings.open_source else 1,
        horizontal=True,
        help="Open Source has slightly lower quality and takes longer but must be used for personal/sensitive documents (Ollama). Closed Source = GPT/Azure OpenAI."
    )
    settings.open_source = (model_type == "Open Source (Ollama)")
    
    query = st.text_area(
        "Process to be modelled",
        placeholder="e.g., 'Building permit application process from submission to permit decision' or 'Residence registration process at the local registration office'",
        height=100,
        help="Enter what process you would like to be modelled based on the documents you just uploaded. Be specific: what (scope), how detailed, and what must be included. Less detailed prompts tend to produce more generic models."
    )
    
    init_rag_system()
    
    generate_button = st.button("🚀 Generate Process", type="primary", use_container_width=True)
    
    if generate_button and query:
        init_rag_system()
        
        with st.spinner("🤖 Generating BPMN model... This may take a few seconds."):
            try:
                bpmn_xml, filename, info = asyncio.run(generate_bpmn(query))
                
                # Store in session state for persistence
                st.session_state.bpmn_xml = bpmn_xml
                st.session_state.bpmn_filename = filename
                st.session_state.bpmn_info = info
                
                # Check if there was an error from the service
                if info and info.get("error"):
                    st.error(f"❌ Error with BPMN service: {info.get('error')}")
                    if info.get("process_name"):
                        st.info(f"BPMN model was generated, but the external service could not process it.")
                elif bpmn_xml:
                    st.success("✅ BPMN model successfully generated!")
                
                else:
                    st.warning("⚠️ No BPMN model generated. Try a different query or upload more documents.")
                    # Clear session state if no model generated
                    st.session_state.bpmn_xml = None
                    st.session_state.bpmn_filename = None
                    st.session_state.bpmn_info = None
            
            except Exception as e:
                st.error(f"❌ Error during generation: {str(e)}")
                if settings.debug:
                    import traceback
                    st.code(traceback.format_exc())
                # Clear session state on error
                st.session_state.bpmn_xml = None
                st.session_state.bpmn_filename = None
                st.session_state.bpmn_info = None
    
    elif generate_button:
        st.warning("⚠️ Please enter a process to be modelled.")
    
    # Display BPMN visualization if available in session state
    if st.session_state.bpmn_xml and st.session_state.bpmn_info:
        bpmn_xml = st.session_state.bpmn_xml
        filename = st.session_state.bpmn_filename
        info = st.session_state.bpmn_info
        
        # BPMN Visualization with bpmn-js (Camunda) - Always in edit mode
        st.markdown("---")
        st.markdown('<h2 style="font-size: 1.5rem; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; color: #262730;">📊 BPMN Model Editor</h2>', unsafe_allow_html=True)
        st.caption("Powered by bpmn-js - developed by Camunda")
        st.info("✏️ **You can edit the model and download it as BPMN, XML, PNG or PDF file below the model editor.**")
        
        # Display the BPMN modeler with download buttons (all inside iframe - same document)
        viewer_html = create_bpmn_viewer_html(bpmn_xml, editable=True, filename=filename)
        st.components.v1.html(viewer_html, height=1150, scrolling=False)

