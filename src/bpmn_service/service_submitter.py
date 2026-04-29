"""Submit BPMN JSON to the BPMN Generator Service and save the returned BPMN XML.

This module provides functionality to submit BPMN JSON to the external
BPMN Generator Service (https://bpmnchatbot.aau.at/bpmn-generator/generate)
and save the returned BPMN XML file to the Final_BPMN directory.
"""

from typing import Optional
from pathlib import Path
from pydantic import BaseModel
import requests
import json
import logging
from datetime import datetime
import hashlib
from ..config import settings

logger = logging.getLogger(__name__)


class SubmitToServiceInput(BaseModel):
    """Input for submitting BPMN JSON to the BPMN Generator Service."""
    bpmn_json: dict  # The BPMN model as JSON dict
    process_name: Optional[str] = None
    user_query: Optional[str] = None  # User query/request for unique filename generation


class SubmitToServiceResult(BaseModel):
    """Result from BPMN Generator Service submission."""
    success: bool
    message: str
    bpmn_xml: Optional[str] = None
    file_path: Optional[str] = None


def _get_service_url() -> str:
    """Get the BPMN Generator Service URL (local Docker or remote)."""
    url = settings.bpmn_service_url
    if url:
        url = url.rstrip("/")
        if not url.endswith("/generate"):
            url = f"{url}/generate"
        return url
    return "https://bpmnchatbot.aau.at/bpmn-generator/generate"


def _ensure_final_bpmn_directory() -> Path:
    """Ensure the Final_BPMN directory exists and return its path."""
    # Get the project root (this module lives under src/bpmn_service)
    project_root = Path(__file__).parent.parent.parent
    final_bpmn_dir = project_root / "Final_BPMN"
    final_bpmn_dir.mkdir(exist_ok=True)
    return final_bpmn_dir


def generate_unique_bpmn_filename(query: Optional[str] = None, process_name: Optional[str] = None) -> str:
    """
    Generate a unique filename for BPMN XML files.
    
    Combines current timestamp with a hash of the query/process name to create
    a unique, meaningful filename.
    
    Args:
        query: User query/request (optional)
        process_name: Process name (optional, used as fallback if query not provided)
        
    Returns:
        Unique filename string (without extension), e.g., "20241215_143022_corona_quarantane"
    """
    # Get current timestamp in format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use query or process_name to create a meaningful identifier
    identifier = query or process_name or "process"
    
    # Normalize identifier: lowercase, replace spaces with underscores, remove special chars
    normalized = "".join(
        c if c.isalnum() or c in (' ', '-', '_') else ''
        for c in identifier
    ).strip().lower().replace(' ', '_')
    
    # Limit length to avoid too long filenames (max 50 chars for identifier part)
    if len(normalized) > 50:
        # Use first 40 chars + hash of full string for uniqueness
        hash_suffix = hashlib.md5(identifier.encode('utf-8')).hexdigest()[:8]
        normalized = normalized[:40] + "_" + hash_suffix
    elif len(normalized) == 0:
        # Fallback if normalization results in empty string
        normalized = "process"
    
    # Combine timestamp and normalized identifier
    filename = f"{timestamp}_{normalized}"
    
    return filename


def submit_to_bpmn_service(input: SubmitToServiceInput) -> SubmitToServiceResult:
    """
    Submit BPMN JSON to the BPMN Generator Service and save the returned BPMN XML.
    
    This function:
    1. Sends the BPMN JSON to the service endpoint
    2. Receives the BPMN XML response
    3. Saves the XML to the Final_BPMN directory
    
    Args:
        input: SubmitToServiceInput containing the BPMN JSON and process name
        
    Returns:
        SubmitToServiceResult with success status, message, and file path
    """
    service_url = _get_service_url()
    
    try:
        # Send POST request to the service
        logger.info(f"Submitting BPMN JSON to {service_url}")
        verify_ssl = settings.bpmn_service_verify_ssl
        if not verify_ssl:
            logger.warning("SSL certificate verification is disabled for BPMN service requests")
        response = requests.post(
            service_url,
            data=json.dumps(input.bpmn_json),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            timeout=60,  # 60 second timeout
            verify=verify_ssl  # SSL certificate verification
        )
        
        if response.status_code != 200:
            error_msg = f"Service returned status code {response.status_code}: {response.text}"
            logger.error(error_msg)
            return SubmitToServiceResult(
                success=False,
                message=error_msg
            )
        
        # Extract BPMN XML from response
        data = response.json()
        bpmn_xml = data.get("bpmnXML")
        
        if not bpmn_xml:
            error_msg = "Response did not contain 'bpmnXML' field"
            logger.error(error_msg)
            return SubmitToServiceResult(
                success=False,
                message=error_msg
            )
        
        # Ensure Final_BPMN directory exists
        final_bpmn_dir = _ensure_final_bpmn_directory()
        
        # Generate unique filename from timestamp and query/process name
        unique_name = generate_unique_bpmn_filename(
            query=input.user_query,
            process_name=input.process_name
        )
        filename = f"{unique_name}.bpmn"
        
        file_path = final_bpmn_dir / filename
        
        # Save BPMN XML to file
        file_path.write_text(bpmn_xml, encoding='utf-8')
        logger.info(f"BPMN XML saved to {file_path}")
        
        return SubmitToServiceResult(
            success=True,
            message=f"BPMN XML successfully generated and saved to {file_path}",
            bpmn_xml=bpmn_xml,
            file_path=str(file_path)
        )
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error connecting to service: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return SubmitToServiceResult(
            success=False,
            message=error_msg
        )
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing JSON response: {str(e)}"
        logger.error(error_msg)
        return SubmitToServiceResult(
            success=False,
            message=error_msg
        )
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return SubmitToServiceResult(
            success=False,
            message=error_msg
        )


__all__ = [
    "submit_to_bpmn_service",
    "SubmitToServiceInput",
    "SubmitToServiceResult",
    "generate_unique_bpmn_filename",
]

