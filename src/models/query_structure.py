"""Query structure model for structured query extraction."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class QueryStructure(BaseModel):
    """Structured representation of a user query for process modeling."""
    
    original_query: str = Field(..., description="Original unmodified user query")
    process_name: str = Field(..., description="Speakable process name, domain-specific")
    domain: str = Field(..., description="Administrative domain, e.g., Feuerwehr, Bildung, Soziales")
    procedure_type: str = Field(..., description="Type of procedure, e.g., Antragsverfahren, Bewilligungsverfahren, Antragseinreichung")
    perspective: Optional[str] = Field(default=None, description="Process perspective, e.g., Antragsteller, Sachbearbeitung, Fachabteilung")
    granularity: Optional[str] = Field(default=None, description="Level of detail: high_level, medium, detailed")
    scope_start: Optional[str] = Field(default=None, description="Start of the described process scope")
    scope_end: Optional[str] = Field(default=None, description="End of the described process scope")
    notes: Optional[str] = Field(default=None, description="Free-form additional information, e.g., important terms, synonyms")
    key_terms: Optional[List[str]] = Field(default=None, description="Extracted key terms from the query (typically 3 terms)")
    synonyms: Optional[Dict[str, List[str]]] = Field(default=None, description="Synonyms per key term (term -> list of synonyms)")



