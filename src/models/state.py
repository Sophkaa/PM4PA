"""Global state model for graph runs."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from .artifacts import ProcessDraft, ValidationResult, ValidationResultSetting4
from .bpmn import BPMNModelJsonNested
from .query_structure import QueryStructure
from .relevance import RelevanceEvaluationResult


class ProcessState(BaseModel):
    """Global state of a graph run."""
    user_request: str
    setting_name: str
    draft: Optional[ProcessDraft] = None
    bpmn: Optional[BPMNModelJsonNested] = None
    validation: Optional[ValidationResult] = None
    
    # Additional fields for retrieval and intermediate data
    retrieved_documents: List[str] = Field(default_factory=list)
    retrieved_metadatas: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_ids: List[str] = Field(default_factory=list)  # ChromaDB IDs for chunks
    relevance_scores: List[float] = Field(default_factory=list)
    expanded_queries: List[str] = Field(default_factory=list)  # Query expansion variants used for retrieval
    file_filter: Optional[Dict[str, Any]] = None
    query_structure: Optional[QueryStructure] = None  # Structured query information for Setting 2

    # Prompt used for draft generation (for experiment tracking)
    draft_user_prompt: Optional[str] = None
    
    # Fields for structured validation and revision loops
    validation_setting4: Optional[ValidationResultSetting4] = None
    bpmn_original: Optional[BPMNModelJsonNested] = None  # Original BPMN before revision
    revision_iteration_count: int = 0  # Track number of revision iterations (safety mechanism)

    # Relevance evaluation result
    relevance_evaluation: Optional[RelevanceEvaluationResult] = None

