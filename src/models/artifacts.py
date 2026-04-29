"""Pydantic models for domain artifacts."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProcessDraft(BaseModel):
    """Text description of a process draft."""
    text_description: str


class ValidationResult(BaseModel):
    """Result of BPMN model validation."""
    is_valid: bool
    issues: List[str]


class MissingElement(BaseModel):
    """A missing process element that should be present."""
    element_type: str
    element_label: str
    source_chunk_reference: str
    description: str


class HallucinatedElement(BaseModel):
    """A process element that is not backed by source chunks."""
    element_type: str
    element_label: str
    description: str


class ConsistencyIssue(BaseModel):
    """An inconsistency between BPMN model and source chunks."""
    element_type: str
    element_label: str
    source_chunk_reference: str
    description: str


class StructuralIssue(BaseModel):
    """A structural issue in the BPMN model (e.g., missing gateways, incorrect event placement)."""
    issue_type: str
    element_label: str
    description: str


class ValidationResultSetting4(BaseModel):
    """Structured validation result with detailed feedback."""
    missing_elements: List[MissingElement] = Field(default_factory=list)
    hallucinated_elements: List[HallucinatedElement] = Field(default_factory=list)
    structural_issues: List[StructuralIssue] = Field(default_factory=list)
    overall_assessment: Dict[str, bool] = Field(
        default_factory=lambda: {"iteration_recommended": False}
    )
    assessment_statement: str = Field(
        default="",
        description="Short overall assessment and rationale for the validation outcome"
    )


class LLMJudgeResult(BaseModel):
    """LLM-as-a-judge evaluation result for a BPMN model."""
    semantic_alignment_score: int
    justification: str
    chain_of_thought: str = ""  # Optional: LLM's reasoning process

