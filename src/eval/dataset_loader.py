"""Pydantic models for CLI-driven BPMN evaluation (gold model + sample)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models.bpmn import BPMNModelJsonNested, BPMNElement


class GoldBPMNModel(BaseModel):
    """Gold-standard BPMN representation for a single process.

    Supports both nested JSON format (legacy) and flat JSON format (preferred for gold standards).
    """

    process_id: str = Field(..., description="Unique identifier for the gold model (e.g., filename).")
    process_name: str = Field(..., description="Human-readable process name.")
    bpmn: Optional[BPMNModelJsonNested] = Field(
        default=None,
        description="Structured BPMN elements in nested format (legacy, for backward compatibility).",
    )
    flat_elements: Optional[Dict[str, List[BPMNElement]]] = Field(
        default=None,
        description="Flat BPMN elements for evaluation matching (preferred format for gold standards).",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        if "bpmn" not in data and "flat_elements" not in data:
            raise ValueError("GoldBPMNModel must have either 'bpmn' (nested) or 'flat_elements' (flat) format")
        super().__init__(**data)


class EvalSample(BaseModel):
    """Single evaluation sample with gold BPMN model (built by the eval CLI from paths)."""

    sample_id: str = Field(..., description="Unique identifier for the evaluation sample.")
    query: str = Field(..., description="Query that should reproduce the gold process.")
    description: Optional[str] = Field(default=None, description="Optional description of the sample.")
    gold_model: GoldBPMNModel
    submit_to_service: bool = Field(
        default=True,
        description="Whether to submit generated BPMN to the external conversion service.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "EvalSample",
    "GoldBPMNModel",
]
