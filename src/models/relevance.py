"""Pydantic models for relevance evaluation of retrieved chunks."""

from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class ChunkInput(BaseModel):
    """Input chunk payload for relevance evaluation."""

    chunk_nr: int  # Sequential number for enumeration (1, 2, 3, ...)
    chunk_id: str  # Unique ChromaDB ID, e.g. "procedure_manual_chunk_0"
    text: str  # Chunk text content


class EvidenceSpan(BaseModel):
    """Evidence span extracted from a chunk."""

    text: str = Field(..., description="Exact snippet from the chunk (max ~25 words)")


class ChunkAssessment(BaseModel):
    """Relevance assessment for a single chunk."""

    chunk_nr: int  # Sequential number (1, 2, 3, ...) instead of chunk_id
    relevance: Literal["high", "medium", "low", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)
    why_not_relevant: Optional[str] = None


class RelevanceEvaluationResult(BaseModel):
    """Complete relevance evaluation result for all chunks."""

    query: str
    decision: Literal["proceed", "refine_query", "fetch_neighbors"] = "proceed"
    chunk_assessments: List[ChunkAssessment]

