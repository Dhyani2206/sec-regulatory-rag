from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User regulatory question")
    company: Optional[str] = Field(default=None, description="Optional filing company/ticker scope")
    form_folder: Optional[str] = Field(default=None, description="Optional filing type scope, e.g. 10-K or 10-Q")
    year: Optional[int] = Field(default=None, description="Optional filing year scope")


class EvidenceItem(BaseModel):
    source_type: Optional[str] = None
    citation: Optional[str] = None
    section: Optional[str] = None
    company: Optional[str] = None
    form: Optional[str] = None
    year: Optional[int] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceItem(BaseModel):
    label: str
    section: Optional[str] = None
    company: Optional[str] = None
    form: Optional[str] = None
    year: Optional[int] = None


class EvidencePreviewItem(BaseModel):
    label: str
    snippet: Optional[str] = None
    section: Optional[str] = None
    score: Optional[float] = None


class ReviewResult(BaseModel):
    finding: str
    label: str
    confidence: str
    evidence_strength: str


class TechnicalTrace(BaseModel):
    route: Optional[Union[str, Dict[str, Any]]] = None
    guard_reason: Optional[Union[str, Dict[str, Any]]] = None


class QueryResponse(BaseModel):
    status: str
    review_result: ReviewResult
    answer: str
    plain_english_explanation: Optional[str] = None
    why_it_matters: Optional[str] = None
    review_guidance: List[str] = Field(default_factory=list)
    sources: List[SourceItem] = Field(default_factory=list)
    evidence_preview: List[EvidencePreviewItem] = Field(default_factory=list)
    technical_trace: Optional[TechnicalTrace] = None
    notes: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)