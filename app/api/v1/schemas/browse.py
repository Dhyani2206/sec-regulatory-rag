"""
Pydantic schemas for the document browse and corpus stats endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SectionItem(BaseModel):
    """One filing section with a chunk count and a short text preview."""

    section: str
    chunk_count: int
    preview_text: Optional[str] = None
    title: Optional[str] = None


class BrowseResponse(BaseModel):
    """Sections available in a single indexed filing."""

    company: str
    form_folder: str
    year: int
    total_sections: int
    sections: List[SectionItem] = Field(default_factory=list)


class ChunkItem(BaseModel):
    """A single text chunk from a filing section."""

    chunk_id: Optional[str] = None
    section: Optional[str] = None
    title: Optional[str] = None
    text: str


class ChunksResponse(BaseModel):
    """Paginated chunks for a filing section."""

    company: str
    form_folder: str
    year: int
    section: str
    page: int
    page_size: int
    total_chunks: int
    chunks: List[ChunkItem] = Field(default_factory=list)


class CorpusStatsResponse(BaseModel):
    """Aggregate statistics over the entire indexed corpus."""

    total_companies: int
    total_forms: int
    total_filings: int
    total_filing_chunks: int
    total_rule_chunks: int
    total_rule_citations: int
    total_rule_parts: int
    companies: List[str] = Field(default_factory=list)
    forms: List[str] = Field(default_factory=list)
    years_covered: List[int] = Field(default_factory=list)
    chunks_per_company: Dict[str, int] = Field(default_factory=dict)
