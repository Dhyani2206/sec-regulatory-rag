"""
Module: app/api/v1/endpoints/browse.py

Read-only corpus browsing endpoints.

Endpoints
---------
GET /api/v1/browse           — sections for a company / form / year filing
GET /api/v1/browse/chunks    — paginated chunks for a specific section
GET /api/v1/corpus-stats     — aggregate statistics over the indexed corpus
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.v1.schemas.browse import (
    BrowseResponse,
    ChunkItem,
    ChunksResponse,
    CorpusStatsResponse,
    SectionItem,
)
from app.services.browse_service import get_corpus_stats, list_chunks, list_sections

router = APIRouter()


@router.get(
    "/browse",
    response_model=BrowseResponse,
    summary="List sections in a filing",
    description=(
        "Returns all indexed sections for the specified company / form / year, "
        "each with a chunk count and a short text preview. "
        "Use the /api/v1/options endpoint to discover valid scope combinations."
    ),
    tags=["browse"],
)
def browse_filing(
    company: str = Query(..., description="Company ticker, e.g. AAPL"),
    form_folder: str = Query(..., description="Form type, e.g. 10-K"),
    year: int = Query(..., description="Filing year, e.g. 2023"),
) -> BrowseResponse:
    sections_raw = list_sections(company, form_folder, year)
    return BrowseResponse(
        company=company.upper(),
        form_folder=form_folder,
        year=year,
        total_sections=len(sections_raw),
        sections=[SectionItem(**s) for s in sections_raw],
    )


@router.get(
    "/browse/chunks",
    response_model=ChunksResponse,
    summary="Get chunks for a filing section",
    description=(
        "Returns paginated text chunks for a specific company / form / year / section. "
        "Use page and page_size to navigate large sections."
    ),
    tags=["browse"],
)
def browse_chunks(
    company: str = Query(..., description="Company ticker"),
    form_folder: str = Query(..., description="Form type, e.g. 10-K"),
    year: int = Query(..., description="Filing year"),
    section: str = Query(..., description="Section label, e.g. ITEM 1A"),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=10, ge=1, le=50, description="Chunks per page"),
) -> ChunksResponse:
    result = list_chunks(company, form_folder, year, section, page, page_size)
    return ChunksResponse(
        company=company.upper(),
        form_folder=form_folder,
        year=year,
        section=section.upper(),
        page=result["page"],
        page_size=result["page_size"],
        total_chunks=result["total_chunks"],
        chunks=[ChunkItem(**c) for c in result["chunks"]],
    )


@router.get(
    "/corpus-stats",
    response_model=CorpusStatsResponse,
    summary="Aggregate corpus statistics",
    description=(
        "Returns aggregate counts for the entire indexed corpus: "
        "companies, filings, filing chunks, rule chunks, citations, and years covered."
    ),
    tags=["browse"],
)
def corpus_stats() -> CorpusStatsResponse:
    stats = get_corpus_stats()
    return CorpusStatsResponse(**stats)
