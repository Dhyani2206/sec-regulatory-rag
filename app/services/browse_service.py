"""
Module: app/services/browse_service.py

Provides read-only browsing over the indexed corpus stored in storage/.

Responsibilities
----------------
- List sections available for a specific company / form / year filing
- Return paginated chunks for a specific section
- Compute aggregate corpus statistics

All results are derived from ``storage/chunks.jsonl`` and
``storage/rules_chunks.jsonl``.  No embeddings are loaded.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FILING_CHUNKS_PATH = _PROJECT_ROOT / "storage" / "chunks.jsonl"
_RULES_CHUNKS_PATH = _PROJECT_ROOT / "storage" / "rules_chunks.jsonl"

# Module-level caches — re-read only once per process lifetime.
_FILING_ROWS: Optional[List[Dict[str, Any]]] = None
_RULES_ROWS: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_filing_rows() -> List[Dict[str, Any]]:
    global _FILING_ROWS
    if _FILING_ROWS is None:
        _FILING_ROWS = _read_jsonl(_FILING_CHUNKS_PATH)
    return _FILING_ROWS


def _load_rules_rows() -> List[Dict[str, Any]]:
    global _RULES_ROWS
    if _RULES_ROWS is None:
        _RULES_ROWS = _read_jsonl(_RULES_CHUNKS_PATH)
    return _RULES_ROWS


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
    return rows


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_company(val: Any) -> str:
    return str(val or "").strip().upper()


def _norm_form(val: Any) -> str:
    v = str(val or "").strip().lower().replace("_", "-")
    if v in {"10k", "10-k"}:
        return "10-K"
    if v in {"10q", "10-q"}:
        return "10-Q"
    return v.upper() if v else ""


def _norm_year(val: Any) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _extract_row_scope(row: Dict[str, Any]) -> Tuple[str, str, Optional[int]]:
    company = _norm_company(
        row.get("company") or row.get("ticker") or row.get("symbol")
    )
    form = _norm_form(
        row.get("form") or row.get("form_folder") or row.get("form_type") or row.get("doc_type")
    )
    year = _norm_year(row.get("year"))
    return company, form, year


def _row_text(row: Dict[str, Any]) -> str:
    for key in ("text", "content", "chunk_text", "body", "passage"):
        v = row.get(key, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ---------------------------------------------------------------------------
# list_sections
# ---------------------------------------------------------------------------

def list_sections(
    company: str,
    form_folder: str,
    year: int,
) -> List[Dict[str, Any]]:
    """
    Return sections for a single filing, sorted by label.

    Each item: ``{section, chunk_count, preview_text, title}``
    """
    company_n = _norm_company(company)
    form_n = _norm_form(form_folder)
    year_n = year

    rows = _load_filing_rows()

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        c, f, y = _extract_row_scope(row)
        if c != company_n or f != form_n or y != year_n:
            continue
        section = str(row.get("section") or "").strip().upper() or "UNKNOWN"
        grouped[section].append(row)

    sections: List[Dict[str, Any]] = []
    for section, items in sorted(grouped.items()):
        first_text = _row_text(items[0])
        title = str(items[0].get("title") or "").strip()
        sections.append(
            {
                "section": section,
                "chunk_count": len(items),
                "preview_text": first_text[:320] + ("..." if len(first_text) > 320 else ""),
                "title": title or None,
            }
        )

    return sections


# ---------------------------------------------------------------------------
# list_chunks
# ---------------------------------------------------------------------------

def list_chunks(
    company: str,
    form_folder: str,
    year: int,
    section: str,
    page: int = 1,
    page_size: int = 10,
) -> Dict[str, Any]:
    """
    Return paginated chunks for a single filing section.

    Returns ``{total_chunks, page, page_size, chunks: [{chunk_id, section, title, text}]}``.
    """
    company_n = _norm_company(company)
    form_n = _norm_form(form_folder)
    year_n = year
    section_n = section.strip().upper()

    rows = _load_filing_rows()
    matched: List[Dict[str, Any]] = []
    for row in rows:
        c, f, y = _extract_row_scope(row)
        s = str(row.get("section") or "").strip().upper()
        if c == company_n and f == form_n and y == year_n and s == section_n:
            matched.append(row)

    total = len(matched)
    page = max(1, page)
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = matched[start:end]

    chunks = []
    for row in page_rows:
        chunks.append(
            {
                "chunk_id": row.get("chunk_id"),
                "section": str(row.get("section") or "").strip().upper(),
                "title": str(row.get("title") or "").strip() or None,
                "text": _row_text(row),
            }
        )

    return {
        "total_chunks": total,
        "page": page,
        "page_size": page_size,
        "chunks": chunks,
    }


# ---------------------------------------------------------------------------
# get_corpus_stats
# ---------------------------------------------------------------------------

def get_corpus_stats() -> Dict[str, Any]:
    """
    Compute aggregate statistics over the indexed corpus.

    Returns a dict with counts, company list, year range, and per-company
    chunk counts for charting.
    """
    filing_rows = _load_filing_rows()
    rules_rows = _load_rules_rows()

    companies: set = set()
    forms: set = set()
    years: set = set()
    per_company: Dict[str, int] = defaultdict(int)
    filings: set = set()  # (company, form, year) triples

    for row in filing_rows:
        c, f, y = _extract_row_scope(row)
        if c:
            companies.add(c)
            per_company[c] += 1
        if f:
            forms.add(f)
        if y is not None:
            years.add(y)
        if c and f and y is not None:
            filings.add((c, f, y))

    # Rules corpus: count citations and parts
    rule_parts: set = set()
    rule_citations: set = set()
    for row in rules_rows:
        part = str(row.get("part") or "").strip()
        cit = str(row.get("citation") or "").strip()
        if part:
            rule_parts.add(part)
        if cit:
            rule_citations.add(cit)

    return {
        "total_companies": len(companies),
        "total_forms": len(forms),
        "total_filings": len(filings),
        "total_filing_chunks": len(filing_rows),
        "total_rule_chunks": len(rules_rows),
        "total_rule_citations": len(rule_citations),
        "total_rule_parts": len(rule_parts),
        "companies": sorted(companies),
        "forms": sorted(forms),
        "years_covered": sorted(years),
        "chunks_per_company": dict(sorted(per_company.items())),
    }
