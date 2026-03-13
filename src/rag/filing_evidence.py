"""
Module: filing_evidence.py

Purpose
-------
Provides deterministic filing evidence lookup utilities.

Responsibilities
----------------
- Normalize filing form names
- Load scoped chunk evidence for a company/form/year/section

Pipeline Position
-----------------
Shared evidence utility layer used by answer generation,
gap reporting, and evaluation.

Notes
-----
This module is intentionally deterministic. It does not perform
semantic retrieval. It only loads exact filing evidence for a
specific company / form / year / section scope.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.rag.config import DEFAULT_CONFIG


cfg = DEFAULT_CONFIG
_CHUNKS_CACHE: Optional[List[Dict[str, Any]]] = None


def _load_chunks() -> List[Dict[str, Any]]:
    """
    Load filing chunks once and cache them in memory.
    """
    global _CHUNKS_CACHE

    if _CHUNKS_CACHE is None:
        rows: List[Dict[str, Any]] = []

        if not cfg.filing_chunks_path.exists() or cfg.filing_chunks_path.stat().st_size == 0:
            _CHUNKS_CACHE = []
            return _CHUNKS_CACHE

        with cfg.filing_chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        _CHUNKS_CACHE = rows

    return _CHUNKS_CACHE


def normalize_form(value: str) -> str:
    v = str(value or "").strip().lower().replace("_", "-")
    if v in {"10k", "10-k"}:
        return "10-k"
    if v in {"10q", "10-q"}:
        return "10-q"
    return v


def load_scoped_section_chunk_evidence(
    company: str,
    form_folder: str,
    year: str,
    section: str,
    max_hits: int = 5,
) -> List[Dict[str, Any]]:
    """
    Load exact chunk evidence for a single filing section.

    Returns
    -------
    List[dict]
        Deterministic evidence items in the shape:
        {
            "score": 1.0,
            "chunk": {...}
        }
    """
    if max_hits <= 0:
        raise ValueError("max_hits must be >= 1")

    rows = _load_chunks()
    hits: List[Dict[str, Any]] = []

    company_norm = str(company).upper()
    form_norm = normalize_form(form_folder)
    year_norm = str(year)
    section_norm = str(section).upper().strip()

    for rec in rows:
        rec_company = str(rec.get("company", "")).upper()
        rec_year = str(rec.get("year", ""))
        rec_section = str(rec.get("section", "")).upper().strip()

        rec_form = rec.get("form")
        if rec_form is None:
            rec_form = rec.get("form_type") or rec.get("doc_type") or rec.get("form_folder")
        rec_form = normalize_form(rec_form)

        if rec_company != company_norm:
            continue
        if rec_form != form_norm:
            continue
        if rec_year != year_norm:
            continue
        if rec_section != section_norm:
            continue

        normalized = rec.copy()
        normalized["company"] = rec_company
        normalized["year"] = rec_year
        normalized["section"] = rec_section
        normalized["form"] = rec_form

        hits.append(
            {
                "score": 1.0,
                "chunk": normalized,
            }
        )

        if len(hits) >= max_hits:
            break

    return hits