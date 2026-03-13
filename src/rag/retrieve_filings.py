"""
Module: retrieve_filings.py

Purpose
-------
Retrieves relevant filing text chunks from the FAISS filings index.

Responsibilities
----------------
- Load chunk corpus and filings FAISS index
- Embed user query
- Perform vector similarity search
- Normalize schema fields for downstream modules
- Apply company/form/year/section scoping
- Expand common regulatory queries
- Re-rank results using section-aware boosts

Inputs
------
User query string
storage/faiss.index
storage/chunks.jsonl

Outputs
-------
List of filing evidence chunks with similarity scores.

Pipeline Position
-----------------
Retrieval layer of the RAG pipeline.

Notes
-----
Returned chunks include metadata such as company, form type,
year, and section for audit traceability.

This module supports:
- strict scope filtering
- section-aware reranking
- query expansion for common filing concepts
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

from src.rag.config import (
    DEFAULT_CONFIG,
    DEFAULT_FILING_TOP_K,
    FILING_CHUNKS_PATH,
    FILING_EMBEDDING_MODEL,
    FILING_INDEX_PATH,
)
from src.rag.embeddings import embed_texts
cfg = DEFAULT_CONFIG

# MODULE LEVEL CACHE
_CHUNKS_CACHE: Optional[List[Dict[str, Any]]] = None
_INDEX_CACHE = None

# LOADERS
def _load_chunks() -> List[Dict[str, Any]]:
    """
    Load filing chunks once and cache them in memory.
    """
    global _CHUNKS_CACHE

    if _CHUNKS_CACHE is None:
        chunks: List[Dict[str, Any]] = []
        with FILING_CHUNKS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
        _CHUNKS_CACHE = chunks

    return _CHUNKS_CACHE

def _load_index():
    """
    Load FAISS filings index once and cache it in memory.
    """
    global _INDEX_CACHE

    if _INDEX_CACHE is None:
        if not FILING_INDEX_PATH.exists():
            raise FileNotFoundError(f"Missing filings index: {FILING_INDEX_PATH}")
        _INDEX_CACHE = faiss.read_index(str(FILING_INDEX_PATH))

    return _INDEX_CACHE

# Normalization helpers
def normalize_form(value: Optional[str]) -> str:
    if not value:
        return ""

    v = str(value).strip().lower().replace("_", "-")
    if v in {"10k", "10-k"}:
        return "10-k"
    if v in {"10q", "10-q"}:
        return "10-q"
    return v

def normalize_chunk_schema(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize chunk metadata so downstream code can rely on stable keys.
    """
    out = chunk.copy()

    if out.get("form") is None:
        if out.get("form_type"):
            out["form"] = out["form_type"]
        elif out.get("doc_type"):
            out["form"] = out["doc_type"]
        elif out.get("form_folder"):
            out["form"] = out["form_folder"]

    out["form"] = normalize_form(out.get("form"))

    if out.get("company") is not None:
        out["company"] = str(out["company"]).upper()

    if out.get("year") is not None:
        out["year"] = str(out["year"])

    if out.get("section") is not None:
        out["section"] = str(out["section"]).upper().strip()

    return out

def chunk_matches_scope(
    chunk: Dict[str, Any],
    company: Optional[str] = None,
    form: Optional[str] = None,
    year: Optional[str] = None,
    section: Optional[str] = None,
) -> bool:
    if company and str(chunk.get("company", "")).upper() != str(company).upper():
        return False

    if form and normalize_form(chunk.get("form")) != normalize_form(form):
        return False

    if year and str(chunk.get("year", "")) != str(year):
        return False

    if section and str(chunk.get("section", "")).upper().strip() != str(section).upper().strip():
        return False

    return True

# Query expansion and section preference
def expand_query(query: str) -> str:
    """
    Expand common filing-related queries to align better with SEC section language.
    """
    q = str(query or "").strip()
    q_lower = q.lower()
    expansions: List[str] = []

    if "risk factor" in q_lower:
        expansions.append("item 1a risk factors")

    if "management discussion" in q_lower or "md&a" in q_lower:
        expansions.append("item 7 management discussion and analysis")
        expansions.append("item 2 management discussion and analysis")

    if "liquidity" in q_lower:
        expansions.append("item 7 liquidity capital resources cash flows")
        expansions.append("item 2 liquidity capital resources cash flows")

    if "cyber" in q_lower or "cybersecurity" in q_lower:
        expansions.append("item 1a cybersecurity risk information security")

    if "legal proceeding" in q_lower or "litigation" in q_lower:
        expansions.append("item 3 legal proceedings litigation")

    if "supply chain" in q_lower or "supplier" in q_lower or "vendor" in q_lower:
        expansions.append("item 1a supply chain supplier vendor third party")

    if not expansions:
        return q

    return q + " " + " ".join(expansions)


def infer_preferred_sections(query: str, form: Optional[str]) -> List[str]:
    """
    Infer likely target sections from the query and filing form.
    """
    q = str(query or "").lower()
    form_norm = normalize_form(form)

    preferred: List[str] = []

    if "risk factor" in q:
        preferred.append("ITEM 1A")

    if "management discussion" in q or "md&a" in q:
        if form_norm == "10-k":
            preferred.append("ITEM 7")
        elif form_norm == "10-q":
            preferred.append("ITEM 2")
        else:
            preferred.extend(["ITEM 7", "ITEM 2"])

    if "liquidity" in q or "capital resources" in q or "cash flow" in q:
        if form_norm == "10-k":
            preferred.append("ITEM 7")
        elif form_norm == "10-q":
            preferred.append("ITEM 2")
        else:
            preferred.extend(["ITEM 7", "ITEM 2"])

    if "cyber" in q or "cybersecurity" in q or "information security" in q:
        preferred.append("ITEM 1A")

    if "legal proceeding" in q or "litigation" in q:
        preferred.append("ITEM 3")

    return list(dict.fromkeys(preferred))


def section_boost(
    section: str,
    preferred_sections: List[str],
    explicit_section: Optional[str],
) -> float:
    """
    Compute a reranking boost based on section preference.
    """
    sec = str(section or "").upper().strip()

    if explicit_section and sec == str(explicit_section).upper().strip():
        return 0.25

    if sec in preferred_sections:
        return 0.20

    return 0.0

# Main Retrieval Function
def retrieve_filings(
    query: str,
    top_k: int = DEFAULT_FILING_TOP_K,
    company: Optional[str] = None,
    form: Optional[str] = None,
    year: Optional[str] = None,
    section: Optional[str] = None,
    candidate_pool: int = 80,
) -> List[Dict[str, Any]]:
    """
    Retrieve filing evidence from the filings FAISS index.

    Strategy
    --------
    1. Expand the query for common filing concepts.
    2. Retrieve a larger semantic candidate pool.
    3. Filter by scope.
    4. Re-rank using section-aware boosts.
    5. Return top-k reranked hits.

    Notes
    -----
    This is designed to improve semantic retrieval quality for
    section-oriented filing questions without sacrificing scope integrity.
    """
    q = str(query or "").strip()
    if not q:
        raise ValueError("query is required")

    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    if candidate_pool < top_k:
        candidate_pool = top_k

    chunks = _load_chunks()
    index = _load_index()

    expanded_query = expand_query(q)
    preferred_sections = infer_preferred_sections(expanded_query, form)

    qvec = embed_texts([expanded_query], FILING_EMBEDDING_MODEL)
    qvec = np.asarray(qvec, dtype=np.float32)

    if qvec.shape[1] != index.d:
        raise ValueError(
            f"Filing query embedding dimension mismatch: got {qvec.shape[1]}, expected {index.d}. "
            f"Check FILING_EMBEDDING_MODEL and how faiss.index was built."
        )

    scores, ids = index.search(qvec, candidate_pool)

    scoped_hits: List[Dict[str, Any]] = []
    fallback_same_scope_any_section: List[Dict[str, Any]] = []

    for idx, score in zip(ids[0], scores[0]):
        if idx < 0:
            continue
        if idx >= len(chunks):
            continue

        c = normalize_chunk_schema(chunks[idx])

        adjusted_score = float(score) + section_boost(
            section=c.get("section", ""),
            preferred_sections=preferred_sections,
            explicit_section=section,
        )

        item = {
            "score": float(score),
            "adjusted_score": adjusted_score,
            "chunk": c,
        }

        # Priority: strict full scope match, including section when requested
        if chunk_matches_scope(c, company=company, form=form, year=year, section=section):
            scoped_hits.append(item)
            continue

        # Fallback: same filing scope, but different section
        if chunk_matches_scope(c, company=company, form=form, year=year, section=None):
            fallback_same_scope_any_section.append(item)

    scoped_hits.sort(key=lambda x: x["adjusted_score"], reverse=True)
    fallback_same_scope_any_section.sort(key=lambda x: x["adjusted_score"], reverse=True)

    # Priority 1: exact scoped section matches
    if scoped_hits:
        return scoped_hits[:top_k]

    # Priority 2: same company/form/year but different section
    if fallback_same_scope_any_section:
        return fallback_same_scope_any_section[:top_k]

    # Priority 3: no scope provided → global semantic retrieval
    if not any([company, form, year, section]):
        global_results: List[Dict[str, Any]] = []

        for idx, score in zip(ids[0], scores[0]):
            if idx < 0:
                continue
            if idx >= len(chunks):
                continue

            c = normalize_chunk_schema(chunks[idx])

            adjusted_score = float(score) + section_boost(
                section=c.get("section", ""),
                preferred_sections=preferred_sections,
                explicit_section=section,
            )

            global_results.append(
                {
                    "score": float(score),
                    "adjusted_score": adjusted_score,
                    "chunk": c,
                }
            )

        global_results.sort(key=lambda x: x["adjusted_score"], reverse=True)
        return global_results[:top_k]

    return []