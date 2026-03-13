"""
Module: retrieve_filing_sections.py

Purpose
-------
Performs section-level retrieval inside a single scoped filing.

Responsibilities
----------------
- Load filing chunks from chunks.jsonl
- Restrict to a single company / form / year
- Group chunks by section
- Build section-level text representations
- Score sections against a query using embeddings
- Return ranked sections for downstream chunk retrieval

Inputs
------
- User query
- Filing scope (company, form, year)
- storage/chunks.jsonl

Outputs
-------
List of ranked filing sections with similarity scores.

Pipeline Position
-----------------
Intermediate retrieval layer between scoped filing filtering
and chunk-level evidence retrieval.

Notes
-----
This module is designed to solve section discovery more reliably than
flat chunk-level semantic retrieval. It is especially useful for:
- risk factors
- MD&A
- liquidity
- legal proceedings
- cybersecurity

This module does not retrieve final answer chunks. It ranks sections first.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from src.rag.config import DEFAULT_CONFIG
from src.rag.embeddings import embed_texts
from src.rag.filing_evidence import normalize_form


cfg = DEFAULT_CONFIG

_CHUNKS_CACHE: Optional[List[Dict[str, Any]]] = None


def load_chunks() -> List[Dict[str, Any]]:
    """
    Load filing chunks once and cache them in memory.
    """
    global _CHUNKS_CACHE

    if _CHUNKS_CACHE is None:
        rows: List[Dict[str, Any]] = []
        with cfg.filing_chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        _CHUNKS_CACHE = rows

    return _CHUNKS_CACHE


def normalize_chunk_schema(chunk: Dict[str, Any]) -> Dict[str, Any]:
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


def chunk_matches_filing_scope(
    chunk: Dict[str, Any],
    company: str,
    form: str,
    year: str,
) -> bool:
    return (
        str(chunk.get("company", "")).upper() == str(company).upper()
        and normalize_form(chunk.get("form")) == normalize_form(form)
        and str(chunk.get("year", "")) == str(year)
    )


def expand_section_query(query: str, form: Optional[str]) -> str:
    """
    Expand common section-discovery queries so they align better with SEC filing language.
    """
    q = str(query or "").strip()
    ql = q.lower()
    form_norm = normalize_form(form)

    expansions: List[str] = []

    if "risk factor" in ql:
        expansions.extend(["item 1a", "risk factors"])

    if "management discussion" in ql or "md&a" in ql:
        if form_norm == "10-k":
            expansions.extend(["item 7", "management discussion and analysis"])
        elif form_norm == "10-q":
            expansions.extend(["item 2", "management discussion and analysis"])
        else:
            expansions.extend(["item 7", "item 2", "management discussion and analysis"])

    if "liquidity" in ql or "capital resources" in ql or "cash flow" in ql:
        if form_norm == "10-k":
            expansions.extend(["item 7", "liquidity", "capital resources", "cash flows"])
        elif form_norm == "10-q":
            expansions.extend(["item 2", "liquidity", "capital resources", "cash flows"])
        else:
            expansions.extend(["item 7", "item 2", "liquidity", "capital resources", "cash flows"])

    if "cyber" in ql or "cybersecurity" in ql or "information security" in ql:
        expansions.extend(["item 1a", "item 1c", "cybersecurity", "information security"])

    if "legal proceeding" in ql or "litigation" in ql:
        expansions.extend(["item 3", "legal proceedings", "litigation"])

    if not expansions:
        return q

    return q + " " + " ".join(expansions)


def section_prior_score(section: str, query: str, form: Optional[str]) -> float:
    """
    Stronger structural prior for section-aware retrieval.

    This is intentionally more decisive than a weak semantic hint,
    because section discovery in SEC filings is a structured problem.
    """
    sec = str(section or "").upper().strip()
    ql = str(query or "").lower()
    form_norm = normalize_form(form)

    score = 0.0

    # Risk Factors
    if "risk factor" in ql:
        if sec == "ITEM 1A":
            score += 0.35

    # MD&A
    if "management discussion" in ql or "md&a" in ql:
        if form_norm == "10-k":
            if sec == "ITEM 7":
                score += 0.40
            elif sec == "ITEM 7A":
                score += 0.08
            elif sec == "ITEM 9A":
                score -= 0.05
        elif form_norm == "10-q":
            if sec == "ITEM 2":
                score += 0.40
            elif sec == "ITEM 4":
                score -= 0.05

    # Liquidity usually lives inside MD&A
    if "liquidity" in ql or "capital resources" in ql or "cash flow" in ql:
        if form_norm == "10-k":
            if sec == "ITEM 7":
                score += 0.35
            elif sec == "ITEM 7A":
                score += 0.05
        elif form_norm == "10-q":
            if sec == "ITEM 2":
                score += 0.35

    # Cybersecurity
    if "cyber" in ql or "cybersecurity" in ql or "information security" in ql:
        if sec == "ITEM 1C":
            score += 0.30
        elif sec == "ITEM 1A":
            score += 0.18

    # Legal proceedings
    if "legal proceeding" in ql or "litigation" in ql:
        if sec == "ITEM 3":
            score += 0.35

    return score


def build_section_documents(
    company: str,
    form: str,
    year: str,
) -> List[Dict[str, Any]]:
    """
    Build one retrieval document per section for a scoped filing.
    """
    rows = load_chunks()

    scoped: List[Dict[str, Any]] = []
    for row in rows:
        rr = normalize_chunk_schema(row)
        if chunk_matches_filing_scope(rr, company, form, year):
            scoped.append(rr)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in scoped:
        sec = row.get("section", "")
        if not sec:
            continue
        grouped.setdefault(sec, []).append(row)

    docs: List[Dict[str, Any]] = []
    for sec, chunks in grouped.items():
        chunks = sorted(chunks, key=lambda x: x.get("chunk_id", ""))

        preview_parts: List[str] = []
        for chunk in chunks[:3]:
            txt = str(chunk.get("text", "")).strip()
            if txt:
                preview_parts.append(txt[:1200])

        preview_text = "\n".join(preview_parts).strip()
        section_doc = f"{sec}\n{preview_text}"

        docs.append(
            {
                "company": company.upper(),
                "form": normalize_form(form),
                "year": str(year),
                "section": sec,
                "chunk_count": len(chunks),
                "text": section_doc,
            }
        )

    return docs


def retrieve_filing_sections(
    query: str,
    company: str,
    form: str,
    year: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rank sections for a single scoped filing.
    """
    q = str(query or "").strip()
    if not q:
        raise ValueError("query is required")

    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    section_docs = build_section_documents(company, form, year)
    if not section_docs:
        return []

    expanded_query = expand_section_query(q, form)

    query_vec = embed_texts([expanded_query], cfg.filing_embedding_model)
    query_vec = np.asarray(query_vec, dtype=np.float32)[0]

    doc_vecs = embed_texts(
        [doc["text"] for doc in section_docs],
        cfg.filing_embedding_model,
    )
    doc_vecs = np.asarray(doc_vecs, dtype=np.float32)

    scores = np.dot(doc_vecs, query_vec)

    ranked: List[Dict[str, Any]] = []
    for doc, base_score in zip(section_docs, scores):
        prior = section_prior_score(doc["section"], q, form)
        adjusted = float(base_score) + prior

        ranked.append(
            {
                "company": doc["company"],
                "form": doc["form"],
                "year": doc["year"],
                "section": doc["section"],
                "chunk_count": doc["chunk_count"],
                "score": float(base_score),
                "adjusted_score": adjusted,
            }
        )

    ranked.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return ranked[:top_k]