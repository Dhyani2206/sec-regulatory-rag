"""
Module: src/rag/agent/tools.py

LangChain @tool wrappers around the existing RAG retrieval functions.

Each tool accepts simple string / integer arguments and returns a
formatted evidence string so the LLM can reason over them.

Tools exposed
-------------
retrieve_filing_chunks     — semantic retrieval from the filings FAISS index
retrieve_rule_chunks       — citation-aware retrieval from the rules FAISS index
load_filing_section        — deterministic section-level chunk retrieval
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from langchain.tools import tool

from src.rag.filing_evidence import load_scoped_section_chunk_evidence
from src.rag.retrieve_filings import retrieve_filings
from src.rag.rules_router import retrieve_rules_routed

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 800
_SEPARATOR = "\n---\n"


def _format_chunk(hit: dict, idx: int) -> str:
    """
    Render a single evidence hit as a concise text block.

    Each block includes the chunk ID (for citation), metadata, and an
    excerpt of the text truncated to ``_MAX_TEXT_CHARS`` characters.
    """
    chunk = hit.get("chunk", hit)
    chunk_id = chunk.get("chunk_id") or chunk.get("citation") or f"chunk_{idx}"
    score = hit.get("score", hit.get("adjusted_score"))

    parts = [f"[{chunk_id}]"]

    meta_fields = [
        ("company", chunk.get("company")),
        ("form", chunk.get("form") or chunk.get("form_folder")),
        ("year", chunk.get("year")),
        ("section", chunk.get("section")),
        ("citation", chunk.get("citation")),
    ]
    meta = ", ".join(f"{k}={v}" for k, v in meta_fields if v)
    if meta:
        parts.append(f"Meta: {meta}")
    if score is not None:
        parts.append(f"Score: {score:.4f}")

    text = (
        chunk.get("text")
        or chunk.get("content")
        or chunk.get("chunk_text")
        or ""
    ).strip()
    if text:
        parts.append(text[:_MAX_TEXT_CHARS] + ("..." if len(text) > _MAX_TEXT_CHARS else ""))

    return "\n".join(parts)


@tool
def retrieve_filing_chunks(
    query: str,
    company: Optional[str] = None,
    form: Optional[str] = None,
    year: Optional[str] = None,
    top_k: int = 6,
) -> str:
    """
    Retrieve relevant filing text chunks from the FAISS filings index.

    Use this tool when you need evidence about a specific SEC filing
    (risk factors, MD&A, liquidity, legal proceedings, etc.).

    Parameters
    ----------
    query:
        The user's question or retrieval query.
    company:
        Optional company ticker to restrict retrieval scope (e.g. "AAPL").
    form:
        Optional form type filter, e.g. "10-K" or "10-Q".
    year:
        Optional filing year as a string, e.g. "2023".
    top_k:
        Number of top evidence chunks to return (default 6).

    Returns
    -------
    str
        Formatted evidence chunks with chunk IDs for citation.
    """
    try:
        hits = retrieve_filings(
            query=query,
            company=company or None,
            form=form or None,
            year=str(year) if year else None,
            top_k=top_k,
        )
    except Exception as exc:
        logger.warning("retrieve_filing_chunks failed: %s", exc)
        return f"Error retrieving filing chunks: {exc}"

    if not hits:
        return "No filing chunks found for the given query and scope."

    return _SEPARATOR.join(_format_chunk(h, i) for i, h in enumerate(hits))


@tool
def retrieve_rule_chunks(query: str, top_k: int = 8) -> str:
    """
    Retrieve relevant regulatory rule chunks from the FAISS rules index.

    Use this tool when you need evidence about SEC / CFR regulatory
    requirements, obligations, or rule text.  It prioritises exact CFR
    citation matches (e.g. "17 CFR 229.105") before falling back to
    semantic retrieval.

    Parameters
    ----------
    query:
        The user's question, optionally containing a CFR citation.
    top_k:
        Number of top rule evidence chunks to return (default 8).

    Returns
    -------
    str
        Formatted rule chunks with chunk IDs / citations.
    """
    try:
        hits = retrieve_rules_routed(query=query, top_k=top_k)
    except Exception as exc:
        logger.warning("retrieve_rule_chunks failed: %s", exc)
        return f"Error retrieving rule chunks: {exc}"

    if not hits:
        return "No regulatory rule chunks found for the given query."

    return _SEPARATOR.join(_format_chunk(h, i) for i, h in enumerate(hits))


@tool
def load_filing_section(
    company: str,
    form_folder: str,
    year: str,
    section: str,
    max_hits: int = 5,
) -> str:
    """
    Load deterministic chunk evidence for a specific section of a single filing.

    Use this tool when you already know the exact company / form / year / section
    and want the literal text from that section rather than a semantic match.

    Parameters
    ----------
    company:
        Company ticker, e.g. "AAPL".
    form_folder:
        Form type, e.g. "10-K" or "10-Q".
    year:
        Filing year as a string, e.g. "2023".
    section:
        SEC filing section label, e.g. "ITEM 1A" or "ITEM 7".
    max_hits:
        Maximum number of chunks to return (default 5).

    Returns
    -------
    str
        Formatted evidence chunks from the specified section.
    """
    try:
        hits = load_scoped_section_chunk_evidence(
            company=company,
            form_folder=form_folder,
            year=str(year),
            section=section,
            max_hits=max_hits,
        )
    except Exception as exc:
        logger.warning("load_filing_section failed: %s", exc)
        return f"Error loading filing section: {exc}"

    if not hits:
        return (
            f"No chunks found for {company} / {form_folder} / {year} / {section}. "
            "Check that this scope exists in the indexed dataset."
        )

    return _SEPARATOR.join(_format_chunk(h, i) for i, h in enumerate(hits))
