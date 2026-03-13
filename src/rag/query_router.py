"""
Module: query_router.py

Purpose
-------
Routes filing-related queries into the correct retrieval strategy.

Responsibilities
----------------
- Detect structural section lookup questions
- Detect semantic topic lookup questions
- Detect clearly not-applicable requests only when structurally certain
- Return a simple routing decision for downstream retrieval

Inputs
------
- query
- filing form (10-k / 10-q)

Outputs
-------
Dictionary with:
- intent
- target_section (optional)
- allow_semantic_fallback
- reason

Pipeline Position
-----------------
Lightweight routing layer before section retrieval.

Design Notes
------------
This module is intentionally rule-based.

To reduce hallucination risk:
- Use structural routing only for known form/section mappings.
- Do not refuse unless the request is truly structurally inapplicable.
- Prefer semantic fallback when the form is known but the section may be absent or only partially populated.
- Keep routing deterministic and avoid making answer-level claims here.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

from src.rag.filing_evidence import normalize_form


Route = Dict[str, object]


def normalize_query_text(query: str) -> str:
    q = str(query or "").strip().lower()
    q = q.replace("’", "'")
    q = re.sub(r"[^a-z0-9\s&\-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    return any(p in text for p in phrases)


def _structural_route(
    target_section: str,
    reason: str,
    allow_semantic_fallback: bool = False,
) -> Route:
    return {
        "intent": "structural_section_lookup",
        "target_section": target_section,
        "allow_semantic_fallback": allow_semantic_fallback,
        "reason": reason,
    }


def _semantic_route(reason: str) -> Route:
    return {
        "intent": "semantic_topic_lookup",
        "target_section": None,
        "allow_semantic_fallback": True,
        "reason": reason,
    }


def _not_applicable_route(reason: str) -> Route:
    return {
        "intent": "not_applicable",
        "target_section": None,
        "allow_semantic_fallback": False,
        "reason": reason,
    }


def route_filing_query(query: str, form: Optional[str]) -> Route:
    q = normalize_query_text(query)
    form_norm = normalize_form(form)

    # Risk Factors
    # 10-K: Item 1A
    # 10-Q: Part II Item 1A, often for material changes only.
    # For 10-Q we allow semantic fallback because some quarterly filings may
    # have sparse/short disclosures or parser extraction may not surface a
    # strong deterministic section hit.
    if _contains_any(q, ("risk factor", "risk factors")):
        if form_norm == "10-k":
            return _structural_route(
                target_section="ITEM 1A",
                reason="Risk factors map structurally to ITEM 1A in 10-K.",
                allow_semantic_fallback=False,
            )
        if form_norm == "10-q":
            return _structural_route(
                target_section="ITEM 1A",
                reason="Risk factors map structurally to ITEM 1A in 10-Q, with semantic fallback allowed if deterministic section evidence is absent.",
                allow_semantic_fallback=True,
            )
        return _semantic_route(
            reason="Risk factors are filing-structured for known forms, but the supplied form is not explicitly mapped."
        )

    # MD&A
    if _contains_any(
        q,
        (
            "management discussion",
            "discussion and analysis",
            "management s discussion and analysis",
            "md&a",
            "mda",
        ),
    ):
        if form_norm == "10-k":
            return _structural_route(
                target_section="ITEM 7",
                reason="MD&A maps structurally to ITEM 7 in 10-K.",
            )
        if form_norm == "10-q":
            return _structural_route(
                target_section="ITEM 2",
                reason="MD&A maps structurally to ITEM 2 in 10-Q.",
            )
        return _semantic_route(
            reason="MD&A is filing-structured for known forms, but the supplied form is not explicitly mapped."
        )

    # Legal Proceedings
    if _contains_any(q, ("legal proceeding", "legal proceedings", "litigation")):
        if form_norm == "10-k":
            return _structural_route(
                target_section="ITEM 3",
                reason="Legal proceedings map structurally to ITEM 3 in 10-K.",
            )
        if form_norm == "10-q":
            return _structural_route(
                target_section="ITEM 1",
                reason="Legal proceedings map structurally to ITEM 1 in 10-Q.",
            )
        return _semantic_route(
            reason="Legal proceedings are filing-structured for known forms, but the supplied form is not explicitly mapped."
        )

    # Cybersecurity / information security
    if _contains_any(
        q,
        (
            "cyber",
            "cybersecurity",
            "information security",
            "data breach",
            "ransomware",
            "incident response",
        ),
    ):
        return _semantic_route(
            reason="Cybersecurity is a topical lookup and may vary by filing."
        )

    # Liquidity / capital resources / cash flow
    if _contains_any(
        q,
        (
            "liquidity",
            "capital resources",
            "cash flow",
            "cash flows",
        ),
    ):
        return _semantic_route(
            reason="Liquidity is a topical lookup, usually inside MD&A."
        )

    # Supply chain / supplier / vendor / third party
    if _contains_any(
        q,
        (
            "supply chain",
            "supplier",
            "vendor",
            "third party",
            "third-party",
        ),
    ):
        return _semantic_route(
            reason="Supply chain is a topical lookup."
        )

    return _semantic_route(
        reason="Default semantic lookup."
    )