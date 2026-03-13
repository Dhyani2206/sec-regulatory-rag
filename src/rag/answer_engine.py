"""
Module: answer_engine.py

Purpose
-------
Provides the final user-facing answer workflow for the regulatory RAG system.

Responsibilities
----------------
- Route user queries into the correct retrieval strategy
- Detect rule-only questions robustly
- Retrieve rule evidence, filing section evidence, or semantic filing evidence
- Rank retrieved evidence
- Apply a hallucination guard before answering
- Enforce explicit citation matching for rule-only CFR queries
- Add controlled semantic fallback for eligible structural misses
- Emit retrieval debug logs without affecting answer safety
- Return a grounded answer payload with evidence and refusal behavior

Notes
-----
This module does not use an LLM. It is deterministic and evidence-first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.rag.debug_logging import log_retrieval_event
from src.rag.evidence_ranker import build_ranked_evidence_pack
from src.rag.filing_evidence import load_scoped_section_chunk_evidence, normalize_form
from src.rag.query_router import route_filing_query
from src.rag.retrieve_filing_sections import retrieve_filing_sections
from src.rag.retrieve_rules import retrieve_rules
from src.rag.rules_router import retrieve_rules_routed


# ---------------------------------------------------------
# Rule query detection
# ---------------------------------------------------------
RULE_CITATION_RE = re.compile(
    r"(17\s*cfr\s*\d+\.\d+[a-zA-Z0-9\-]*|§\s*229\.\d+[a-zA-Z0-9\-]*|item\s+\d{3,4})",
    re.IGNORECASE,
)

EXPLICIT_CFR_RE = re.compile(
    r"\b17\s*cfr\s+(\d+)\.(\d+[a-zA-Z0-9\-]*)\b",
    re.IGNORECASE,
)


@dataclass
class AnswerConfig:
    max_rule_evidence: int = 3
    max_filing_evidence: int = 3
    min_rule_hits: int = 1
    min_filing_hits: int = 1
    filing_min_score: float = 0.30
    rule_min_score: float = 0.30
    semantic_top_k_sections: int = 5


DEFAULT_CONFIG = AnswerConfig()


# ---------------------------------------------------------
# Query helpers
# ---------------------------------------------------------
def is_rule_only_query(query: str) -> bool:
    q = str(query or "").strip().lower()

    if RULE_CITATION_RE.search(q):
        return True

    rule_prefixes = (
        "what does",
        "explain",
        "summarize",
        "what is required by",
        "what does section",
        "what does item",
    )

    if any(q.startswith(p) for p in rule_prefixes) and (
        "17 cfr" in q or "§" in q or "item " in q
    ):
        return True

    return False


def _extract_explicit_cfr_citation(query: str) -> Optional[str]:
    q = str(query or "").strip()
    m = EXPLICIT_CFR_RE.search(q)
    if not m:
        return None
    return f"17 CFR {m.group(1)}.{m.group(2)}"


def _clean_heading(text: Optional[str]) -> str:
    s = str(text or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(" .")
    return s


def _friendly_section_label(section: Optional[str]) -> str:
    s = str(section or "").upper().strip()

    mapping = {
        "ITEM 1": "Item 1 (Legal Proceedings)",
        "ITEM 1A": "Item 1A (Risk Factors)",
        "ITEM 1C": "Item 1C (Cybersecurity)",
        "ITEM 2": "Item 2 (Management's Discussion and Analysis)",
        "ITEM 3": "Item 3 (Legal Proceedings)",
        "ITEM 4": "Item 4",
        "ITEM 7": "Item 7 (Management's Discussion and Analysis)",
        "ITEM 7A": "Item 7A (Market Risk Disclosures)",
    }

    return mapping.get(s, s)


def _extract_text_from_chunk(chunk: Dict[str, Any]) -> Optional[str]:
    for key in ("text", "content", "chunk_text", "body", "passage", "section_text", "raw_text"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_log(**kwargs: Any) -> None:
    try:
        log_retrieval_event(**kwargs)
    except Exception:
        pass


def _should_try_semantic_fallback(route: Dict[str, Any], form_folder: Optional[str]) -> bool:
    if not isinstance(route, dict):
        return False
    if route.get("intent") != "structural_section_lookup":
        return False
    if not bool(route.get("allow_semantic_fallback", False)):
        return False
    return normalize_form(form_folder) == "10-q"

def _is_executive_compensation_topic(query: str) -> bool:
    ql = str(query or "").lower()
    phrases = (
        "ceo compensation",
        "executive compensation",
        "named executive officer",
        "named executive officers",
        "compensation discussion and analysis",
        "cd&a",
        "summary compensation table",
        "director compensation",
        "proxy compensation",
    )
    return any(p in ql for p in phrases)


def _is_environmental_or_climate_topic(query: str) -> bool:
    ql = str(query or "").lower()
    phrases = (
        "esg",
        "environmental policy",
        "environment policy",
        "climate",
        "climate risk",
        "climate disclosure",
        "ghg",
        "greenhouse gas",
        "carbon emissions",
        "sustainability",
        "net zero",
    )
    return any(p in ql for p in phrases)


def _allowed_semantic_sections_for_topic(
    query: str,
    form_folder: Optional[str],
) -> Optional[set[str]]:
    """
    Return a set of acceptable filing sections for sensitive topics.

    If None is returned, section names alone are not enough to validate the topic.
    In that case, chunk-text topic support is required instead.
    """
    form_norm = normalize_form(form_folder)

    if _is_executive_compensation_topic(query):
        if form_norm == "10-k":
            # Executive compensation in a 10-K context is generally Part III / Item 11,
            # often incorporated by reference from the definitive proxy statement.
            return {"ITEM 11", "PART III ITEM 11", "ITEM 10", "PART III"}
        # Do not force a 10-Q section for this topic.
        return set()

    # For ESG / climate topics, do not rely only on section labels.
    return None


def _topic_keywords(query: str) -> List[str]:
    """
    Conservative keyword expansion used to verify that filing chunk text
    actually supports the topic, not just the retrieval score.
    """
    ql = str(query or "").lower()

    if _is_executive_compensation_topic(query):
        return [
            "compensation",
            "executive",
            "named executive officer",
            "named executive officers",
            "salary",
            "bonus",
            "stock awards",
            "option awards",
            "summary compensation table",
            "compensation discussion and analysis",
            "proxy statement",
            "director compensation",
        ]

    if _is_environmental_or_climate_topic(query):
        return [
            "environmental",
            "climate",
            "sustainability",
            "carbon",
            "emissions",
            "ghg",
            "greenhouse gas",
            "net zero",
            "renewable",
            "environment",
        ]

    return []


def _filing_hits_support_topic(query: str, hits: List[Dict[str, Any]]) -> bool:
    """
    Verify that retrieved filing chunk text actually contains topical support.

    This is intentionally conservative:
    - if we have no special-topic keywords, return True
    - otherwise require at least 2 keyword matches across retrieved filing text
    """
    keywords = _topic_keywords(query)
    if not keywords:
        return True

    combined_text_parts: List[str] = []
    for hit in hits:
        chunk = hit.get("chunk", {}) if isinstance(hit, dict) else {}
        text = _extract_text_from_chunk(chunk)
        if text:
            combined_text_parts.append(text.lower())

    if not combined_text_parts:
        return False

    combined_text = "\n".join(combined_text_parts)

    matched = 0
    for kw in keywords:
        if kw in combined_text:
            matched += 1

    return matched >= 2


def _filter_semantic_sections_for_topic(
    query: str,
    form_folder: Optional[str],
    ranked_sections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Filter ranked semantic sections using topic-aware constraints.

    For some topics (e.g. executive compensation), only certain filing sections
    are acceptable. For others, section names are not enough, so we leave the
    list intact and rely on chunk-text verification later.
    """
    allowed = _allowed_semantic_sections_for_topic(query, form_folder)

    # None means section-name filtering is not useful for this topic.
    if allowed is None:
        return ranked_sections

    # Empty set means there is no acceptable section for this form/topic pairing.
    if not allowed:
        return []

    out: List[Dict[str, Any]] = []
    for s in ranked_sections:
        sec = str(s.get("section", "")).upper().strip()
        if sec in allowed:
            out.append(s)
    return out


def _build_topic_alignment_refusal(
    query: str,
    company: str,
    form_folder: str,
    year: str,
    route: Dict[str, Any],
    ranked_rules: List[Dict[str, Any]],
    notes: Optional[List[str]] = None,
    config: Optional[AnswerConfig] = None,
) -> Dict[str, Any]:
    """
    Return a grounded refusal for cases where rule evidence exists but filing
    section alignment cannot be safely verified.
    """
    config = config or DEFAULT_CONFIG
    form_norm = normalize_form(form_folder)
    form_label = "10-K" if form_norm == "10-k" else "10-Q"
    company_label = str(company).upper()

    if _is_executive_compensation_topic(query) and form_norm == "10-k":
        answer = (
            f"Executive compensation is governed by Item 402 of Regulation S-K. "
            f"For a {form_label} context, that disclosure is often handled in Part III "
            f"and may be incorporated by reference from the company's definitive proxy statement. "
            f"I could not verify a matching compensation section in {company_label}'s {form_label} {year} filing."
        )
        guard_reason = "Rule evidence was found, but filing-section alignment for executive compensation was not verified."
    elif _is_environmental_or_climate_topic(query):
        answer = (
            f"I found potentially related regulatory or filing evidence, but I could not verify a filing section "
            f"in {company_label}'s {form_label} {year} filing that clearly supports this environmental/climate topic."
        )
        guard_reason = "Topic-specific filing evidence was not sufficiently verified."
    else:
        answer = "I found some related evidence, but I could not verify a filing section that safely supports this answer."
        guard_reason = "Filing-section alignment could not be verified."

    return {
        "status": "REFUSE",
        "answer": answer,
        "route": route,
        "guard_reason": guard_reason,
        "evidence": {
            "rule_evidence": _compact_rule_evidence(ranked_rules, config.max_rule_evidence),
            "filing_evidence": [],
            "ranked_sections": [],
        },
        "notes": notes or [
            "The system found related rule evidence but refused to name a filing section without grounded support.",
            "This helps prevent false rule-to-filing alignment.",
        ],
    }

def _is_acceptable_fallback_section(
    query: str,
    form_folder: Optional[str],
    section: Optional[str],
) -> bool:
    """
    Validate whether a semantic fallback section is acceptable for this query.

    This prevents semantically similar but regulatorily wrong sections
    (for example ITEM 4 controls/procedures) from being treated as valid
    risk-factor evidence.
    """
    ql = str(query or "").lower()
    form_norm = normalize_form(form_folder)
    sec = str(section or "").upper().strip()

    if "risk factor" in ql or "risk factors" in ql:
        if form_norm in {"10-k", "10-q"}:
            return sec == "ITEM 1A"
        return False

    if "management discussion" in ql or "md&a" in ql or "discussion and analysis" in ql:
        if form_norm == "10-k":
            return sec == "ITEM 7"
        if form_norm == "10-q":
            return sec == "ITEM 2"
        return False

    if "legal proceeding" in ql or "legal proceedings" in ql or "litigation" in ql:
        if form_norm == "10-k":
            return sec == "ITEM 3"
        if form_norm == "10-q":
            return sec == "ITEM 1"
        return False

    # For other query types, allow semantic fallback sections.
    return True


# ---------------------------------------------------------
# Evidence compaction helpers
# ---------------------------------------------------------
def _compact_rule_evidence(hits: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits[:limit]:
        c = h.get("chunk", {})
        out.append(
            {
                "citation": c.get("citation"),
                "heading": c.get("heading"),
                "chunk_id": c.get("chunk_id"),
                "text": _extract_text_from_chunk(c),
                "score": round(float(h.get("score", 0.0)), 4),
                "final_score": round(float(h.get("final_score", h.get("score", 0.0))), 4),
            }
        )
    return out


def _compact_filing_evidence(hits: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits[:limit]:
        c = h.get("chunk", {})
        out.append(
            {
                "chunk_id": c.get("chunk_id"),
                "company": c.get("company"),
                "form": c.get("form"),
                "year": c.get("year"),
                "section": c.get("section"),
                "text": _extract_text_from_chunk(c),
                "score": round(float(h.get("score", 0.0)), 4),
                "final_score": round(float(h.get("final_score", h.get("score", 0.0))), 4),
            }
        )
    return out


def _compact_ranked_sections(hits: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits[:limit]:
        out.append(
            {
                "section": h.get("section"),
                "company": h.get("company"),
                "form": h.get("form"),
                "year": h.get("year"),
                "chunk_count": h.get("chunk_count"),
                "score": round(float(h.get("score", 0.0)), 4),
                "adjusted_score": round(float(h.get("adjusted_score", h.get("score", 0.0))), 4),
            }
        )
    return out


# ---------------------------------------------------------
# Guard
# ---------------------------------------------------------
def _hallucination_guard(
    route: Dict[str, Any],
    ranked_rules: List[Dict[str, Any]],
    ranked_filings: List[Dict[str, Any]],
    config: AnswerConfig,
    ranked_sections: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    intent = str(route.get("intent", ""))

    if intent == "rule_only":
        if len(ranked_rules) < config.min_rule_hits:
            return {"allow_answer": False, "reason": "Insufficient rule evidence."}
        return {"allow_answer": True, "reason": "Sufficient rule evidence."}

    if intent == "structural_section_lookup":
        if len(ranked_filings) < config.min_filing_hits:
            return {
                "allow_answer": False,
                "reason": "No deterministic filing evidence found for the target section.",
            }
        return {"allow_answer": True, "reason": "Deterministic filing evidence found."}

    if intent == "semantic_topic_lookup":
        if not ranked_sections:
            return {
                "allow_answer": False,
                "reason": "No reliable filing section was identified for this topic.",
            }
        if len(ranked_filings) < config.min_filing_hits:
            return {
                "allow_answer": False,
                "reason": "Insufficient filing evidence for semantic topic lookup.",
            }
        return {"allow_answer": True, "reason": "Sufficient semantic filing evidence found."}

    if intent == "not_applicable":
        return {
            "allow_answer": False,
            "reason": str(route.get("reason", "Not applicable for this filing form.")),
        }

    return {"allow_answer": False, "reason": "Unsupported routing intent."}


# ---------------------------------------------------------
# Answer phrasing helpers
# ---------------------------------------------------------
def _answer_for_rule_only(query: str, ranked_rules: List[Dict[str, Any]]) -> str:
    top = ranked_rules[0]["chunk"]
    citation = str(top.get("citation", "the requested rule")).strip()
    heading = _clean_heading(top.get("heading"))
    ql = str(query or "").lower()

    if "229.105" in citation or "item 105" in ql:
        return (
            f"{citation} requires companies to disclose material risk factors "
            f"so investors can understand the main risks facing the business."
        )

    if "229.303" in citation or "item 303" in ql:
        return (
            f"{citation} requires management to discuss the company's financial condition, "
            f"results of operations, and important trends or uncertainties."
        )

    if heading:
        return f"{citation} addresses {heading}."

    return f"{citation} is the most relevant rule for this question."


def _answer_for_structural_lookup(
    query: str,
    route: Dict[str, Any],
    company: str,
    form_folder: str,
    year: str,
    ranked_filings: List[Dict[str, Any]],
) -> str:
    section = str(route.get("target_section", "")).upper().strip()
    section_label = _friendly_section_label(section)

    form_norm = normalize_form(form_folder)
    form_label = "10-K" if form_norm == "10-k" else "10-Q"
    company_label = company.upper()
    ql = str(query or "").lower()

    if "risk factor" in ql:
        return (
            f"Risk factors are disclosed in {section_label} of {company_label}'s "
            f"{form_label} {year} filing."
        )

    if "management discussion" in ql or "md&a" in ql or "discussion and analysis" in ql:
        return (
            f"Management's Discussion and Analysis appears in {section_label} of "
            f"{company_label}'s {form_label} {year} filing."
        )

    if "legal proceeding" in ql or "litigation" in ql:
        return (
            f"Legal proceedings are disclosed in {section_label} of "
            f"{company_label}'s {form_label} {year} filing."
        )

    return (
        f"The relevant information appears in {section_label} of "
        f"{company_label}'s {form_label} {year} filing."
    )


def _answer_for_semantic_topic(
    query: str,
    company: str,
    form_folder: str,
    year: str,
    ranked_sections: List[Dict[str, Any]],
    ranked_filings: List[Dict[str, Any]],
) -> str:
    top_section = ranked_sections[0]["section"] if ranked_sections else None
    section_label = _friendly_section_label(top_section)

    form_norm = normalize_form(form_folder)
    form_label = "10-K" if form_norm == "10-k" else "10-Q"
    company_label = company.upper()
    ql = str(query or "").lower()

    if not top_section:
        return (
            f"I could not identify a reliable section for this topic in "
            f"{company_label}'s {form_label} {year} filing."
        )

    if "cyber" in ql or "cybersecurity" in ql or "information security" in ql:
        return (
            f"Cybersecurity-related disclosure appears most clearly in "
            f"{section_label} of {company_label}'s {form_label} {year} filing."
        )

    if "liquidity" in ql or "capital resources" in ql or "cash flow" in ql or "cash flows" in ql:
        return (
            f"Liquidity-related disclosure appears most clearly in "
            f"{section_label} of {company_label}'s {form_label} {year} filing."
        )

    if "supply chain" in ql or "supplier" in ql or "vendor" in ql or "third party" in ql or "third-party" in ql:
        return (
            f"Supply-chain-related disclosure appears most clearly in "
            f"{section_label} of {company_label}'s {form_label} {year} filing."
        )

    if "risk factor" in ql:
        return (
            f"Risk-factor-related disclosure appears most clearly in "
            f"{section_label} of {company_label}'s {form_label} {year} filing."
        )

    return (
        f"This topic appears most clearly in {section_label} of "
        f"{company_label}'s {form_label} {year} filing."
    )


# ---------------------------------------------------------
# Main answer workflow
# ---------------------------------------------------------
def answer_query(
    query: str,
    company: Optional[str] = None,
    form_folder: Optional[str] = None,
    year: Optional[str] = None,
    config: Optional[AnswerConfig] = None,
) -> Dict[str, Any]:
    if config is None:
        config = DEFAULT_CONFIG

    q = str(query or "").strip()
    if not q:
        raise ValueError("query is required")

    filing_scope_provided = bool(company and form_folder and year)

    # ---------------------------------------------------------
    # Path A: Rule-only query
    # ---------------------------------------------------------
    if is_rule_only_query(q):
        explicit_cfr = _extract_explicit_cfr_citation(q)

        raw_rule_hits = retrieve_rules_routed(q, top_k=10)
        ranked = build_ranked_evidence_pack(
            filing_hits=[],
            rule_hits=raw_rule_hits,
            expected_section=None,
            expected_citation=explicit_cfr,
            filing_min_score=config.filing_min_score,
            rule_min_score=config.rule_min_score,
        )
        ranked_rules = ranked["rule_evidence"]

        route = {
            "intent": "rule_only",
            "target_section": None,
            "allow_semantic_fallback": False,
            "reason": "Rule-only query pattern detected.",
        }

        _safe_log(
            query=q,
            company=company,
            form_folder=form_folder,
            year=year,
            route=route,
            stage="after_rule_retrieval",
            structural_hits=[],
            semantic_hits=ranked_rules,
            used_semantic_fallback=False,
            guard_reason=None,
            status=None,
            notes=[],
        )

        if explicit_cfr:
            exact = [
                h for h in ranked_rules
                if str(h.get("chunk", {}).get("citation", "")).strip().lower() == explicit_cfr.lower()
            ]
            if exact:
                ranked_rules = exact + [h for h in ranked_rules if h not in exact]
            else:
                _safe_log(
                    query=q,
                    company=company,
                    form_folder=form_folder,
                    year=year,
                    route=route,
                    stage="final",
                    structural_hits=[],
                    semantic_hits=ranked_rules,
                    used_semantic_fallback=False,
                    guard_reason="Exact citation match not found.",
                    status="REFUSE",
                    notes=["Explicit CFR citation requested but exact match was not found."],
                )
                return {
                    "status": "REFUSE",
                    "answer": f"I could not verify {explicit_cfr} in the current rules index.",
                    "route": route,
                    "guard_reason": "Exact citation match not found.",
                    "evidence": {
                        "rule_evidence": _compact_rule_evidence(ranked_rules, config.max_rule_evidence),
                        "filing_evidence": [],
                        "ranked_sections": [],
                    },
                    "notes": [
                        "The system refuses explicit rule citations when an exact citation match is not verified.",
                        "This helps prevent answering with a nearby but incorrect rule.",
                    ],
                }

        guard = _hallucination_guard(route, ranked_rules, [], config)
        if not guard["allow_answer"]:
            _safe_log(
                query=q,
                company=company,
                form_folder=form_folder,
                year=year,
                route=route,
                stage="final",
                structural_hits=[],
                semantic_hits=ranked_rules,
                used_semantic_fallback=False,
                guard_reason=guard["reason"],
                status="REFUSE",
                notes=["Rule-only path refused due to insufficient evidence."],
            )
            return {
                "status": "REFUSE",
                "answer": "I do not have enough rule evidence to answer this safely.",
                "route": route,
                "guard_reason": guard["reason"],
                "evidence": {
                    "rule_evidence": _compact_rule_evidence(ranked_rules, config.max_rule_evidence),
                    "filing_evidence": [],
                    "ranked_sections": [],
                },
                "notes": [
                    "This response is evidence-gated.",
                    "No answer is returned when rule evidence is insufficient.",
                ],
            }

        _safe_log(
            query=q,
            company=company,
            form_folder=form_folder,
            year=year,
            route=route,
            stage="final",
            structural_hits=[],
            semantic_hits=ranked_rules,
            used_semantic_fallback=False,
            guard_reason=guard["reason"],
            status="PASS",
            notes=["Rule-only path answered successfully."],
        )

        return {
            "status": "PASS",
            "answer": _answer_for_rule_only(q, ranked_rules),
            "route": route,
            "guard_reason": guard["reason"],
            "evidence": {
                "rule_evidence": _compact_rule_evidence(ranked_rules, config.max_rule_evidence),
                "filing_evidence": [],
                "ranked_sections": [],
            },
            "notes": [
                "Answer generated from rule retrieval only.",
                "This is not a legal opinion.",
            ],
        }

    # ---------------------------------------------------------
    # Filing-scoped questions require scope
    # ---------------------------------------------------------
    if not filing_scope_provided:
        route = {
            "intent": "missing_scope",
            "target_section": None,
            "allow_semantic_fallback": False,
            "reason": "Filing-scoped questions need company/form/year.",
        }

        _safe_log(
            query=q,
            company=company,
            form_folder=form_folder,
            year=year,
            route=route,
            stage="final",
            structural_hits=[],
            semantic_hits=[],
            used_semantic_fallback=False,
            guard_reason="Missing filing scope.",
            status="REFUSE",
            notes=["Filing-scoped query refused because scope was incomplete."],
        )

        return {
            "status": "REFUSE",
            "answer": "Please provide the company, filing type, and year so I can look up the correct filing.",
            "route": route,
            "guard_reason": "Missing filing scope.",
            "evidence": {
                "rule_evidence": [],
                "filing_evidence": [],
                "ranked_sections": [],
            },
            "notes": [
                "Provide company, form_folder (10-k or 10-q), and year.",
            ],
        }

    # ---------------------------------------------------------
    # Filing query routing
    # ---------------------------------------------------------
    route = route_filing_query(q, form_folder)

    # ---------------------------------------------------------
    # Path B: Not applicable / refusal
    # ---------------------------------------------------------
    if route["intent"] == "not_applicable":
        reason = str(route.get("reason", "This query is not applicable for the selected filing form."))

        _safe_log(
            query=q,
            company=company,
            form_folder=form_folder,
            year=year,
            route=route,
            stage="final",
            structural_hits=[],
            semantic_hits=[],
            used_semantic_fallback=False,
            guard_reason=reason,
            status="REFUSE",
            notes=["Refusal based on filing structure, not retrieval failure."],
        )

        return {
            "status": "REFUSE",
            "answer": f"This question is not applicable in the way requested. {reason}",
            "route": route,
            "guard_reason": reason,
            "evidence": {
                "rule_evidence": [],
                "filing_evidence": [],
                "ranked_sections": [],
            },
            "notes": [
                "This refusal is based on filing structure, not a retrieval failure.",
            ],
        }

    # ---------------------------------------------------------
    # Path C: Structural section lookup
    # ---------------------------------------------------------
    if route["intent"] == "structural_section_lookup":
        target_section = str(route["target_section"]).upper().strip()

        structural_hits = load_scoped_section_chunk_evidence(
            company=company,
            form_folder=form_folder,
            year=str(year),
            section=target_section,
            max_hits=config.max_filing_evidence,
        )

        _safe_log(
            query=q,
            company=company,
            form_folder=form_folder,
            year=year,
            route=route,
            stage="after_structural_lookup",
            structural_hits=structural_hits,
            semantic_hits=[],
            used_semantic_fallback=False,
            guard_reason=None,
            status=None,
            notes=[],
        )

        ranked = build_ranked_evidence_pack(
            filing_hits=structural_hits,
            rule_hits=[],
            expected_section=target_section,
            expected_citation=None,
            filing_min_score=config.filing_min_score,
            rule_min_score=config.rule_min_score,
        )
        ranked_filings = ranked["filing_evidence"]

        used_semantic_fallback = False
        ranked_sections: List[Dict[str, Any]] = []
        semantic_hits: List[Dict[str, Any]] = []

        if not ranked_filings and _should_try_semantic_fallback(route, form_folder):
            ranked_sections = retrieve_filing_sections(
                query=q,
                company=company,
                form=form_folder,
                year=str(year),
                top_k=config.semantic_top_k_sections,
            )
            used_semantic_fallback = True

            acceptable_sections = [
                s for s in ranked_sections
                if _is_acceptable_fallback_section(q, form_folder, s.get("section"))
            ]

            if acceptable_sections:
                top_section = acceptable_sections[0]["section"]

                semantic_hits = load_scoped_section_chunk_evidence(
                    company=company,
                    form_folder=form_folder,
                    year=str(year),
                    section=top_section,
                    max_hits=config.max_filing_evidence,
                )

                ranked_fallback = build_ranked_evidence_pack(
                    filing_hits=semantic_hits,
                    rule_hits=[],
                    expected_section=top_section,
                    expected_citation=None,
                    filing_min_score=config.filing_min_score,
                    rule_min_score=config.rule_min_score,
                )
                ranked_filings = ranked_fallback["filing_evidence"]
                ranked_sections = acceptable_sections
            else:
                semantic_hits = []
                ranked_filings = []
                ranked_sections = []

            _safe_log(
                query=q,
                company=company,
                form_folder=form_folder,
                year=year,
                route=route,
                stage="after_semantic_fallback",
                structural_hits=structural_hits,
                semantic_hits=semantic_hits,
                used_semantic_fallback=True,
                guard_reason=None,
                status=None,
                notes=["Semantic fallback attempted after structural miss."],
            )

        guard = _hallucination_guard(route, [], ranked_filings, config)

        if not guard["allow_answer"]:
            final_notes = ["Structural query could not be verified in the scoped filing."]
            if used_semantic_fallback:
                final_notes.append("Semantic fallback was attempted but did not produce acceptable evidence.")

            _safe_log(
                query=q,
                company=company,
                form_folder=form_folder,
                year=year,
                route=route,
                stage="final",
                structural_hits=structural_hits,
                semantic_hits=semantic_hits,
                used_semantic_fallback=used_semantic_fallback,
                guard_reason=guard["reason"],
                status="REFUSE",
                notes=final_notes,
            )

            return {
                "status": "REFUSE",
                "answer": "I could not verify the expected filing section in the selected filing.",
                "route": route,
                "guard_reason": guard["reason"],
                "evidence": {
                    "rule_evidence": [],
                    "filing_evidence": _compact_filing_evidence(ranked_filings, config.max_filing_evidence),
                    "ranked_sections": _compact_ranked_sections(ranked_sections, config.semantic_top_k_sections),
                },
                "notes": final_notes,
            }

        if used_semantic_fallback and ranked_sections:
            answer_text = _answer_for_semantic_topic(
                q, company, form_folder, str(year), ranked_sections, ranked_filings
            )
            notes = [
                "Structural routing was attempted first.",
                "Semantic fallback was used because deterministic section evidence was not found.",
                "This is not a legal opinion.",
            ]
            guard_reason = "Semantic fallback evidence found after structural miss."
        else:
            answer_text = _answer_for_structural_lookup(
                q, route, company, form_folder, str(year), ranked_filings
            )
            notes = [
                "Answer generated from structural routing and deterministic filing evidence.",
                "This is not a legal opinion.",
            ]
            guard_reason = guard["reason"]

        _safe_log(
            query=q,
            company=company,
            form_folder=form_folder,
            year=year,
            route=route,
            stage="final",
            structural_hits=structural_hits,
            semantic_hits=semantic_hits,
            used_semantic_fallback=used_semantic_fallback,
            guard_reason=guard_reason,
            status="PASS",
            notes=notes,
        )

        return {
            "status": "PASS",
            "answer": answer_text,
            "route": route,
            "guard_reason": guard_reason,
            "evidence": {
                "rule_evidence": [],
                "filing_evidence": _compact_filing_evidence(ranked_filings, config.max_filing_evidence),
                "ranked_sections": _compact_ranked_sections(ranked_sections, config.semantic_top_k_sections),
            },
            "notes": notes,
        }

    # ---------------------------------------------------------
    # Path D: Semantic topic lookup
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Path D: Semantic topic lookup
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Path D: Semantic topic lookup
    # ---------------------------------------------------------
    ranked_sections = retrieve_filing_sections(
        query=q,
        company=company,
        form=form_folder,
        year=str(year),
        top_k=config.semantic_top_k_sections,
    )

    # Topic-aware section filtering before loading filing evidence
    filtered_ranked_sections = _filter_semantic_sections_for_topic(
        query=q,
        form_folder=form_folder,
        ranked_sections=ranked_sections,
    )

    filing_hits: List[Dict[str, Any]] = []
    if filtered_ranked_sections:
        top_section = filtered_ranked_sections[0]["section"]
        filing_hits = load_scoped_section_chunk_evidence(
            company=company,
            form_folder=form_folder,
            year=str(year),
            section=top_section,
            max_hits=config.max_filing_evidence,
        )

    raw_rule_hits = retrieve_rules(q, top_k=5)

    ranked = build_ranked_evidence_pack(
        filing_hits=filing_hits,
        rule_hits=raw_rule_hits,
        expected_section=filtered_ranked_sections[0]["section"] if filtered_ranked_sections else None,
        expected_citation=None,
        filing_min_score=config.filing_min_score,
        rule_min_score=config.rule_min_score,
    )

    ranked_rules = ranked["rule_evidence"]
    ranked_filings = ranked["filing_evidence"]

    _safe_log(
        query=q,
        company=company,
        form_folder=form_folder,
        year=year,
        route=route,
        stage="after_semantic_lookup",
        structural_hits=ranked_filings,
        semantic_hits=filtered_ranked_sections,
        used_semantic_fallback=False,
        guard_reason=None,
        status=None,
        notes=[],
    )

    # ---------------------------------------------------------
    # Topic-specific filing-text verification
    # ---------------------------------------------------------
    # For sensitive topics like executive compensation or ESG/climate,
    # rule evidence alone is not enough to name a filing section.
    # We require filing chunk text to support the topic.
    topic_requires_strict_alignment = (
        _is_executive_compensation_topic(q)
        or _is_environmental_or_climate_topic(q)
    )

    if topic_requires_strict_alignment and not _filing_hits_support_topic(q, filing_hits):
        refusal = _build_topic_alignment_refusal(
            query=q,
            company=company,
            form_folder=form_folder,
            year=str(year),
            route=route,
            ranked_rules=ranked_rules,
            notes=[
                "Semantic topic lookup found related rule evidence but did not verify a topic-appropriate filing section.",
                "The system refused rather than naming an unsupported filing location.",
            ],
            config=config,
        )

        _safe_log(
            query=q,
            company=company,
            form_folder=form_folder,
            year=year,
            route=route,
            stage="final",
            structural_hits=ranked_filings,
            semantic_hits=filtered_ranked_sections,
            used_semantic_fallback=False,
            guard_reason=refusal["guard_reason"],
            status="REFUSE",
            notes=refusal["notes"],
        )

        return refusal

    guard = _hallucination_guard(
        route,
        ranked_rules,
        ranked_filings,
        config,
        ranked_sections=filtered_ranked_sections,
    )

    if not guard["allow_answer"]:
        _safe_log(
            query=q,
            company=company,
            form_folder=form_folder,
            year=year,
            route=route,
            stage="final",
            structural_hits=ranked_filings,
            semantic_hits=filtered_ranked_sections,
            used_semantic_fallback=False,
            guard_reason=guard["reason"],
            status="REFUSE",
            notes=[
                "Semantic topic lookup was attempted.",
                "The system refused because evidence was too weak.",
            ],
        )

        return {
            "status": "REFUSE",
            "answer": "I do not have enough grounded filing evidence to answer this topic safely.",
            "route": route,
            "guard_reason": guard["reason"],
            "evidence": {
                "rule_evidence": _compact_rule_evidence(ranked_rules, config.max_rule_evidence),
                "filing_evidence": _compact_filing_evidence(ranked_filings, config.max_filing_evidence),
                "ranked_sections": _compact_ranked_sections(filtered_ranked_sections, config.semantic_top_k_sections),
            },
            "notes": [
                "Semantic topic lookup was attempted.",
                "The system refused because evidence was too weak.",
            ],
        }

    _safe_log(
        query=q,
        company=company,
        form_folder=form_folder,
        year=year,
        route=route,
        stage="final",
        structural_hits=ranked_filings,
        semantic_hits=filtered_ranked_sections,
        used_semantic_fallback=False,
        guard_reason=guard["reason"],
        status="PASS",
        notes=[
            "Answer generated from semantic topic lookup plus evidence ranking.",
            "This is not a legal opinion.",
        ],
    )

    return {
        "status": "PASS",
        "answer": _answer_for_semantic_topic(
            q, company, form_folder, str(year), filtered_ranked_sections, ranked_filings
        ),
        "route": route,
        "guard_reason": guard["reason"],
        "evidence": {
            "rule_evidence": _compact_rule_evidence(ranked_rules, config.max_rule_evidence),
            "filing_evidence": _compact_filing_evidence(ranked_filings, config.max_filing_evidence),
            "ranked_sections": _compact_ranked_sections(filtered_ranked_sections, config.semantic_top_k_sections),
        },
        "notes": [
            "Answer generated from semantic topic lookup plus evidence ranking.",
            "This is not a legal opinion.",
        ],
    }