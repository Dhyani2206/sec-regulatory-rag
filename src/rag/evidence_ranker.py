"""
Module: evidence_ranker.py

Purpose
-------
Ranks retrieved rule and filing evidence before answer generation
or compliance reporting.

Responsibilities
----------------
- Score filing evidence using retrieval score and section relevance
- Score rule evidence using retrieval score and citation strength
- Sort evidence from strongest to weakest
- Filter weak evidence using configurable thresholds

Inputs
------
- Filing evidence hits
- Rule evidence hits
- Optional context such as expected section or expected citation

Outputs
-------
Ranked evidence lists suitable for downstream answer generation
or compliance workflows.

Pipeline Position
-----------------
Post-retrieval / pre-answer layer.

Notes
-----
This module is intentionally lightweight and rule-based.
Its goal is to improve evidence precision and reduce hallucination risk,
not to replace retrieval.
"""

from typing import Dict, List, Optional


def filing_section_bonus(
    section: Optional[str],
    expected_section: Optional[str],
) -> float:
    """
    Bonus for matching the expected filing section.
    """
    sec = str(section or "").upper().strip()
    exp = str(expected_section or "").upper().strip()

    if not sec or not exp:
        return 0.0

    if sec == exp:
        return 0.20

    return 0.0


def rule_citation_bonus(
    citation: Optional[str],
    expected_citation: Optional[str],
) -> float:
    """
    Bonus for exact rule citation match.
    """
    cit = str(citation or "").strip()
    exp = str(expected_citation or "").strip()

    if not cit or not exp:
        return 0.0

    if cit == exp:
        return 0.25

    return 0.0


def score_filing_hit(
    hit: Dict,
    expected_section: Optional[str] = None,
) -> float:
    """
    Compute final filing evidence score.
    """
    base = float(hit.get("adjusted_score", hit.get("score", 0.0)))
    chunk = hit.get("chunk", {})

    sec_bonus = filing_section_bonus(
        section=chunk.get("section"),
        expected_section=expected_section,
    )

    return base + sec_bonus


def score_rule_hit(
    hit: Dict,
    expected_citation: Optional[str] = None,
) -> float:
    """
    Compute final rule evidence score.
    """
    base = float(hit.get("score", 0.0))
    chunk = hit.get("chunk", {})

    cit_bonus = rule_citation_bonus(
        citation=chunk.get("citation"),
        expected_citation=expected_citation,
    )

    return base + cit_bonus


def rank_filing_evidence(
    hits: List[Dict],
    expected_section: Optional[str] = None,
    min_score: float = 0.30,
) -> List[Dict]:
    """
    Rank and filter filing evidence.
    """
    ranked = []

    for h in hits:
        final_score = score_filing_hit(h, expected_section=expected_section)
        if final_score < min_score:
            continue

        item = dict(h)
        item["final_score"] = final_score
        ranked.append(item)

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked


def rank_rule_evidence(
    hits: List[Dict],
    expected_citation: Optional[str] = None,
    min_score: float = 0.30,
) -> List[Dict]:
    """
    Rank and filter rule evidence.
    """
    ranked = []

    for h in hits:
        final_score = score_rule_hit(h, expected_citation=expected_citation)
        if final_score < min_score:
            continue

        item = dict(h)
        item["final_score"] = final_score
        ranked.append(item)

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked


def build_ranked_evidence_pack(
    filing_hits: List[Dict],
    rule_hits: List[Dict],
    expected_section: Optional[str] = None,
    expected_citation: Optional[str] = None,
    filing_min_score: float = 0.30,
    rule_min_score: float = 0.30,
) -> Dict[str, List[Dict]]:
    """
    Rank and filter filing + rule evidence together.
    """
    ranked_filings = rank_filing_evidence(
        filing_hits,
        expected_section=expected_section,
        min_score=filing_min_score,
    )

    ranked_rules = rank_rule_evidence(
        rule_hits,
        expected_citation=expected_citation,
        min_score=rule_min_score,
    )

    return {
        "filing_evidence": ranked_filings,
        "rule_evidence": ranked_rules,
    }