"""
Module: evidence_pack.py

Purpose
-------
Builds an auditable evidence package for a regulatory obligation by retrieving
supporting evidence from both SEC rules and scoped company filings.

Responsibilities
----------------
- Construct rule queries from obligation metadata
- Construct filing queries from obligation metadata
- Retrieve rule evidence from the rules index
- Retrieve filing evidence from the filings index using strict scope filters
- Return structured evidence used by gap_report and other workflows

Inputs
------
- Obligation metadata (rule_citation, rule_query, mapped_section, filing_query)
- Filing scope (company, form, year)
- rules index + filings index

Outputs
-------
Dictionary containing:
- rule_query
- filing_query
- rules_hits
- filings_hits

Pipeline Position
-----------------
Retrieval utility layer used by compliance workflows.

Notes
-----
This module performs retrieval only. It does not assign compliance status.
Downstream modules apply evidence thresholds and PASS/WARN/REVIEW logic.
"""

from typing import Dict, List, Optional

from .retrieve_filings import retrieve_filings
from .rules_router import retrieve_rules_routed


def build_rule_query(rule_citation: Optional[str], rule_query: Optional[str]) -> str:
    if rule_citation:
        return f"{rule_citation} disclosure requirements"
    if rule_query:
        return rule_query
    return "SEC disclosure requirement"


def build_filing_query(mapped_section: Optional[str], filing_query: Optional[str], description: str) -> str:
    if filing_query:
        return filing_query
    if mapped_section:
        return f"{mapped_section} {description}"
    return description


def evidence_pack_for_obligation(
    description: str,
    mapped_section: Optional[str] = None,
    rule_citation: Optional[str] = None,
    rule_query: Optional[str] = None,
    filing_query: Optional[str] = None,
    company: Optional[str] = None,
    form: Optional[str] = None,
    year: Optional[str] = None,
    k_rules: int = 4,
    k_filings: int = 6,
) -> Dict[str, List[dict]]:

    rule_q = build_rule_query(rule_citation, rule_query)
    filing_q = build_filing_query(mapped_section, filing_query, description)

    rules_hits = retrieve_rules_routed(rule_q, top_k=k_rules)

    filings_hits = retrieve_filings(
        filing_q,
        top_k=k_filings,
        company=company,
        form=form,
        year=year,
        section=mapped_section,
        candidate_pool=80,
    )

    return {
        "rule_query": rule_q,
        "filing_query": filing_q,
        "rules_hits": rules_hits,
        "filings_hits": filings_hits,
    }