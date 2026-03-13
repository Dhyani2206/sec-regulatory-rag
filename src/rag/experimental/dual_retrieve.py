"""
Module: dual_retrieve.py

Purpose
-------
Combines rule retrieval and filing retrieval to produce a unified
evidence set for answering regulatory questions.

Responsibilities
----------------
- Execute rules retrieval
- Execute filings retrieval
- Combine evidence sets
- Provide guardrails to ensure sufficient evidence exists

Inputs
------
User query

Outputs
-------
Dictionary containing rule evidence and filing evidence.

Pipeline Position
-----------------
Core retrieval orchestration layer of the RAG system.

Notes
-----
Dual retrieval ensures that answers are grounded in both regulatory
rules and actual company filings, reducing hallucination risk.
"""
from src.rag.retrieve_filings import retrieve_filings
from src.rag.rules_router import retrieve_rules_routed

def dual_retrieve(query: str, k_filings: int = 6, k_rules: int = 6, min_hits: int = 2):
    filings = retrieve_filings(query, top_k=k_filings)
    rules   = retrieve_rules_routed(query, top_k=k_rules)

    return {
        "query": query,
        "filings_hits": filings,
        "rules_hits": rules,
        "total_hits": len(filings) + len(rules),
        "min_hits_required": min_hits,
    }