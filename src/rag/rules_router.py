"""
Module: rules_router.py

Purpose
-------
Routes rule-related queries to improve retrieval when explicit
regulatory citations are present.

Responsibilities
----------------
- Detect explicit CFR citations in user queries
- Prioritize exact rule matches when possible
- Fall back to semantic retrieval when no citation is detected

Inputs
------
User query string

Outputs
-------
List of rule evidence chunks with scores.

Pipeline Position
-----------------
Sits between query parsing and rules retrieval.

Notes
-----
This module improves reliability for rule questions by prioritizing
deterministic citation matches before relying purely on semantic search.
"""
import re
from typing import List, Dict, Optional
from src.rag.retrieve_rules import retrieve_rules

# Citation Detection
CIT_RE = re.compile(
    r"\b17\s*CFR\s+(\d+)\.(\d+[a-zA-Z0-9\-]*)\b",
    re.IGNORECASE,
)


def extract_citation(query: str) -> Optional[str]:
    """
    Extract a CFR citation from the query if present.
    Example
    "What does 17 CFR 229.105 require?"
    "17 CFR 229.105"
    """
    m = CIT_RE.search(query)
    if not m:
        return None

    return f"17 CFR {m.group(1)}.{m.group(2)}"


# Routed rules retireval
def retrieve_rules_routed(query: str, top_k: int = 8) -> List[Dict[str, object]]:
    """
    Retrieve rules with citation-aware prioritization.

    Strategy
    1. Detect explicit CFR citation in query.
    2. Run standard semantic retrieval.
    3. Prioritize exact citation matches if present.
    4. If no exact match appears in top results, expand search slightly.
    """
    cit = extract_citation(query)

    hits = retrieve_rules(query, top_k=top_k)

    if not cit:
        return hits

# Exact citation match
    exact = [
        h for h in hits
        if h["chunk"].get("citation", "").lower() == cit.lower()
    ]

    if exact:
        rest = [h for h in hits if h not in exact]
        return (exact + rest)[:top_k]

# Fallback expanded search
    hits2 = retrieve_rules(
        cit + " requirements",
        top_k=max(top_k, 16),
    )

    exact2 = [
        h for h in hits2
        if h["chunk"].get("citation", "").lower() == cit.lower()
    ]
    return exact2[:top_k] if exact2 else hits2[:top_k]