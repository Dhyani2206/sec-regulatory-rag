"""
Module: smoke_test_section_retrieval.py

Purpose
-------
Quick manual smoke test for section-first retrieval.
"""

from src.rag.retrieve_filing_sections import retrieve_filing_sections

tests = [
    {
        "company": "VISA",
        "form": "10-k",
        "year": "2024",
        "query": "Where are risk factors disclosed?",
    },
    {
        "company": "VISA",
        "form": "10-k",
        "year": "2024",
        "query": "Where is Management's Discussion and Analysis disclosed?",
    },
    {
        "company": "AAPL",
        "form": "10-q",
        "year": "2024",
        "query": "Where is Management's Discussion and Analysis disclosed?",
    },
    {
        "company": "AAPL",
        "form": "10-k",
        "year": "2024",
        "query": "Where is cybersecurity discussed?",
    },
]

for t in tests:
    hits = retrieve_filing_sections(
        query=t["query"],
        company=t["company"],
        form=t["form"],
        year=t["year"],
        top_k=5,
    )
    print("\nQUERY:", t["query"])
    print("SCOPE:", t["company"], t["form"], t["year"])
    for h in hits[:5]:
        print(" ", h["section"], "| adjusted:", round(h["adjusted_score"], 4), "| base:", round(h["score"], 4))