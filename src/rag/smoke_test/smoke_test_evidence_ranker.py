"""
Module: smoke_test_evidence_ranker.py

Purpose
-------
Quick manual test for evidence ranking.
"""

from src.rag.evidence_ranker import build_ranked_evidence_pack
from src.rag.retrieve_rules import retrieve_rules
from src.rag.filing_evidence import load_scoped_section_chunk_evidence

rule_hits = retrieve_rules("17 CFR 229.105", top_k=5)

filing_hits = load_scoped_section_chunk_evidence(
    company="VISA",
    form_folder="10-k",
    year="2024",
    section="ITEM 1A",
    max_hits=5,
)

ranked = build_ranked_evidence_pack(
    filing_hits=filing_hits,
    rule_hits=rule_hits,
    expected_section="ITEM 1A",
    expected_citation="17 CFR 229.105",
)

print("\nRULE EVIDENCE")
for x in ranked["rule_evidence"][:5]:
    print(
        x["chunk"].get("citation"),
        "| base:", round(x.get("score", 0.0), 4),
        "| final:", round(x.get("final_score", 0.0), 4),
    )

print("\nFILING EVIDENCE")
for x in ranked["filing_evidence"][:5]:
    print(
        x["chunk"].get("chunk_id"),
        "| sec:", x["chunk"].get("section"),
        "| base:", round(x.get("score", 0.0), 4),
        "| final:", round(x.get("final_score", 0.0), 4),
    )