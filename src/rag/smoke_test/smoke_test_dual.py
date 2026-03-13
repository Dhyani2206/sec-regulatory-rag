from src.rag.dual_retrieval import dual_retrieve

tests = [
    "Where do we disclose cybersecurity risk?",
    "What does 17 CFR 229.105 require?",
    "Management’s discussion and analysis disclosure requirements",
]

for q in tests:
    res = dual_retrieve(q)
    print("\nQUERY:", q)
    print("rules:", len(res["rules_hits"]), "filings:", len(res["filings_hits"]), "total:", res["total_hits"])
    if res["rules_hits"]:
        print(" top rule:", res["rules_hits"][0]["chunk"].get("citation"), res["rules_hits"][0]["score"])
    if res["filings_hits"]:
        c = res["filings_hits"][0]["chunk"]
        print(" top filing:", c.get("company"), c.get("form"), c.get("year"), c.get("section"), res["filings_hits"][0]["score"])