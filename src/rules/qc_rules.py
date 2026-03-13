import json
import re
from pathlib import Path
from collections import Counter

from src.rag.config import RAGConfig

CFG = RAGConfig()

CIT_RE = re.compile(r"\b17\s*CFR\s+\d+\.\d+|\b17\s*CFR\s+\d+\.\d+-\d+", re.IGNORECASE)

def qc_one(path: Path) -> dict:
    total = 0
    empty_text = 0
    missing_citation = 0
    parts = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            text = (rec.get("text") or "").strip()
            cit = (rec.get("citation") or "").strip()
            part = (rec.get("part") or "").strip()

            if not text:
                empty_text += 1
            if not cit and not CIT_RE.search(rec.get("heading", "") or ""):
                missing_citation += 1
            if part:
                parts[part] += 1

    return {
        "file": str(path),
        "total_sections": total,
        "empty_text": empty_text,
        "missing_citation": missing_citation,
        "parts": dict(parts),
    }

def main():
    rules_files = [
        CFG.regulations_processed_dir / "regsk_229_2025.jsonl",
        CFG.regulations_processed_dir / "regsx_210_2025.jsonl",
    ]
    out_dir = CFG.regulations_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"files": [], "notes": []}
    for fp in rules_files:
        if not fp.exists():
            report["notes"].append(f"Missing: {fp}")
            continue
        report["files"].append(qc_one(fp))

    out_path = out_dir / "qc_rules_2025.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()