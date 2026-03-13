import json
import re
from pathlib import Path
from typing import List, Dict

from src.rag.config import RAGConfig

CFG = RAGConfig()

def clean_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(text: str, chunk_chars: int = 1600, overlap: int = 200) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main(year: str = "2025"):
    inputs = [
        CFG.regulations_processed_dir / f"regsk_229_{year}.jsonl",
        CFG.regulations_processed_dir / f"regsx_210_{year}.jsonl",
    ]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"Missing rules file: {p}. Run rules ingestion first.")

    CFG.rules_chunks_path.parent.mkdir(parents=True, exist_ok=True)

    out = CFG.rules_chunks_path
    wrote = 0
    per_part = {"229": 0, "210": 0}

    with out.open("w", encoding="utf-8") as w:
        for p in inputs:
            for doc in iter_jsonl(p):
                part = str(doc.get("part", "")).strip()
                section = str(doc.get("section", "")).strip()
                citation = str(doc.get("citation", "")).strip()
                heading = str(doc.get("heading", "")).strip()
                text = str(doc.get("text", "")).strip()
                source = str(doc.get("source", "govinfo")).strip()

                # Guardrail: skip junk
                if not citation or not text or len(text) < 300:
                    continue

                chunks = chunk_text(text, chunk_chars=1600, overlap=200)

                for idx, ch in enumerate(chunks, start=1):
                    rec = {
                        "chunk_id": f"RULE|{citation}|{year}|chunk{idx}",
                        "corpus": "rules",
                        "year": year,
                        "part": part,
                        "section": section,
                        "citation": citation,
                        "heading": heading[:300],
                        "text": ch,
                        "source": source,
                    }
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wrote += 1
                    if part in per_part:
                        per_part[part] += 1

    # Checkpoint manifest
    manifest = {
        "year": year,
        "inputs": [str(x) for x in inputs],
        "output": str(out),
        "rule_chunks_written": wrote,
        "per_part_chunks": per_part,
    }
    mf = CFG.store_dir / f"rules_chunks_manifest_{year}.json"
    mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {wrote} rules chunks -> {out}")
    print(f"Saved manifest -> {mf}")
    print("Per-part:", per_part)

if __name__ == "__main__":
    main("2025")