import json
import re
from pathlib import Path
from typing import Dict, Any, Iterator, List
from rag.config import RAGConfig
cfg = RAGConfig()


YEAR_RE = re.compile(r"(19|20)\d{2}")

def guess_year(name: str) -> str:
    m = YEAR_RE.search(name)
    return m.group(0) if m else ""

def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks

def iter_docs(cfg: RAGConfig) -> Iterator[Dict[str, Any]]:
    root = Path(cfg.processed_dir)
    for fp in root.rglob("*.json"):
        data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
        company = data.get("company") or fp.parts[-3]
        form_folder = fp.parts[-2]  # 10-k or 10-q
        form_type = data.get("form_type") or form_folder.upper().replace("-", "")
        source_file = data.get("source_file", str(fp))
        year = guess_year(fp.name)

        sections = data.get("sections", {})
        if not isinstance(sections, dict):
            continue

        for section, obj in sections.items():
            if not isinstance(obj, dict):
                continue
            title = obj.get("title", "")
            text = obj.get("text", "")
            yield {
                "company": str(company).upper(),
                "form_type": str(form_type).upper(),
                "form_folder": form_folder.lower(),  # keep for filtering
                "year": year,
                "section": str(section).upper(),
                "title": title,
                "text": text,
                "source_file": source_file,
                "doc_id": f"{str(company).upper()}|{str(form_type).upper()}|{year}|{str(section).upper()}|{fp.stem}",
            }

def build_chunks(cfg: RAGConfig) -> int:
    store_dir = Path(cfg.store_dir).resolve()
    store_dir.mkdir(parents=True, exist_ok=True)

    out = Path(cfg.chunks_path).resolve()
    print(f"[build_corpus] Writing chunks to: {out}")

    n_docs = 0
    n_chunks = 0

    with out.open("w", encoding="utf-8", newline="\n") as f:
        for doc in iter_docs(cfg):
            n_docs += 1
            parts = chunk_text(doc["text"], cfg.chunk_chars, cfg.overlap_chars)
            for i, chunk in enumerate(parts):
                rec = {
                    "chunk_id": f"{doc['doc_id']}|chunk{i}",
                    "company": doc["company"],
                    "form_type": doc["form_type"],
                    "form_folder": doc["form_folder"],
                    "year": doc["year"],
                    "section": doc["section"],
                    "title": doc["title"],
                    "source_file": doc["source_file"],
                    "text": chunk,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1

        # FORCE flush to disk
        f.flush()

    print(f"[build_corpus] Docs used: {n_docs}")
    return n_chunks

if __name__ == "__main__":
    cfg = RAGConfig()
    total = build_chunks(cfg)
    print(f"Wrote {total} chunks -> {cfg.chunks_path}")