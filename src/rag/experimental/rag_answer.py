import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from dotenv import load_dotenv

from config import RAGConfig
from llm_prompts import SYSTEM_PROMPT, USER_TEMPLATE

load_dotenv()

@dataclass
class Hit:
    score: float
    rec: Dict[str, Any]

def load_chunks(cfg: RAGConfig) -> List[dict]:
    chunks = []
    with open(cfg.chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def embed_query(cfg: RAGConfig, q: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model=cfg.embedding_model, input=[q])
    v = np.asarray([resp.data[0].embedding], dtype=np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

def conservative_section_hint(question: str) -> Optional[str]:
    q = question.lower()
    # Only hint when it’s very obvious; otherwise don’t filter.
    if "risk" in q:
        return "ITEM 1A"
    if "md&a" in q or "management’s discussion" in q or "management's discussion" in q:
        return None  # could be Item 7 (10-K) or Item 2 (10-Q)
    return None

def format_evidence(hits: List[Hit]) -> str:
    parts = []
    for h in hits:
        c = h.rec
        header = f"[{c['chunk_id']}] {c['company']} {c['form_folder']} {c.get('year','')} {c['section']} — {c.get('title','')}"
        parts.append(header + "\n" + c["text"].strip())
    return "\n\n---\n\n".join(parts)

def call_llm(cfg: RAGConfig, question: str, evidence: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    msg = USER_TEMPLATE.format(question=question, evidence=evidence)
    resp = client.chat.completions.create(
        model=cfg.answer_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content

def answer_question(question: str,
                    company: Optional[str] = None,
                    form_folder: Optional[str] = None,  # "10-k" or "10-q"
                    year: Optional[str] = None) -> str:
    cfg = RAGConfig()
    chunks = load_chunks(cfg)
    index = faiss.read_index(cfg.faiss_index_path)

    qv = embed_query(cfg, question)
    scores, ids = index.search(qv, cfg.oversample_k)

    section_hint = conservative_section_hint(question)

    hits: List[Hit] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0:
            continue
        rec = chunks[idx]

        if company and rec["company"].upper() != company.upper():
            continue
        if form_folder and rec["form_folder"].lower() != form_folder.lower():
            continue
        if year and str(rec.get("year","")) != str(year):
            continue
        if section_hint and rec["section"].upper() != section_hint:
            continue

        hits.append(Hit(score=float(score), rec=rec))
        if len(hits) >= cfg.top_k:
            break

    if not hits or max(h.score for h in hits) < cfg.min_score:
        return (
            "Insufficient evidence in the indexed SEC filings to answer safely.\n"
            "Try specifying company (ticker), filing type (10-k/10-q), and year."
        )

    evidence = format_evidence(hits)
    return call_llm(cfg, question, evidence)

if __name__ == "__main__":
    q = input("Ask: ").strip()
    print(answer_question(q))