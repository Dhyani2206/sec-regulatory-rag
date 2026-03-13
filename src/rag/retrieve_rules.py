"""
Module: retrieve_rules.py

Purpose
-------
Retrieves relevant regulatory rule chunks from the FAISS rules index.

Responsibilities
----------------
- Load rules index and rule chunk corpus
- Embed user query using the rules embedding model
- Perform vector similarity search
- Return top rule evidence chunks with scores

Inputs
------
User query string
storage/faiss_rules.index
storage/rules_chunks.jsonl

Outputs
-------
List of rule evidence chunks with similarity scores.

Pipeline Position
-----------------
Retrieval layer of the RAG pipeline.

Notes
-----
Used together with filings retrieval to form a dual evidence
retrieval system that reduces hallucination risk.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from src.rag.config import (
    DEFAULT_RULE_TOP_K,
    RULES_CHUNKS_PATH,
    RULES_EMBEDDING_MODEL,
    RULES_INDEX_PATH,
)
from src.rag.embeddings import embed_texts


# ---------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------
_RULES_CHUNKS_CACHE: Optional[List[Dict[str, Any]]] = None
_RULES_INDEX_CACHE = None


# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------
def _load_rules_chunks() -> List[Dict[str, Any]]:
    """
    Load rules chunks once and cache them in memory.
    """
    global _RULES_CHUNKS_CACHE

    if _RULES_CHUNKS_CACHE is None:
        rows: List[Dict[str, Any]] = []
        with RULES_CHUNKS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        _RULES_CHUNKS_CACHE = rows

    return _RULES_CHUNKS_CACHE


def _load_rules_index():
    """
    Load FAISS rules index once and cache it in memory.
    """
    global _RULES_INDEX_CACHE

    if _RULES_INDEX_CACHE is None:
        if not RULES_INDEX_PATH.exists():
            raise FileNotFoundError(f"Missing rules index: {RULES_INDEX_PATH}")
        _RULES_INDEX_CACHE = faiss.read_index(str(RULES_INDEX_PATH))

    return _RULES_INDEX_CACHE


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------
def retrieve_rules(query: str, top_k: int = DEFAULT_RULE_TOP_K) -> List[Dict[str, Any]]:
    """
    Retrieve top rule evidence chunks from the rules FAISS index.
    """
    q = str(query or "").strip()
    if not q:
        raise ValueError("query is required")

    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    rows = _load_rules_chunks()
    index = _load_rules_index()

    qvec = embed_texts([q], RULES_EMBEDDING_MODEL)
    qvec = np.asarray(qvec, dtype=np.float32)

    if qvec.shape[1] != index.d:
        raise ValueError(
            f"Rules query embedding dimension mismatch: got {qvec.shape[1]}, expected {index.d}. "
            f"Check RULES_EMBEDDING_MODEL and how faiss_rules.index was built."
        )
    scores, ids = index.search(qvec, top_k)

    results: List[Dict[str, Any]] = []

    for idx, score in zip(ids[0], scores[0]):
        if idx < 0:
            continue
        if idx >= len(rows):
            continue

        results.append(
            {
                "score": float(score),
                "chunk": rows[idx],
            }
        )

    return results