import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import faiss
from dotenv import load_dotenv

from .config import RAGConfig

load_dotenv()


def require_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Put it in .env and restart terminal.")


def load_chunks(cfg: RAGConfig) -> List[dict]:
    p = Path(cfg.chunks_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"chunks file not found: {p}")
    if p.stat().st_size == 0:
        raise RuntimeError(f"chunks file is 0 bytes: {p}")

    chunks = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"[build_index] Chunks: {len(chunks)} from {p} ({p.stat().st_size} bytes)")
    return chunks


def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(np.float32)


def embed_openai_batch(texts: List[str], model: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
    return normalize(vecs)


def main():
    cfg = RAGConfig()
    require_api_key()

    Path(cfg.store_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.store_dir) / "embeddings.npy"
    ckpt_meta = Path(cfg.store_dir) / "embeddings_meta.json"
    index_path = Path(cfg.faiss_index_path).resolve()

    chunks = load_chunks(cfg)
    texts = [c["text"] for c in chunks]
    n = len(texts)

    # ---- Resume support ----
    start_i = 0
    vecs: Optional[np.ndarray] = None

    if ckpt_path.exists() and ckpt_meta.exists():
        meta = json.loads(ckpt_meta.read_text(encoding="utf-8"))
        done = int(meta.get("done", 0))
        model_used = meta.get("model", "")
        if model_used == cfg.embedding_model and 0 < done <= n:
            print(f"[build_index] Resuming from checkpoint: {done}/{n}")
            vecs = np.load(ckpt_path)
            start_i = done
        else:
            print("[build_index] Checkpoint exists but model changed or invalid; starting fresh.")

    if vecs is None:
        # Initialize empty array when we know embedding dim (after first batch)
        vecs = None

    batch_size = 256  # faster + fewer API calls; safe
    print(f"[build_index] Embedding model: {cfg.embedding_model}")
    print(f"[build_index] Batch size: {batch_size}")

    i = start_i
    t0 = time.time()

    while i < n:
        batch = texts[i:i + batch_size]

        # Retry loop with exponential backoff (handles rate limits cleanly)
        attempt = 0
        while True:
            try:
                bvec = embed_openai_batch(batch, cfg.embedding_model)
                break
            except Exception as e:
                attempt += 1
                wait = min(60, 2 ** attempt)
                print(f"[build_index] ERROR on batch starting at {i}: {type(e).__name__}: {e}")
                print(f"[build_index] Retrying in {wait}s (attempt {attempt})...")
                time.sleep(wait)

        if vecs is None:
            dim = bvec.shape[1]
            vecs = np.zeros((n, dim), dtype=np.float32)

        vecs[i:i + len(batch)] = bvec
        i += len(batch)

        # Progress
        elapsed = time.time() - t0
        rate = i / max(elapsed, 1e-6)
        eta = (n - i) / max(rate, 1e-6)
        print(f"[build_index] Embedded {i}/{n} | {rate:.1f} chunks/s | ETA ~{eta/60:.1f} min")

        # Checkpoint every ~1024 chunks
        if i % 1024 == 0 or i == n:
            np.save(ckpt_path, vecs)
            ckpt_meta.write_text(json.dumps({"done": i, "model": cfg.embedding_model}, indent=2), encoding="utf-8")
            print(f"[build_index] Checkpoint saved: {i}/{n}")

    # Build FAISS
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(index_path))
    print(f"[build_index] Saved FAISS index: {index_path} (n={n}, dim={dim})")

    print("[build_index] Done.")


if __name__ == "__main__":
    main()