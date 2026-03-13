import json
import numpy as np
from pathlib import Path

RULES_CHUNKS = Path("storage/rules_chunks.jsonl")
RULES_VECS   = Path("storage/rules_vectors.npy")
RULES_META   = Path("storage/rules_embeddings_meta.json")

def main():
    assert RULES_CHUNKS.exists(), f"Missing {RULES_CHUNKS}"
    assert RULES_VECS.exists(),   f"Missing {RULES_VECS}"
    assert RULES_META.exists(),   f"Missing {RULES_META}"

    # Count chunks
    n_chunks = 0
    with RULES_CHUNKS.open("r", encoding="utf-8") as f:
        for _ in f:
            n_chunks += 1

    vecs = np.load(RULES_VECS)
    meta = json.loads(RULES_META.read_text(encoding="utf-8"))

    print(f"[qc] chunks.jsonl lines: {n_chunks}")
    print(f"[qc] vectors shape: {vecs.shape} dtype={vecs.dtype}")
    print(f"[qc] meta entries: {len(meta)} (meta is list)")

    if vecs.ndim != 2:
        raise RuntimeError("rules_vectors.npy must be 2D (N, dim).")
    if vecs.shape[0] != n_chunks:
        raise RuntimeError(f"Mismatch: vectors N={vecs.shape[0]} vs chunks lines={n_chunks}")
    if len(meta) != n_chunks:
        raise RuntimeError(f"Mismatch: meta={len(meta)} vs chunks={n_chunks}")

    # Spot-check first/last meta
    print("[qc] meta[0] keys:", sorted(meta[0].keys()))
    print("[qc] meta[-1] keys:", sorted(meta[-1].keys()))
    print("[qc] OK")

if __name__ == "__main__":
    main()