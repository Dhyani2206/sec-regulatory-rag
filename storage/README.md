# Storage Artifacts

This folder contains runtime artifacts required by the RAG system.

Examples include:

- FAISS indexes
- embedding vectors
- chunk metadata

These files are intentionally **not stored in Git** because they are large.

To run the system locally, regenerate them using the preprocessing pipelines in `src/`.

Typical artifacts:

- faiss.index
- faiss_filings.index
- faiss_rules.index
- embeddings.npy
- rules_vectors.npy
- chunks.jsonl