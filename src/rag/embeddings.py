"""
Module: embeddings.py

Purpose
-------
Provides the shared embedding interface used across the RAG system.

Responsibilities
----------------
- Generate vector embeddings for text using the OpenAI embeddings API
- Normalize vectors for cosine similarity search
- Serve as the single embedding entrypoint for the retrieval pipeline

Inputs
------
List[str] of text inputs
Embedding model name (from config)

Outputs
-------
NumPy array of normalized float32 vectors with shape (N, D)

Pipeline Position
-----------------
Core infrastructure layer used by:

- build/build_index.py
- retrieve_filings.py
- retrieve_rules.py
- any semantic evaluation utilities

Design Notes
------------
This module intentionally centralizes all embedding generation so that:

1. The same embedding logic is used for both indexing and retrieval.
2. Vector normalization is consistent across the system.
3. Changes to the embedding model require modification in only one place.

Vectors are L2-normalized so FAISS cosine similarity behaves correctly.

Security
--------
Requires OPENAI_API_KEY to be set in environment variables (.env file).
"""

from __future__ import annotations
import os
from typing import List
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Enviornment Validation
def require_api_key() -> None:
    """
    Ensure the OpenAI API key is present before making requests.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to your .env file and restart your shell."
        )

# Vector Normalization
def normalize(vecs: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors for cosine similarity search.
    FAISS cosine search assumes vectors are normalized.
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(np.float32)

# Embedding Function
def embed_texts(texts: List[str], model: str) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    Parameters
    texts : List[str]
        Text inputs to embed.

    model : str
        OpenAI embedding model name.

    Returns
    np.ndarray
        Normalized float32 embedding matrix of shape (N, D).

    Notes
    This function is used for both:
    - Building FAISS indexes
    - Embedding user queries during retrieval
    """
    require_api_key()
    client = OpenAI()

    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    vectors = np.asarray(
        [item.embedding for item in response.data],
        dtype=np.float32,
    )

    return normalize(vectors)