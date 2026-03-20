"""
Module: embeddings.py

Purpose
-------
Provides the shared embedding interface used across the RAG system.

Responsibilities
----------------
- Generate vector embeddings for text using the Azure OpenAI embeddings API
- Normalize vectors for cosine similarity search
- Serve as the single embedding entrypoint for the retrieval pipeline

Inputs
------
List[str] of text inputs
Embedding model name (used as fallback deployment name)

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
3. Changes to the embedding provider require modification in only one place.

Vectors are L2-normalized so FAISS cosine similarity behaves correctly.

Security
--------
Requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and
AZURE_OPENAI_API_VERSION to be set in environment variables (.env file).
"""

from __future__ import annotations
import os
from typing import List

import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


def _require_azure_config() -> None:
    """Ensure required Azure OpenAI environment variables are present."""
    missing = [
        v for v in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
        )
        if not os.getenv(v)
    ]
    if missing:
        raise RuntimeError(
            f"Missing Azure OpenAI env vars: {', '.join(missing)}. "
            "Add them to your .env file and restart."
        )


def _get_client() -> AzureOpenAI:
    """Build an AzureOpenAI client from environment variables."""
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )


def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize vectors for cosine similarity search with FAISS IndexFlatIP."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(np.float32)


def embed_texts(texts: List[str], model: str) -> np.ndarray:
    """
    Generate embeddings for a list of texts via Azure OpenAI.

    Parameters
    ----------
    texts : List[str]
        Text inputs to embed.
    model : str
        Fallback deployment name if AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not set.

    Returns
    -------
    np.ndarray
        Normalized float32 embedding matrix of shape (N, D).
    """
    _require_azure_config()

    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or model
    client = _get_client()

    response = client.embeddings.create(
        model=deployment,
        input=texts,
    )

    vectors = np.asarray(
        [item.embedding for item in response.data],
        dtype=np.float32,
    )

    return normalize(vectors)


# Keep for backwards compatibility with any code that calls require_api_key()
def require_api_key() -> None:
    _require_azure_config()
