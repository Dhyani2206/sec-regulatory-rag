from __future__ import annotations
import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = PROJECT_ROOT / "storage"

# Runtime artifacts. Legacy faiss_filings.index is not required (retrieval uses faiss.index).
ARTIFACTS = {
    "CHUNKS_JSONL_URL": "chunks.jsonl",
    "FAISS_INDEX_URL": "faiss.index",
    "FAISS_RULES_INDEX_URL": "faiss_rules.index",
    "EMBEDDINGS_NPY_URL": "embeddings.npy",
    "RULES_CHUNKS_JSONL_URL": "rules_chunks.jsonl",
    "RULES_VECTORS_NPY_URL": "rules_vectors.npy",
}
class ArtifactBootstrapError(RuntimeError):
    pass


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()

        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def ensure_storage_artifacts() -> None:
    missing: list[tuple[str, Path]] = []

    for env_name, filename in ARTIFACTS.items():
        path = STORAGE_DIR / filename
        if not path.exists():
            missing.append((env_name, path))

    if not missing:
        logger.debug("All required storage artifacts are present under %s", STORAGE_DIR)
        return

    for env_name, path in missing:
        url = os.getenv(env_name)
        if not url:
            raise ArtifactBootstrapError(
                f"Missing required artifact '{path.name}' and no URL was provided in env var '{env_name}'."
            )

        print(f"[artifact_bootstrap] Downloading {path.name} ...")
        _download_file(url, path)
        print(f"[artifact_bootstrap] Downloaded {path.name}")