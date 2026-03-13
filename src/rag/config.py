from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List


# PROJECT ROOTS
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TESTS_DIR = PROJECT_ROOT / "tests"

# DATA DIRECTORIES
COMPANIES_DATA_DIR = DATA_DIR / "companies_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
REGULATIONS_RAW_DIR = DATA_DIR / "regulations_raw"
REGULATIONS_PROCESSED_DIR = DATA_DIR / "regulations_processed"
OBLIGATIONS_DIR = DATA_DIR / "obligations"

# OUTPUT DIRECTORIES
DIFF_REPORTS_DIR = OUTPUTS_DIR / "diff_reports"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
GAP_REPORTS_DIR = OUTPUTS_DIR / "gap_reports"
LOGS_DIR = OUTPUTS_DIR / "logs"
QC_DIR = OUTPUTS_DIR / "qc"
RUNS_DIR = OUTPUTS_DIR / "runs"

# STORAGE FILES — FILINGS
FILING_CHUNKS_PATH = STORAGE_DIR / "chunks.jsonl"
FILING_EMBEDDINGS_PATH = STORAGE_DIR / "embeddings.npy"
FILING_EMBEDDINGS_META_PATH = STORAGE_DIR / "embeddings_meta.json"
FILING_INDEX_PATH = STORAGE_DIR / "faiss.index"

# STORAGE FILES — RULES
RULES_CHUNKS_PATH = STORAGE_DIR / "rules_chunks.jsonl"
RULES_VECTORS_PATH = STORAGE_DIR / "rules_vectors.npy"
RULES_EMBEDDINGS_META_PATH = STORAGE_DIR / "rules_embeddings_meta.json"
RULES_INDEX_PATH = STORAGE_DIR / "faiss_rules.index"
RULES_INDEX_MANIFEST_PATH = STORAGE_DIR / "rules_index_manifest.json"
RULES_CHUNKS_MANIFEST_PATH = STORAGE_DIR / "rules_chunks_manifest_2025.json"
RULES_CKPT_META_PATH = STORAGE_DIR / "rules_ckpt_meta.json"

# EXPERIMENTAL
LEGACY_FILING_INDEX_PATH = STORAGE_DIR / "faiss_filings.index"

# EMBEDDING / RETRIEVAL SETTINGS
FILING_EMBEDDING_MODEL = "text-embedding-3-large"
RULES_EMBEDDING_MODEL = "text-embedding-3-large"

FILING_VECTOR_DIM = 3072   # text-embedding-3-large
RULES_VECTOR_DIM = 3072   

FAISS_METRIC = "cosine"
DEFAULT_TOP_K = 10
DEFAULT_RULE_TOP_K = 10
DEFAULT_FILING_TOP_K = 10
DEFAULT_SECTION_TOP_K = 5

MIN_RULE_SCORE = 0.30
MIN_FILING_SCORE = 0.30

MAX_RULE_EVIDENCE = 3
MAX_FILING_EVIDENCE = 3

# APP / PIPELINE SETTINGS
SUPPORTED_FORMS = {"10-k", "10-q"}
DEFAULT_ENCODING = "utf-8"
JSONL_ENCODING = "utf-8"

# RUNTIME CONFIG OBJECT
@dataclass(frozen=True)
class RAGConfig:
    project_root: Path = PROJECT_ROOT
    src_dir: Path = SRC_DIR
    data_dir: Path = DATA_DIR
    storage_dir: Path = STORAGE_DIR
    outputs_dir: Path = OUTPUTS_DIR

    filing_chunks_path: Path = FILING_CHUNKS_PATH
    filing_embeddings_path: Path = FILING_EMBEDDINGS_PATH
    filing_embeddings_meta_path: Path = FILING_EMBEDDINGS_META_PATH
    filing_index_path: Path = FILING_INDEX_PATH

    rules_chunks_path: Path = RULES_CHUNKS_PATH
    rules_vectors_path: Path = RULES_VECTORS_PATH
    rules_embeddings_meta_path: Path = RULES_EMBEDDINGS_META_PATH
    rules_index_path: Path = RULES_INDEX_PATH
    rules_index_manifest_path: Path = RULES_INDEX_MANIFEST_PATH
    rules_chunks_manifest_path: Path = RULES_CHUNKS_MANIFEST_PATH
    rules_ckpt_meta_path: Path = RULES_CKPT_META_PATH

    filing_embedding_model: str = FILING_EMBEDDING_MODEL
    rules_embedding_model: str = RULES_EMBEDDING_MODEL

    filing_vector_dim: int = FILING_VECTOR_DIM
    rules_vector_dim: int = RULES_VECTOR_DIM

    default_top_k: int = DEFAULT_TOP_K
    default_rule_top_k: int = DEFAULT_RULE_TOP_K
    default_filing_top_k: int = DEFAULT_FILING_TOP_K
    default_section_top_k: int = DEFAULT_SECTION_TOP_K

    min_rule_score: float = MIN_RULE_SCORE
    min_filing_score: float = MIN_FILING_SCORE

    max_rule_evidence: int = MAX_RULE_EVIDENCE
    max_filing_evidence: int = MAX_FILING_EVIDENCE

DEFAULT_CONFIG = RAGConfig()


# HELPERS
def ensure_output_dirs() -> None:
    required_dirs = [
        OUTPUTS_DIR,
        DIFF_REPORTS_DIR,
        EVALUATION_DIR,
        GAP_REPORTS_DIR,
        LOGS_DIR,
        QC_DIR,
        RUNS_DIR,
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)


def required_runtime_files() -> List[Path]:
    """
    Files required for request-time RAG runtime.
    """
    return [
        FILING_CHUNKS_PATH,
        FILING_EMBEDDINGS_PATH,
        FILING_EMBEDDINGS_META_PATH,
        FILING_INDEX_PATH,
        RULES_CHUNKS_PATH,
        RULES_VECTORS_PATH,
        RULES_EMBEDDINGS_META_PATH,
        RULES_INDEX_PATH,
    ]


def missing_runtime_files() -> List[Path]:
    """
    Return missing required runtime files.
    """
    return [p for p in required_runtime_files() if not p.exists()]


def validate_runtime_files() -> None:
    """
    Raise a helpful error if required runtime artifacts are missing.
    """
    missing = missing_runtime_files()
    if missing:
        joined = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Required runtime storage files are missing:\n{joined}"
        )

def using_legacy_filing_index() -> bool:
    """
    Returns True if the legacy filing index exists.
    Useful during migration from faiss.index -> faiss_filings.index.
    """
    return LEGACY_FILING_INDEX_PATH.exists()


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("STORAGE_DIR:", STORAGE_DIR)
    print("FILING_INDEX_PATH:", FILING_INDEX_PATH)
    print("RULES_INDEX_PATH:", RULES_INDEX_PATH)
    ensure_output_dirs()

    missing = missing_runtime_files()
    if missing:
        print("\nMissing runtime files:")
        for p in missing:
            print(" -", p)
    else:
        print("\nAll required runtime files are present.")