"""
Smoke tests for the FastAPI layer.

Skipped automatically when `storage/` artifacts are not present (e.g. fresh clone).
Run from project root: pytest tests/test_api_smoke.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE = PROJECT_ROOT / "storage"
_REQUIRED = (
    STORAGE / "chunks.jsonl",
    STORAGE / "faiss.index",
    STORAGE / "faiss_rules.index",
    STORAGE / "rules_chunks.jsonl",
    STORAGE / "rules_vectors.npy",
    STORAGE / "embeddings.npy",
)


def _artifacts_ready() -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in _REQUIRED)


pytestmark = pytest.mark.skipif(
    not _artifacts_ready(),
    reason="storage artifacts missing; build indexes or download artifacts first",
)


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def test_health_ok(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_documentation_page_renders(client: TestClient) -> None:
    for path in ("/documentation", "/api/v1/documentation"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "text/html" in r.headers.get("content-type", "")
        body = r.text
        assert "SEC Regulatory RAG" in body or "application works" in body.lower()
        assert "OpenAPI" in body or "Swagger" in body


def test_options_returns_companies(client: TestClient) -> None:
    r = client.get("/api/v1/options")
    assert r.status_code == 200
    data = r.json()
    assert "companies" in data
    assert isinstance(data["companies"], list)
    assert len(data["companies"]) >= 1


@pytest.mark.skipif(
    not os.getenv("AZURE_OPENAI_API_KEY"),
    reason="AZURE_OPENAI_API_KEY not set (embeddings required for rule retrieval)",
)
def test_query_rule_only_pass(client: TestClient) -> None:
    r = client.post(
        "/api/v1/query",
        json={"query": "What does 17 CFR 229.105 require?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") in ("PASS", "REFUSE")
    assert "answer" in body
    if body["status"] == "PASS":
        assert "229.105" in body["answer"] or "risk" in body["answer"].lower()


def test_query_invalid_scope_refuse(client: TestClient) -> None:
    r = client.post(
        "/api/v1/query",
        json={
            "query": "Where are risk factors?",
            "company": "MSFT",
            "form_folder": "10-K",
            "year": 1999,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "REFUSE"
    assert body.get("review_result", {}).get("finding") == "invalid_scope"


def test_agent_query_invalid_scope_refuse(client: TestClient) -> None:
    """
    The agent endpoint applies the same scope guard as the deterministic engine.

    This test does not require Azure credentials because the request is
    stopped before the agent is invoked.
    """
    r = client.post(
        "/api/v1/agent-query",
        json={
            "query": "Where are risk factors?",
            "company": "MSFT",
            "form_folder": "10-K",
            "year": 1999,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "REFUSE"
    assert body.get("review_result", {}).get("finding") == "invalid_scope"


@pytest.mark.skipif(
    not os.getenv("AZURE_OPENAI_API_KEY"),
    reason="AZURE_OPENAI_API_KEY not set (LLM required for agent invocation)",
)
def test_agent_query_rule_only_pass(client: TestClient) -> None:
    """
    End-to-end agent call for a rule-only question (no filing scope).

    Requires a valid Azure OpenAI key and built FAISS indexes.
    """
    r = client.post(
        "/api/v1/agent-query",
        json={"query": "What does 17 CFR 229.105 require?"},
        timeout=120,
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") in ("PASS", "ERROR")
    assert "answer" in body
    if body["status"] == "PASS":
        assert len(body["answer"]) > 20
