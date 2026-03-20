from __future__ import annotations
import os

APP_TITLE = "Regulatory Reporting Assistant"
LAYOUT = "wide"

API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://localhost:8000").rstrip("/")
OPTIONS_ENDPOINT = f"{API_BASE_URL}/api/v1/options"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"
AGENT_QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/agent-query"
BROWSE_ENDPOINT = f"{API_BASE_URL}/api/v1/browse"
CORPUS_STATS_ENDPOINT = f"{API_BASE_URL}/api/v1/corpus-stats"
DOCUMENTATION_URL = f"{API_BASE_URL}/documentation"
REQUEST_TIMEOUT_SECONDS = 60