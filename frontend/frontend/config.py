from __future__ import annotations
import os

APP_TITLE = "Regulatory Reporting Assistant"
LAYOUT = "wide"

API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://localhost:8000").rstrip("/")
OPTIONS_ENDPOINT = f"{API_BASE_URL}/api/v1/options"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"
REQUEST_TIMEOUT_SECONDS = 60