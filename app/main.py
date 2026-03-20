from __future__ import annotations
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.v1.endpoints.documentation import get_documentation_response
from app.api.v1.router import api_v1_router
from app.services.artifact_bootstrap import ensure_storage_artifacts

# Load .env for embedding keys (Azure OpenAI) and optional download URLs.
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_storage_artifacts()
    yield


app = FastAPI(
    title="Regulatory Reporting AI API",
    version="1.0.0",
    summary="Thin FastAPI wrapper over the canonical regulatory RAG engine",
    description=(
        "Provides production API access to the regulatory reporting RAG backend "
        "without moving retrieval or answer logic into the web layer. "
        "**[How the app works](/documentation)** — architecture, query flow, and compliance notes."
    ),
    lifespan=lifespan,
)

# Local dev: allow Streamlit (8501) or other UIs on localhost to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8501",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Documentation: register on the app directly so routes are always present after reload.
# Two paths: root (bookmark-friendly) and under /api/v1 (API-only gateways / proxies).
@app.get(
    "/documentation",
    response_class=HTMLResponse,
    tags=["documentation"],
    summary="How the application works",
    name="application_documentation",
)
@app.get(
    "/api/v1/documentation",
    response_class=HTMLResponse,
    tags=["documentation"],
    summary="How the application works (API prefix)",
    name="application_documentation_api_v1",
)
def serve_application_documentation() -> HTMLResponse:
    return get_documentation_response()


app.include_router(api_v1_router)