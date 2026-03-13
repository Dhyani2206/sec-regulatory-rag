from __future__ import annotations
from fastapi import FastAPI
from app.api.v1.router import api_v1_router

app = FastAPI(
    title="Regulatory Reporting AI API",
    version="1.0.0",
    summary="Thin FastAPI wrapper over the canonical regulatory RAG engine",
    description=(
        "Provides production API access to the regulatory reporting RAG backend "
        "without moving retrieval or answer logic into the web layer."
    ),
)

app.include_router(api_v1_router)