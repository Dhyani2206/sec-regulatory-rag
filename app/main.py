from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_v1_router
from app.services.artifact_bootstrap import ensure_storage_artifacts


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
        "without moving retrieval or answer logic into the web layer."
    ),
    lifespan=lifespan,
)

app.include_router(api_v1_router)