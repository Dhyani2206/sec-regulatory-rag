from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.schemas.common import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
def readiness_check() -> ReadinessResponse:
    """
    Keep this lightweight.
    Ready = app can import runtime dependencies and serve requests.
    You can later expand checks for indexes/models/config if needed.
    """
    checks = {
        "api": "ok",
        "answer_engine_import": "ok",
    }
    return ReadinessResponse(status="ready", checks=checks)