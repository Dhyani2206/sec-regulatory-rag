from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.api.v1.schemas.query import QueryRequest, QueryResponse
from app.services.query_service import run_query_service

router = APIRouter()
@router.post("/query", response_model=QueryResponse, summary="Query the Intelligent Regulatory Review Assistant")
def query_endpoint(payload: QueryRequest) -> QueryResponse:
    try:
        return run_query_service(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing query.",
        ) from exc