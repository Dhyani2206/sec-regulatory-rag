"""
Module: app/api/v1/endpoints/agent_query.py

POST /api/v1/agent-query

Runs the user's question through the LangChain Deep Agent and returns the
same ``QueryResponse`` schema as the deterministic /api/v1/query endpoint,
so the Streamlit frontend can render the result without changes.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.v1.schemas.query import QueryRequest, QueryResponse
from app.services.agent_query_service import run_agent_query_service

router = APIRouter()


@router.post(
    "/agent-query",
    response_model=QueryResponse,
    summary="LangChain Deep Agent query",
    description=(
        "Execute a regulatory question through the LangChain Deep Agent. "
        "The agent autonomously calls retrieval tools (filings, rules, sections) "
        "and synthesises a grounded answer. "
        "Every run is traced to LangSmith when LANGCHAIN_TRACING_V2=true."
    ),
    tags=["agent"],
)
def agent_query(payload: QueryRequest) -> QueryResponse:
    """
    Run a regulatory question through the LangChain Deep Agent.

    The agent decides which retrieval tools to call, iterates until it has
    enough evidence, then returns a synthesised answer in the standard
    ``QueryResponse`` format.
    """
    return run_agent_query_service(payload)
