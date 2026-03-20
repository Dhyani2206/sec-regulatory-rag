"""
Module: src/rag/agent/agent.py

Factory for the LangChain Deep Agent that powers the /api/v1/agent-query endpoint.

The agent is a ``CompiledStateGraph`` (LangChain 1.x create_agent) that receives
a list of messages, calls the three retrieval tools in a loop, and returns a
final synthesised answer grounded entirely in retrieved evidence.

Every invocation is automatically traced to LangSmith when the env vars
LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set.

Usage
-----
    from src.rag.agent import build_regulatory_agent

    agent = build_regulatory_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content
"""
from __future__ import annotations

import logging
import os

from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI

from src.rag.agent.tools import (
    load_filing_section,
    retrieve_filing_chunks,
    retrieve_rule_chunks,
)
from src.rag.llm_prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_AGENT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

Tool usage guide:
- retrieve_rule_chunks: use for regulatory questions (CFR, SEC rules, requirements).
- retrieve_filing_chunks: use for company-specific filing questions (risk factors, MD&A, etc.).
- load_filing_section: use when you already know the exact company / form / year / section.

Always retrieve evidence before answering. Never answer from memory.
"""
)

_TOOLS = [retrieve_filing_chunks, retrieve_rule_chunks, load_filing_section]


def _build_llm() -> AzureChatOpenAI:
    """
    Instantiate the Azure OpenAI chat model used by the agent.

    Reads configuration from environment variables so no secrets are
    hard-coded in the source.
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    missing = [
        name
        for name, val in [
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
            ("AZURE_OPENAI_API_VERSION", api_version),
        ]
        if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required Azure OpenAI environment variables: {', '.join(missing)}"
        )

    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=0,
    )


def build_regulatory_agent():
    """
    Build and return the SEC Regulatory LangChain Deep Agent.

    Returns
    -------
    CompiledStateGraph
        A compiled agent graph that can be invoked with
        ``{"messages": [{"role": "user", "content": <question>}]}``.

    Raises
    ------
    EnvironmentError
        If any required Azure OpenAI environment variables are missing.
    """
    llm = _build_llm()
    logger.info(
        "Building regulatory agent with deployment=%s",
        os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    )
    return create_agent(
        model=llm,
        tools=_TOOLS,
        system_prompt=_AGENT_SYSTEM_PROMPT,
    )
