"""
Package: src.rag.agent

LangChain Deep Agent layer for the SEC Regulatory RAG system.

Exports
-------
build_regulatory_agent  — factory that returns a ready-to-invoke agent executor
"""
from src.rag.agent.agent import build_regulatory_agent

__all__ = ["build_regulatory_agent"]
