"""
Module: app/services/agent_query_service.py

Service layer that bridges the FastAPI agent endpoint and the LangChain Deep Agent.

Responsibilities
----------------
- Enrich bare citations into proper questions before sending to the agent
- Validate the requested filing scope (same guard as the deterministic engine)
- Build a natural-language user message that includes scope context
- Invoke the LangChain agent and collect its final synthesised answer
- Extract evidence items from ToolMessages so the response is grounded
- Extract agent steps (tool call sequence) for the technical trace
- Map the agent output to the existing ``QueryResponse`` schema

The agent is built lazily on first use and cached for the lifetime of the process.

Notes on tool-call format compatibility
----------------------------------------
Some Azure OpenAI deployments return tool calls as raw JSON text in the message
``content`` rather than using the structured ``tool_calls`` API field.  The
extraction helpers handle both formats:

1. Standard format  : ``AIMessage.tool_calls`` is non-empty list, ``content``
   is empty or a text preamble.
2. Legacy/text format: ``AIMessage.additional_kwargs["tool_calls"]`` is populated
   but ``AIMessage.tool_calls`` may still be empty in LangChain parsing.
3. Raw-text format  : The LLM prints tool-call JSON into ``content`` directly,
   with ``tool_calls`` empty.  We detect this with ``_is_tool_call_json()`` and
   skip those messages when looking for the final answer.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.api.v1.schemas.query import (
    EvidenceItem,
    EvidencePreviewItem,
    QueryRequest,
    QueryResponse,
    ReviewResult,
    SourceItem,
    TechnicalTrace,
)
from app.services.options_service import is_valid_scope

logger = logging.getLogger(__name__)

_AGENT = None

_SNIPPET_MAX = 400
_STEP_OUTPUT_MAX = 600

_BARE_CFR_RE = re.compile(
    r"^\s*(?:17\s+CFR\s+\d+\.\d+\S*|item\s+\d+[a-z]?|rule\s+\d+\S*)\s*$",
    re.IGNORECASE,
)

# Heuristic: a JSON dict whose keys are common tool argument names.
_TOOL_ARG_KEYS = frozenset(
    {"query", "top_k", "company", "form", "form_folder", "year", "section", "max_hits"}
)


# ---------------------------------------------------------------------------
# Lazy agent singleton
# ---------------------------------------------------------------------------

def _get_agent():
    global _AGENT
    if _AGENT is None:
        from src.rag.agent import build_regulatory_agent
        _AGENT = build_regulatory_agent()
    return _AGENT


# ---------------------------------------------------------------------------
# Query enrichment
# ---------------------------------------------------------------------------

def _enrich_query(query: str) -> str:
    """
    Convert a bare CFR citation or Item reference into a proper question.

    Examples
    --------
    "17 CFR 229.105"          → "What does 17 CFR 229.105 require?"
    "Item 303"                → "What does Item 303 require?"
    "What does ... require?"  → unchanged
    """
    q = query.strip()
    if _BARE_CFR_RE.match(q):
        return f"What does {q} require?"
    return q


# ---------------------------------------------------------------------------
# Tool-call JSON detection
# ---------------------------------------------------------------------------

def _is_tool_call_json(text: str) -> bool:
    """
    Return True if *text* looks like raw tool-call argument JSON rather than a
    natural-language answer.

    This handles two observed patterns:
    - Single JSON object: ``{"query": "...", "top_k": 8}``
    - Two objects concatenated (parallel calls): ``{...} {...}``
    """
    t = text.strip()
    if not t.startswith("{"):
        return False

    # Try a single object first.
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            if _TOOL_ARG_KEYS.intersection(obj.keys()):
                return True
    except json.JSONDecodeError:
        pass

    # Try splitting on "} {" for concatenated objects.
    parts = re.split(r"\}\s*\{", t)
    if len(parts) > 1:
        try:
            obj = json.loads(parts[0] + "}")
            if isinstance(obj, dict) and _TOOL_ARG_KEYS.intersection(obj.keys()):
                return True
        except json.JSONDecodeError:
            pass

    return False


def _has_any_tool_calls(msg: AIMessage) -> bool:
    """
    Return True if *msg* has tool calls via any supported mechanism:
    - ``AIMessage.tool_calls`` (LangChain normalised field)
    - ``AIMessage.additional_kwargs["tool_calls"]`` (raw OpenAI/Azure format)
    """
    if msg.tool_calls:
        return True
    if msg.additional_kwargs.get("tool_calls"):
        return True
    return False


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def _build_user_message(payload: QueryRequest) -> str:
    enriched_query = _enrich_query(payload.query)
    parts = [enriched_query]

    scope_parts: List[str] = []
    if payload.company:
        scope_parts.append(f"company={payload.company}")
    if payload.form_folder:
        scope_parts.append(f"form={payload.form_folder}")
    if payload.year is not None:
        scope_parts.append(f"year={payload.year}")

    if scope_parts:
        parts.append(f"\nFiling scope: {', '.join(scope_parts)}")
    else:
        parts.append(
            "\nNo specific filing scope provided — answer from regulatory rules only."
        )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_text_from_content(content: Any) -> str:
    """Pull a text string from an AIMessage content (str or list-of-blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(t for t in texts if t.strip()).strip()
    return ""


def _extract_final_answer(result: Any) -> str:
    """
    Extract the plain-text synthesised answer from the agent invocation result.

    Strategy (in priority order):
    1. Last AIMessage with no tool calls AND non-JSON content → synthesis turn.
    2. Fall back to any AIMessage with non-JSON content.
    3. Return a descriptive fallback if nothing suitable is found.
    """
    messages = result.get("messages", [])

    # Pass 1: strict — no tool calls, non-JSON content.
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if _has_any_tool_calls(msg):
            continue
        text = _extract_text_from_content(msg.content)
        if text and not _is_tool_call_json(text):
            return text

    # Pass 2: relax tool_calls check — look for any non-JSON AIMessage content.
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        text = _extract_text_from_content(msg.content)
        if text and not _is_tool_call_json(text):
            return text

    # Diagnostic: report what messages we found.
    msg_types = [type(m).__name__ for m in messages]
    logger.warning(
        "Could not extract a natural-language answer. Message types: %s", msg_types
    )
    return (
        "The agent completed its retrieval steps but did not produce a synthesised "
        "natural-language answer. This may indicate a compatibility issue between the "
        "Azure OpenAI deployment and the tool-calling protocol. "
        "Check the Technical Trace below for the raw agent steps."
    )


# ---------------------------------------------------------------------------
# Agent steps extraction
# ---------------------------------------------------------------------------

def _parse_tool_calls_from_msg(msg: AIMessage) -> List[Dict[str, Any]]:
    """
    Return a list of ``{name, args, id}`` dicts from an AIMessage, checking
    both the LangChain-normalised ``tool_calls`` field and the raw
    ``additional_kwargs["tool_calls"]`` field.
    """
    # LangChain normalised format.
    if msg.tool_calls:
        return [
            {"name": tc.get("name", "unknown"), "args": tc.get("args", {}), "id": tc.get("id", "")}
            for tc in msg.tool_calls
        ]

    # Raw Azure / OpenAI additional_kwargs format.
    raw = msg.additional_kwargs.get("tool_calls", [])
    if not raw:
        return []

    parsed = []
    for item in raw:
        func = item.get("function", {})
        name = func.get("name", "unknown")
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {"raw_arguments": func.get("arguments", "")}
        parsed.append({"name": name, "args": args, "id": item.get("id", "")})
    return parsed


def _extract_agent_steps(result: Any) -> List[Dict[str, Any]]:
    """
    Walk the messages list and build a structured list of agent reasoning steps.

    Handles both the LangChain normalised ``tool_calls`` field and the raw
    ``additional_kwargs["tool_calls"]`` Azure/OpenAI format.
    """
    messages = result.get("messages", [])

    tool_outputs: Dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tid = getattr(msg, "tool_call_id", None) or ""
            content = msg.content or ""
            if isinstance(content, list):
                content = " ".join(
                    str(b.get("text", b)) if isinstance(b, dict) else str(b)
                    for b in content
                )
            tool_outputs[tid] = str(content)

    steps: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        calls = _parse_tool_calls_from_msg(msg)
        for call in calls:
            call_id = call["id"]
            raw_output = tool_outputs.get(call_id, "")
            steps.append(
                {
                    "tool": call["name"],
                    "args": call["args"],
                    "output_snippet": (
                        raw_output[:_STEP_OUTPUT_MAX] + "..."
                        if len(raw_output) > _STEP_OUTPUT_MAX
                        else raw_output
                    ),
                }
            )

    return steps


# ---------------------------------------------------------------------------
# Evidence parsing
# ---------------------------------------------------------------------------

_CHUNK_ID_RE = re.compile(r"^\[(.+?)\]")
_META_RE = re.compile(r"Meta:\s*(.+)")
_SCORE_RE = re.compile(r"Score:\s*([\d.]+)")
_TOOL_SEPARATOR = "\n---\n"


def _parse_single_chunk_block(block: str) -> Optional[EvidenceItem]:
    lines = block.strip().splitlines()
    if not lines:
        return None

    chunk_id: Optional[str] = None
    company: Optional[str] = None
    form: Optional[str] = None
    year: Optional[int] = None
    section: Optional[str] = None
    citation: Optional[str] = None
    score: Optional[float] = None
    text_lines: List[str] = []

    for line in lines:
        m_id = _CHUNK_ID_RE.match(line)
        if m_id and chunk_id is None:
            chunk_id = m_id.group(1)
            continue

        m_meta = _META_RE.match(line)
        if m_meta:
            for kv in m_meta.group(1).split(","):
                kv = kv.strip()
                if "=" not in kv:
                    continue
                k, _, v = kv.partition("=")
                k, v = k.strip(), v.strip()
                if k == "company":
                    company = v or None
                elif k == "form":
                    form = v or None
                elif k == "year":
                    try:
                        year = int(v)
                    except ValueError:
                        pass
                elif k == "section":
                    section = v or None
                elif k == "citation":
                    citation = v or None
            continue

        m_score = _SCORE_RE.match(line)
        if m_score:
            try:
                score = float(m_score.group(1))
            except ValueError:
                pass
            continue

        text_lines.append(line)

    text = "\n".join(text_lines).strip()
    if not text and not chunk_id:
        return None

    return EvidenceItem(
        chunk_id=chunk_id,
        company=company,
        form=form,
        year=year,
        section=section,
        citation=citation,
        score=score,
        text=text or None,
    )


def _parse_tool_outputs_to_evidence(
    steps: List[Dict[str, Any]],
) -> Tuple[List[EvidenceItem], List[EvidencePreviewItem], List[SourceItem]]:
    evidence: List[EvidenceItem] = []
    previews: List[EvidencePreviewItem] = []
    seen_sources: set = set()
    sources: List[SourceItem] = []

    for step in steps:
        raw_output = step.get("output_snippet", "")
        blocks = raw_output.split(_TOOL_SEPARATOR)

        for block in blocks:
            item = _parse_single_chunk_block(block)
            if item is None:
                continue
            evidence.append(item)

            label_parts = [
                p for p in [
                    item.company,
                    item.form,
                    str(item.year) if item.year else None,
                    item.section,
                ]
                if p
            ]
            label = ", ".join(label_parts) if label_parts else (item.citation or "Retrieved evidence")

            snippet = item.text or ""
            if len(snippet) > _SNIPPET_MAX:
                snippet = snippet[:_SNIPPET_MAX].rstrip() + "..."
            previews.append(
                EvidencePreviewItem(
                    label=label,
                    snippet=snippet or None,
                    section=item.section,
                    score=item.score,
                )
            )

            src_key = (item.company, item.form, item.year, item.section)
            if src_key not in seen_sources:
                seen_sources.add(src_key)
                sources.append(
                    SourceItem(
                        label=label,
                        section=item.section,
                        company=item.company,
                        form=item.form,
                        year=item.year,
                    )
                )

    return evidence, previews, sources


# ---------------------------------------------------------------------------
# Scope refusal response
# ---------------------------------------------------------------------------

def _build_invalid_scope_response(payload: QueryRequest) -> QueryResponse:
    return QueryResponse(
        status="REFUSE",
        review_result=ReviewResult(
            finding="invalid_scope",
            label="Requested filing scope is not available",
            confidence="high",
            evidence_strength="not_found",
        ),
        answer=(
            "The selected company, form, and year combination is not available "
            "in the indexed dataset."
        ),
        plain_english_explanation=(
            "The agent checks available filing metadata before retrieval, and this "
            "specific scope combination was not found."
        ),
        why_it_matters=(
            "This prevents unsupported answers by ensuring the agent only runs "
            "against filings that actually exist in the indexed dataset."
        ),
        review_guidance=[
            "Choose a company, form, and year combination from the available options.",
            "If you want a rule-only answer, submit the query without filing scope.",
        ],
        sources=[],
        evidence_preview=[],
        technical_trace=None,
        notes=[
            "Query stopped before agent invocation because the requested scope is not available."
        ],
        evidence=[],
    )


# ---------------------------------------------------------------------------
# Direct RAG fallback (used when agent's tool-calling protocol is unsupported)
# ---------------------------------------------------------------------------

def _run_direct_rag(
    payload: QueryRequest,
    enriched_query: str,
) -> Tuple[str, List[Dict[str, Any]], List[EvidenceItem], List[EvidencePreviewItem], List[SourceItem]]:
    """
    Directly invoke retrieval tools and call the LLM for synthesis.

    Used as a fallback when the Azure deployment does not support the
    structured tool-calling protocol (agent calls 0 tools).  This approach
    is compatible with any Azure OpenAI deployment that can generate text.
    """
    from langchain_core.messages import HumanMessage as HM
    from langchain_core.messages import SystemMessage

    from src.rag.agent.agent import _build_llm
    from src.rag.agent.tools import retrieve_filing_chunks, retrieve_rule_chunks
    from src.rag.llm_prompts import SYSTEM_PROMPT, USER_TEMPLATE

    steps: List[Dict[str, Any]] = []

    # --- Rule evidence (always retrieved) ---
    try:
        rule_output = str(
            retrieve_rule_chunks.invoke({"query": enriched_query, "top_k": 8})
        )
    except Exception as exc:
        rule_output = f"Rule retrieval failed: {exc}"

    steps.append(
        {
            "tool": "retrieve_rule_chunks",
            "args": {"query": enriched_query, "top_k": 8},
            "output_snippet": rule_output[:_STEP_OUTPUT_MAX]
            + ("..." if len(rule_output) > _STEP_OUTPUT_MAX else ""),
        }
    )

    # --- Filing evidence (when scope is provided) ---
    if payload.company and payload.form_folder and payload.year is not None:
        filing_args = {
            "query": enriched_query,
            "company": payload.company,
            "form": payload.form_folder,
            "year": str(payload.year),
            "top_k": 6,
        }
        try:
            filing_output = str(retrieve_filing_chunks.invoke(filing_args))
        except Exception as exc:
            filing_output = f"Filing retrieval failed: {exc}"

        steps.append(
            {
                "tool": "retrieve_filing_chunks",
                "args": filing_args,
                "output_snippet": filing_output[:_STEP_OUTPUT_MAX]
                + ("..." if len(filing_output) > _STEP_OUTPUT_MAX else ""),
            }
        )

    # --- LLM synthesis ---
    evidence_text = "\n\n---\n\n".join(
        f"[Tool: {s['tool']}]\n{s['output_snippet']}" for s in steps
    )
    user_msg = USER_TEMPLATE.format(question=enriched_query, evidence=evidence_text)

    try:
        llm = _build_llm()
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HM(content=user_msg)]
        )
        answer = str(response.content).strip()
        if not answer:
            answer = "The LLM returned an empty response."
    except Exception as exc:
        logger.exception("LLM synthesis call failed")
        answer = f"LLM synthesis failed: {exc}"

    evidence, previews, sources = _parse_tool_outputs_to_evidence(steps)
    return answer, steps, evidence, previews, sources


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _summarise_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Return a compact debug summary of the agent message list."""
    summary = []
    for msg in messages:
        name = type(msg).__name__
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = f"[list of {len(content)} blocks]"
        elif isinstance(content, str) and len(content) > 120:
            content = content[:120] + "..."
        tc_count = len(getattr(msg, "tool_calls", []))
        ak_tc_count = len((getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls", []))
        summary.append(
            {
                "type": name,
                "content_preview": content,
                "tool_calls": tc_count,
                "additional_kwargs_tool_calls": ak_tc_count,
            }
        )
    return summary


# ---------------------------------------------------------------------------
# Main service entry point
# ---------------------------------------------------------------------------

def run_agent_query_service(payload: QueryRequest) -> QueryResponse:
    """
    Execute a regulatory query using the LangChain Deep Agent.

    Parameters
    ----------
    payload:
        Validated ``QueryRequest`` with query and optional filing scope.

    Returns
    -------
    QueryResponse
        Structured response compatible with the deterministic engine's output.
    """
    if not is_valid_scope(payload.company, payload.form_folder, payload.year):
        return _build_invalid_scope_response(payload)

    user_message = _build_user_message(payload)
    enriched_query = _enrich_query(payload.query)

    used_fallback = False
    try:
        agent = _get_agent()
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]}
        )

        messages = result.get("messages", [])
        msg_summary = _summarise_messages(messages)
        logger.info("Agent returned %d messages: %s", len(messages), msg_summary)

        answer = _extract_final_answer(result)
        steps = _extract_agent_steps(result)
        evidence, previews, sources = _parse_tool_outputs_to_evidence(steps)

        status = "PASS"
        notes: List[str] = []

        if not steps:
            # The Azure deployment does not support the tool-calling protocol used
            # by create_agent.  Fall back to direct retrieval + LLM synthesis.
            logger.info(
                "Agent called 0 tools — falling back to direct RAG synthesis."
            )
            answer, steps, evidence, previews, sources = _run_direct_rag(
                payload, enriched_query
            )
            used_fallback = True
            notes.append(
                "The LangChain agent was bypassed because the Azure deployment does "
                "not support the tool-calling protocol in the current LangChain version. "
                "Evidence was retrieved directly and synthesised with the LLM."
            )
        elif not evidence:
            notes.append(
                "The agent called tools but no structured evidence chunks were "
                "extracted from tool outputs."
            )

    except EnvironmentError as exc:
        logger.error("Agent configuration error: %s", exc)
        answer = f"Agent is not configured: {exc}"
        status = "ERROR"
        steps, evidence, previews, sources = [], [], [], []
        notes = [str(exc)]
        msg_summary = []
        used_fallback = False

    except Exception as exc:
        logger.exception("Agent invocation failed")
        answer = f"Agent invocation failed: {exc}"
        status = "ERROR"
        steps, evidence, previews, sources = [], [], [], []
        notes = [str(exc)]
        msg_summary = []
        used_fallback = False

    if len(evidence) >= 3:
        evidence_strength = "strong"
        confidence = "medium"
    elif len(evidence) >= 1:
        evidence_strength = "moderate"
        confidence = "medium"
    else:
        evidence_strength = "not_found" if status != "PASS" else "moderate"
        confidence = "low" if status != "PASS" else "medium"

    review_result = ReviewResult(
        finding="supported_answer" if status == "PASS" else "manual_review_recommended",
        label=(
            "Answer synthesised by LangChain Deep Agent from retrieved evidence"
            if status == "PASS"
            else "Agent error — human review required"
        ),
        confidence=confidence,
        evidence_strength=evidence_strength,
    )

    if used_fallback:
        plain_english = (
            "The answer was produced by directly calling SEC regulatory retrieval tools "
            "and synthesising the evidence with the Azure OpenAI language model. "
            f"{len(steps)} retrieval tool(s) were used to gather evidence."
        )
    else:
        plain_english = (
            "The answer was produced by a LangChain Deep Agent that autonomously "
            "called retrieval tools and synthesised evidence grounded in SEC filings "
            f"and regulatory rules. The agent called {len(steps)} tool(s)."
        )

    return QueryResponse(
        status=status,
        review_result=review_result,
        answer=answer,
        plain_english_explanation=plain_english,
        why_it_matters=(
            "The agent approach allows multi-step reasoning and tool chaining, "
            "surfacing more nuanced answers than a single-pass retrieval."
        ),
        review_guidance=[
            "Verify the cited chunk IDs in the evidence against the original filing.",
            "Cross-check any CFR citations the agent produced against the rules corpus.",
            "Human review is recommended before using this output in a compliance decision.",
        ],
        sources=sources[:5],
        evidence_preview=previews[:5],
        technical_trace=TechnicalTrace(
            route={
                "intent": "langchain_agent",
                "enriched_query": enriched_query,
                "tools_available": [
                    "retrieve_filing_chunks",
                    "retrieve_rule_chunks",
                    "load_filing_section",
                ],
                "message_summary": msg_summary,
            },
            guard_reason=None,
            agent_steps=steps if steps else None,
        ),
        notes=notes,
        evidence=evidence,
    )
