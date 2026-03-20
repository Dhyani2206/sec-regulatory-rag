from __future__ import annotations
import json
import streamlit as st
from frontend.components.helpers import as_dict, as_list

def render_status_banner(status: str) -> None:
    normalized = str(status or "").upper()

    if normalized == "PASS":
        st.success("Grounded answer returned.")
    elif normalized == "REFUSE":
        st.warning("Grounded refusal returned.")
    else:
        st.info(f"Response status: {normalized or 'UNKNOWN'}")

def render_review_result(review_result: dict) -> None:
    review = as_dict(review_result)
    st.markdown("### Review Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Finding", str(review.get("finding", "—")))
    c2.metric("Label", str(review.get("label", "—")))
    c3.metric("Confidence", str(review.get("confidence", "—")))
    c4.metric("Evidence Strength", str(review.get("evidence_strength", "—")))

def render_answer_block(response: dict) -> None:
    status = str(response.get("status", "")).upper()
    answer = response.get("answer") or ""
    explanation = response.get("plain_english_explanation")
    why_it_matters = response.get("why_it_matters")

    if status == "REFUSE":
        st.markdown("### Response")
        st.markdown(answer or "_No response returned._")

        st.markdown("### Why this was refused")
        st.markdown(
            explanation or "The system could not find strong enough support to answer safely."
        )
    else:
        st.markdown("### Answer")
        st.markdown(answer or "_No answer returned._")

        if explanation:
            st.markdown("### Plain-English Explanation")
            st.markdown(explanation)

    if why_it_matters:
        st.markdown("### Why It Matters")
        st.markdown(why_it_matters)

def render_review_guidance(review_guidance: list) -> None:
    items = as_list(review_guidance)
    if not items:
        return
    st.markdown("### Review Guidance")
    for item in items:
        st.markdown(f"- {item}")

def render_sources(sources: list) -> None:
    items = as_list(sources)
    if not items:
        return
    st.markdown("### Sources")
    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            title = item.get("title") or item.get("source") or f"Source {idx}"
            st.markdown(f"**{idx}. {title}**")

            lines = []
            for key in ("source", "section", "citation", "type"):
                value = item.get(key)
                if value:
                    lines.append(f"**{key.title()}:** {value}")

            if lines:
                st.markdown("  \n".join(lines))
        else:
            st.markdown(f"- {item}")

def render_evidence_preview(evidence_preview: list) -> None:
    items = as_list(evidence_preview)
    if not items:
        return

    st.markdown("### Evidence Preview")
    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            title = (
                item.get("title")
                or item.get("section")
                or item.get("source")
                or f"Preview {idx}"
            )
            snippet = item.get("snippet") or item.get("text") or "_No snippet available._"
            with st.expander(f"{idx}. {title}", expanded=(idx == 1)):
                meta = []
                for key in ("source", "type", "score", "citation"):
                    value = item.get(key)
                    if value is not None and value != "":
                        meta.append(f"{key.title()}: {value}")
                if meta:
                    st.caption(" | ".join(meta))
                st.markdown(snippet)
        else:
            st.markdown(f"- {item}")

def render_notes(notes: list) -> None:
    items = as_list(notes)
    if not items:
        return
    st.markdown("### Notes")
    for item in items:
        st.markdown(f"- {item}")

def render_evidence(evidence: list) -> None:
    items = as_list(evidence)
    if not items:
        return

    st.markdown("### Evidence")
    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            title = (
                item.get("title")
                or item.get("section")
                or item.get("source")
                or f"Evidence {idx}"
            )
            snippet = item.get("snippet") or item.get("text") or "_No snippet available._"
            with st.expander(f"{idx}. {title}", expanded=False):
                meta_lines = []
                for key in ("source", "type", "score", "section", "citation"):
                    value = item.get(key)
                    if value is not None and value != "":
                        meta_lines.append(f"**{key.title()}:** {value}")
                if meta_lines:
                    st.markdown("  \n".join(meta_lines))
                st.markdown(snippet)
        else:
            st.markdown(f"- {item}")

def _render_agent_steps(steps: list) -> None:
    """Render agent tool call steps as a readable numbered list."""
    if not steps:
        st.caption("No tool calls recorded.")
        return

    for i, step in enumerate(steps, start=1):
        tool_name = step.get("tool", "unknown")
        args = step.get("args", {})
        output = step.get("output_snippet", "")

        with st.expander(f"Step {i}: `{tool_name}`", expanded=False):
            # Arguments
            st.markdown("**Arguments**")
            arg_lines = [f"- `{k}`: {v}" for k, v in args.items() if v is not None and v != ""]
            if arg_lines:
                st.markdown("\n".join(arg_lines))
            else:
                st.caption("_(no arguments)_")

            # Output snippet
            if output:
                st.markdown("**Evidence retrieved** _(truncated)_")
                st.code(output, language=None)
            else:
                st.caption("_(no output recorded)_")


def render_technical_trace(technical_trace: dict | None) -> None:
    trace = as_dict(technical_trace)
    if not trace:
        return

    agent_steps = trace.get("agent_steps")

    st.markdown("### Technical Trace")

    if agent_steps:
        # Agent mode: show a human-readable tool call breakdown first.
        enriched_q = (trace.get("route") or {}).get("enriched_query")
        if enriched_q:
            st.caption(f"Enriched query sent to agent: _{enriched_q}_")

        st.markdown(
            f"The agent called **{len(agent_steps)} tool(s)** to gather evidence:"
        )
        _render_agent_steps(agent_steps)

        with st.expander("Raw trace JSON", expanded=False):
            st.json({k: v for k, v in trace.items() if k != "agent_steps"})
    else:
        # Deterministic mode: plain JSON trace.
        with st.expander("Show retrieval and routing trace", expanded=False):
            st.json(trace)

def render_download_panel(response: dict) -> None:
    json_bytes = json.dumps(response, indent=2).encode("utf-8")
    st.download_button(
        label="Download response JSON",
        data=json_bytes,
        file_name="query_response.json",
        mime="application/json",
        use_container_width=False,
    )

def render_query_response(response: dict) -> None:
    status = response.get("status", "UNKNOWN")
    review_result = as_dict(response.get("review_result"))
    render_status_banner(status)
    render_review_result(review_result)
    render_answer_block(response)
    render_review_guidance(response.get("review_guidance"))
    render_sources(response.get("sources"))
    render_evidence_preview(response.get("evidence_preview"))
    render_notes(response.get("notes"))
    render_evidence(response.get("evidence"))
    render_technical_trace(response.get("technical_trace"))
    render_download_panel(response)