from __future__ import annotations
import streamlit as st
from frontend.api_client import APIClientError, get_options, submit_agent_query, submit_query
from frontend.config import API_BASE_URL, APP_TITLE, DOCUMENTATION_URL, LAYOUT
from frontend.state import initialize_state
from frontend.styles import inject_base_styles
from frontend.components.scope_panel import render_scope_panel
from frontend.components.response_renderer import render_query_response

st.set_page_config(
    page_title=APP_TITLE,
    layout=LAYOUT,
)
initialize_state()
inject_base_styles()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Backend")
    st.code(API_BASE_URL, language="text")
    st.caption("Override with env var `RAG_API_BASE_URL` before starting Streamlit.")

    st.markdown("### Engine")
    use_agent = st.toggle(
        "Use LangChain Agent",
        value=False,
        help=(
            "When ON, queries are routed to the LangChain Deep Agent "
            "(/api/v1/agent-query) which autonomously calls retrieval tools "
            "and synthesises answers. Traces are sent to LangSmith.\n\n"
            "When OFF, the deterministic engine (/api/v1/query) is used."
        ),
    )
    if use_agent:
        st.caption("Agent mode: autonomous tool-calling via LangChain + LangSmith tracing.")
    else:
        st.caption("Deterministic mode: rule-based retrieval and structured answers.")

    st.markdown("### Documentation")
    st.link_button("How the app works (browser)", DOCUMENTATION_URL, use_container_width=True)
    st.caption("Architecture, query flow, storage, and compliance notes.")
    with st.expander("Example questions"):
        st.markdown(
            """
**Rules only**
- What does 17 CFR 229.105 require?
- Explain Item 303 MD&A disclosure requirements.

**Filing-scoped** (pick company, form, year first)
- Where are risk factors disclosed?
- Where is Management's Discussion and Analysis?
- Where does the company discuss cybersecurity risk?
"""
        )

    # Query history
    history: list = st.session_state.get("query_history", [])
    if history:
        st.markdown("### Recent queries")
        for entry in reversed(history[-8:]):
            status = entry.get("status", "?")
            icon = "✅" if status == "PASS" else "⚠️" if status == "REFUSE" else "ℹ️"
            q_text = entry.get("query", "")
            short_q = q_text[:50] + ("…" if len(q_text) > 50 else "")
            scope_label = entry.get("scope_label", "")
            with st.container():
                st.markdown(f"{icon} `{status}` — {short_q}")
                if scope_label:
                    st.caption(scope_label)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Regulatory Reporting Assistant")
st.caption("Evidence-first answers grounded in SEC filings and regulatory rules.")

try:
    options_payload = get_options()
except APIClientError as exc:
    st.error(f"Failed to load available scope options from backend: {exc}")
    st.stop()
with st.container():
    scope = render_scope_panel(options_payload)

st.markdown("### Ask a question")
query = st.text_area(
    label="Query",
    label_visibility="collapsed",
    placeholder="Example: Where does Apple disclose CEO compensation?",
    height=140,
)

action_col, info_col = st.columns([1, 5])

with action_col:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)
with info_col:
    st.caption(
        "The frontend displays backend-grounded output and does not reinterpret answer logic."
    )

_submit_fn = submit_agent_query if use_agent else submit_query
_spinner_msg = (
    "Agent reasoning over evidence (this may take longer)..."
    if use_agent
    else "Retrieving grounded response..."
)


def _record_history(q: str, response: dict, scope_label: str) -> None:
    """Append an entry to the query history (max 20 entries)."""
    entry = {
        "query": q,
        "status": response.get("status", "UNKNOWN"),
        "scope_label": scope_label,
        "engine": "agent" if use_agent else "deterministic",
    }
    history = st.session_state.get("query_history", [])
    history.append(entry)
    st.session_state["query_history"] = history[-20:]


if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question before submitting.")
    elif scope.get("mode") == "rules_only":
        with st.spinner(_spinner_msg):
            try:
                response = _submit_fn(
                    query=query.strip(),
                    company=None,
                    form_folder=None,
                    year=None,
                )
                st.session_state["last_query"] = query.strip()
                st.session_state["last_response"] = response
                _record_history(query.strip(), response, "Rules only")
            except APIClientError as exc:
                st.error(f"Query request failed: {exc}")
    elif not scope.get("company") or not scope.get("form_folder") or scope.get("year") is None:
        st.warning("Please select a valid company, form, and year.")
    else:
        scope_label = f"{scope['company']} · {scope['form_folder']} · {scope['year']}"
        with st.spinner(_spinner_msg):
            try:
                response = _submit_fn(
                    query=query.strip(),
                    company=scope["company"],
                    form_folder=scope["form_folder"],
                    year=scope["year"],
                )
                st.session_state["last_query"] = query.strip()
                st.session_state["last_response"] = response
                _record_history(query.strip(), response, scope_label)
            except APIClientError as exc:
                st.error(f"Query request failed: {exc}")

response = st.session_state.get("last_response")
if response:
    st.divider()
    render_query_response(response)
else:
    st.info(
        "Choose **Rules only** or a **filing scope** from backend options, enter a question, and click **Ask**."
    )
