from __future__ import annotations
import streamlit as st
from frontend.api_client import APIClientError, get_options, submit_query
from frontend.config import APP_TITLE, LAYOUT
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

st.title("Regulatory Reporting Assistant")
st.caption("Deterministic, evidence-first answers grounded in filings and rules.")

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

if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question before submitting.")
    elif not scope["company"] or not scope["form_folder"] or scope["year"] is None:
        st.warning("Please select a valid company, form, and year.")
    else:
        with st.spinner("Retrieving grounded response..."):
            try:
                response = submit_query(
                    query=query.strip(),
                    company=scope["company"],
                    form_folder=scope["form_folder"],
                    year=scope["year"],
                )
                st.session_state["last_query"] = query.strip()
                st.session_state["last_response"] = response
            except APIClientError as exc:
                st.error(f"Query request failed: {exc}")

response = st.session_state.get("last_response")
if response:
    st.divider()
    render_query_response(response)
else:
    st.info(
        "Choose a valid filing scope from backend-provided options, enter a question, and submit."
    )