"""
Page: Compare Filings

Ask the same question against two different filing scopes side-by-side.
Useful for comparing disclosures across years, forms, or companies.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FRONTEND_ROOT = Path(__file__).resolve().parents[1]
if str(_FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_ROOT))

import streamlit as st
from frontend.api_client import APIClientError, get_options, submit_query
from frontend.components.response_renderer import render_answer_block, render_evidence_preview
from frontend.config import API_BASE_URL, APP_TITLE

st.set_page_config(
    page_title=f"Compare Filings — {APP_TITLE}",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scope_selector(options_payload: dict, key_prefix: str, label: str) -> dict:
    """Render a compact scope selector (company / form / year) and return selections."""
    st.markdown(f"#### {label}")

    companies_raw = options_payload.get("companies", [])
    ticker_options = [
        str(c["ticker"])
        for c in companies_raw
        if isinstance(c, dict) and c.get("ticker")
    ]

    if not ticker_options:
        st.warning("No filings available.")
        return {"company": None, "form_folder": None, "year": None}

    company = st.selectbox("Company", ticker_options, key=f"{key_prefix}_company")

    company_record = next(
        (c for c in companies_raw if isinstance(c, dict) and str(c.get("ticker")) == company),
        {},
    )
    available_forms = company_record.get("available_forms", {})
    form_options = sorted(available_forms.keys()) if isinstance(available_forms, dict) else []

    form = st.selectbox("Form type", form_options, key=f"{key_prefix}_form") if form_options else None

    year_options: list[int] = []
    if form and isinstance(available_forms, dict):
        raw = available_forms.get(form, [])
        year_options = sorted([int(y) for y in raw if str(y).isdigit()])

    year = (
        st.selectbox("Year", year_options, index=len(year_options) - 1, key=f"{key_prefix}_year")
        if year_options
        else None
    )

    return {"company": company, "form_folder": form, "year": year}


def _status_badge(status: str) -> str:
    s = str(status or "").upper()
    if s == "PASS":
        return "✅ PASS"
    if s == "REFUSE":
        return "⚠️ REFUSE"
    return f"ℹ️ {s}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Backend")
    st.code(API_BASE_URL, language="text")
    st.markdown(
        "Compare the same question against two different filing scopes. "
        "Useful for year-over-year or company-to-company comparisons."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Compare Filings")
st.caption(
    "Ask the same regulatory question against two different filings and see the answers side-by-side."
)
st.divider()

# ---------------------------------------------------------------------------
# Load options
# ---------------------------------------------------------------------------
try:
    options_payload = get_options()
except APIClientError as exc:
    st.error(f"Could not load filing options from backend: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Scope selectors
# ---------------------------------------------------------------------------
scope_col_a, div_col, scope_col_b = st.columns([5, 1, 5])

with scope_col_a:
    scope_a = _scope_selector(options_payload, key_prefix="a", label="Scope A")

with div_col:
    st.markdown("<div style='text-align:center; margin-top: 3.5rem; font-size: 1.4rem;'>vs</div>", unsafe_allow_html=True)

with scope_col_b:
    scope_b = _scope_selector(options_payload, key_prefix="b", label="Scope B")

st.divider()

# ---------------------------------------------------------------------------
# Shared query
# ---------------------------------------------------------------------------
st.markdown("### Question")
query = st.text_area(
    label="Question",
    label_visibility="collapsed",
    placeholder="e.g. Where does the company discuss cybersecurity risk?",
    height=100,
    key="compare_query",
)

compare_clicked = st.button(
    "Compare",
    type="primary",
    disabled=not query.strip(),
)

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------
if compare_clicked and query.strip():
    missing_a = not scope_a["company"] or not scope_a["form_folder"] or scope_a["year"] is None
    missing_b = not scope_b["company"] or not scope_b["form_folder"] or scope_b["year"] is None

    if missing_a and missing_b:
        st.warning("Please select a valid scope for at least one side.")
    else:
        result_a, result_b = None, None
        error_a, error_b = None, None

        with st.spinner("Running both queries..."):
            # Query A
            if not missing_a:
                try:
                    result_a = submit_query(
                        query=query.strip(),
                        company=scope_a["company"],
                        form_folder=scope_a["form_folder"],
                        year=scope_a["year"],
                    )
                except APIClientError as exc:
                    error_a = str(exc)

            # Query B
            if not missing_b:
                try:
                    result_b = submit_query(
                        query=query.strip(),
                        company=scope_b["company"],
                        form_folder=scope_b["form_folder"],
                        year=scope_b["year"],
                    )
                except APIClientError as exc:
                    error_b = str(exc)

        st.session_state["compare_result_a"] = result_a
        st.session_state["compare_result_b"] = result_b
        st.session_state["compare_error_a"] = error_a
        st.session_state["compare_error_b"] = error_b
        st.session_state["compare_query_text"] = query.strip()
        st.session_state["compare_scope_a"] = scope_a
        st.session_state["compare_scope_b"] = scope_b

# ---------------------------------------------------------------------------
# Render side-by-side results
# ---------------------------------------------------------------------------
result_a = st.session_state.get("compare_result_a")
result_b = st.session_state.get("compare_result_b")
error_a = st.session_state.get("compare_error_a")
error_b = st.session_state.get("compare_result_b")  # intentional for display

if result_a is not None or result_b is not None or error_a or error_b:
    last_query = st.session_state.get("compare_query_text", "")
    saved_scope_a = st.session_state.get("compare_scope_a", scope_a)
    saved_scope_b = st.session_state.get("compare_scope_b", scope_b)

    st.divider()
    st.markdown(f"**Query:** _{last_query}_")
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        label_a = f"{saved_scope_a.get('company')} · {saved_scope_a.get('form_folder')} · {saved_scope_a.get('year')}"
        st.markdown(f"### Scope A — {label_a}")
        if error_a:
            st.error(f"Request failed: {error_a}")
        elif result_a:
            status_a = result_a.get("status", "UNKNOWN")
            st.markdown(f"**Status:** {_status_badge(status_a)}")

            confidence = (result_a.get("review_result") or {}).get("confidence", "—")
            evidence_strength = (result_a.get("review_result") or {}).get("evidence_strength", "—")
            m1, m2 = st.columns(2)
            m1.metric("Confidence", confidence)
            m2.metric("Evidence", evidence_strength)

            render_answer_block(result_a)
            render_evidence_preview(result_a.get("evidence_preview", []))
        else:
            st.info("No scope A was provided.")

    with col_b:
        label_b = f"{saved_scope_b.get('company')} · {saved_scope_b.get('form_folder')} · {saved_scope_b.get('year')}"
        st.markdown(f"### Scope B — {label_b}")
        if st.session_state.get("compare_error_b"):
            st.error(f"Request failed: {st.session_state['compare_error_b']}")
        elif result_b:
            status_b = result_b.get("status", "UNKNOWN")
            st.markdown(f"**Status:** {_status_badge(status_b)}")

            confidence = (result_b.get("review_result") or {}).get("confidence", "—")
            evidence_strength = (result_b.get("review_result") or {}).get("evidence_strength", "—")
            m1, m2 = st.columns(2)
            m1.metric("Confidence", confidence)
            m2.metric("Evidence", evidence_strength)

            render_answer_block(result_b)
            render_evidence_preview(result_b.get("evidence_preview", []))
        else:
            st.info("No scope B was provided.")
