"""
Page: Document Browser

Browse the indexed SEC filing corpus by company, form type, and year.
Select any section to see actual text chunks as they appear in the index.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the frontend package is importable when Streamlit runs from the pages/ dir.
_FRONTEND_ROOT = Path(__file__).resolve().parents[1]
if str(_FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_ROOT))

import streamlit as st
from frontend.api_client import APIClientError, browse_chunks, browse_sections, get_options
from frontend.config import API_BASE_URL, APP_TITLE

st.set_page_config(
    page_title=f"Document Browser — {APP_TITLE}",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Backend")
    st.code(API_BASE_URL, language="text")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Document Browser")
st.caption(
    "Browse all indexed SEC filings. Select a company, form, and year to see "
    "the sections available, then drill into any section to read the actual text chunks."
)
st.divider()

# ---------------------------------------------------------------------------
# Scope selector
# ---------------------------------------------------------------------------
try:
    options_payload = get_options()
except APIClientError as exc:
    st.error(f"Could not load filing options from backend: {exc}")
    st.stop()

companies_raw = options_payload.get("companies", [])
ticker_options = [
    str(c["ticker"])
    for c in companies_raw
    if isinstance(c, dict) and c.get("ticker")
]

if not ticker_options:
    st.warning("No filings are indexed yet. Build the corpus first.")
    st.stop()

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    selected_company = st.selectbox("Company", ticker_options, key="browser_company")

# Find the record for the selected company.
company_record = next(
    (c for c in companies_raw if isinstance(c, dict) and str(c.get("ticker")) == selected_company),
    {},
)
available_forms: dict = company_record.get("available_forms", {})
form_options = sorted(available_forms.keys()) if isinstance(available_forms, dict) else []

with col2:
    if form_options:
        selected_form = st.selectbox("Form type", form_options, key="browser_form")
    else:
        selected_form = None
        st.warning("No forms available.")

year_options: list[int] = []
if selected_form and isinstance(available_forms, dict):
    raw_years = available_forms.get(selected_form, [])
    year_options = sorted([int(y) for y in raw_years if str(y).isdigit()])

with col3:
    if year_options:
        selected_year = st.selectbox(
            "Year",
            year_options,
            index=len(year_options) - 1,
            key="browser_year",
        )
    else:
        selected_year = None
        st.warning("No years available.")

with col4:
    st.markdown("&nbsp;")  # vertical alignment spacer
    browse_clicked = st.button(
        "Browse filing",
        type="primary",
        use_container_width=True,
        disabled=(selected_form is None or selected_year is None),
    )

# ---------------------------------------------------------------------------
# Section list
# ---------------------------------------------------------------------------
if browse_clicked and selected_form and selected_year:
    with st.spinner(f"Loading sections for {selected_company} {selected_form} {selected_year}..."):
        try:
            browse_data = browse_sections(selected_company, selected_form, selected_year)
            st.session_state["browser_data"] = browse_data
            st.session_state["browser_scope"] = (selected_company, selected_form, selected_year)
            st.session_state["browser_chunks"] = {}
            st.session_state["browser_section_page"] = {}
        except APIClientError as exc:
            st.error(f"Browse request failed: {exc}")
            st.session_state.pop("browser_data", None)

browse_data = st.session_state.get("browser_data")
browser_scope = st.session_state.get("browser_scope")

if browse_data and browser_scope:
    company_s, form_s, year_s = browser_scope
    sections = browse_data.get("sections", [])
    total_sections = browse_data.get("total_sections", len(sections))

    st.markdown(
        f"### {company_s} · {form_s} · {year_s} — "
        f"**{total_sections} section{'s' if total_sections != 1 else ''}** indexed"
    )

    # Optional section filter.
    filter_text = st.text_input(
        "Filter sections",
        placeholder="e.g. ITEM 1A",
        label_visibility="collapsed",
    ).strip().upper()

    filtered = [
        s for s in sections
        if not filter_text or filter_text in s.get("section", "").upper()
    ]

    if not filtered:
        st.info("No sections match the filter.")
    else:
        for sec in filtered:
            section_label = sec.get("section", "UNKNOWN")
            title = sec.get("title") or ""
            chunk_count = sec.get("chunk_count", 0)
            preview = sec.get("preview_text") or "_No preview available._"

            header = f"**{section_label}**"
            if title:
                header += f" — {title}"
            header += f"  `{chunk_count} chunk{'s' if chunk_count != 1 else ''}`"

            with st.expander(header, expanded=False):
                st.markdown("**Preview**")
                st.markdown(preview)

                chunks_key = f"chunks_{company_s}_{form_s}_{year_s}_{section_label}"
                page_key = f"page_{chunks_key}"

                if st.button(
                    f"Load all chunks for {section_label}",
                    key=f"load_{chunks_key}",
                    use_container_width=False,
                ):
                    st.session_state[page_key] = 1
                    with st.spinner("Loading chunks..."):
                        try:
                            result = browse_chunks(
                                company_s,
                                form_s,
                                year_s,
                                section_label,
                                page=1,
                                page_size=10,
                            )
                            st.session_state[chunks_key] = result
                        except APIClientError as exc:
                            st.error(f"Could not load chunks: {exc}")

                chunks_data = st.session_state.get(chunks_key)
                if chunks_data:
                    total_chunks = chunks_data.get("total_chunks", 0)
                    current_page = st.session_state.get(page_key, 1)
                    page_size = 10
                    total_pages = max(1, (total_chunks + page_size - 1) // page_size)

                    st.caption(
                        f"{total_chunks} chunk{'s' if total_chunks != 1 else ''} · "
                        f"Page {current_page} of {total_pages}"
                    )

                    for i, chunk in enumerate(chunks_data.get("chunks", []), start=1):
                        chunk_id = chunk.get("chunk_id") or f"chunk_{i}"
                        chunk_title = chunk.get("title") or ""
                        text = chunk.get("text") or "_Empty chunk._"

                        with st.container(border=True):
                            badge = f"`{chunk_id}`"
                            if chunk_title:
                                badge += f"  _{chunk_title}_"
                            st.markdown(badge)
                            st.markdown(text)

                    # Pagination controls.
                    if total_pages > 1:
                        prev_col, page_col, next_col = st.columns([1, 2, 1])
                        with prev_col:
                            if current_page > 1 and st.button(
                                "← Previous", key=f"prev_{chunks_key}"
                            ):
                                new_page = current_page - 1
                                st.session_state[page_key] = new_page
                                try:
                                    result = browse_chunks(
                                        company_s, form_s, year_s, section_label,
                                        page=new_page, page_size=page_size,
                                    )
                                    st.session_state[chunks_key] = result
                                except APIClientError as exc:
                                    st.error(str(exc))
                                st.rerun()
                        with page_col:
                            st.caption(f"Page {current_page} / {total_pages}")
                        with next_col:
                            if current_page < total_pages and st.button(
                                "Next →", key=f"next_{chunks_key}"
                            ):
                                new_page = current_page + 1
                                st.session_state[page_key] = new_page
                                try:
                                    result = browse_chunks(
                                        company_s, form_s, year_s, section_label,
                                        page=new_page, page_size=page_size,
                                    )
                                    st.session_state[chunks_key] = result
                                except APIClientError as exc:
                                    st.error(str(exc))
                                st.rerun()
