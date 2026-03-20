"""
Page: Corpus Stats

Dashboard showing aggregate statistics for the indexed SEC filing and rules corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FRONTEND_ROOT = Path(__file__).resolve().parents[1]
if str(_FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_ROOT))

import pandas as pd
import streamlit as st
from frontend.api_client import APIClientError, get_corpus_stats, get_options
from frontend.config import API_BASE_URL, APP_TITLE

st.set_page_config(
    page_title=f"Corpus Stats — {APP_TITLE}",
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
st.title("Corpus Statistics")
st.caption("Aggregate view of all indexed SEC filings and regulatory rules.")
st.divider()

# ---------------------------------------------------------------------------
# Load stats
# ---------------------------------------------------------------------------
with st.spinner("Loading corpus statistics..."):
    try:
        stats = get_corpus_stats()
    except APIClientError as exc:
        st.error(f"Could not load corpus statistics: {exc}")
        st.stop()

# ---------------------------------------------------------------------------
# Top-level metric tiles
# ---------------------------------------------------------------------------
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Companies", stats.get("total_companies", 0))
m2.metric("Unique filings", stats.get("total_filings", 0))
m3.metric("Filing chunks", f"{stats.get('total_filing_chunks', 0):,}")
m4.metric("Rule chunks", f"{stats.get('total_rule_chunks', 0):,}")
m5.metric("CFR citations", stats.get("total_rule_citations", 0))
m6.metric("CFR parts", stats.get("total_rule_parts", 0))

st.divider()

# ---------------------------------------------------------------------------
# Chunks per company bar chart
# ---------------------------------------------------------------------------
chunks_per_company: dict = stats.get("chunks_per_company", {})
if chunks_per_company:
    st.markdown("### Filing chunks per company")
    df_bar = pd.DataFrame(
        {"Company": list(chunks_per_company.keys()), "Chunks": list(chunks_per_company.values())}
    ).sort_values("Chunks", ascending=False)
    st.bar_chart(df_bar.set_index("Company"))

# ---------------------------------------------------------------------------
# Indexed filings table
# ---------------------------------------------------------------------------
try:
    options_payload = get_options()
    companies_raw = options_payload.get("companies", [])
except APIClientError:
    companies_raw = []

if companies_raw:
    st.markdown("### Indexed filings")

    rows = []
    for company in companies_raw:
        ticker = company.get("ticker", "")
        available_forms: dict = company.get("available_forms", {})
        if not isinstance(available_forms, dict):
            continue
        for form, years in sorted(available_forms.items()):
            for year in sorted(years):
                rows.append({"Company": ticker, "Form": form, "Year": int(year)})

    if rows:
        df_filings = pd.DataFrame(rows).sort_values(["Company", "Form", "Year"])
        st.dataframe(
            df_filings,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Company": st.column_config.TextColumn("Company", width="small"),
                "Form": st.column_config.TextColumn("Form", width="small"),
                "Year": st.column_config.NumberColumn("Year", format="%d", width="small"),
            },
        )
    else:
        st.info("No filing index data available.")

# ---------------------------------------------------------------------------
# Year coverage
# ---------------------------------------------------------------------------
years_covered: list = stats.get("years_covered", [])
if years_covered:
    st.divider()
    st.markdown("### Years covered")
    st.markdown(
        " · ".join(f"`{y}`" for y in sorted(years_covered))
    )

# ---------------------------------------------------------------------------
# Forms indexed
# ---------------------------------------------------------------------------
forms: list = stats.get("forms", [])
companies: list = stats.get("companies", [])

info_col1, info_col2 = st.columns(2)
with info_col1:
    if forms:
        st.markdown("### Form types")
        for f in sorted(forms):
            st.markdown(f"- `{f}`")

with info_col2:
    if companies:
        st.markdown("### Companies")
        st.markdown(", ".join(f"`{c}`" for c in sorted(companies)))
