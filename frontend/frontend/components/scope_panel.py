from __future__ import annotations

from typing import Any
import streamlit as st


def _safe_companies(options_payload: dict) -> list[dict[str, Any]]:
    companies = options_payload.get("companies", [])
    return companies if isinstance(companies, list) else []


def _find_company_record(
    companies: list[dict[str, Any]],
    ticker: str,
) -> dict[str, Any]:
    for item in companies:
        if isinstance(item, dict) and str(item.get("ticker")) == str(ticker):
            return item
    return {}


def render_scope_panel(options_payload: dict) -> dict:
    st.markdown("### Query mode")

    mode = st.radio(
        "How should the backend scope this question?",
        options=("filing", "rules_only"),
        format_func=lambda x: (
            "Filing-scoped — company, form, and year (10-K / 10-Q)"
            if x == "filing"
            else "Rules only — CFR / Regulation S-K (no filing scope)"
        ),
        horizontal=True,
        help=(
            "Use **Rules only** for questions like “What does 17 CFR 229.105 require?” "
            "Use **Filing-scoped** when the answer must refer to a specific filing."
        ),
    )

    if mode == "rules_only":
        st.info(
            "No company or year is sent to the API. The backend retrieves **regulatory rule** "
            "evidence only (deterministic, evidence-first)."
        )
        return {
            "mode": "rules_only",
            "company": None,
            "form_folder": None,
            "year": None,
        }

    st.markdown("### Filing scope")

    companies = _safe_companies(options_payload)
    ticker_options = [
        str(item["ticker"])
        for item in companies
        if isinstance(item, dict) and item.get("ticker")
    ]

    if not ticker_options:
        st.error("No company options were returned by the backend.")
        return {
            "mode": "filing",
            "company": None,
            "form_folder": None,
            "year": None,
        }

    c1, c2, c3 = st.columns(3)

    with c1:
        selected_company = st.selectbox(
            "Company",
            options=ticker_options,
            index=0,
        )

    company_record = _find_company_record(companies, selected_company)
    available_forms = company_record.get("available_forms", {})
    if not isinstance(available_forms, dict):
        available_forms = {}

    form_options = sorted(available_forms.keys())

    with c2:
        if form_options:
            selected_form = st.selectbox(
                "Form",
                options=form_options,
                index=0,
            )
        else:
            selected_form = None
            st.warning("No valid forms available for the selected company.")

    year_options: list[int] = []
    if selected_form is not None:
        raw_years = available_forms.get(selected_form, [])
        if isinstance(raw_years, list):
            year_options = sorted(
                [int(y) for y in raw_years if isinstance(y, int) or str(y).isdigit()]
            )

    with c3:
        if year_options:
            selected_year = st.selectbox(
                "Year",
                options=year_options,
                index=len(year_options) - 1,
            )
        else:
            selected_year = None
            st.warning("No valid years available for the selected company/form.")

    return {
        "mode": "filing",
        "company": selected_company,
        "form_folder": selected_form,
        "year": selected_year,
    }