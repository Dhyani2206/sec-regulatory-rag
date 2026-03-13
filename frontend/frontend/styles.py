from __future__ import annotations
import streamlit as st


def inject_base_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            padding: 0.75rem;
            border-radius: 0.75rem;
        }
        div[data-testid="stExpander"] {
            border-radius: 0.75rem;
            border: 1px solid #e5e7eb;
        }
        .small-muted {
            color: #6b7280;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )