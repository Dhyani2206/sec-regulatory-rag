from __future__ import annotations
import streamlit as st

def initialize_state() -> None:
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = None
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = None
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []  # list of {query, status, scope}