# SEC Regulatory RAG

A deterministic, evidence-first regulatory reporting assistant built with FastAPI and Streamlit.

## Overview

This project provides a grounded regulatory question-answering system over SEC filing content and regulatory rule text.  
It is designed to prioritize evidence-backed answers, support valid filing scope selection, and preserve grounded refusal behavior when strong support is not available.

The system includes:

- a FastAPI backend for query and options APIs
- a Streamlit frontend for interactive use
- a deterministic retrieval pipeline for filings and rules
- evidence ranking, evidence preview, and technical trace support
- evaluation, QC, and reporting utilities

## Main Features

- Evidence-first regulatory Q&A
- Filing scope selection by company, form, and year
- Rule queries, filing section queries, and semantic topic queries
- Grounded refusal behavior
- Technical trace for debugging and review
- Streamlit frontend for demo and product presentation

## Project Structure

```text
SEC_REGULATORY/
├── app/                  # FastAPI application
├── src/                  # Core RAG, extraction, and rules logic
├── frontend/             # Streamlit frontend
├── storage/              # Retrieval artifacts and metadata
├── data/                 # Source and processed data
├── outputs/              # Reports, logs, and evaluation outputs
├── tests/                # Test cases
├── run.py
└── requirements.txt