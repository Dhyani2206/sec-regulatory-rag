# SEC Regulatory RAG

Deterministic, evidence-first regulatory question answering over SEC filings and rule text.

## Overview

SEC Regulatory RAG is a grounded regulatory assistant built with FastAPI and Streamlit. It is designed to answer filing and rule-related questions conservatively, using retrieved evidence and preserving refusal behavior when strong support is not available.

The system supports:

- rule queries
- filing section queries
- semantic topic queries
- grounded refusals
- evidence preview
- technical trace for debugging and review

## Architecture

The project is split into two main layers:

### 1. FastAPI backend
The backend is the canonical system of truth. It handles:

- scope validation
- query routing
- rule retrieval
- filing retrieval
- evidence ranking
- answer generation
- grounded refusal behavior

### 2. Streamlit frontend
The frontend is a thin client over the backend. It:

- loads valid filing scope options
- sends queries to the backend
- renders answers, review results, evidence preview, and technical trace
- does not move business logic out of the backend

## Key Principles

- Deterministic and evidence-first
- Thin API integration
- Options-driven filing scope
- Professional answer presentation
- Preserved refusal behavior
- No frontend hallucination shortcuts

## API

### `GET /api/v1/options`
Returns available filing scope options derived from real filing metadata.

### `POST /api/v1/query`
Returns a structured response such as:

- `status`
- `review_result`
- `answer`
- `plain_english_explanation`
- `why_it_matters`
- `review_guidance`
- `sources`
- `evidence_preview`
- `technical_trace`
- `notes`
- `evidence`

## Project Structure

```text
SEC_REGULATORY/
├── app/                  # FastAPI app and API layer
├── src/                  # Core RAG and retrieval logic
├── frontend/             # Streamlit frontend
├── storage/              # Runtime retrieval artifacts (not fully committed)
├── data/                 # Data inputs and processing assets
├── tests/                # Test cases
├── run.py
├── requirements.txt
└── README.md
