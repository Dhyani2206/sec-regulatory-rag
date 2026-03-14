# SEC Regulatory RAG

Deterministic, evidence-first regulatory question answering over SEC filings and rule text.

## Overview

SEC Regulatory RAG is a **domain-specialized Retrieval-Augmented Generation (RAG) system** designed for regulatory research and compliance intelligence. The system retrieves evidence from **SEC filings (10-K / 10-Q)** and **CFR regulatory rules**, ranks the evidence, and generates grounded answers using an LLM.

Instead of relying purely on language model reasoning, the system enforces an **evidence-first approach**, ensuring answers are supported by primary regulatory sources. When sufficient evidence cannot be retrieved, the system preserves **refusal behavior** to avoid unsupported responses.

The system supports:

* rule queries
* filing section queries
* semantic topic queries
* grounded refusals
* evidence preview
* technical trace for debugging and review

## Architecture

The system follows a **Deep RAG pipeline for regulatory compliance intelligence**, integrating frontend interaction, backend orchestration, retrieval pipelines, and structured answer generation.

### System Architecture

![RAG Architecture](docs/rag_pipeline.png)

### End-to-End Pipeline

```text
User Question
      ↓
Frontend UI (Web App)
      ↓
Query Classification & Routing
(Filing Query vs Rule Query)

      ↓                        ↓
Retrieve Filing Data    Retrieve Rule Data
(SEC Filings FAISS)     (CFR Rules FAISS)

      ↓                        ↓
Get Filing Snippets     Get Rule Snippets

            ↓
     Evidence Guard
(Citation + Hallucination Check)

           ↓
     LLM Answer Generation

           ↓
      Structured Response

• Final Answer
• Filing Evidence
• Rule Evidence
```

## The project is split into two main layers:

### 1. FastAPI backend

The backend is the **canonical system of truth** and implements the core regulatory RAG engine. It handles:

* scope validation
* query routing
* rule retrieval
* filing retrieval
* FAISS vector search
* evidence ranking
* answer synthesis
* grounded refusal behavior
* citation enforcement

The backend ensures that responses are **generated only from retrieved regulatory evidence**.

### 2. Streamlit frontend

The frontend acts as a **thin client over the backend API**. It:

* loads valid filing scope options
* sends user queries to the backend
* renders structured responses
* displays evidence previews
* shows technical traces for debugging
* preserves backend decision logic

All regulatory reasoning and retrieval logic remains inside the backend.

## Key Principles

The system is built around several core design principles:

* Deterministic and evidence-first
* Thin API integration
* Options-driven filing scope
* Professional answer presentation
* Preserved refusal behavior
* No frontend hallucination shortcuts
* Regulatory evidence grounding

## API

The backend exposes a small set of endpoints for interacting with the regulatory RAG engine.

### `GET /api/v1/options`

Returns available filing scope options derived from real filing metadata.

Example response includes:

* available companies
* filing forms (10-K / 10-Q)
* available years

These options are used by the frontend to construct valid regulatory queries.

### `POST /api/v1/query`

Accepts a user query and optional filing scope and returns a structured regulatory answer.

The response includes fields such as:

* `status`
* `review_result`
* `answer`
* `plain_english_explanation`
* `why_it_matters`
* `review_guidance`
* `sources`
* `evidence_preview`
* `technical_trace`
* `notes`
* `evidence`

This structured output allows both **human interpretation and automated analysis**.

## Data Architecture

The system maintains **two independent vector retrieval indexes** using FAISS.

### Filing Vector Database

Used for retrieving relevant sections from SEC filings.

```text
storage/
├── chunks.jsonl
├── embeddings.npy
└── faiss.index
```

### Regulatory Rules Vector Database

Used for retrieving relevant CFR rule text.

```text
storage/
├── rules_chunks.jsonl
├── rules_vectors.npy
└── faiss_rules.index
```

Both indexes enable **fast semantic retrieval across regulatory corpora**.

## Project Structure

```text
SEC_REGULATORY/
├── app/                  # FastAPI app and API layer
├── src/                  # Core RAG and retrieval logic
├── frontend/             # Streamlit frontend
├── storage/              # Runtime retrieval artifacts (not fully committed)
├── data/                 # Data inputs and processing assets
├── tests/                # Test cases
├── docs/                 # Architecture diagrams and documentation
├── run.py
├── requirements.txt
└── README.md
```

## Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the backend

```bash
uvicorn app:app --reload
```

### Launch the frontend

```bash
streamlit run frontend/app.py
```

The system will then be accessible through the Streamlit interface.

## Example Queries

Examples of supported regulatory queries:

```text
What risk factor disclosure is required under SEC rules?

What risks does Microsoft report in its 2024 10-K?

Explain disclosure requirements under Regulation S-K.

What are the major risk factors reported by Visa in its 2023 10-Q?
```

## Project Status

### Completed

* Regulatory data ingestion
* SEC filing parsing
* CFR rule extraction
* FAISS vector indexing
* Query routing and classification
* Dual retrieval pipelines
* Evidence ranking
* Evidence guard (citation + hallucination control)
* FastAPI backend
* Streamlit interface

### Next Steps

Future development directions include:

* regulatory research agents
* compliance analysis workflows
* regulatory change monitoring
* automated compliance reporting


MIT License
