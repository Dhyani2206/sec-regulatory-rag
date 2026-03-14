# Intelligent Regulatory Review Assistant — Architecture Map

## 1. System Purpose

This project is a **deterministic, evidence-first regulatory RAG system** for SEC reporting.

It is designed to:

- Answer SEC rule questions
- Locate disclosure sections in company filings
- Retrieve filing and rule evidence
- Rank and package evidence
- Refuse unsupported answers
- Remain audit-friendly and low-hallucination

This system is **not a free-form chatbot**.  
It is a **structured retrieval system** with explicit routing, retrieval, ranking, and refusal logic.

---

# 2. Core Design Principles

The backend is intentionally:

- Deterministic
- Evidence-first
- Low-hallucination
- Refusal-safe
- Audit-friendly
- FastAPI-wrapped but backend-driven

**Important architectural rule**

> The FastAPI layer is thin.  
> The canonical answer logic lives in `src/rag/answer_engine.py`.

---

# 3. Canonical Production Path
Client / Frontend / Swagger
↓
FastAPI
↓
app/services/query_service.py
↓
src/rag/answer_engine.py
↓
query_router.py / rules_router.py
↓
retrieve_filing_sections.py / retrieve_rules.py / filing_evidence.py
↓
evidence_ranker.py / evidence_pack.py
↓
hallucination / refusal guard
↓
structured response


This is the **real production serving path**.

---

# 4. System Layers

## API Layer
Handles HTTP requests and responses.

## Service Layer
Validates scope and formats API responses.

## RAG Runtime Layer
Handles routing, retrieval, ranking, and answer generation.

## Storage Layer
Contains filing chunks, rule chunks, and FAISS indexes.

## Observability Layer
Captures debugging logs and traces.

---

# 5. FastAPI Layer

## `app/main.py`

**Role**

FastAPI application entrypoint.

**Responsibilities**

- Create FastAPI app
- Register routers
- Start API service

**Connected To**
app/api/v1/router.py


---

## `app/api/v1/router.py`

---

## `app/api/v1/router.py`

**Role**

Top-level API router.

**Responsibilities**

- Mount `/api/v1`
- Register endpoint groups

**Connected To**
endpoints/query.py
endpoints/options.py
endpoints/health.py
endpoints/ready.py


---

## `app/api/v1/endpoints/query.py`

**Role**

Main query endpoint.

**Endpoint**
POST/api/v1/query


---

## `app/api/v1/endpoints/options.py`

**Role**

Options endpoint.

**Endpoint**

GET/api/v1/options


**Responsibilities**

Return available:

- tickers
- forms
- years

**Connected To**
schemas/options.py
services/options_service.py


---

## `app/api/v1/endpoints/health.py`

**Role**

Liveness check endpoint.

---

## `app/api/v1/endpoints/ready.py`

**Role**

Readiness check endpoint.

---

# 6. API Schemas

## `app/api/v1/schemas/query.py`

Defines API contract.

### Key Models

- `QueryRequest`
- `QueryResponse`
- `ReviewResult`
- `EvidenceItem`
- `EvidencePreviewItem`
- `SourceItem`
- `TechnicalTrace`

These are used for **frontend rendering and API responses**.

---

## `app/api/v1/schemas/options.py`

Defines schema for options endpoint.

Used for:

- ticker list
- forms
- years

---

# 7. Service Layer

## `app/services/query_service.py`

**Role**

API integration layer.

**Responsibilities**

- Resolve ticker from query text
- Validate scope
- Call `answer_engine.py`
- Normalize evidence
- Build final response

**Connected To**
src/rag/answer_engine.py
app/services/options_service.py


**Important**

This layer **must not contain retrieval logic**.

---

## `app/services/options_service.py`

**Role**

Source of truth for available filing scope.

**Responsibilities**

- Read filing metadata
- Extract valid ticker/form/year combinations
- Validate scope
- Power `/api/v1/options`

**Connected To**
storage/chunks.jsonl


---

# 8. Canonical RAG Runtime Layer

## `src/rag/answer_engine.py`

**Most important file in the system**

**Role**

Canonical answer orchestrator.

**Responsibilities**

- Detect rule-only queries
- Route filing queries
- Retrieve evidence
- Rank evidence
- Apply hallucination guard
- Apply refusal logic
- Produce structured response

**Connected To**
query_router.py
rules_router.py
retrieve_filing_sections.py
retrieve_rules.py
filing_evidence.py
evidence_ranker.py
evidence_pack.py
debug_logging.py


---

# 9. Query Routing

## `src/rag/query_router.py`

**Role**

Routes filing questions.

**Query Types**
structural_section_lookup
semantic_topic_lookup
not_applicable


**Examples**

| Topic | 10-K Section |
|------|--------------|
Risk Factors | Item 1A  
MD&A | Item 7  
Legal Proceedings | Item 3  

| Topic | 10-Q Section |
|------|--------------|
Risk Factors | Item 1A  
MD&A | Item 2  
Legal Proceedings | Item 1  

---

## `src/rag/rules_router.py`

**Role**

Routes regulatory rule questions.

**Responsibilities**

- Detect CFR citations
- Route rule retrieval
- Support semantic fallback

---

# 10. Retrieval Modules

## `src/rag/retrieve_rules.py`

**Role**

Retrieve SEC rule evidence.

**Storage**
storage/rules_chunks.jsonl
storage/faiss_rules.index


---

## `src/rag/retrieve_filing_sections.py`

**Role**

Retrieve filing sections using semantic search.

**Storage**
storage/faiss.index
storage/chunks.jsonl


---

## `src/rag/filing_evidence.py`

**Role**

Load filing evidence.

**Responsibilities**

- Load chunks
- Filter by ticker/form/year
- Filter by section

**Storage**
storage/chunks.jsonl


---

# 11. Evidence Processing

## `src/rag/evidence_ranker.py`

Ranks evidence based on:

- relevance
- section priority
- rule citations

---

## `src/rag/evidence_pack.py`

Packages evidence into a structured format.

Used by the answer engine.

---

# 12. Observability

## `src/rag/debug_logging.py`

Logs:

- routing decisions
- retrieval attempts
- fallback behavior
- refusal events

Logs stored in:
outputs/logs/


---

# 13. Storage Layer

## Filing Corpus

### `storage/chunks.jsonl`

Contains:

- ticker
- form
- year
- section
- chunk text
- metadata

Used by:

- filing retrieval
- options service
- evidence loading

---

### `storage/faiss.index`

Vector index for filings.

Used for semantic retrieval.

---

## Rules Corpus

### `storage/rules_chunks.jsonl`

Contains:

- rule citations
- rule text
- headings

---

### `storage/faiss_rules.index`

Vector index for rules.

---

# 14. Legacy Files (Do Not Use)

These exist in the repository but **must not be used in production**.

## `src/rag/dual_retrieve.py`

Legacy retrieval helper.

---

## `src/rag/rag_answer.py`

Old answer wrapper.

---

# 15. Public API Endpoints

## Health
GET/api/v1/health
## READY
GET/api/v1/ready
## OPTIONS
GET/api/v1/options

Returns:
- ticker
- forms
- years

## Query
POST/api/v1/query

Handles:
- rule questions
- filing questions
- refusal cases

# 16. Query Flow Examples
## rule Query 
Example: What does 17 CFR 229.105 require?
Flow
API -> query_service
-> answer_engine
-> rules_router
-> retrieve_rules
-> evidence_ranker
-> response

## Filing Structural Query
Example: Where are the risk factors disclosed in AAPL 2024 10-k
Flow
API -> query_service
-> answer_engine
-> rules_router
-> retrieve_rules
-> evidence_ranker
-> response


---

# 17. Engineering Rules

1. Never move core logic out of `answer_engine.py`.
2. Keep FastAPI routes thin.
3. Do not hardcode frontend scope options.
4. Always use `/api/v1/options`.
5. Preserve refusal behavior.
6. Never rely on FAISS index to determine valid scope.
7. `chunks.jsonl` is the source of truth.

---

# 18. Short Summary

| Component                  | Purpose                    |
|----------------------------|----------------------------|
`answer_engine.py`           | canonical answer pipeline  |
`query_router.py`            | filing routing             |
`rules_router.py`            | rule routing               |
`retrieve_filing_sections.py`| semantic filing retrieval  |
`retrieve_rules.py`          | rule retrieval             |
`filing_evidence.py`         | filing evidence loader     |
`evidence_ranker.py`         | evidence ranking           |
`evidence_pack.py`           | evidence packaging         |
`query_service.py`           | API integration            |
`options_service.py`         | ticker/form/year discovery |
`chunks.jsonl`               | filing corpus              |
`rules_chunks.jsonl`         | rules corpus               |
`faiss.index`                | filing vector index        |
`faiss_rules.index`          | rule vector index          |

---
