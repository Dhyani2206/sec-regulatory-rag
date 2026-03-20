# How the SEC Regulatory RAG application works

This document explains the **end-to-end behavior** of the system: what each layer does, how queries are processed, and how to run and configure it. It supports **audit and compliance review** by describing evidence-first design explicitly.

---

## 1. What the system does

The application answers **regulatory and SEC filing questions** using:

- **Retrieved evidence** from two indexes: **SEC filings** (10-K / 10-Q chunks) and **CFR / Regulation S-K rule** text.
- **Deterministic routing and guards** so answers are not invented when evidence is weak.

It is **not** a general chatbot: the backend **prefers refusal or conservative answers** when grounding is insufficient.

---

## 2. Architecture overview

```text
┌─────────────────┐     HTTP      ┌──────────────────┐
│  Streamlit UI   │ ────────────► │  FastAPI backend │
│  (frontend/)    │               │  (app/)          │
└─────────────────┘               └────────┬─────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
            ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
            │ Query service │       │ Options API   │       │ Lifespan      │
            │ + schemas     │       │ (valid scope) │       │ Artifact      │
            └───────┬───────┘       └───────────────┘       │ bootstrap     │
                    │                                       └───────────────┘
                    ▼
            ┌───────────────┐
            │ Answer engine │  ◄── src/rag/answer_engine.py
            │ (no LLM for   │      Routing, retrieval, ranking, guards
            │  final text)  │
            └───────┬───────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
 ┌──────────┐ ┌──────────┐ ┌──────────┐
 │ Filing   │ │ Rules    │ │ Azure    │
 │ FAISS +  │ │ FAISS +  │ │ OpenAI   │
 │ chunks   │ │ chunks   │ │ embeddings│
 └──────────┘ └──────────┘ └──────────┘
        ▲           ▲
        └───────────┴── storage/ (indexes, jsonl, npy)
```

- **Frontend** only displays what the API returns; it does **not** re-interpret regulatory logic.
- **Backend** owns all routing, retrieval, evidence ranking, and refusal behavior.
- **Embeddings** at query time use **Azure OpenAI** (see `.env`), aligned with how indexes were built.

---

## 3. Main components

| Layer | Location | Role |
|--------|----------|------|
| **API** | `app/main.py`, `app/api/v1/` | HTTP endpoints, request/response models, startup checks |
| **Query orchestration** | `app/services/query_service.py` | Scope validation, maps engine output to API response (review labels, evidence preview, etc.) |
| **Options** | `app/services/options_service.py` | Builds allowed **company / form / year** from `storage/chunks.jsonl` |
| **Artifact bootstrap** | `app/services/artifact_bootstrap.py` | On startup, ensures required `storage/` files exist or downloads them from URLs in env vars |
| **Answer engine** | `src/rag/answer_engine.py` | Core logic: rule vs filing paths, structural vs semantic retrieval, hallucination-style guards |
| **Retrieval** | `src/rag/retrieve_filings.py`, `retrieve_rules.py`, `retrieve_filing_sections.py`, etc. | FAISS search + optional query expansion / section boosts |
| **Embeddings** | `src/rag/embeddings.py` | Single entry point for **Azure OpenAI** embedding calls |
| **Frontend** | `frontend/app.py`, `frontend/frontend/` | Scope panel (**Rules only** vs **Filing-scoped**), query box, response rendering |

---

## 4. API endpoints (summary)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v1/health` | Liveness |
| `GET` | `/api/v1/ready` | Readiness |
| `GET` | `/api/v1/options` | Valid filing scopes for the UI |
| `POST` | `/api/v1/query` | Submit a question + optional scope (deterministic engine) |
| `POST` | `/api/v1/agent-query` | Submit a question to the **LangChain Deep Agent** |
| `GET` | `/documentation` | **This guide** rendered as HTML (root URL) |
| `GET` | `/api/v1/documentation` | **Same guide** (use if only `/api/v1/*` is exposed) |

Interactive OpenAPI docs: `/docs` (Swagger UI).

---

## 5. How a query is processed

### 5.1 Request body (`POST /api/v1/query`)

- **`query`** (required): Natural language question.
- **`company`**, **`form_folder`**, **`year`** (optional): Filing scope. If all three are omitted or null, the backend can still answer **rule-only** questions.

The Streamlit app sends **no scope** when you choose **Rules only**; it sends all three when you choose **Filing-scoped**.

### 5.2 Scope validation

- If company, form, and year are provided, they must match a combination returned by **`GET /api/v1/options`**. Otherwise the API returns **`REFUSE`** with finding **`invalid_scope`** (no retrieval run).
- Partial scope may be filled conservatively from the query text in some cases (see `query_service`); invalid combinations are still blocked.

### 5.3 Answer engine paths (high level)

1. **Rule-only detection**  
   If the question looks like a **CFR / Item / rule** question, the engine primarily retrieves **rules** evidence. With enough evidence, status is **`PASS`** and the answer summarizes **grounded rule text**. Explicit citation mismatches can trigger **`REFUSE`**.

2. **Missing filing scope**  
   If the question is **not** rule-only but no complete filing scope is available, the engine returns **`REFUSE`** with guidance to provide company, form, and year.

3. **Filing-scoped**  
   - **Structural routing**: e.g. “where are risk factors?” maps to a target section (e.g. Item 1A) and loads **deterministic** chunks for that section.  
   - **Semantic topic routing**: e.g. “cybersecurity” may use **embedding search** over sections/chunks, then guards to avoid weak matches.  
   - **Not applicable** routes (e.g. wrong form for the question type) return **`REFUSE`** with a structural reason.

4. **Guards**  
   Before returning **`PASS`**, the engine checks that evidence meets minimum thresholds for the detected **intent** (rule-only, structural section lookup, semantic topic, etc.).

### 5.4 Response shape

The API returns structured fields including **`status`**, **`answer`**, **`review_result`**, **`evidence`**, **`evidence_preview`**, **`technical_trace`** (route and guard reason), and **`notes`**. This supports **human review** and **downstream tooling**.

---

## 5b. LangChain Deep Agent + LangSmith

### What the agent does differently

The **deterministic engine** (`/api/v1/query`) follows a fixed routing tree and never calls a generative LLM to produce answer text. It is fast and highly reproducible.

The **LangChain Deep Agent** (`/api/v1/agent-query`) is a `CompiledStateGraph` created with `langchain.agents.create_agent`. It receives the user question (and optional filing scope) as a message, autonomously decides which tools to call, and synthesises a final answer from the retrieved evidence.

### Tool descriptions

| Tool | Module | What it does |
|------|--------|--------------|
| `retrieve_filing_chunks` | `src/rag/agent/tools.py` | Semantic FAISS search over SEC filing chunks; supports company/form/year scope |
| `retrieve_rule_chunks` | `src/rag/agent/tools.py` | Citation-aware FAISS search over CFR/Regulation S-K rule chunks; prioritises exact "17 CFR x.x" matches |
| `load_filing_section` | `src/rag/agent/tools.py` | Deterministic lookup of all chunks for a known company/form/year/section (no embedding call) |

The agent decides autonomously which tools to call, how many times, and in what order. It can combine filing evidence and rule evidence in a single answer.

### LangSmith tracing

LangSmith is zero-code: you only need three environment variables:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<from smith.langchain.com>
LANGCHAIN_PROJECT=sec-regulatory-rag
```

With these set, every agent invocation is automatically traced: tool call sequence, token usage, latency, and full messages are visible in the LangSmith dashboard at https://smith.langchain.com.

The `LANGCHAIN_API_KEY` placeholder in `.env` is `ls__your-key-here`; replace it with your real key from smith.langchain.com. Without a valid key, tracing is silently disabled and the agent still works.

### Agent vs deterministic engine — when to use each

| | Deterministic engine | LangChain Agent |
|-|----------------------|-----------------|
| Speed | Fast (no LLM call for answer text) | Slower (LLM + multi-tool loop) |
| Answer style | Structured, template-driven | Free-form, LLM-synthesised |
| Tracing | Not traced | LangSmith traces |
| Multi-step reasoning | Not supported | Supported |
| Reproducibility | Fully deterministic | Non-deterministic (LLM sampling) |
| Compliance suitability | High (grounded, auditable) | Moderate — human review required |

Both endpoints return the same `QueryResponse` schema, so the Streamlit UI handles both transparently.

---

## 6. Storage artifacts (`storage/`)

Runtime files (large; often **not** in git) include:

| File | Role |
|------|------|
| `chunks.jsonl` | Filing text chunks + metadata |
| `embeddings.npy` | Filing embedding matrix (build time) |
| `faiss.index` | Filing vector index |
| `rules_chunks.jsonl` | Rule chunks |
| `rules_vectors.npy` | Rule embedding matrix |
| `faiss_rules.index` | Rules vector index |

If a file is missing at startup, **`artifact_bootstrap`** can download it when the matching `*_URL` environment variable is set (see `app/services/artifact_bootstrap.py`).

---

## 7. Configuration (`.env`)

**Azure OpenAI** (embeddings + agent LLM):

| Variable | Purpose |
|----------|---------|
| `AZURE_OPENAI_API_KEY` | Azure key for embeddings and the agent LLM |
| `AZURE_OPENAI_ENDPOINT` | Azure Cognitive Services endpoint URL |
| `AZURE_OPENAI_API_VERSION` | API version, e.g. `2024-12-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | Chat deployment name used by the agent (e.g. `gpt-5-chat`) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding deployment name — **must match** how indexes were built |

**LangSmith** (optional, agent tracing only):

| Variable | Purpose |
|----------|---------|
| `LANGCHAIN_TRACING_V2` | Set to `true` to enable tracing |
| `LANGCHAIN_API_KEY` | Key from smith.langchain.com |
| `LANGCHAIN_PROJECT` | Project name shown in the LangSmith dashboard |

**Frontend → API URL:**

- `RAG_API_BASE_URL` (default `http://localhost:8000`)

---

## 8. How to run locally

1. Create a venv and `pip install -r requirements.txt`.
2. Place `.env` with Azure (and optional artifact URLs) in the **project root**.
3. Ensure `storage/` contains required artifacts (build pipelines under `src/rag/build/` and `src/rules/`, or download URLs).
4. Start API: `python run.py` (or `uvicorn app.main:app --reload`).
5. Start UI: `streamlit run frontend/app.py`.

**Automated checks:** `pytest tests/test_api_smoke.py -v` (requires artifacts; embedding test needs Azure).

---

## 9. Compliance-oriented notes

- **Evidence-first:** Answers are tied to retrieved chunks/rules; weak evidence leads to **refusal** or **manual review** messaging.
- **Traceability:** `technical_trace`, citations, chunk ids, and structured evidence support **audit trails**.
- **No legal advice:** Disclaimers in API notes and UI copy remind users this is **not** a substitute for legal or compliance judgment.

---

## 10. Further reading

- `README.md` — setup, examples, project map  
- `ARCHITECTURE_MAP.md` (if present) — deeper module map  
- OpenAPI **`/docs`** — request/response schemas  

For questions about changing behavior, start with **`answer_engine.py`** and **`query_service.py`**, then the retrieval modules under **`src/rag/`**.
