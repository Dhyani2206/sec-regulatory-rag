# SEC Regulatory RAG

> Deterministic, evidence-first regulatory question answering over SEC filings and CFR rules — built with FastAPI, Streamlit, LangChain, and Azure OpenAI.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-orange)](https://www.langchain.com/)

---

## Overview

SEC Regulatory RAG is a **domain-specialized Retrieval-Augmented Generation (RAG) system** designed for regulatory research and compliance intelligence. The system retrieves evidence from **SEC filings (10-K / 10-Q)** and **CFR regulatory rules**, ranks the evidence, and generates grounded answers using an LLM.

Instead of relying purely on language model reasoning, the system enforces an **evidence-first approach**, ensuring answers are supported by primary regulatory sources. When sufficient evidence cannot be retrieved, the system preserves **refusal behavior** to avoid unsupported responses.

### Supported query types

| Query type | Example |
|---|---|
| CFR / Rule citation | `What does 17 CFR 229.105 require?` |
| Filing section lookup | `Where does AAPL discuss cybersecurity in its 2025 10-K?` |
| Semantic topic | `What liquidity risks does JPM disclose in its 2024 10-K?` |
| Year-over-year comparison | Compare AAPL cybersecurity disclosure 2024 vs 2025 |
| LangChain Deep Agent | Multi-step tool-chaining across filings and rules |

---

## Architecture

The system follows a **Deep RAG pipeline for regulatory compliance intelligence**, integrating frontend interaction, backend orchestration, retrieval pipelines, and structured answer generation.

### End-to-End Pipeline

```
User Question
      ↓
Frontend UI (Streamlit)
      ↓
FastAPI Backend — Query Router
      ↓                        ↓
 Rule-only path          Filing-scoped path
      ↓                        ↓
 CFR Rules FAISS         SEC Filings FAISS
 (rules_chunks.jsonl)    (chunks.jsonl)
      ↓                        ↓
 Rule snippets           Section ranking
                              ↓
                         Chunk evidence
                  ↓
           Evidence Guard
   (Citation + Hallucination Check)
                  ↓
     Deterministic answer  OR  LangChain Deep Agent
                  ↓
         Structured Response
          • Final Answer
          • Evidence Preview
          • Sources
          • Technical Trace
```

### Two main layers

#### 1. FastAPI backend (`app/`, `src/`)

The backend is the **canonical system of truth** and implements the core regulatory RAG engine. It handles:

- Scope validation and query routing
- Rule retrieval (CFR / Regulation S-K)
- Filing retrieval with year-accurate scoping
- FAISS vector search over two independent indexes
- Evidence ranking and citation enforcement
- Hallucination guard — refusal when evidence is insufficient
- LangChain Deep Agent with Azure OpenAI + direct RAG fallback
- Corpus browsing and statistics endpoints

#### 2. Streamlit frontend (`frontend/`)

The frontend is a **thin client over the backend API**. It:

- Loads valid filing scope options from the backend
- Supports both deterministic engine and LangChain agent modes
- Renders structured responses, evidence previews, and technical traces
- Provides four dedicated pages:
  - **Main** — query interface with rules / filing-scoped modes
  - **Document Browser** — navigate indexed filings by company / form / year / section
  - **Compare Filings** — side-by-side year-over-year or company-to-company comparison
  - **Corpus Statistics** — aggregate metrics, charts, and indexed filing table

All regulatory reasoning and retrieval logic remains inside the backend.

---

## Project Structure

```
sec-regulatory-rag/
├── app/                          # FastAPI application
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   ├── agent_query.py    # LangChain agent endpoint
│   │   │   ├── browse.py         # Corpus browsing endpoints
│   │   │   ├── documentation.py  # In-browser docs endpoint
│   │   │   └── query.py          # Deterministic query endpoint
│   │   ├── schemas/
│   │   │   ├── browse.py
│   │   │   └── query.py
│   │   └── router.py
│   ├── services/
│   │   ├── agent_query_service.py
│   │   ├── browse_service.py
│   │   ├── options_service.py
│   │   └── query_service.py
│   └── main.py
│
├── src/rag/                      # Core retrieval and reasoning
│   ├── agent/                    # LangChain Deep Agent
│   │   ├── agent.py              # Agent factory (create_agent + AzureChatOpenAI)
│   │   └── tools.py              # @tool wrappers for retrieval functions
│   ├── build/                    # Index building scripts
│   ├── answer_engine.py          # Deterministic answer workflow
│   ├── embeddings.py             # Azure OpenAI embedding client
│   ├── filing_evidence.py        # Year-scoped chunk retrieval
│   ├── retrieve_filing_sections.py
│   ├── retrieve_rules.py
│   └── llm_prompts.py
│
├── src/rules/                    # CFR rules corpus and index builders
│
├── frontend/
│   ├── app.py                    # Main Streamlit page
│   ├── pages/
│   │   ├── 1_Document_Browser.py
│   │   ├── 2_Compare_Filings.py
│   │   └── 3_Corpus_Stats.py
│   └── frontend/
│       ├── api_client.py
│       ├── config.py
│       ├── state.py
│       └── components/
│
├── storage/                      # Runtime FAISS indexes and chunk data
│   ├── chunks.jsonl
│   ├── embeddings.npy
│   ├── faiss.index
│   ├── rules_chunks.jsonl
│   ├── rules_vectors.npy
│   └── faiss_rules.index
│
├── data/                         # Raw data and processing assets
├── docs/                         # Application guide and architecture docs
├── tests/                        # Pytest smoke tests
├── run.py                        # Backend entry point
├── pytest.ini
└── requirements.txt
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/options` | Available companies, forms, and years |
| `POST` | `/api/v1/query` | Deterministic RAG query |
| `POST` | `/api/v1/agent-query` | LangChain Deep Agent query |
| `GET` | `/api/v1/browse` | List sections for a filing scope |
| `GET` | `/api/v1/browse/chunks` | Paginated chunks for a section |
| `GET` | `/api/v1/corpus-stats` | Aggregate corpus statistics |
| `GET` | `/api/v1/documentation` | Rendered application guide |

### `POST /api/v1/query` — response fields

| Field | Description |
|-------|-------------|
| `status` | `PASS` / `REFUSE` / `ERROR` |
| `review_result` | Finding, label, confidence, evidence strength |
| `answer` | Grounded regulatory answer |
| `plain_english_explanation` | Non-technical summary of how the answer was produced |
| `why_it_matters` | Compliance relevance note |
| `review_guidance` | Bullet-point human review checklist |
| `sources` | Deduplicated source list |
| `evidence_preview` | Top evidence snippets with scores |
| `technical_trace` | Route, guard reason, agent steps |
| `notes` | Retrieval and safety notes |
| `evidence` | Full evidence list with chunk metadata |

---

## Data Architecture

The system maintains **two independent FAISS vector indexes**.

### Filing index

```
storage/
├── chunks.jsonl        # Parsed SEC filing chunks (company, year, section, text)
├── embeddings.npy      # Pre-computed chunk embeddings
└── faiss.index         # FAISS flat index
```

### Regulatory rules index

```
storage/
├── rules_chunks.jsonl  # Parsed CFR rule text (citation, heading, text)
├── rules_vectors.npy   # Pre-computed rule embeddings
└── faiss_rules.index   # FAISS flat index
```

### Indexed companies and years

| Ticker | Available years |
|--------|----------------|
| AAPL | 2021 – 2025 |
| AMZN | 2021 – 2025 |
| BA | 2021 – 2025 |
| BAC | 2021 – 2025 |
| JPM | 2021 – 2025 |
| MSFT | 2021 – 2025 |
| PFE | 2021 – 2025 |
| PYPL | 2021 – 2025 |
| VISA | 2021 – 2025 |
| WMT | 2021 – 2025 |
| XOM | 2021 – 2025 |

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- Azure OpenAI account with:
  - A **chat** deployment (e.g. `gpt-4o`)
  - An **embedding** deployment (e.g. `text-embedding-3-large`)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/ankitjawla/sec-regulatory-rag.git
cd sec-regulatory-rag
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root (never commit this file):

```env
# Azure OpenAI — required
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=<your-chat-deployment>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<your-embedding-deployment>

# LangSmith tracing — optional
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-key>
LANGCHAIN_PROJECT=sec-regulatory-rag
```

### 4. Build or restore storage artifacts

The `storage/` FAISS indexes and chunk files are not committed to git. Either:

**Option A — Build locally** (requires raw data in `data/`):

```bash
# Build filing corpus
python src/rag/build/build_corpus.py
python src/rag/build/build_index.py

# Build rules corpus
python src/rules/build_rules_corpus.py
python src/rules/build_rules_index.py
```

**Option B — Download via environment URLs**: set `CHUNKS_JSONL_URL`, `FAISS_INDEX_URL`, etc. in `.env` as documented in `app/services/artifact_bootstrap.py`. The API will download them automatically on first startup.

### 5. Start the backend

```bash
source venv/bin/activate
python run.py
# or: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend available at: `http://localhost:8000`
Interactive API docs: `http://localhost:8000/docs`

### 6. Launch the frontend

```bash
source venv/bin/activate
streamlit run frontend/app.py --server.port 8501
```

Frontend available at: `http://localhost:8501`

---

## Example Queries

### Rule queries

```
What does 17 CFR 229.105 require?
What are the disclosure requirements under Regulation S-K Item 303?
Explain 17 CFR 229.106 cybersecurity disclosure rules.
```

### Filing-scoped queries

```
Where does AAPL discuss cybersecurity risk in its 2025 10-K?
What liquidity risks does JPM report in its 2024 10-K?
Where does AMZN disclose supply chain risk factors?
What does MSFT say about legal proceedings in its 2023 10-K?
```

### Comparison queries (Compare Filings page)

```
Compare AAPL cybersecurity disclosures: 2024 vs 2025
Compare risk factor language: JPM 2023 vs JPM 2024
```

---

## Running Tests

```bash
source venv/bin/activate
pytest tests/test_api_smoke.py -v
```

Tests that require storage artifacts or Azure credentials are automatically skipped when those are not available.

---

## Documentation

- **In-browser** (while backend is running): [http://localhost:8000/documentation](http://localhost:8000/documentation)
- **In the repository**: [`docs/application-guide.md`](docs/application-guide.md)

---

## Contributing

Contributions are welcome. Please follow the steps below to keep the codebase consistent and auditable.

### Development workflow

#### 1. Fork and clone

```bash
git clone https://github.com/<your-username>/sec-regulatory-rag.git
cd sec-regulatory-rag
git remote add upstream https://github.com/ankitjawla/sec-regulatory-rag.git
```

#### 2. Create a feature branch

Always branch off `main`. Use a descriptive prefix:

| Prefix | Use for |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code restructuring without behaviour change |
| `test/` | Adding or improving tests |

```bash
git checkout main
git pull upstream main
git checkout -b feat/your-feature-name
```

#### 3. Set up the environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your Azure credentials
```

#### 4. Make your changes

Follow these conventions:

- **Evidence-first**: new answer paths must surface retrieved evidence — never return answers from LLM memory alone.
- **Refusal behavior**: preserve the existing guard logic — if evidence is insufficient, the system must refuse rather than hallucinate.
- **Backend is canonical**: regulatory reasoning lives in `src/` and `app/`. The frontend is a thin client and must not duplicate business logic.
- **Docstrings**: every new module and public function should have a docstring explaining its purpose, inputs, and outputs.
- **No hard-coded secrets**: use environment variables loaded via `python-dotenv`.
- **Type hints**: use them throughout; avoid `Any` unless necessary.

#### 5. Test your changes

```bash
pytest tests/test_api_smoke.py -v
```

Start both the backend and frontend and manually test the affected query paths before opening a PR.

#### 6. Commit with a clear message

Follow the format below. Keep the subject line under 72 characters.

```
<type>(<scope>): <short summary>

<body — what changed and why>
```

Examples:

```
fix(filing_evidence): skip form filter when chunk has no form metadata

Chunks in chunks.jsonl carry form=None. The previous equality check
normalize_form(None) == normalize_form("10-K") always evaluated to False,
filtering out all filing evidence and returning only rule chunks.
```

```
feat(agent): add direct RAG fallback when tool-calling is unsupported

Some Azure deployments output tool-call JSON as plain text content instead
of using the function-calling API. This adds a fallback that calls retrieval
functions directly and synthesises an answer via LLM when the agent calls
zero tools.
```

#### 7. Open a pull request

Push your branch and open a PR against `main`:

```bash
git push origin feat/your-feature-name
```

In the PR description, include:
- **What** the change does
- **Why** it is needed
- **How** you tested it
- Any **evidence screenshots** or API response samples if applicable

#### 8. Code review

- Respond to review comments within the PR thread
- Do not force-push after a review has started — add new commits instead
- Once approved, squash-merge into `main`

### Reporting issues

Open a [GitHub Issue](https://github.com/ankitjawla/sec-regulatory-rag/issues) with:
- A clear title
- Steps to reproduce
- Expected vs actual behaviour
- Relevant logs or API responses

---

## Project Status

### Completed

- Regulatory data ingestion (10-K filings for 11 companies, 2021–2025)
- CFR rules extraction (Regulation S-K, selected 17 CFR rules)
- FAISS vector indexing — filing and rules indexes
- Query routing and classification (rule-only, structural, semantic)
- Dual retrieval pipelines with year-accurate scoping
- Evidence ranking and hallucination guard
- Grounded refusal behavior for insufficient evidence
- FastAPI backend with full API documentation
- Streamlit multi-page frontend
- LangChain Deep Agent with Azure OpenAI + direct RAG fallback
- LangSmith tracing integration
- Document Browser, Compare Filings, Corpus Statistics pages
- API smoke tests with pytest

### Roadmap

- Regulatory change monitoring (diff across filing years)
- Automated compliance gap reporting
- Support for 10-Q filings alongside 10-K
- Expanding the corpus to additional companies and CFR parts
- Evaluation harness with ground-truth regulatory Q&A pairs
- Docker Compose deployment configuration

---

## License

This project is provided for educational and research purposes.

---

## Acknowledgements

- [SEC EDGAR](https://www.sec.gov/edgar/) — public filing data
- [eCFR](https://www.ecfr.gov/) — electronic Code of Federal Regulations
- [LangChain](https://www.langchain.com/) — agent and tool framework
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [FastAPI](https://fastapi.tiangolo.com/) — backend framework
- [Streamlit](https://streamlit.io/) — frontend framework
