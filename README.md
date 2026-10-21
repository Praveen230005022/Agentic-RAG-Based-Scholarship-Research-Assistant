# Agentic RAG-Based Scholarship Research Assistant

An AI assistant that researches a scholarship on the web, compiles the findings into a
cited PDF report, and answers follow-up questions with a retrieval-augmented (RAG) chatbot
that only uses the collected evidence.

> **Status: starter codebase (v0.1).** The architecture, RAG pipeline, PDF reports, demo
> mode and tests are implemented. Live research and LLM answers need API keys (see
> [What works without API keys](#what-works-without-api-keys)).

## Architecture

```
Streamlit UI (app.py)
        │
        ▼
ScholarshipAssistant  (services.py — application controller)
        │
   ┌────┴──────────────────────────┐
   ▼                               ▼
ResearchAgent (research_agent.py)  RAGPipeline (rag_pipeline.py)
   │ plan queries                  │ follow-up-aware query
   │ search (Tavily, pluggable)    │ per-scholarship FAISS retrieval
   │ rank: official sources first  │ cosine-similarity threshold
   │ read pages / PDFs (robots.txt)│ grounded prompt → Groq LLM
   │ extract fields (Groq / rules) │ answer + numbered citations
   │ gap-filling follow-up search  │ or explicit "insufficient evidence"
   │ conflict + verification checks│
   ▼                               │
document_processor.py ──► chunks + metadata ──► embeddings ──► FAISS index (one per scholarship)
   │
   ▼
pdf_generator.py ──► downloadable PDF report (ReportLab)
```

See [`docs/architecture.md`](docs/architecture.md) for the data flow and design decisions.

## Module responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI: Research, Ask Questions and Sources tabs, setup status, demo mode |
| `src/scholarship_assistant/config.py` | Settings from environment / `.env`, validation, per-backend defaults |
| `src/scholarship_assistant/models.py` | Dataclasses: sources, reports, verification status, retrieval and chat results |
| `src/scholarship_assistant/research_agent.py` | Search-provider interface, source classification, bounded research loop, extraction, conflict and deadline checks |
| `src/scholarship_assistant/document_processor.py` | PDF/HTML/text loading, LangChain documents, chunking with provenance metadata |
| `src/scholarship_assistant/rag_pipeline.py` | Embeddings, per-scholarship FAISS index management, retrieval, grounded answers |
| `src/scholarship_assistant/pdf_generator.py` | ReportLab report with verification labels, page numbers, clickable links |
| `src/scholarship_assistant/services.py` | Controller used by the UI; turns failures into user-facing errors |
| `src/scholarship_assistant/utils.py` | Slugs, timestamps, sentence splitting, logging |
| `data/sample/` | Two **fictional** sample scholarships for demo mode |
| `data/eval/eval_questions.json` | Small evaluation set (questions + expected supporting source) |
| `tests/` | Offline pytest suite (no keys or network needed) |

## Technology stack

Python · LangChain (LCEL, text splitters, FAISS integration) · Groq (`llama-3.3-70b-versatile` by default) ·
Tavily search · Sentence-Transformers `all-MiniLM-L6-v2` · FAISS · PyMuPDF · BeautifulSoup · ReportLab ·
Streamlit · pytest

## Prerequisites

- Python 3.10 or newer (developed on 3.13)
- About 1 GB of disk for dependencies (PyTorch CPU build) plus ~90 MB for the MiniLM model,
  downloaded automatically from Hugging Face on first use
- Optional API keys: [Groq](https://console.groq.com/keys) and [Tavily](https://app.tavily.com)

## Installation

```bash
git clone <your-repo-url> scholarship-assistant
cd scholarship-assistant

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Optional but recommended without an NVIDIA GPU: CPU-only PyTorch (much smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

## Configuration

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Then edit `.env`:

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | LLM answers and structured extraction | not set |
| `TAVILY_API_KEY` | Live web research | not set |
| `GROQ_MODEL` | Groq model ID ([current list](https://console.groq.com/docs/models)) | `llama-3.3-70b-versatile` |
| `EMBEDDING_BACKEND` | `sentence-transformers`, or `hashing` for fully offline use | `sentence-transformers` |
| `MAX_SEARCH_QUERIES` / `MAX_RESULTS_PER_QUERY` / `MAX_PAGES` | Bounds on research cost and time | 4 / 5 / 8 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking in characters | 1000 / 150 |
| `TOP_K` | Passages retrieved per question | 4 |
| `RELEVANCE_THRESHOLD` | Minimum cosine similarity to count as evidence | 0.35 (MiniLM), 0.15 (hashing) |

Values that still start with `your_` are treated as unset. Never commit `.env`.

## Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501. Without a Tavily key the app starts in **demo mode**: pick a
sample scholarship, click **Research**, then use the **Ask Questions** and **Sources** tabs.

## Run the tests

```bash
pytest                                   # offline suite (uses hashing embeddings)
RUN_MODEL_TESTS=1 pytest -k minilm       # also test the real MiniLM model (downloads it)
```

On Windows PowerShell: `$env:RUN_MODEL_TESTS=1; pytest -k minilm`.

## What works without API keys

| Feature | No keys | Groq only | Tavily only | Both |
|---|---|---|---|---|
| App UI, setup status, demo mode | ✅ | ✅ | ✅ | ✅ |
| Sample research → FAISS index → PDF report | ✅ | ✅ | ✅ | ✅ |
| Chat retrieval with citations and insufficient-evidence detection | ✅ | ✅ | ✅ | ✅ |
| Chat answers | retrieved passages | LLM answers | retrieved passages | LLM answers |
| Live web research | ❌ | ❌ | ✅ rule-based extraction | ✅ LLM extraction |

The MiniLM embedding model is downloaded from Hugging Face on first use. If that is not
possible, set `EMBEDDING_BACKEND=hashing` (word-overlap vectors, lower retrieval quality).

## How the assistant stays honest

- Official government (`.gov`, `.gov.in`, `.nic.in`, …) and provider sources are read first; Wikipedia is background only and never a source for amounts, steps or terms.
- Each report field is labelled **Verified (official source)**, **Secondary source only**, **Conflicting** or **Not verified**.
- Different amounts or deadlines across sources are flagged as conflicts.
- Past deadlines are marked as probably belonging to a previous cycle.
- If no retrieved passage passes the similarity threshold, the LLM is **not called** and the user gets an explicit insufficient-evidence reply.
- The research agent respects `robots.txt`, does not bypass logins or 401/403 responses, and stays within query/page limits.
- Sample data is fictional, uses `example.org` URLs and is labelled in the UI and PDF.

## Known limitations

- Rule-based extraction (no Groq key) is keyword-driven and can include irrelevant sentences.
- Conflict detection is heuristic (maximum benefit amount and deadline dates per source).
- Pages that need JavaScript to render may return little text.
- FAISS indexes are local files; the app stores them under `data/indexes/` and loads only indexes it wrote itself.
- `langchain-community` (used for the FAISS integration) is being sunset by LangChain and prints a deprecation warning; it still works. Swap in a standalone FAISS integration when one is published.
- Not yet tested against live Groq/Tavily APIs in this repository's test suite (all external calls are mocked).

## Planned extensions

- LLM-generated search plans and query reformulation
- Hybrid retrieval (BM25 + vectors) and re-ranking
- Scheduled re-research to refresh deadlines
- Evaluation with answer-faithfulness metrics on real scholarships
- Deployment (Streamlit Community Cloud) with secrets management

## Origin

This project started from [krishnaik06/AWS-Bedrock](https://github.com/krishnaik06/AWS-Bedrock)
("Chat with PDF using AWS Bedrock"). Its ingestion → splitting → FAISS → "say you don't know"
prompt flow was kept and rebuilt; see `docs/architecture.md` for details.

## License

The original repository is licensed under GPL-3.0, and this project is derived from it, so it is distributed under the same license (see `LICENSE`).
