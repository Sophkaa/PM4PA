# Process Modeling for Public Administration (PM4PA)

## Overview


PM4PA implements an **agentic retrieval-augmented generation (RAG) pipeline** that supports **document-grounded BPMN 2.0 process modeling**. Given heterogeneous text sources (e.g. regulations, handbooks, PDF or Office documents), the system **retrieves** relevant passages, **drafts** a process description, **generates** a nested BPMN JSON representation, and **validates** (and optionally **revises**) the model against the query and sources. The design targets **public administration** knowledge bases but is applicable wherever processes must be modelled from different sources.

**Source annotation** on generated process elements (e.g. links to documents, pages, or retrieved chunks in model metadata) **increases transparency and explainability**, so users can trace model content back to the underlying source document.

Interaction is supported through a **Streamlit** web interface and through **command-line** entrypoints for single runs and for **quantitative evaluation** against gold BPMN artifacts (see [src/eval/evaluation.md](src/eval/evaluation.md)). The usual way to run the web app is **[Docker](#run-with-docker)** (reproducible environment).

## Table of contents

- [Web interface](#web-interface)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Repository structure](#repository-structure)
- [License](#license)

## Web interface

**Document ingestion.** Users can upload or select source documents (PDF, Word, text, etc.); the app ingests them into the configured Chroma collection so retrieval is grounded in the corpus.

![Document Upload](images/Document%20Upload.png)

**Process modeling view.** After the documents are uploaded, the user inserts a query about the process which should be modelled based on the uploaded documents. The user can also select, if open source or closed source models should be used for generation.

![Process modeler](images/Process%20modeller.png)

**Resulting process model.** The final generated process model is displayed and can be edited and downloaded in .bpmn, .xml, .pdf, or .png format. When a modeled process element is selected, the corresponding reference aka the source document from which the process element was extracted is listed on the right, increasing transparency.

![Modelled process](images/Modelled%20process.png)


## Setup

### Prerequisites

- **Docker & Docker Compose** (or compatible tooling) if you follow [Run with Docker](#run-with-docker); **Python** 3.11 or newer if you install [locally without Docker](#local-installation-without-docker) (aligned with the `Dockerfile`).
- **Working directory:** repository root (the directory containing `src/`) for CLI modules and relative paths.
- **Network:** the runtime host or container must reach whichever **LLM** and **embedding** endpoints are configured (`OPEN_SOURCE` switches Ollama vs Azure; see [.env.example](.env.example)). Local `localhost` URLs in `.env` often need rewriting for containers (e.g. `host.docker.internal` on Docker Desktop).

### Configuration

1. Copy `.env.example` to `.env`.
2. Set endpoints, keys, `OPEN_SOURCE`, embedding behavior, timeouts, **`CHROMA_DB_PATH`** / **`CHROMA_COLLECTION_NAME`** (and optionally model names), as documented in `.env.example`. For Compose, **`CHROMA_DB_PATH`** is overridden inside the container to `/data/chroma`; keep other variables aligned with how you expose Ollama or Azure from the host or network.

### Run with Docker

Recommended for a reproducible runtime. `docker-compose.yml` mounts a named volume for Chroma (`chroma_data` → `/data/chroma`) and loads `.env`. Build context honours [`.dockerignore`](.dockerignore) so `.env` and other ignores are **not copied into image layers**. Services referenced in `.env` must be reachable **from inside the container** (often not plain `localhost` on the machine running Docker unless you bridge with `host.docker.internal` / host networking).

Build and run the Streamlit service:

```bash
docker compose up --build
```

Then open **http://localhost:8501**.

Build image only:

```bash
docker compose build
```

Run built image **without Compose** (example):

```bash
docker build -t agentic-rag .
docker run --rm -p 8501:8501 --env-file .env \
  -v agentic_chroma:/data/chroma \
  -e CHROMA_DB_PATH=/data/chroma \
  agentic-rag
```

Browser uploads are ephemeral unless your deployment persists them; the durable vector index lives at the mounted Chroma path.

### Local installation (without Docker)

Install dependencies from [`requirements.txt`](requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `.env` is not present yet: `cp .env.example .env` and edit endpoints and paths as above. Heavy stacks (e.g. local embeddings via `sentence-transformers`) use the same file; GPU is not assumed.

**Optional host tools.** For **scanned PDFs** or text in images, **Tesseract** and **Poppler** improve extraction compared to the slim Docker image, which does not include them.

#### Web UI (Streamlit)

Theme: [`src/web/.streamlit/config.toml`](src/web/.streamlit/config.toml). From `src/web` so Streamlit resolves configuration:

```bash
cd src/web
streamlit run streamlit_app.py
```

Then open **http://localhost:8501** (default).

#### CLI usage

Default pipeline graph mirrors the main web flow unless `--setting` overrides.

Ingest documents (after configuring paths/collection):

```bash
python -m src.app.ingestion
python -m src.app.ingestion --skip-existing   # skip already embedded files
```

Run BPMN generation for a query:

```bash
python -m src.app.run_request --query "The process to be modelled"
```

**Evaluation pipeline** — metrics and how to run: [src/eval/evaluation.md](src/eval/evaluation.md).

## Pipeline

End-to-end behavior, summarized:

1. **Ingestion** — Documents are parsed, chunked, embedded, and stored in **ChromaDB**.
2. **Retrieval** — The user query drives **hybrid** retrieval (e.g. dense + lexical components)
3. **Drafting** — An intermediate **process draft** is produced from retrieved context.
4. **BPMN generation** — Structured **nested BPMN JSON** is generated and is automatically submitted to an external service for XML layout.
5. **Validation and revision** — Configurable validators compare the model to sources and the query; the graph may loop for **revision** within iteration limits.

## BPMN conversion

The **external BPMN layout / conversion** step (client code in `src/bpmn_service/`) implements an approach from Safan & Köpke, 2025 [*A Framework for LLM-Based Conceptual Modeling: Application to BPMN Collaboration Diagrams*](https://ceur-ws.org/Vol-4099/forum_paper7.pdf) (Companion Proceedings of the 44th International Conference on Conceptual Modeling: Industrial Track, ER Forum, Poitiers, France, 8th SCME, 177-190). It is **not** part of drafting or nested JSON generation: it is invoked **only** to produce the **final** BPMN XML (layout and serialization) from the structured model the pipeline has already built.

## Repository structure

Application code is packaged under **`src/`**:

```
.
├── src/
│   ├── app/                    # CLI entrypoints and orchestration
│   │   ├── pipeline.py         # GraphRAGSystem: wiring, graph execution
│   │   ├── ingestion.py        # Document → Chroma ingestion CLI
│   │   └── run_request.py      # Single full pipeline run from the terminal
│   ├── agents/                 # LLM agents (draft, BPMN, retrieval, relevance, validation, judge)
│   ├── bpmn_service/           # HTTP client helpers for external BPMN layout / conversion
│   ├── eval/                   # Evaluation CLI, metrics, evaluation.md, run artifacts
│   ├── graphs/
│   │   └── pipeline_graphs.py  # Pydantic Graph nodes and get_graph_for_setting
│   ├── infrastructure/         # API clients, ingestion, retrieval, Chroma store
│   ├── models/                 # Pydantic domain models
│   ├── web/                    # Streamlit UI and .streamlit theme
│   └── config.py               # Environment-backed settings
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

**Runtime artifacts** (commonly gitignored): vector store at `CHROMA_DB_PATH` (default `./chroma_db`), evaluation outputs under `eval_output/`, optional local corpora (e.g. `data/`).

## License

This project is licensed under the [MIT License](LICENSE).
