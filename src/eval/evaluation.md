# Running evaluation (CLI)

This guide describes **how to execute** the evaluation command and where results are stored. For a detailed explanation of the evaluation metric pipeline, please refer to the paper. 

## What the command does (in one sentence)

For a single **query**, the pipeline generates a BPMN model using your configured **graph setting** and **Chroma** index, compares the result to a **gold JSON** (for automated metrics) and uses **gold XML** plus a **predicted BPMN XML** (from the layout service) for the LLM judge step, then writes the evaluation results under `eval_output/`.

## Before you run

1. **Working directory:** Run commands from the **repository root** (the directory that contains `src/`).
2. **Environment:** Copy and edit `.env` from `.env.example` so LLM/embeddings and `BPMN_SERVICE_URL` match your setup (same as for Streamlit or `run_request`).
3. **Vector store:** The evaluation run uses whatever is already in **Chroma** at `CHROMA_DB_PATH` with your `CHROMA_COLLECTION_NAME`. Ingest documents first (`python -m src.app.ingestion` and/or your usual data pipeline) so retrieval is meaningful.
4. **Gold artifacts:**
   - **`--gold-json`:** Reference BPMN in **JSON** form for element-level metrics.  
     - **Nested** shape: like LLM nested BPMN (see `gold_template_nested.json`).  
     - **Flat** shape: top-level keys `pools`, `lanes`, `activities`, `events`, `gateways` (see `gold_template_flat.json`).  
     The loader picks the format automatically.
   - **`--gold-xml`:** Static Reference **BPMN 2.0 XML** file used as the reference for the **LLM judge** (must stay versioned/reproducible for fair comparison).

## Pipeline settings

The **`--setting`** CLI flag is **required**. It selects which **graph pipeline** runs; the same identifiers are used in this evaluation CLI, in `python -m src.app.run_request`, and in the Streamlit app. Implementation: **`get_graph_for_setting`** in `src/graphs/pipeline_graphs.py`, which connects **Pydantic Graph** nodes (retrieval, drafting, BPMN generation, validation, revision) for the chosen variant.

| Setting | Description |
|---------|-------------|
| **`setting_1`** | **Baseline:** combined retrieval and BPMN generation in a single graph step. |
| **`setting_2`** | **Enhanced retrieval:** query-structure extraction, expansion, **BM25** over the collection, hybrid vector + lexical scoring, relevance filtering / grouping; then BPMN generation from retrieved context. |
| **`setting_3`** | **Setting 2 + structured BPMN:** same retrieval as setting 2, then **two-stage** modeling using an explicit **query_structure** in draft and BPMN agents. |
| **`setting_4`** | **Setting 3 + validate & revise:** after nested BPMN JSON is produced, **validators** compare the model to the query and chunks; the graph may **loop** and call the revision agent while `iteration_recommended` and limits allow. |
| **`setting_5`** | **Setting 3 + three validators:** separate agents for **scope/completeness**, **factual fidelity** (source alignment), and **process logic / modeling**; results are **aggregated**; optional revision loop like setting 4. |


## Command

```bash
python -m src.eval.run_evaluation \
  --query "Your evaluation prompt about which process should be modelled based on the ingested documents (same style as production)" \
  --gold-json path/to/gold.json \
  --gold-xml path/to/gold.bpmn \
  --setting setting_2 \
  --runs 1
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--query` | Yes | Natural-language task; drives retrieval and generation. |
| `--gold-json` | Yes | Path to gold JSON (nested or flat). |
| `--gold-xml` | Yes | Path to gold BPMN XML for the judge. |
| `--setting` | Yes | Pipeline graph id: **`setting_1`** … **`setting_5`**. See [Pipeline settings](#pipeline-settings). |
| `--runs` | No | Default `1`. Use a value greater than `1` to repeat the run and emit aggregated statistics. |

Paths may be **absolute** or **relative**. Relative paths are resolved against the **current shell directory** first; if the file is not found there, the resolver also tries the **repository root** (useful when you run from the repo root with paths like `eval_data/Gold_….json`).

## Console output

The script prints progress (generation, service submission, judge), then **precision / recall / F1**, **LLM judge score** (if available), timing, and a short **model configuration** summary.

## Output files

Each successful run creates a subdirectory:

```text
eval_output/<setting>_YYYYMMDD_HHMMSS/
```

Example: `eval_output/setting_2_20260421_143022/`.

Typical files:

| File | When |
|------|------|
| `meta.json` | Always: timestamp, CLI args, model configuration (subset). |
| `summary.json` | Always: dataset-level summary for the last completed run. |
| `per_sample.json` | Always: per-sample evaluation payload (includes predicted BPMN and extras when available). |
| `aggregated_statistics.json` | Only if `--runs` is greater than `1`. |
| `per_run_per_sample.json` | Only if `--runs` is greater than `1` (per-run breakdown). |

If writing these files fails, check logs for a warning; metrics may still have been printed to the terminal.

## Multiple runs

```bash
python -m src.eval.run_evaluation \
  --query "…" \
  --gold-json path/to/gold.json \
  --gold-xml path/to/gold.bpmn \
  --setting setting_2 \
  --runs 3
```

Use this when you need mean/variance-style aggregation over stochastic generation; see `aggregated_statistics.json` and console summaries.

## Docker note

The default **Docker Compose** setup targets the **Streamlit** app. Evaluation is normally run **on the host** with the same `.env` and a local Python venv. If you run it inside a container, ensure `CHROMA_DB_PATH` points at data visible in that container and that all API URLs are reachable from inside the container (not only `localhost` on the host).
