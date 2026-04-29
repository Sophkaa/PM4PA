"""Lightweight logging for evaluation output artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any

from .metrics import DatasetEvaluationSummary, SampleEvaluation, AggregatedRunSummary
from ..models.bpmn import BPMNModelJsonNested


def _ensure_eval_output_dir() -> Path:
    base = Path("eval_output")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _generate_experiment_name() -> str:
    return datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S")


# Not persisted in meta.json: duplicate of embedding_model_rag in this project.
_META_MODEL_CONFIG_EXCLUDE = frozenset({"embedding_model"})


def _namespace_to_dict(args) -> dict:
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _slim_metadata(meta: dict) -> dict:
    """Extract only essential chunk metadata: title, chunk_nr, page number, heading."""
    if not meta:
        return {}
    title = (
        meta.get("file_name")
        or meta.get("document_title")
        or meta.get("title")
        or (meta.get("file_path", "").split("/")[-1] if meta.get("file_path") else None)
    )
    return {
        "title": title,
        "chunk_nr": meta.get("chunk_nr"),
        "page": meta.get("page_number"),
        "heading": meta.get("heading"),
    }


def _prepare_per_sample_payload(
    evaluations: List[SampleEvaluation],
    bpmn_results: Optional[List[BPMNModelJsonNested]] = None,
    process_states: Optional[List[Any]] = None,
) -> List[dict]:
    payload = []
    bpmn_results = bpmn_results or []
    process_states = process_states or []

    for i, evaluation in enumerate(evaluations):
        entry = evaluation.model_dump()

        # Add predicted BPMN
        if i < len(bpmn_results):
            try:
                entry["predicted_bpmn"] = bpmn_results[i].model_dump()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to serialize BPMN for sample {evaluation.sample_id}: {e}")
                entry["predicted_bpmn"] = None
        else:
            entry["predicted_bpmn"] = None

        # Add intermediate results from process_state (if available)
        if i < len(process_states) and process_states[i] is not None:
            state = process_states[i]
            try:
                # Add draft (ProcessDraft)
                if hasattr(state, "draft") and state.draft is not None:
                    entry["draft"] = (
                        state.draft.model_dump()
                        if hasattr(state.draft, "model_dump")
                        else {"text_description": state.draft.text_description if hasattr(state.draft, "text_description") else str(state.draft)}
                    )

                # Add expanded queries (if available and not empty)
                if hasattr(state, "expanded_queries") and state.expanded_queries:
                    entry["expanded_queries"] = state.expanded_queries

                # Add query_structure (if available)
                if hasattr(state, "query_structure") and state.query_structure is not None:
                    entry["query_structure"] = state.query_structure.model_dump() if hasattr(state.query_structure, "model_dump") else None

                # Do NOT store draft_system_prompt or draft_user_prompt

                # Add retrieved documents: count, slim metadata (title, chunk_nr, page, heading), scores
                if hasattr(state, "retrieved_documents") and state.retrieved_documents:
                    entry["retrieved_documents_count"] = len(state.retrieved_documents)
                    if hasattr(state, "retrieved_metadatas") and state.retrieved_metadatas:
                        entry["retrieved_chunks"] = [
                            _slim_metadata(m) for m in state.retrieved_metadatas
                        ]
                    if hasattr(state, "relevance_scores") and state.relevance_scores:
                        entry["relevance_scores"] = state.relevance_scores

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to serialize process_state for sample {evaluation.sample_id}: {e}")

        payload.append(entry)
    return payload


def record_experiment_run(
    experiment_name: Optional[str],
    args,
    summary: DatasetEvaluationSummary,
    evaluations: List[SampleEvaluation],
    bpmn_results: List[BPMNModelJsonNested],
    process_states: Optional[List[Any]] = None,
    model_configuration: Optional[Any] = None,
    aggregated_summary: Optional[AggregatedRunSummary] = None,
    all_runs_evaluations: Optional[List[List[SampleEvaluation]]] = None,
    all_runs_bpmn_results: Optional[List[List[BPMNModelJsonNested]]] = None,
    all_runs_process_states: Optional[List[List[Any]]] = None,
    run_timestamp: Optional[datetime] = None,
) -> Path:
    """Store evaluation artifacts for later inspection.

    Args:
        experiment_name: Name of the experiment
        args: CLI arguments
        summary: Evaluation summary (from last run)
        evaluations: List of per-sample evaluations (from last run)
        bpmn_results: List of generated BPMN models (from last run)
        process_states: Optional list of process states (from last run)
        model_configuration: Optional model configuration dictionary
        aggregated_summary: Optional aggregated statistics over multiple runs
        all_runs_evaluations: Optional list of runs, each a list of SampleEvaluation
        all_runs_bpmn_results: Optional list of runs, each a list of BPMNModelJsonNested
        all_runs_process_states: Optional list of runs, each a list of process states
        run_timestamp: Moment for meta.json and (if caller builds the name from it) folder suffix; UTC recommended.
    """
    base_dir = _ensure_eval_output_dir()
    name = experiment_name or _generate_experiment_name()
    exp_dir = base_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    ts = run_timestamp if run_timestamp is not None else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    timestamp = ts.isoformat()
    cli_args = _namespace_to_dict(args)

    meta = {
        "experiment_name": name,
        "timestamp": timestamp,
        "cli_args": cli_args,
    }

    # Add model configuration if provided (omit embedding fields from disk metadata)
    if model_configuration:
        meta["model_configuration"] = {
            k: v
            for k, v in model_configuration.items()
            if k not in _META_MODEL_CONFIG_EXCLUDE
        }

    (exp_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (exp_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    if aggregated_summary is not None:
        (exp_dir / "aggregated_statistics.json").write_text(
            aggregated_summary.model_dump_json(indent=2), encoding="utf-8"
        )

    # Per-sample for last run (backward compatible)
    per_sample_payload = _prepare_per_sample_payload(evaluations, bpmn_results, process_states)
    (exp_dir / "per_sample.json").write_text(
        json.dumps(per_sample_payload, indent=2), encoding="utf-8"
    )

    # Per-run per-sample when multiple runs
    if all_runs_evaluations and all_runs_bpmn_results and all_runs_process_states:
        runs_payload = []
        for run_idx, (run_evals, run_bpmn, run_states) in enumerate(
            zip(all_runs_evaluations, all_runs_bpmn_results, all_runs_process_states)
        ):
            run_data = _prepare_per_sample_payload(run_evals, run_bpmn, run_states)
            runs_payload.append({"run": run_idx + 1, "samples": run_data})
        (exp_dir / "per_run_per_sample.json").write_text(
            json.dumps({"runs": runs_payload}, indent=2), encoding="utf-8"
        )

    return exp_dir
