"""CLI helper to evaluate the Graph-based Agentic RAG pipeline against gold BPMN models."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Any, Dict
from datetime import datetime, timezone

# Support both direct script execution and package/module execution
try:
    from ..config import settings
    from ..app.pipeline import GraphRAGSystem
    from ..logging_config import configure_third_party_logging
    from ..models.bpmn import BPMNModelJsonNested, BPMNElement
    from ..agents.judge import run_llm_judge_agent
    from .dataset_loader import EvalSample, GoldBPMNModel
    from .tracker import record_experiment_run
    from .metrics import (
        EvaluationConfig,
        evaluate_sample,
        summarize_dataset_results,
        aggregate_run_statistics,
        SampleEvaluation,
        AggregatedRunSummary,
    )
except ImportError:
    # If relative imports fail, try absolute imports (for direct execution)
    import sys

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import settings
    from src.app.pipeline import GraphRAGSystem
    from src.logging_config import configure_third_party_logging
    from src.models.bpmn import BPMNModelJsonNested, BPMNElement
    from src.agents.judge import run_llm_judge_agent
    from src.eval.dataset_loader import EvalSample, GoldBPMNModel
    from src.eval.tracker import record_experiment_run
    from src.eval.metrics import (
        EvaluationConfig,
        evaluate_sample,
        summarize_dataset_results,
        aggregate_run_statistics,
        SampleEvaluation,
        AggregatedRunSummary,
    )


logger = logging.getLogger(__name__)


_SPINNER_FRAMES = ("|", "/", "-", "\\")


async def _run_with_spinner(coro: Any, label: str) -> Any:
    """Run an awaitable with a simple CLI spinner."""
    if not sys.stdout.isatty():
        return await coro

    done = False
    result: Any = None
    error: Optional[BaseException] = None
    start = time.perf_counter()

    async def _runner() -> None:
        nonlocal done, result, error
        try:
            result = await coro
        except BaseException as exc:  # keep original exception semantics
            error = exc
        finally:
            done = True

    task = asyncio.create_task(_runner())
    frame_idx = 0

    while not done:
        elapsed = time.perf_counter() - start
        frame = _SPINNER_FRAMES[frame_idx % len(_SPINNER_FRAMES)]
        print(f"\r{frame} {label} ({elapsed:.1f}s)", end="", flush=True)
        frame_idx += 1
        await asyncio.sleep(0.12)

    await task
    elapsed = time.perf_counter() - start
    if error is None:
        print(f"\r✓ {label} ({elapsed:.1f}s)")
        return result

    print(f"\r✗ {label} ({elapsed:.1f}s)")
    raise error


def _project_root() -> Path:
    """Repository root (directory that contains `src/`)."""
    return Path(__file__).resolve().parents[2]


def _resolve_cli_path(raw: str, *, what: str) -> Path:
    """Resolve a user-supplied path robustly.

    Relative paths are first resolved against the current working directory (shell CWD).
    If that file does not exist, we also try resolving the same relative path against the
    project root (e.g. ``eval_data/Gold_....json`` when the CLI is run from the repo root).
    """
    raw_path = Path(raw).expanduser()
    candidates: List[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((Path.cwd() / raw_path).resolve())
        candidates.append((_project_root() / raw_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"{what} not found. Tried: "
        + ", ".join(str(p) for p in candidates)
        + ". Tip: use a path relative to the repo root (e.g. eval_data/...) or an absolute path."
    )


def _is_flat_json_format(payload: Dict[str, Any]) -> bool:
    required_keys = {"pools", "lanes", "activities", "events", "gateways"}
    return required_keys.issubset(set(payload.keys()))


def _convert_flat_json_to_elements(flat_json: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[BPMNElement]]:
    result: Dict[str, List[BPMNElement]] = {}
    for element_type in ["pools", "lanes", "activities", "events", "gateways"]:
        if element_type in flat_json:
            result[element_type] = [BPMNElement(**elem_dict) for elem_dict in flat_json[element_type]]
        else:
            result[element_type] = []
    return result


def _load_gold_model_from_json(gold_json_path: Path, sample_id: str, query: str) -> GoldBPMNModel:
    with gold_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
        if isinstance(payload, str):
            payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid gold JSON in {gold_json_path}: expected object, got {type(payload).__name__}")

    if _is_flat_json_format(payload):
        flat_elements = _convert_flat_json_to_elements(payload)
        process_name = query
        if flat_elements.get("pools") and flat_elements["pools"]:
            process_name = flat_elements["pools"][0].name or query
        return GoldBPMNModel(
            process_id=payload.get("process_id", sample_id),
            process_name=payload.get("process_name", process_name),
            flat_elements=flat_elements,
            metadata=payload.get("metadata", {}),
        )

    gold_bpmn = BPMNModelJsonNested(**payload)
    return GoldBPMNModel(
        process_id=payload.get("process_id", sample_id),
        process_name=payload.get("process_name", gold_bpmn.process_name or query),
        bpmn=gold_bpmn,
        metadata=payload.get("metadata", {}),
    )


def _build_single_sample(args: argparse.Namespace) -> EvalSample:
    gold_json_path = _resolve_cli_path(args.gold_json, what="Gold JSON file")
    gold_xml_path = _resolve_cli_path(args.gold_xml, what="Gold XML file")

    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", gold_json_path.stem).strip("_") or "sample"
    sample_id = f"{safe_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    gold_model = _load_gold_model_from_json(gold_json_path, sample_id=sample_id, query=args.query)

    return EvalSample(
        sample_id=sample_id,
        query=args.query,
        description=args.query,
        gold_model=gold_model,
        submit_to_service=True,
        metadata={
            "gold_json_path": str(gold_json_path),
            "gold_xml_path": str(gold_xml_path),
        },
    )


def collect_model_configuration(
    rag_system: GraphRAGSystem,
    enable_llm_judge: bool = False,
    judge_model_override: Optional[str] = None,
    eval_config: Optional[EvaluationConfig] = None,
) -> Dict[str, Any]:
    """
    Collect model configuration from the RAG system settings.

    Args:
        rag_system: The GraphRAGSystem instance
        enable_llm_judge: Whether LLM judge is enabled
        judge_model_override: Optional override for judge model
        eval_config: Optional EvaluationConfig for semantic matching embedding model

    Returns:
        Dictionary with model configuration for each agent/step
    """
    settings = rag_system.settings
    setting_name = rag_system.setting_name

    # Retrieval method depends on setting
    if setting_name == "setting_1":
        retrieval_method = "vector_only"
    else:
        retrieval_method = "hybrid_bm25_vector"

    model_config = {
        "embedding_model": settings.embedding_model,
        "embedding_model_rag": settings.embedding_model,
        "embedding_model_semantic_matching": (
            eval_config.embedding_model
            if eval_config
            else settings.eval_semantic_embedding_model
        ),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieval_method": retrieval_method,
        "temperature": settings.agentic_temperature,
        "open_source": settings.open_source,
        "setting": setting_name
    }

    if settings.open_source:
        # Ollama: store per-agent models
        model_config["agents"] = {
            "retrieval": {
                "model": settings.agentic_retrieval_model,
                "description": "Query expansion and document retrieval"
            },
            "draft": {
                "model": settings.agentic_draft_model,
                "description": "Process draft generation"
            },
            "bpmn": {
                "model": settings.agentic_bpmn_model,
                "description": "BPMN model generation"
            },
            "validation": {
                "model": settings.agentic_validation_model,
                "description": "BPMN validation"
            },
            "relevance": {
                "model": settings.agentic_judge_model,
                "description": "Relevance assessment of retrieved chunks (used in Settings 2, 3, 4)"
            }
        }
        if enable_llm_judge:
            judge_model = judge_model_override or settings.agentic_judge_model
            model_config["agents"]["judge"] = {
                "model": judge_model,
                "description": "LLM-as-a-judge evaluation (final quality assessment)"
            }
    else:
        # Azure/GPT: GPT used for all agents, store only the GPT model
        model_config["gpt_model"] = settings.azure_deployment or "gpt-4o-2024-11-20"

    return model_config


def _print_aggregated_summary(agg: AggregatedRunSummary) -> None:
    """Print aggregated run statistics (mean ± std)."""

    def _fmt(stats) -> str:
        return f"{stats.mean:.3f} ± {stats.std:.3f}"

    print("\n" + "=" * 60)
    print("AGGREGATED STATISTICS (%d Runs)" % agg.n_runs)
    print("=" * 60)
    print(f"Setting: {agg.setting} | Open Source: {agg.open_source}")
    print(f"Dataset: {agg.dataset_name} | Samples: {agg.n_samples}")
    print()
    print("--- Metrics ---")
    print(f"  Precision: {_fmt(agg.precision)}")
    print(f"  Recall:    {_fmt(agg.recall)}")
    print(f"  F1:        {_fmt(agg.f1)}")
    print()
    print("Generation Time (avg per sample):")
    print(f"  {_fmt(agg.avg_generation_time_seconds)} seconds")
    if agg.judge_semantic_alignment_score is not None:
        print()
        print("LLM Judge (Semantic Alignment Score 0-100):")
        print(f"  {_fmt(agg.judge_semantic_alignment_score)}")
    print("=" * 60)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BPMN generation quality.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Single query to evaluate.",
    )
    parser.add_argument(
        "--gold-json",
        type=str,
        required=True,
        help="Path to static/versioned gold JSON used for metrics.",
    )
    parser.add_argument(
        "--gold-xml",
        type=str,
        required=True,
        help="Path to static/versioned gold BPMN XML used for LLM judge.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["setting_1", "setting_2", "setting_3", "setting_4", "setting_5"],
        help=(
            "Graph setting to use for evaluation "
            "setting_1: combined retrieval & BPMN generation, "
            "setting_2: enhanced retrieval with query extraction + BM25 + hybrid search, "
            "setting_3: enhanced retrieval (Setting 2) + two-stage BPMN generation with query_structure, "
            "setting_4: enhanced retrieval + BPMN validation after nested JSON generation + optional BPMN revision loop, "
            "setting_5: enhanced retrieval + three specialized validators (scope/completeness, factual fidelity/sources, process logic/modeling)."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation runs for mean/variance (default: 1). Use >1 for aggregated statistics.",
    )
    return parser


async def _evaluate_sample(
    rag_system: GraphRAGSystem,
    sample: EvalSample,
    config: EvaluationConfig,
    gold_xml_path: Path,
) -> Optional[tuple[SampleEvaluation, BPMNModelJsonNested, Any]]:
    logger.debug("Evaluating sample '%s' (%s)", sample.sample_id, sample.query)
    
    # Start timing for BPMN generation
    generation_start_time = time.perf_counter()
    print("▶ Step 1/3: BPMN generation pipeline")
    
    try:
        state = await rag_system.orchestrator.run(
            query=sample.query,
            file_filter=None,
            progress_callback=None,
        )
    except Exception as exc:
        logger.exception("Pipeline failed for sample '%s': %s", sample.sample_id, exc)
        return None
    
    # Calculate generation time (from start until BPMN model is ready)
    generation_time = time.perf_counter() - generation_start_time

    if not state.bpmn_result:
        error_msg = state.error if state.error else "Unknown error"
        logger.warning(
            "Skipping sample '%s': BPMN generation returned no result (error=%s)",
            sample.sample_id,
            error_msg,
        )
        return None


    # BPMN service: XML is stored under Final_BPMN; judge reads that path.
    predicted_xml_path: Optional[Path] = None
    if sample.submit_to_service:
        print("▶ Step 2/3: Submit BPMN to service")
        try:
            try:
                from ..bpmn_service.service_submitter import SubmitToServiceInput, submit_to_bpmn_service
            except ImportError:
                from src.bpmn_service.service_submitter import SubmitToServiceInput, submit_to_bpmn_service

            logger.debug("Submitting generated BPMN to BPMN service for sample '%s'...", sample.sample_id)
            service_result = await _run_with_spinner(
                asyncio.to_thread(
                    submit_to_bpmn_service,
                    SubmitToServiceInput(
                        bpmn_json=state.bpmn_result.model_dump(),
                        process_name=state.bpmn_result.process_name,
                        user_query=sample.query,
                    ),
                ),
                "Submitting BPMN to service",
            )

            if service_result.success and service_result.file_path:
                predicted_xml_path = Path(service_result.file_path)
                logger.debug(
                    "BPMN XML for sample '%s' (Final_BPMN): %s",
                    sample.sample_id,
                    predicted_xml_path,
                )
            else:
                logger.warning(
                    "Failed to generate BPMN XML via BPMN service for sample '%s': %s",
                    sample.sample_id,
                    service_result.message,
                )
        except Exception as exc:
            logger.warning(
                "Error submitting BPMN to BPMN service for sample '%s': %s",
                sample.sample_id,
                exc,
            )
    else:
        logger.debug("Skipping service submission for sample '%s' (submit_to_service=False)", sample.sample_id)

    evaluation = evaluate_sample(
        sample_id=sample.sample_id,
        predicted=state.bpmn_result,
        gold=sample.gold_model,
        config=config,
    )
    
    # Add generation time to evaluation
    evaluation.generation_time_seconds = generation_time
    logger.debug("Sample '%s' generation time: %.2f seconds (%.2f minutes)", 
                sample.sample_id, generation_time, generation_time / 60)

    try:
        if not predicted_xml_path or not predicted_xml_path.is_file():
            logger.warning(
                "LLM judge skipped for sample '%s': Predicted BPMN XML not found at %s.",
                sample.sample_id,
                predicted_xml_path,
            )
        else:
            print("▶ Step 3/3: Run LLM judge")
            logger.debug(
                "Running LLM judge for sample '%s': gold=%s, predicted=%s",
                sample.sample_id,
                gold_xml_path,
                predicted_xml_path,
            )
            judge_result = await run_llm_judge_agent(
                task_description=sample.description or sample.query,
                gold_bpmn_xml=gold_xml_path,
                predicted_bpmn_xml=predicted_xml_path,
                model_override=None,
            )
            evaluation.judge = judge_result
    except Exception as exc:
        logger.warning(
            "LLM judge failed for sample '%s': %s",
            sample.sample_id,
            exc,
        )

    # Extract process_state from the wrapper
    process_state = state.process_state if hasattr(state, 'process_state') else None
    
    return evaluation, state.bpmn_result, process_state


async def evaluate_dataset(args: argparse.Namespace) -> None:
    sample = _build_single_sample(args)
    dataset_name = f"single_sample:{Path(args.gold_json).name}"
    logger.debug("Prepared single evaluation sample '%s' from %s", sample.sample_id, args.gold_json)

    config = EvaluationConfig(
        embedding_model=settings.eval_semantic_embedding_model
    )
    rag_system = GraphRAGSystem(setting_name=args.setting)
    gold_xml_path = Path(sample.metadata["gold_xml_path"])

    n_runs = max(1, args.runs or 1)
    all_runs_evaluations: List[List[SampleEvaluation]] = []
    all_runs_bpmn_results: List[List[BPMNModelJsonNested]] = []
    all_runs_process_states: List[List[Any]] = []

    for run_idx in range(n_runs):
        if n_runs > 1:
            logger.debug("=== Run %d/%d ===", run_idx + 1, n_runs)
        evaluations: List[SampleEvaluation] = []
        bpmn_results: List[BPMNModelJsonNested] = []
        process_states: List[Any] = []

        result = await _evaluate_sample(
            rag_system=rag_system,
            sample=sample,
            config=config,
            gold_xml_path=gold_xml_path,
        )
        if result:
            evaluation, bpmn_result, process_state = result
            evaluations.append(evaluation)
            bpmn_results.append(bpmn_result)
            process_states.append(process_state)

        if not evaluations:
            logger.warning("Run %d: No successful evaluations.", run_idx + 1)
            continue
        all_runs_evaluations.append(evaluations)
        all_runs_bpmn_results.append(bpmn_results)
        all_runs_process_states.append(process_states)

    if not all_runs_evaluations:
        logger.error("No successful evaluations across all runs. Aborting.")
        return

    evaluations = all_runs_evaluations[-1]
    bpmn_results = all_runs_bpmn_results[-1]
    process_states = all_runs_process_states[-1]

    summary = summarize_dataset_results(dataset_name, evaluations, config.element_types)
    
    # Collect model configuration
    model_config = collect_model_configuration(
        rag_system=rag_system,
        enable_llm_judge=True,
        judge_model_override=None,
        eval_config=config,
    )

    aggregated: Optional[AggregatedRunSummary] = None
    if n_runs > 1:
        aggregated = aggregate_run_statistics(
            runs=all_runs_evaluations,
            dataset_name=dataset_name,
            setting=args.setting,
            open_source=rag_system.settings.open_source,
            element_types=config.element_types,
        )
        _print_aggregated_summary(aggregated)

    print(f"\n=== Metrics ===")
    print(f"  Precision: {summary.precision:.3f}")
    print(f"  Recall: {summary.recall:.3f}")
    print(f"  F1-score: {summary.f1:.3f}")
    runtime_mode = "open source" if model_config.get("open_source") else "closed source"
    print(f"  Runtime mode: {runtime_mode}")
    
    judge_scores: List[float] = []
    judge_justifications: List[str] = []
    for run_evals in all_runs_evaluations:
        for ev in run_evals:
            if ev.judge is not None:
                judge_scores.append(float(ev.judge.semantic_alignment_score))
                justification = (ev.judge.justification or "").strip()
                if justification:
                    judge_justifications.append(justification)

    if judge_scores:
        if len(judge_scores) == 1:
            llm_score_text = f"{judge_scores[0]}"
        else:
            mean_j = sum(judge_scores) / len(judge_scores)
            llm_score_text = f"{mean_j:.2f}"
        llm_justification_text = (
            " | ".join(judge_justifications) if judge_justifications else "N/A"
        )
    else:
        llm_score_text = "N/A"
        llm_justification_text = "N/A"

    print(f"  LLM judge score: {llm_score_text}")
    print(f"  LLM justification: {llm_justification_text}")
    
    # Print timing summary
    if summary.generation_time_seconds is not None:
        print("\n=== TIMING SUMMARY ===")
        gt = summary.generation_time_seconds
        print(f"Generation Time: {gt:.2f} seconds ({gt / 60:.2f} minutes)")
    elif summary.avg_generation_time_seconds is not None:
        print("\n=== TIMING SUMMARY ===")
        print(
            f"Average Generation Time: {summary.avg_generation_time_seconds:.2f} seconds "
            f"({summary.avg_generation_time_seconds / 60:.2f} minutes)"
        )
        if summary.min_generation_time_seconds is not None:
            print(
                f"Minimum Generation Time: {summary.min_generation_time_seconds:.2f} seconds "
                f"({summary.min_generation_time_seconds / 60:.2f} minutes)"
            )
        if summary.max_generation_time_seconds is not None:
            print(
                f"Maximum Generation Time: {summary.max_generation_time_seconds:.2f} seconds "
                f"({summary.max_generation_time_seconds / 60:.2f} minutes)"
            )
        if summary.total_generation_time_seconds is not None:
            print(
                f"Total Generation Time: {summary.total_generation_time_seconds:.2f} seconds "
                f"({summary.total_generation_time_seconds / 60:.2f} minutes)"
            )
    
    # Print model configuration summary
    print(f"\n=== MODEL CONFIGURATION ===")
    print(f"Setting: {model_config['setting']}")
    print(f"Embedding Model: {model_config['embedding_model']}")
    print(f"Temperature: {model_config['temperature']}")
    if model_config.get("open_source"):
        print(f"\nAgent Models:")
        for agent_name, agent_info in model_config.get("agents", {}).items():
            print(f"  - {agent_name.capitalize()}: {agent_info['model']}")
            print(f"    ({agent_info['description']})")
    else:
        print(f"\nLLM Model (Azure GPT): {model_config.get('gpt_model', 'N/A')}")

    try:
        run_at = datetime.now(timezone.utc)
        exp_name = f"{args.setting}_{run_at.strftime('%Y%m%d_%H%M%S')}"

        exp_dir = record_experiment_run(
            experiment_name=exp_name,
            args=args,
            summary=summary,
            evaluations=evaluations,
            bpmn_results=bpmn_results,
            process_states=process_states,
            model_configuration=model_config,
            aggregated_summary=aggregated,
            all_runs_evaluations=all_runs_evaluations if n_runs > 1 else None,
            all_runs_bpmn_results=all_runs_bpmn_results if n_runs > 1 else None,
            all_runs_process_states=all_runs_process_states if n_runs > 1 else None,
            run_timestamp=run_at,
        )
        logger.debug("Stored experiment artifacts in %s", exp_dir)
    except Exception as exc:
        logger.warning("Failed to store experiment artifacts: %s", exc)
    
def main(argv: Optional[List[str]] = None) -> None:
    root_level = logging.DEBUG if settings.debug else logging.WARNING
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,
    )
    configure_third_party_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    asyncio.run(evaluate_dataset(args))


if __name__ == "__main__":
    main()

