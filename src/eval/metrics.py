"""Precision/Recall/F1 metrics for BPMN evaluation."""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Iterable, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import Hungarian algorithm (optimal bipartite matching)
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    linear_sum_assignment = None

from ..config import DEFAULT_EVAL_SEMANTIC_EMBEDDING_MODEL, settings
from ..models.bpmn import BPMNElement, BPMNModelJsonNested
from ..models.artifacts import LLMJudgeResult
from .dataset_loader import GoldBPMNModel


NODE_ELEMENT_TYPES = ("roles", "activities", "events", "gateways")
DEFAULT_ELEMENT_TYPES = NODE_ELEMENT_TYPES


class EvaluationConfig(BaseModel):
    """Configuration for BPMN evaluation."""

    element_types: Tuple[str, ...] = Field(
        default=DEFAULT_ELEMENT_TYPES,
        description="BPMN element groups that should be considered during evaluation.",
    )
    use_semantic_matching: bool = Field(
        default=True,
        description="Enable semantic similarity matching for element names.",
    )
    semantic_threshold: float = Field(
        default=0.7,
        description="Minimum cosine similarity (0-1) for semantic matches when enabled.",
    )
    embedding_model: str = Field(
        default=DEFAULT_EVAL_SEMANTIC_EMBEDDING_MODEL,
        description="Sentence transformer for eval semantic matching (env: EVAL_SEMANTIC_EMBEDDING_MODEL).",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether to treat element names as case-sensitive.",
    )
    strip_diacritics: bool = Field(
        default=True,
        description="Normalize umlauts/accents before comparison.",
    )
    collapse_whitespace: bool = Field(
        default=True,
        description="Replace repeated whitespace with single spaces.",
    )
    debug_matching: bool = Field(
        default=False,
        description="Enable debug output showing detailed matching information.",
    )

    @field_validator("element_types")
    @classmethod
    def _restrict_element_types_to_nodes(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if tuple(value) != NODE_ELEMENT_TYPES:
            raise ValueError(
                f"element_types must be exactly {NODE_ELEMENT_TYPES}; sequence flows are no longer supported."
            )
        return tuple(value)


@dataclass
class ElementMetrics:
    element_type: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


class SampleEvaluation(BaseModel):
    """Per-sample evaluation record."""

    sample_id: str
    process_name: str
    metrics: Dict[str, ElementMetrics]
    judge: Optional[LLMJudgeResult] = None
    generation_time_seconds: Optional[float] = Field(
        default=None,
        description="Time taken to generate BPMN model (in seconds), excluding evaluation time."
    )


class LLMJudgeSummaryEntry(BaseModel):
    sample_id: str
    semantic_alignment_score: int
    justification: str


class LLMJudgeDatasetSummary(BaseModel):
    mean_semantic_alignment_score: float
    samples: List[LLMJudgeSummaryEntry]


class DatasetEvaluationSummary(BaseModel):
    """Aggregated evaluation for a dataset."""

    dataset_name: str
    sample_count: int
    precision: float
    recall: float
    f1: float
    generation_time_seconds: Optional[float] = Field(
        default=None,
        description="Single-run generation time when only one sample contributed timing (no min/avg/max spread)."
    )
    avg_generation_time_seconds: Optional[float] = Field(
        default=None,
        description="Average time taken to generate BPMN models (in seconds), excluding evaluation time."
    )
    min_generation_time_seconds: Optional[float] = Field(
        default=None,
        description="Minimum generation time across all samples (in seconds)."
    )
    max_generation_time_seconds: Optional[float] = Field(
        default=None,
        description="Maximum generation time across all samples (in seconds)."
    )
    total_generation_time_seconds: Optional[float] = Field(
        default=None,
        description="Total time taken for all BPMN generations (in seconds)."
    )
    llm_judge: Optional[LLMJudgeDatasetSummary] = Field(
        default=None,
        description="LLM-as-judge scores and justifications when judge was run per sample.",
    )


class RunStatistics(BaseModel):
    """Mean and standard deviation for a metric over N runs."""

    mean: float
    std: float = Field(description="Standard deviation.")
    n_runs: int


class AggregatedRunSummary(BaseModel):
    """Summary of metrics over multiple evaluation runs (mean ± std)."""

    dataset_name: str
    setting: str
    open_source: bool
    n_runs: int
    n_samples: int

    precision: RunStatistics
    recall: RunStatistics
    f1: RunStatistics

    avg_generation_time_seconds: RunStatistics

    judge_semantic_alignment_score: Optional[RunStatistics] = Field(
        default=None,
        description="LLM judge semantic alignment score (0-100) mean ± std, when --enable-llm-judge is used.",
    )

    per_sample: Dict[str, Dict[str, RunStatistics]] = Field(
        default_factory=dict,
        description="sample_id -> metric_name -> RunStatistics",
    )


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------

# Global cache for embedding model
_embedding_model_cache: Dict[str, SentenceTransformer] = {}


def _fix_jina_cache_if_needed(error: Exception) -> bool:
    """Attempt to fix corrupted jina-embeddings-v3 cache by downloading missing files.
    
    Downloads all required files from the xlm-roberta-flash-implementation repository.
    
    Returns:
        True if fix was attempted, False otherwise
    """
    import os
    import urllib.request
    import json
    
    error_str = str(error)
    if "xlm_hyphen_roberta_hyphen_flash_hyphen_implementation" not in error_str:
        return False
    
    # Try to find the cache directory
    hf_modules_root = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "modules",
        "transformers_modules",
        "jinaai",
        "xlm_hyphen_roberta_hyphen_flash_hyphen_implementation",
    )
    cache_paths = [hf_modules_root]
    
    # List of all files that might be needed from the repository
    # Based on: https://huggingface.co/jinaai/xlm-roberta-flash-implementation
    required_files = [
        "block.py",
        "mlp.py",
        "mha.py",
        "rotary.py",
        "embedding.py",
        "stochastic_depth.py",
        "xlm_padding.py",
        "configuration_xlm_roberta.py",
        "modeling_lora.py",
        "modeling_xlm_roberta.py",
    ]
    
    for base_path in cache_paths:
        if not os.path.exists(base_path):
            continue
        
        # Find the commit hash directory
        subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d != "__pycache__"]
        if not subdirs:
            continue
        
        commit_dir = os.path.join(base_path, subdirs[0])
        
        # Check which files are missing
        missing_files = []
        for filename in required_files:
            filepath = os.path.join(commit_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
        
        if not missing_files:
            continue  # No files missing in this directory
        
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Attempting to fix corrupted cache by downloading {len(missing_files)} missing file(s) to {commit_dir}")
            
            # Download each missing file
            base_url = "https://huggingface.co/jinaai/xlm-roberta-flash-implementation/raw/main"
            for filename in missing_files:
                filepath = os.path.join(commit_dir, filename)
                url = f"{base_url}/{filename}"
                try:
                    urllib.request.urlretrieve(url, filepath)
                    logger.info(f"Successfully downloaded {filename}")
                except Exception as download_error:
                    logger.warning(f"Failed to download {filename}: {download_error}")
            
            return True
        except Exception as fix_error:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to auto-fix cache: {fix_error}")
            return False
    
    return False


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """Get or load embedding model (cached).
    
    Supports models that require trust_remote_code (e.g., jinaai/jina-embeddings-v3).
    """
    if model_name not in _embedding_model_cache:
        import logging
        logger = logging.getLogger(__name__)
        
        # Some models (like jinaai/jina-embeddings-v3) require trust_remote_code=True
        try:
            # For jina-embeddings-v3, we need to ensure trust_remote_code is set
            if "jina-embeddings-v3" in model_name.lower():
                logger.info(f"Loading jina-embeddings-v3 model (requires trust_remote_code=True)...")
                # jina-embeddings-v3 requires trust_remote_code=True
                # The custom_st module should be automatically loaded from the model's custom code
                _embedding_model_cache[model_name] = SentenceTransformer(
                    model_name, 
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded {model_name}")
            else:
                # Try with trust_remote_code first for other models that might need it
                try:
                    _embedding_model_cache[model_name] = SentenceTransformer(
                        model_name, 
                        trust_remote_code=True
                    )
                except Exception:
                    # Fallback: try without trust_remote_code for models that don't need it
                    _embedding_model_cache[model_name] = SentenceTransformer(model_name)
        except (ModuleNotFoundError, FileNotFoundError, OSError) as e:
            error_str = str(e)
            if "custom_st" in error_str or "block.py" in error_str or "mlp.py" in error_str or "xlm_hyphen_roberta_hyphen_flash_hyphen_implementation" in error_str or "No such file or directory" in error_str:
                # Try to auto-fix the cache
                if _fix_jina_cache_if_needed(e):
                    logger.info("Cache fixed, retrying model load...")
                    try:
                        # Retry loading after fix
                        if "jina-embeddings-v3" in model_name.lower():
                            _embedding_model_cache[model_name] = SentenceTransformer(
                                model_name, 
                                trust_remote_code=True
                            )
                        else:
                            _embedding_model_cache[model_name] = SentenceTransformer(
                                model_name, 
                                trust_remote_code=True
                            )
                        logger.info(f"Successfully loaded {model_name} after cache fix")
                        return _embedding_model_cache[model_name]
                    except Exception as retry_error:
                        logger.error(f"Still failed after cache fix: {retry_error}")
                        # Try one more time - might need to download more files
                        if _fix_jina_cache_if_needed(retry_error):
                            logger.info("Trying to fix cache again...")
                            try:
                                if "jina-embeddings-v3" in model_name.lower():
                                    _embedding_model_cache[model_name] = SentenceTransformer(
                                        model_name, 
                                        trust_remote_code=True
                                    )
                                else:
                                    _embedding_model_cache[model_name] = SentenceTransformer(
                                        model_name, 
                                        trust_remote_code=True
                                    )
                                logger.info(f"Successfully loaded {model_name} after second cache fix attempt")
                                return _embedding_model_cache[model_name]
                            except Exception as second_retry_error:
                                logger.error(f"Failed after second cache fix attempt: {second_retry_error}")
                                # Fall through to raise the original error
                        # If we get here, all retry attempts failed
                        logger.error(f"Failed to load {model_name} after all retry attempts. Original error: {e}, Retry error: {retry_error}")
                        logger.error("This is a known issue with jina-embeddings-v3. The cache may be corrupted.")
                        logger.error("Solutions:")
                        logger.error("1. Clear the Hugging Face cache directory")
                        logger.error("   Default: rm -rf ~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v3")
                        logger.error(
                            "   Modules cache: rm -rf \"$HF_HOME/modules/transformers_modules/jinaai/\" "
                            "(if HF_HOME is set; otherwise use ~/.cache/huggingface/...)"
                        )
                        logger.error("2. Or manually download all required files (block.py, mlp.py, mha.py, etc.)")
                        logger.error("3. Then try loading the model again")
                        raise RuntimeError(
                            f"Failed to load {model_name} after multiple retry attempts.\n"
                            f"Original error: {e}\n"
                            f"Retry error: {retry_error}\n\n"
                            "The Hugging Face cache appears to be corrupted. Try:\n"
                            "1. Clear modules cache: rm -rf ~/.cache/huggingface/modules/transformers_modules/jinaai/\n"
                            "   (or the same path under HF_HOME if you set it)\n"
                            "2. Or use a different embedding model (e.g., 'intfloat/multilingual-e5-large-instruct')\n"
                            "3. Re-run the evaluation"
                        ) from retry_error
                
                # If we get here, the error was not a cache-related error, so raise it
                logger.error(f"Failed to load {model_name}: {e}")
                logger.error("This is a known issue with jina-embeddings-v3. The cache may be corrupted.")
                logger.error("Solutions:")
                logger.error("1. Clear the Hugging Face cache directory")
                logger.error("   Default: rm -rf ~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v3")
                logger.error(
                    "   Modules cache: rm -rf \"$HF_HOME/modules/transformers_modules/jinaai/\" "
                    "(if HF_HOME is set; otherwise use ~/.cache/huggingface/...)"
                )
                logger.error("2. Or manually download all required files (block.py, mlp.py, mha.py, etc.)")
                logger.error("3. Then try loading the model again")
                raise RuntimeError(
                    f"Failed to load {model_name}: {e}\n\n"
                    "The Hugging Face cache appears to be corrupted. Try:\n"
                    "1. Clear modules cache: rm -rf ~/.cache/huggingface/modules/transformers_modules/jinaai/\n"
                    "   (or the same path under HF_HOME if you set it)\n"
                    "2. Or use a different embedding model (e.g., 'intfloat/multilingual-e5-large-instruct')\n"
                    "3. Re-run the evaluation"
                ) from e
        except ModuleNotFoundError as e:
            # Check for common missing dependencies
            error_str = str(e)
            missing_module = error_str.replace("No module named '", "").replace("'", "")
            
            logger.error(f"Failed to load embedding model '{model_name}': Missing Python package '{missing_module}'")
            logger.error(f"Error: {e}")
            
            # Provide specific installation instructions for known dependencies
            if missing_module == "einops":
                logger.error("The 'jinaai/jina-embeddings-v3' model requires 'einops'.")
                logger.error("Install it with: pip install einops")
                logger.error("Or install all requirements: pip install -r requirements.txt")
                raise RuntimeError(
                    f"Missing required package '{missing_module}' for embedding model '{model_name}'. "
                    f"Install it with: pip install {missing_module}\n"
                    f"Or install all requirements: pip install -r requirements.txt"
                ) from e
            else:
                logger.error(f"Missing Python package: {missing_module}")
                logger.error(f"Install it with: pip install {missing_module}")
                raise RuntimeError(
                    f"Missing required package '{missing_module}' for embedding model '{model_name}'. "
                    f"Install it with: pip install {missing_module}"
                ) from e
        except Exception as e:
            import traceback
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.error("Possible causes:")
            logger.error("1. Model name is incorrect (must be a Hugging Face model ID, e.g., 'jinaai/jina-embeddings-v3')")
            logger.error("2. Model is not available on Hugging Face or requires authentication")
            logger.error("3. Internet connection issues preventing model download")
            logger.error("4. Insufficient disk space in Hugging Face cache")
            logger.error("5. Model requires special setup (e.g., trust_remote_code=True)")
            logger.error("6. Missing Python dependencies (e.g., 'einops' for jina-embeddings-v3)")
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                f"Please verify the model name is correct and available on Hugging Face. "
                f"Error: {e}"
            ) from e
    return _embedding_model_cache[model_name]


def _normalize_text(value: Optional[str], config: EvaluationConfig) -> str:
    text = value or ""
    if not config.case_sensitive:
        text = text.lower()
    if config.strip_diacritics:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    if config.collapse_whitespace:
        text = " ".join(text.split())
    return text.strip()


def _get_normalized_name(element: BPMNElement, config: EvaluationConfig) -> str:
    """Get normalized element name for semantic matching."""
    return _normalize_text(element.name, config)


def _calculate_similarity_matrix(
    gold_names: Sequence[str],
    pred_names: Sequence[str],
    config: EvaluationConfig,
) -> np.ndarray:
    """Calculate similarity matrix between all gold and predicted element names.
    
    Returns:
        Matrix of shape (len(gold_names), len(pred_names)) with cosine similarities.
    """
    if not config.use_semantic_matching or not gold_names or not pred_names:
        return np.zeros((len(gold_names), len(pred_names)))
    
    try:
        model = _get_embedding_model(config.embedding_model)
        # Encode all texts at once (more efficient than per-element)
        all_texts = list(gold_names) + list(pred_names)
        
        # Some models support task-specific encoding
        encode_kwargs = {}
        if "jina-embeddings-v3" in config.embedding_model.lower():
            # Use text-matching mode for symmetric similarity tasks (best for comparing two texts)
            # This is optimal for semantic matching between gold and predicted BPMN elements
            encode_kwargs["task"] = "text-matching"
        elif "qwen3-embedding" in config.embedding_model.lower():
            # Qwen3-Embedding supports prompt_name for queries
            # For similarity matching, we can use default encoding or specify prompt_name="query"
            # Using default encoding for now, but could optimize by using prompt_name="query" for gold
            pass
        
        embeddings = model.encode(
            all_texts, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            **encode_kwargs
        )
        
        gold_embeddings = embeddings[:len(gold_names)]
        pred_embeddings = embeddings[len(gold_names):]
        
        similarity_matrix = cosine_similarity(gold_embeddings, pred_embeddings)
        return similarity_matrix
    except Exception as e:
        # Fallback: return zero matrix if embedding fails
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"Embedding calculation failed for model '{config.embedding_model}': {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        logger.error("This will result in all similarities being 0.0, which will affect metrics.")
        logger.error("Please check:")
        logger.error("1. Is the model name correct? (e.g., 'jinaai/jina-embeddings-v3', 'intfloat/multilingual-e5-large-instruct')")
        logger.error("2. Is the model available on Hugging Face?")
        logger.error("3. Do you have internet access to download the model?")
        logger.error("4. Is the Hugging Face cache corrupted? (try clearing it)")
        logger.error("5. Try testing the model with: python test_embedding_model.py <model_name>")
        # Don't silently fail - raise the error so user knows something is wrong
        raise RuntimeError(
            f"Failed to calculate embeddings with model '{config.embedding_model}'. "
            f"This will cause all similarity scores to be 0.0. "
            f"Please fix the embedding model issue before continuing. "
            f"Error: {e}"
        ) from e


def _match_elements(
    predicted: Sequence[BPMNElement],
    gold: Sequence[BPMNElement],
    config: EvaluationConfig,
    element_type: str = "unknown",
) -> Tuple[int, int, int]:
    """Match elements using optimal bipartite matching (Hungarian algorithm) or greedy fallback.
    
    Uses Hungarian algorithm for globally optimal assignment when scipy is available,
    otherwise falls back to greedy matching.
    """
    # Get normalized names for semantic matching
    pred_names = [_get_normalized_name(elem, config) for elem in predicted]
    gold_names = [_get_normalized_name(elem, config) for elem in gold]
    
    if not config.use_semantic_matching or not gold_names or not pred_names:
        # No matching: all are false positives/negatives
        return 0, len(pred_names), len(gold_names)
    
    # Use Hungarian algorithm if available (optimal matching)
    if SCIPY_AVAILABLE:
        return _match_elements_hungarian(gold_names, pred_names, config, element_type)
    else:
        # Fallback to greedy matching
        return _match_elements_greedy(gold_names, pred_names, config, element_type)


def _match_elements_hungarian(
    gold_names: Sequence[str],
    pred_names: Sequence[str],
    config: EvaluationConfig,
    element_type: str = "unknown",
) -> Tuple[int, int, int]:
    """Match elements using Hungarian algorithm for optimal bipartite matching.
    
    Returns:
        Tuple of (tp, fp, fn) where:
        - tp: True positives (predicted elements matched to gold with similarity >= threshold)
        - fp: False positives (predicted elements not matched or matched below threshold)
        - fn: False negatives (gold elements not matched or matched below threshold)
    """
    # Edge cases: empty lists
    if not pred_names:
        # No predictions: all gold are false negatives
        if config.debug_matching:
            print(f"\n[DEBUG {element_type}] No predicted elements. FN={len(gold_names)}")
        return 0, 0, len(gold_names)
    if not gold_names:
        # No gold: all predictions are false positives
        if config.debug_matching:
            print(f"\n[DEBUG {element_type}] No gold elements. FP={len(pred_names)}")
        return 0, len(pred_names), 0
    
    # Calculate similarity matrix: shape (len(gold_names), len(pred_names))
    similarity_matrix = _calculate_similarity_matrix(gold_names, pred_names, config)
    
    # Use Hungarian algorithm for optimal bipartite matching
    # Hungarian minimizes cost, so we convert similarity to cost
    cost_matrix = 1.0 - similarity_matrix
    gold_indices, pred_indices = linear_sum_assignment(cost_matrix)
    
    # Debug output: show all assignments
    if config.debug_matching:
        print(f"\n{'='*80}")
        print(f"[DEBUG {element_type}] Matching: {len(gold_names)} gold vs {len(pred_names)} predicted")
        print(f"Threshold: {config.semantic_threshold}")
        print(f"{'='*80}")
        print(f"\nSimilarity Matrix ({len(gold_names)}x{len(pred_names)}):")
        for i, gold_name in enumerate(gold_names):
            row = [f"{similarity_matrix[i, j]:.3f}" for j in range(len(pred_names))]
            print(f"  Gold[{i}] '{gold_name[:50]}': {row}")
        print(f"\nHungarian Algorithm Assignments:")
    
    # Track matches above threshold
    # A match is valid only if similarity >= threshold
    valid_matches = []
    all_assignments = []
    for gold_idx, pred_idx in zip(gold_indices, pred_indices):
        similarity = float(similarity_matrix[gold_idx, pred_idx])
        is_valid = similarity >= config.semantic_threshold
        all_assignments.append((gold_idx, pred_idx, similarity, is_valid))
        if is_valid:
            valid_matches.append((gold_idx, pred_idx))
        
        if config.debug_matching:
            status = "VALID (TP)" if is_valid else "INVALID (below threshold)"
            print(f"  Gold[{gold_idx}] '{gold_names[gold_idx][:50]}' <-> "
                  f"Pred[{pred_idx}] '{pred_names[pred_idx][:50]}' "
                  f"(similarity={similarity:.4f}) {status}")
    
    # True positives: number of valid matches (similarity >= threshold)
    tp = len(valid_matches)
    
    # False positives: predicted elements that are either:
    # 1. Not assigned to any gold element (if len(pred) > len(gold))
    # 2. Assigned but similarity < threshold
    matched_pred_indices = {pred_idx for _, pred_idx in zip(gold_indices, pred_indices)}
    valid_pred_indices = {pred_idx for _, pred_idx in valid_matches}
    
    # Count false positives
    fp = 0
    fp_details = []
    for pred_idx in range(len(pred_names)):
        if pred_idx not in valid_pred_indices:
            # Either not assigned, or assigned but below threshold
            fp += 1
            if pred_idx in matched_pred_indices:
                # Find the assignment to show why it's FP
                for g_idx, p_idx, sim, _ in all_assignments:
                    if p_idx == pred_idx:
                        fp_details.append(f"Pred[{pred_idx}] '{pred_names[pred_idx][:50]}' (assigned to Gold[{g_idx}], similarity={sim:.4f} < {config.semantic_threshold})")
                        break
            else:
                fp_details.append(f"Pred[{pred_idx}] '{pred_names[pred_idx][:50]}' (not assigned)")
    
    # False negatives: gold elements that are either:
    # 1. Not assigned to any predicted element (if len(gold) > len(pred))
    # 2. Assigned but similarity < threshold
    matched_gold_indices = {gold_idx for gold_idx, _ in zip(gold_indices, pred_indices)}
    valid_gold_indices = {gold_idx for gold_idx, _ in valid_matches}
    
    # Count false negatives
    fn = 0
    fn_details = []
    for gold_idx in range(len(gold_names)):
        if gold_idx not in valid_gold_indices:
            # Either not assigned, or assigned but below threshold
            fn += 1
            if gold_idx in matched_gold_indices:
                # Find the assignment to show why it's FN
                for g_idx, p_idx, sim, _ in all_assignments:
                    if g_idx == gold_idx:
                        fn_details.append(f"Gold[{gold_idx}] '{gold_names[gold_idx][:50]}' (assigned to Pred[{p_idx}], similarity={sim:.4f} < {config.semantic_threshold})")
                        break
            else:
                fn_details.append(f"Gold[{gold_idx}] '{gold_names[gold_idx][:50]}' (not assigned)")
    
    # Debug output: show summary
    if config.debug_matching:
        print(f"\n{'='*80}")
        print(f"[DEBUG {element_type}] Results:")
        print(f"{'='*80}")
        print(f"True Positives (TP): {tp}")
        for g_idx, p_idx in valid_matches:
            sim = float(similarity_matrix[g_idx, p_idx])
            print(f"  OK Gold[{g_idx}] '{gold_names[g_idx][:50]}' <-> Pred[{p_idx}] '{pred_names[p_idx][:50]}' (similarity={sim:.4f})")
        
        print(f"\nFalse Positives (FP): {fp}")
        for detail in fp_details:
            print(f"  - {detail}")
        
        print(f"\nFalse Negatives (FN): {fn}")
        for detail in fn_details:
            print(f"  - {detail}")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\n{'='*80}")
        print(f"[DEBUG {element_type}] Metrics Calculation:")
        print(f"  Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.4f}")
        print(f"  Recall    = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.4f}")
        print(f"  F1        = 2 * P * R / (P + R) = 2 * {precision:.4f} * {recall:.4f} / ({precision:.4f} + {recall:.4f}) = {f1:.4f}")
        print(f"{'='*80}\n")
    
    return tp, fp, fn


def _match_elements_greedy(
    gold_names: Sequence[str],
    pred_names: Sequence[str],
    config: EvaluationConfig,
    element_type: str = "unknown",
) -> Tuple[int, int, int]:
    """Match elements using greedy algorithm (fallback when scipy is not available).
    
    Returns:
        Tuple of (tp, fp, fn) where:
        - tp: True positives (predicted elements matched to gold with similarity >= threshold)
        - fp: False positives (predicted elements not matched or matched below threshold)
        - fn: False negatives (gold elements not matched or matched below threshold)
    """
    # Edge cases: empty lists
    if not pred_names:
        if config.debug_matching:
            print(f"\n[DEBUG {element_type}] No predicted elements. FN={len(gold_names)}")
        return 0, 0, len(gold_names)
    if not gold_names:
        if config.debug_matching:
            print(f"\n[DEBUG {element_type}] No gold elements. FP={len(pred_names)}")
        return 0, len(pred_names), 0
    
    if config.debug_matching:
        print(f"\n{'='*80}")
        print(f"[DEBUG {element_type}] Greedy Matching: {len(gold_names)} gold vs {len(pred_names)} predicted")
        print(f"Threshold: {config.semantic_threshold}")
        print(f"{'='*80}")
    
    # Track which predicted elements have been matched (above threshold)
    pred_matched = [False] * len(pred_names)
    tp = 0
    matches = []
    
    # Greedy matching: for each gold element, find best unmatched predicted element
    for gold_idx, gold_name in enumerate(gold_names):
        best_pred_idx = None
        best_similarity = -1.0
        
        # Find best unmatched predicted element
        for pred_idx, pred_name in enumerate(pred_names):
            if pred_matched[pred_idx]:
                continue  # Already matched
            
            # Calculate similarity
            try:
                model = _get_embedding_model(config.embedding_model)
                # Jina embeddings v3 supports task-specific encoding
                encode_kwargs = {}
                if "jina-embeddings-v3" in config.embedding_model.lower():
                    # Use text-matching mode for symmetric similarity tasks
                    encode_kwargs["task"] = "text-matching"
                embeddings = model.encode(
                    [gold_name, pred_name], 
                    convert_to_numpy=True, 
                    show_progress_bar=False,
                    **encode_kwargs
                )
                similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0])
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Similarity calculation failed: {e}")
                similarity = 0.0
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_pred_idx = pred_idx
        
        # If best match is above threshold, count as true positive
        is_valid = best_pred_idx is not None and best_similarity >= config.semantic_threshold
        if is_valid:
            pred_matched[best_pred_idx] = True
            tp += 1
        
        matches.append((gold_idx, best_pred_idx, best_similarity, is_valid))
        
        if config.debug_matching:
            if best_pred_idx is not None:
                status = "VALID (TP)" if is_valid else "INVALID (below threshold)"
                print(f"  Gold[{gold_idx}] '{gold_names[gold_idx][:50]}' <-> "
                      f"Pred[{best_pred_idx}] '{pred_names[best_pred_idx][:50]}' "
                      f"(similarity={best_similarity:.4f}) {status}")
            else:
                print(f"  Gold[{gold_idx}] '{gold_names[gold_idx][:50]}' <-> NO MATCH")
    
    # False positives: predicted elements not matched above threshold
    fp = sum(1 for matched in pred_matched if not matched)
    
    # False negatives: gold elements not matched above threshold
    fn = len(gold_names) - tp
    
    # Debug output
    if config.debug_matching:
        print(f"\n{'='*80}")
        print(f"[DEBUG {element_type}] Results:")
        print(f"{'='*80}")
        print(f"True Positives (TP): {tp}")
        for g_idx, p_idx, sim, is_valid in matches:
            if is_valid and p_idx is not None:
                print(f"  OK Gold[{g_idx}] '{gold_names[g_idx][:50]}' <-> Pred[{p_idx}] '{pred_names[p_idx][:50]}' (similarity={sim:.4f})")
        
        print(f"\nFalse Positives (FP): {fp}")
        for p_idx, matched in enumerate(pred_matched):
            if not matched:
                print(f"  - Pred[{p_idx}] '{pred_names[p_idx][:50]}' (not matched)")
        
        print(f"\nFalse Negatives (FN): {fn}")
        for g_idx, p_idx, sim, is_valid in matches:
            if not is_valid:
                if p_idx is not None:
                    print(f"  - Gold[{g_idx}] '{gold_names[g_idx][:50]}' (matched to Pred[{p_idx}], similarity={sim:.4f} < {config.semantic_threshold})")
                else:
                    print(f"  - Gold[{g_idx}] '{gold_names[g_idx][:50]}' (no match found)")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\n{'='*80}")
        print(f"[DEBUG {element_type}] Metrics Calculation:")
        print(f"  Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.4f}")
        print(f"  Recall    = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.4f}")
        print(f"  F1        = 2 * P * R / (P + R) = 2 * {precision:.4f} * {recall:.4f} / ({precision:.4f} + {recall:.4f}) = {f1:.4f}")
        print(f"{'='*80}\n")
    
    return tp, fp, fn


def _find_semantic_match(
    query_name: str,
    candidate_names: Sequence[str],
    used_flags: Sequence[bool],
    config: EvaluationConfig,
) -> Optional[int]:
    """Find matching element using semantic similarity.
    
    Args:
        query_name: The name to search for (from gold standard)
        candidate_names: The names to search in (from predicted)
        used_flags: Flags indicating which candidates are already matched
        config: Evaluation configuration
        
    Returns:
        Index of the best matching candidate, or None if no match found
    """
    if not config.use_semantic_matching:
        return None

    # Get unused candidate names
    unused_candidates = [(idx, name) for idx, (name, used) in enumerate(zip(candidate_names, used_flags)) if not used]
    if not unused_candidates:
        return None

    try:
        # Load embedding model
        model = _get_embedding_model(config.embedding_model)

        # Prepare texts for embedding
        texts = [query_name] + [name for _, name in unused_candidates]

        # Generate embeddings (local, no API calls)
        # Use text-matching mode for jina-embeddings-v3 (optimal for symmetric similarity)
        encode_kwargs = {}
        if "jina-embeddings-v3" in config.embedding_model.lower():
            encode_kwargs["task"] = "text-matching"
        
        embeddings = model.encode(
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            **encode_kwargs
        )

        # Calculate cosine similarity
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # Find best match
        best_local_idx = np.argmax(similarities)
        best_score = similarities[best_local_idx]

        if best_score >= config.semantic_threshold:
            return unused_candidates[best_local_idx][0]

    except Exception:
        # Fallback: if embedding fails, return None
        return None

    return None


# ---------------------------------------------------------------------------
# Extraction helpers for nested format
# ---------------------------------------------------------------------------

def _extract_elements_from_nested(model: BPMNModelJsonNested) -> Dict[str, List[BPMNElement]]:
    """Extract flat element lists from nested BPMN model for evaluation.
    
    This function recursively extracts all node elements, including those
    nested within gateway branches.
    """
    pools = []
    lanes = []
    activities = []
    events = []
    gateways = []
    
    element_counter = {"pool": 0, "lane": 0, "activity": 0, "event": 0, "gateway": 0}
    
    from ..models.bpmn import ProcessTask, ProcessEvent, ProcessGateway
    
    def extract_process_element(
        elem: Union[ProcessTask, ProcessEvent, ProcessGateway],
        pool_lanes: List,
    ) -> Optional[str]:
        """Recursively extract a process element and all nested elements.
        
        Returns:
            The extracted element ID, or None if element was not extracted.
        """
        lane_idx = elem.laneIndex if hasattr(elem, 'laneIndex') else 0
        lane_id = f"lane_{lane_idx}" if lane_idx < len(pool_lanes) else None
        
        element_id = None
        
        if isinstance(elem, ProcessTask):
            activity_id = f"activity_{element_counter['activity']}"
            element_counter["activity"] += 1
            element_id = activity_id
            activities.append(BPMNElement(
                element_type="activity",
                id=activity_id,
                name=elem.name,
                properties={"subType": elem.subType},
                lane=lane_id
            ))
        elif isinstance(elem, ProcessEvent):
            event_id = f"event_{element_counter['event']}"
            element_counter["event"] += 1
            element_id = event_id
            events.append(BPMNElement(
                element_type="event",
                id=event_id,
                name=elem.name,
                properties={"subType": elem.subType or "default", "type": "intermediate"},
                lane=lane_id
            ))
        elif isinstance(elem, ProcessGateway):
            gateway_id = f"gateway_{element_counter['gateway']}"
            element_counter["gateway"] += 1
            element_id = gateway_id
            gateway_name = elem.condition or "Gateway"
            gateways.append(BPMNElement(
                element_type="gateway",
                id=gateway_id,
                name=gateway_name,
                properties={"gatewayType": elem.type},
                lane=lane_id
            ))
            
            # Extract elements from gateway branches
            for branch in elem.branches:
                for branch_elem in branch.branch:
                    extract_process_element(branch_elem, pool_lanes)
        
        return element_id
    
    for pool in model.pools:
        # Extract pool
        pool_id = f"pool_{element_counter['pool']}"
        element_counter["pool"] += 1
        pools.append(BPMNElement(
            element_type="pool",
            id=pool_id,
            name=pool.name,
            properties={}
        ))
        
        # Extract lanes
        for lane in pool.lanes:
            lane_id = f"lane_{element_counter['lane']}"
            element_counter["lane"] += 1
            lanes.append(BPMNElement(
                element_type="lane",
                id=lane_id,
                name=lane.name,
                properties={"pool": pool_id}
            ))
        
        # Extract start event
        if pool.startEvent:
            event_id = f"event_{element_counter['event']}"
            element_counter["event"] += 1
            lane_idx = pool.startEvent.laneIndex
            lane_id = f"lane_{lane_idx}" if lane_idx < len(pool.lanes) else None
            events.append(BPMNElement(
                element_type="event",
                id=event_id,
                name=pool.startEvent.name,
                properties={"subType": pool.startEvent.subType or "default", "type": "start"},
                lane=lane_id
            ))
        
        # Extract process elements recursively (including gateway branches)
        for elem in pool.process:
            extract_process_element(elem, pool.lanes)
        
        # Extract end event
        if pool.endEvent:
            event_id = f"event_{element_counter['event']}"
            element_counter["event"] += 1
            if isinstance(pool.endEvent, dict):
                end_name = pool.endEvent.get('name', 'End')
                lane_idx = pool.endEvent.get('laneIndex', 0)
            else:
                end_name = pool.endEvent.name
                lane_idx = pool.endEvent.laneIndex
            lane_id = f"lane_{lane_idx}" if lane_idx < len(pool.lanes) else None
            events.append(BPMNElement(
                element_type="event",
                id=event_id,
                name=end_name,
                properties={"type": "end"},
                lane=lane_id
            ))
        
    return {
        "pools": pools,
        "lanes": lanes,
        "activities": activities,
        "events": events,
        "gateways": gateways,
    }




# ---------------------------------------------------------------------------
# Public evaluation API
# ---------------------------------------------------------------------------

def evaluate_sample(
    sample_id: str,
    predicted: BPMNModelJsonNested,
    gold: GoldBPMNModel,
    config: Optional[EvaluationConfig] = None,
) -> SampleEvaluation:
    config = config or EvaluationConfig(
        embedding_model=settings.eval_semantic_embedding_model
    )

    # Extract elements from nested format for predicted (system output)
    predicted_elements = _extract_elements_from_nested(predicted)
    
    # Extract elements from gold standard (supports both flat and nested formats)
    if gold.flat_elements is not None:
        # Use flat format directly (preferred for gold standards)
        gold_elements = gold.flat_elements
    elif gold.bpmn is not None:
        # Legacy: extract from nested format
        gold_elements = _extract_elements_from_nested(gold.bpmn)
    else:
        raise ValueError(f"Gold model for sample {sample_id} has neither flat_elements nor bpmn")
    
    # Combine pools and lanes into "roles" for evaluation
    if "pools" in predicted_elements or "lanes" in predicted_elements:
        predicted_elements["roles"] = predicted_elements.get("pools", []) + predicted_elements.get("lanes", [])
    if "pools" in gold_elements or "lanes" in gold_elements:
        gold_elements["roles"] = gold_elements.get("pools", []) + gold_elements.get("lanes", [])
    
    per_type_metrics: Dict[str, ElementMetrics] = {}
    
    for element_type in config.element_types:
        pred_elements_list = predicted_elements.get(element_type, [])
        gold_elements_list = gold_elements.get(element_type, [])
        tp, fp, fn = _match_elements(pred_elements_list, gold_elements_list, config, element_type)
        per_type_metrics[element_type] = ElementMetrics(
            element_type=element_type,
            tp=tp,
            fp=fp,
            fn=fn,
        )
    return SampleEvaluation(
        sample_id=sample_id,
        process_name=gold.process_name,
        metrics=per_type_metrics,
    )


def summarize_dataset_results(
    dataset_name: str,
    sample_results: Sequence[SampleEvaluation],
    element_types: Optional[Iterable[str]] = None,
) -> DatasetEvaluationSummary:
    if not sample_results:
        raise ValueError("Cannot summarize empty evaluation results.")

    element_types = tuple(element_types) if element_types else DEFAULT_ELEMENT_TYPES

    aggregated: Dict[str, ElementMetrics] = {}
    for elem_type in element_types:
        tp = sum(result.metrics.get(elem_type, ElementMetrics(elem_type, 0, 0, 0)).tp for result in sample_results)
        fp = sum(result.metrics.get(elem_type, ElementMetrics(elem_type, 0, 0, 0)).fp for result in sample_results)
        fn = sum(result.metrics.get(elem_type, ElementMetrics(elem_type, 0, 0, 0)).fn for result in sample_results)
        aggregated[elem_type] = ElementMetrics(elem_type, tp, fp, fn)

    precision, recall, f1 = _micro_average(aggregated.values())
    

    # Calculate timing statistics
    generation_times = [
        result.generation_time_seconds 
        for result in sample_results 
        if result.generation_time_seconds is not None
    ]
    
    avg_generation_time = None
    min_generation_time = None
    max_generation_time = None
    total_generation_time = None
    
    single_generation_time = None
    if generation_times:
        if len(generation_times) == 1:
            single_generation_time = generation_times[0]
        else:
            avg_generation_time = sum(generation_times) / len(generation_times)
            min_generation_time = min(generation_times)
            max_generation_time = max(generation_times)
            total_generation_time = sum(generation_times)

    judge_entries: List[LLMJudgeSummaryEntry] = []
    for result in sample_results:
        if result.judge is not None:
            judge_entries.append(
                LLMJudgeSummaryEntry(
                    sample_id=result.sample_id,
                    semantic_alignment_score=result.judge.semantic_alignment_score,
                    justification=result.judge.justification,
                )
            )
    llm_judge_summary = None
    if judge_entries:
        llm_judge_summary = LLMJudgeDatasetSummary(
            mean_semantic_alignment_score=sum(e.semantic_alignment_score for e in judge_entries)
            / len(judge_entries),
            samples=judge_entries,
        )

    return DatasetEvaluationSummary(
        dataset_name=dataset_name,
        sample_count=len(sample_results),
        precision=precision,
        recall=recall,
        f1=f1,
        generation_time_seconds=single_generation_time,
        avg_generation_time_seconds=avg_generation_time,
        min_generation_time_seconds=min_generation_time,
        max_generation_time_seconds=max_generation_time,
        total_generation_time_seconds=total_generation_time,
        llm_judge=llm_judge_summary,
    )


def _mean_std(values: Sequence[float]) -> RunStatistics:
    """Compute mean and standard deviation for a list of values."""
    if not values:
        return RunStatistics(mean=0.0, std=0.0, n_runs=0)
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr)) if n > 1 else 0.0
    return RunStatistics(mean=mean_val, std=std_val, n_runs=n)


def aggregate_run_statistics(
    runs: Sequence[Sequence[SampleEvaluation]],
    dataset_name: str,
    setting: str,
    open_source: bool,
    element_types: Optional[Iterable[str]] = None,
) -> AggregatedRunSummary:
    """
    Compute mean and standard deviation of metrics over multiple evaluation runs.

    Args:
        runs: List of runs; each run is a list of SampleEvaluation (one per sample).
        dataset_name: Name of the dataset.
        setting: Setting name (e.g. setting_5).
        open_source: True if using Ollama, False if Azure OpenAI.
        element_types: Element types to include (default: DEFAULT_ELEMENT_TYPES).

    Returns:
        AggregatedRunSummary with mean ± std for each metric.
    """
    from collections import defaultdict

    element_types = tuple(element_types) if element_types else DEFAULT_ELEMENT_TYPES

    run_summaries: List[DatasetEvaluationSummary] = []
    for run_evals in runs:
        if not run_evals:
            continue
        summary = summarize_dataset_results(dataset_name, run_evals, element_types)
        run_summaries.append(summary)

    if not run_summaries:
        raise ValueError("No valid runs to aggregate.")

    n_runs = len(run_summaries)
    n_samples = run_summaries[0].sample_count

    def _stats(getter) -> RunStatistics:
        vals = [getter(s) for s in run_summaries]
        vals_clean = [v if v is not None else 0.0 for v in vals]
        return _mean_std(vals_clean)

    def _per_run_mean_generation_seconds(s: DatasetEvaluationSummary) -> float:
        if s.avg_generation_time_seconds is not None:
            return s.avg_generation_time_seconds
        if s.generation_time_seconds is not None:
            return s.generation_time_seconds
        return 0.0

    sample_to_evals: Dict[str, List[SampleEvaluation]] = defaultdict(list)
    for run_evals in runs:
        for ev in run_evals:
            sample_to_evals[ev.sample_id].append(ev)

    per_sample: Dict[str, Dict[str, RunStatistics]] = {}
    for sample_id, evals_list in sample_to_evals.items():
        if not evals_list:
            continue
        sample_metrics: Dict[str, RunStatistics] = {}

        gen_times = [e.generation_time_seconds for e in evals_list if e.generation_time_seconds is not None]
        if gen_times:
            sample_metrics["generation_time_seconds"] = _mean_std(gen_times)

        judge_scores = [e.judge.semantic_alignment_score for e in evals_list if e.judge is not None]
        if judge_scores:
            sample_metrics["judge_semantic_alignment_score"] = _mean_std(
                [float(s) for s in judge_scores]
            )

        precisions = []
        recalls = []
        f1s = []
        for ev in evals_list:
            all_metrics = [ev.metrics.get(et) for et in element_types]
            all_metrics = [m for m in all_metrics if m is not None]
            if all_metrics:
                tp = sum(m.tp for m in all_metrics)
                fp = sum(m.fp for m in all_metrics)
                fn = sum(m.fn for m in all_metrics)
                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) else 0.0
                precisions.append(p)
                recalls.append(r)
                f1s.append(f1)
        if f1s:
            sample_metrics["precision"] = _mean_std(precisions)
            sample_metrics["recall"] = _mean_std(recalls)
            sample_metrics["f1"] = _mean_std(f1s)

        per_sample[sample_id] = sample_metrics

    # Dataset-level: collect all judge scores across runs and samples
    all_judge_scores: List[float] = []
    for run_evals in runs:
        for ev in run_evals:
            if ev.judge is not None:
                all_judge_scores.append(float(ev.judge.semantic_alignment_score))
    judge_stats = _mean_std(all_judge_scores) if all_judge_scores else None

    return AggregatedRunSummary(
        dataset_name=dataset_name,
        setting=setting,
        open_source=open_source,
        n_runs=n_runs,
        n_samples=n_samples,
        precision=_stats(lambda s: s.precision),
        recall=_stats(lambda s: s.recall),
        f1=_stats(lambda s: s.f1),
        avg_generation_time_seconds=_stats(_per_run_mean_generation_seconds),
        judge_semantic_alignment_score=judge_stats,
        per_sample=per_sample,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _micro_average(metrics: Iterable[ElementMetrics]) -> Tuple[float, float, float]:
    tp = sum(m.tp for m in metrics)
    fp = sum(m.fp for m in metrics)
    fn = sum(m.fn for m in metrics)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


__all__ = [
    "EvaluationConfig",
    "ElementMetrics",
    "SampleEvaluation",
    "LLMJudgeSummaryEntry",
    "LLMJudgeDatasetSummary",
    "DatasetEvaluationSummary",
    "RunStatistics",
    "AggregatedRunSummary",
    "evaluate_sample",
    "summarize_dataset_results",
    "aggregate_run_statistics",
]

