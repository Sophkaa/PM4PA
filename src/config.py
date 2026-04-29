"""Configuration: load and validate environment variables."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


def get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def get_str(key: str, default: str) -> str:
    val = os.getenv(key)
    if val is None:
        return default
    return val


# Eval metrics: semantic element-name matching (P/R/F1). Override via EVAL_SEMANTIC_EMBEDDING_MODEL.
DEFAULT_EVAL_SEMANTIC_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


class Settings:
    """Application settings from environment (with safe defaults for local dev)."""

    def __init__(self) -> None:
        self.open_source = get_bool("OPEN_SOURCE", True)

        self.ollama_base_url = get_str(
            "OLLAMA_BASE_URL", ""
        ).rstrip("/")
        self.open_webui_base_url = get_str(
            "OPEN_WEBUI_BASE_URL", ""
        ).rstrip("/")
        self.open_webui_api_key = get_str("OPEN_WEBUI_API_KEY", "")

        self.azure_base_url = get_str(
            "AZURE_OPENAI_BASE_URL", ""
        ).rstrip("/")
        self.azure_api_key = get_str("AZURE_OPENAI_API_KEY", "")
        self.azure_deployment = get_str("AZURE_DEPLOYMENT_NAME", "gpt-4o-2024-11-20")

        self.chroma_db_path = get_str("CHROMA_DB_PATH", "./chroma_db")
        self.chroma_collection_name = get_str(
            "CHROMA_COLLECTION_NAME", "public_sector_documents"
        )

        self.embedding_model = get_str(
            "EMBEDDING_MODEL", "deepset/mxbai-embed-de-large-v1"
        )
        self.eval_semantic_embedding_model = get_str(
            "EVAL_SEMANTIC_EMBEDDING_MODEL", DEFAULT_EVAL_SEMANTIC_EMBEDDING_MODEL
        )
        self.use_local_embeddings = get_bool("USE_LOCAL_EMBEDDINGS", True)

        self.agentic_retrieval_model = get_str(
            "AGENTIC_RETRIEVAL_MODEL", "mistral-large:123b"
        )
        self.agentic_draft_model = get_str("AGENTIC_DRAFT_MODEL", "mistral-large:123b")
        self.agentic_bpmn_model = get_str("AGENTIC_BPMN_MODEL", "mistral-large:123b")
        self.agentic_revision_model = get_str("AGENTIC_REVISION_MODEL", "mistral-large:123b")
        self.agentic_validation_model = get_str(
            "AGENTIC_VALIDATION_MODEL", "mistral-large:123b"
        )
        self.agentic_judge_model = get_str("AGENTIC_JUDGE_MODEL", "mistral-large:123b")
        self.agentic_temperature = get_float("AGENTIC_TEMPERATURE", 0.0)

        self.api_timeout = get_float("API_TIMEOUT", 600.0)

        self.max_revision_iterations = get_int("MAX_REVISION_ITERATIONS", 2)

        self.debug = get_bool("DEBUG", True)

        self.retrieval_n_results = get_int("RETRIEVAL_N_RESULTS", 50)
        self.retrieval_top_k_setting_1 = get_int("RETRIEVAL_TOP_K_SETTING_1", 20)
        self.retrieval_top_k = get_int("RETRIEVAL_TOP_K", 30)
        self.retrieval_score_threshold = get_float("RETRIEVAL_SCORE_THRESHOLD", 0.3)

        self.bm25_top_n = get_int("BM25_TOP_N", 100)
        self.bm25_weight = get_float("BM25_WEIGHT", 0.3)
        self.vector_weight = get_float("VECTOR_WEIGHT", 0.7)

        self.relevance_top_n = get_int("RELEVANCE_TOP_N", 20)
        self.relevance_min_high_medium = get_int("RELEVANCE_MIN_HIGH_MEDIUM", 4)
        self.relevance_retry_with_new_synonyms = get_bool(
            "RELEVANCE_RETRY_WITH_NEW_SYNONYMS", True
        )
        self.relevance_max_retries = get_int("RELEVANCE_MAX_RETRIES", 1)

        self.bpmn_service_verify_ssl = get_bool("BPMN_SERVICE_VERIFY_SSL", True)
        self.bpmn_service_url = get_str("BPMN_SERVICE_URL", "https://bpmnchatbot.aau.at/bpmn-generator")

        self.chunk_size = get_int("CHUNK_SIZE", 800)
        self.chunk_overlap = get_int("CHUNK_OVERLAP", 150)
        self.min_chunk_size = get_int("MIN_CHUNK_SIZE", 500)
        self.max_chunk_size = get_int("MAX_CHUNK_SIZE", 1000)

        self.validate()

    def validate(self) -> None:
        """Ensure required variables are set for the selected runtime mode."""
        if self.open_source:
            if not self.ollama_base_url:
                raise ValueError(
                    "OPEN_SOURCE=true requires a non-empty OLLAMA_BASE_URL."
                )
        else:
            if not self.azure_base_url:
                raise ValueError(
                    "OPEN_SOURCE=false requires AZURE_OPENAI_BASE_URL."
                )
            if not self.azure_api_key:
                raise ValueError(
                    "OPEN_SOURCE=false requires AZURE_OPENAI_API_KEY."
                )
            if not self.azure_deployment:
                raise ValueError(
                    "OPEN_SOURCE=false requires AZURE_DEPLOYMENT_NAME."
                )


# Global settings instance
settings = Settings()
