"""Open WebUI API client for embeddings and chat completions."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

from ...config import settings

logger = logging.getLogger(__name__)

# Cache for local embedding models (loaded once, reused for all documents)
_local_embedding_model_cache: Dict[str, Any] = {}

# Mapping from Ollama model names to HuggingFace equivalents for local fallback
# Use models that exist on HuggingFace (sentence-transformers or mixedbread-ai)
OLLAMA_TO_HF_EMBEDDING_MODEL = {
    "mxbai-embed-large": "mixedbread-ai/deepset-mxbai-embed-de-large-v1",
    "mxbai-embed-large:335m": "mixedbread-ai/deepset-mxbai-embed-de-large-v1",
    # Support configured aliases/variants for the same German embedding model
    "deepset-mxbai-embed-de-large-v1": "mixedbread-ai/deepset-mxbai-embed-de-large-v1",
    "deepset/mxbai-embed-de-large-v1": "mixedbread-ai/deepset-mxbai-embed-de-large-v1",
    "mixedbread-ai/deepset-mxbai-embed-de-large-v1": "mixedbread-ai/deepset-mxbai-embed-de-large-v1",
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-text:latest": "nomic-ai/nomic-embed-text-v1.5",
}
# Default fallback: well-tested multilingual model (used in eval)
DEFAULT_FALLBACK_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class OpenWebUIClient:
    """Client for interacting with Open WebUI API endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url or settings.open_webui_base_url
        self.api_key = api_key or settings.open_webui_api_key
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def _is_direct_ollama_api(self) -> bool:
        """Check if base_url is a direct Ollama API endpoint."""
        return self.base_url.rstrip("/").endswith("/api")

    def _get_embeddings_local_fallback(self, texts: List[str], model: str) -> List[List[float]]:
        """Fallback to local Sentence-Transformer when API fails."""
        hf_model = OLLAMA_TO_HF_EMBEDDING_MODEL.get(
            model
        ) or OLLAMA_TO_HF_EMBEDDING_MODEL.get(
            model.split(":")[0] if ":" in model else model
        ) or DEFAULT_FALLBACK_EMBEDDING_MODEL
        models_to_try = [hf_model]
        if hf_model != DEFAULT_FALLBACK_EMBEDDING_MODEL:
            models_to_try.append(DEFAULT_FALLBACK_EMBEDDING_MODEL)
        for try_model in models_to_try:
            try:
                from sentence_transformers import SentenceTransformer
                if try_model not in _local_embedding_model_cache:
                    _local_embedding_model_cache[try_model] = SentenceTransformer(try_model)
                st_model = _local_embedding_model_cache[try_model]
                embeddings = st_model.encode(texts, convert_to_numpy=True)
                return [emb.tolist() for emb in embeddings]
            except Exception as e:
                logger.warning("Local embedding fallback failed (%s): %s", try_model, e)
                continue
        raise ValueError(
            f"Embedding API failed and local fallback failed for all models. "
            f"Check: 1) Ollama embedding model '{model}' on server, "
            f"2) OPEN_WEBUI_BASE_URL, 3) sentence-transformers installed, 4) network for HuggingFace."
        )

    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings using Ollama/Open WebUI API, with local fallback on failure.

        Args:
            texts: List of texts to embed
            model: Embedding model name (defaults to configured model)

        Returns:
            List of embedding vectors
        """
        model = model or settings.embedding_model

        if settings.use_local_embeddings:
            return await asyncio.to_thread(self._get_embeddings_local_fallback, texts, model)

        model_variants = [model]
        if ":" in model:
            model_variants.append(model.split(":")[0])  # e.g. mxbai-embed-large:335m -> mxbai-embed-large

        # Build list of (url, model_name) to try
        # Open WebUI uses /api/v1/embeddings (OpenAI-compatible)
        base = self.base_url.rstrip("/")
        if self._is_direct_ollama_api():
            base_host = base.rsplit("/api", 1)[0]
            urls_to_try = [
                f"{base}/v1/embeddings",  # Open WebUI OpenAI-compatible
                f"{base_host}/ollama/api/embed",
                f"{base}/embed",
            ]
        else:
            urls_to_try = [
                f"{base}/v1/embeddings",
                f"{base}/ollama/api/embed",
            ]

        last_error = None
        for url in urls_to_try:
            for model_name in model_variants:
                payload = {"model": model_name, "input": texts}
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            url,
                            json=payload,
                            headers=self.headers if self.api_key else {},
                            timeout=60.0,
                        )
                        response.raise_for_status()
                        result = response.json()
                        # OpenAI format: {"data": [{"embedding": [...]}]}
                        if "data" in result:
                            return [item["embedding"] for item in result["data"]]
                        if isinstance(result, list):
                            return [item.get("embedding", []) for item in result]
                        if "embeddings" in result:
                            return result["embeddings"]
                        if "embedding" in result:
                            return [result["embedding"]]
                        return result.get("embeddings", [])
                except httpx.HTTPStatusError as e:
                    last_error = e
                    continue
                except Exception as e:
                    last_error = e
                    continue

        # Fallback to local Sentence-Transformer (run in thread, CPU-bound)
        logger.warning(
            "Embedding API failed (last: %s). Using local Sentence-Transformer fallback.",
            last_error,
        )
        return await asyncio.to_thread(self._get_embeddings_local_fallback, texts, model)


__all__ = ["OpenWebUIClient"]

