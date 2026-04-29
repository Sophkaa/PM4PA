"""Simple async client for calling LLM chat APIs."""

import asyncio
import logging
import sys
import time
from typing import Dict, List, Optional

import httpx

from ...config import settings

logger = logging.getLogger(__name__)
_SPINNER_FRAMES = ("|", "/", "-", "\\")


async def _post_with_spinner(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: Dict[str, str],
    json_payload: Dict[str, object],
    label: str,
    params: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    """POST request with lightweight CLI spinner for long LLM calls."""
    if not sys.stdout.isatty():
        return await client.post(url, headers=headers, params=params, json=json_payload)

    done = False
    response: Optional[httpx.Response] = None
    error: Optional[BaseException] = None
    start = time.perf_counter()

    async def _runner() -> None:
        nonlocal done, response, error
        try:
            response = await client.post(url, headers=headers, params=params, json=json_payload)
        except BaseException as exc:
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
    if error is None and response is not None:
        print(f"\r✓ {label} ({elapsed:.1f}s)")
        return response

    print(f"\r✗ {label} ({elapsed:.1f}s)")
    raise error if error is not None else RuntimeError("LLM request failed without exception details.")


def _log_ollama_model_missing_hint(status_code: int, error_text: str, model_used: str) -> None:
    """If the gateway returns 404/400 model-not-found, emit one actionable WARNING (not buried in ERROR noise)."""
    if status_code != 400 or not model_used:
        return
    low = error_text.lower()
    if "not found" not in low or "model" not in low:
        return
    logger.warning(
        "Ollama/model gateway reports the model is unavailable (%r). "
        "Use a tag that exists on this server (check the UI or `ollama list`) and set the matching "
        "AGENTIC_*_MODEL values in your environment.",
        model_used,
    )


def _build_chat_url() -> str:
    """Build Ollama chat API URL for OPEN_SOURCE mode."""
    base = settings.ollama_base_url.rstrip("/")
    if base.endswith("/api"):
        return f"{base}/chat"
    return f"{base}/api/chat"


def _get_timeout() -> float:
    """Return global API timeout in seconds."""
    return settings.api_timeout


def _spinner_model_label(request_model: str) -> str:
    """Model/deployment name for CLI progress (matches the HTTP target, not an unused alias)."""
    if settings.open_source:
        return request_model
    deployment = (settings.azure_deployment or "").strip()
    return deployment if deployment else "gpt-4o-2024-11-20"


def _get_azure_timeout(read_timeout: float) -> httpx.Timeout:
    """Create httpx.Timeout configuration for Azure OpenAI calls."""
    return httpx.Timeout(connect=5.0, read=read_timeout, write=10.0, pool=5.0)


def _log_azure_usage(data: Dict) -> None:
    """Log usage information from Azure OpenAI response."""
    usage = data.get("usage")
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        logger.info(
            "Azure OpenAI Usage - Prompt: %d tokens, Completion: %d tokens, Total: %d tokens",
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

        if settings.debug:
            logger.debug(
                "Azure OpenAI Usage (debug) - %d prompt + %d completion = %d total tokens",
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )


def _build_retry_message(status_code: int, attempt: int, retry_after_header: Optional[str]) -> tuple[int, str]:
    """Build retry wait time and message for retryable HTTP status codes."""
    if status_code == 429:
        wait_time = int(retry_after_header) if retry_after_header and retry_after_header.isdigit() else (attempt + 1) * 30
        wait_time = min(wait_time, 120)
        msg = f"429 Too Many Requests - Retrying in {wait_time}s"
    else:
        wait_time = (attempt + 1) * 5
        msg = f"Server Error ({status_code}) - Retrying in {wait_time}s"
    return wait_time, msg


async def call_ollama_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_retries: int = 2,
    timeout: Optional[float] = None,
) -> str:
    """Call chat endpoint and return assistant content text."""
    if not model:
        raise ValueError("model is required for call_ollama_chat().")
    model_name = model
    timeout_value = timeout if timeout is not None else _get_timeout()

    if settings.open_source:
        url = _build_chat_url()
        payload: Dict[str, object] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": float(temperature if temperature is not None else settings.agentic_temperature)},
        }
        headers: Dict[str, str] = {}
        if settings.open_webui_api_key:
            headers["Authorization"] = f"Bearer {settings.open_webui_api_key}"
    else:
        base_url = settings.azure_base_url.rstrip("/")
        deployment_name = settings.azure_deployment or "gpt-4o-2024-11-20"
        url = f"{base_url}/openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-15-preview"
        payload = {
            "messages": messages,
            "temperature": float(temperature if temperature is not None else settings.agentic_temperature),
        }
        headers = {
            "api-key": settings.azure_api_key,
            "Content-Type": "application/json",
        }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            client_timeout = timeout_value if settings.open_source else _get_azure_timeout(timeout_value)
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                resp = await _post_with_spinner(
                    client,
                    url=url,
                    headers=headers,
                    json_payload=payload,
                    label=f"LLM chat call [{_spinner_model_label(model_name)}]",
                )
                resp.raise_for_status()
                data = resp.json()

            if not settings.open_source:
                _log_azure_usage(data)

            if settings.open_source:
                message = data.get("message") or {}
                content = message.get("content")
                if not content:
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices:
                        content = choices[0].get("message", {}).get("content") or choices[0].get("text")
            else:
                choices = data.get("choices")
                content = choices[0].get("message", {}).get("content") if isinstance(choices, list) and choices else None

            if not content:
                raise RuntimeError("LLM response did not contain any content.")
            return str(content).strip()

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 400:
                error_msg = e.response.text if hasattr(e.response, "text") else str(e)
                logger.error("400 Bad Request Error (call_ollama_chat):")
                logger.error("URL: %s", url)
                logger.error("Error details: %s", error_msg)
                _log_ollama_model_missing_hint(status_code, error_msg, model_name)
                if settings.debug:
                    logger.debug("400 Bad Request - Full error: %s", error_msg)

            if status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait_time, msg = _build_retry_message(
                    status_code=status_code,
                    attempt=attempt,
                    retry_after_header=e.response.headers.get("Retry-After"),
                )
                if settings.debug:
                    logger.debug("%s (attempt %d/%d)", msg, attempt + 1, max_retries)
                logger.warning("%s (attempt %d/%d)", msg, attempt + 1, max_retries)
                await asyncio.sleep(wait_time)
                last_error = e
                continue
            raise
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            if attempt < max_retries:
                wait_time = (attempt + 1) * 5
                if settings.debug:
                    logger.debug(
                        "Timeout/Connection error - Retrying in %ds (attempt %d/%d)",
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                await asyncio.sleep(wait_time)
                last_error = e
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Failed to get response from LLM API after retries.")


async def call_ollama_json(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_retries: int = 2,
    timeout: Optional[float] = None,
) -> str:
    """Call chat endpoint with JSON response format and return raw JSON text."""
    if not model:
        raise ValueError("model is required for call_ollama_json().")
    model_name = model
    timeout_value = timeout if timeout is not None else _get_timeout()

    if settings.open_source:
        url = _build_chat_url()
        payload: Dict[str, object] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": float(temperature if temperature is not None else settings.agentic_temperature)},
        }
        headers: Dict[str, str] = {}
        if settings.open_webui_api_key:
            headers["Authorization"] = f"Bearer {settings.open_webui_api_key}"
        params = None
    else:
        base_url = settings.azure_base_url.rstrip("/")
        deployment_name = settings.azure_deployment or "gpt-4o-2024-11-20"
        url = f"{base_url}/openai/deployments/{deployment_name}/chat/completions"
        params = {"api-version": "2024-02-15-preview"}
        payload = {
            "messages": [{"role": "system", "content": "Return ONLY valid JSON. Do not use markdown. Do not add any extra text."}, *messages],
            "temperature": float(temperature if temperature is not None else settings.agentic_temperature),
            "response_format": {"type": "json_object"},
        }
        headers = {
            "api-key": settings.azure_api_key,
            "Content-Type": "application/json",
        }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            client_timeout = timeout_value if settings.open_source else _get_azure_timeout(timeout_value)
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                resp = await _post_with_spinner(
                    client,
                    url=url,
                    headers=headers,
                    params=params,
                    json_payload=payload,
                    label=f"LLM json call [{_spinner_model_label(model_name)}]",
                )
                resp.raise_for_status()
                data = resp.json()

            if not settings.open_source:
                _log_azure_usage(data)

            if settings.open_source:
                message = data.get("message") or {}
                content = message.get("content")
                if not content:
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices:
                        content = choices[0].get("message", {}).get("content") or choices[0].get("text")
            else:
                choices = data.get("choices")
                content = choices[0].get("message", {}).get("content") if isinstance(choices, list) and choices else None

            if not content:
                raise RuntimeError("LLM response did not contain any content.")

            content = str(content).strip()
            if not settings.open_source and content.startswith("```"):
                lines = content.split("\n")
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                content = "\n".join(json_lines).strip()
                if content.endswith("```"):
                    content = content[:-3].strip()

            return content

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 400:
                error_msg = e.response.text if hasattr(e.response, "text") else str(e)
                logger.error("400 Bad Request Error (call_ollama_json):")
                logger.error("URL: %s", url)
                logger.error("Error details: %s", error_msg)
                _log_ollama_model_missing_hint(status_code, error_msg, model_name)
                if settings.debug:
                    logger.debug("400 Bad Request - Full error: %s", error_msg)

            if status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait_time, msg = _build_retry_message(
                    status_code=status_code,
                    attempt=attempt,
                    retry_after_header=e.response.headers.get("Retry-After"),
                )
                if settings.debug:
                    logger.debug("%s (attempt %d/%d)", msg, attempt + 1, max_retries)
                logger.warning("%s (attempt %d/%d)", msg, attempt + 1, max_retries)
                await asyncio.sleep(wait_time)
                last_error = e
                continue
            raise
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            if attempt < max_retries:
                wait_time = (attempt + 1) * 5
                if settings.debug:
                    logger.debug(
                        "Timeout/Connection error - Retrying in %ds (attempt %d/%d)",
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                await asyncio.sleep(wait_time)
                last_error = e
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Failed to get response from LLM API after retries.")


__all__ = ["call_ollama_chat", "call_ollama_json"]

