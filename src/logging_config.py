"""Shared logging configuration helpers."""

from __future__ import annotations

import logging

from .config import settings

_NOISY_THIRD_PARTY_LOGGERS = (
    "httpx",
    "sentence_transformers",
    "urllib3",
    "transformers",
    "chromadb",
)


def configure_third_party_logging(*, debug: bool | None = None) -> None:
    """Set third-party logger levels based on debug mode.

    - debug=True  -> INFO
    - debug=False -> WARNING
    """
    effective_debug = settings.debug if debug is None else debug
    level = logging.INFO if effective_debug else logging.WARNING

    for logger_name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(level)
