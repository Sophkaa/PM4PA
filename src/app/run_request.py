"""Single-request helper for non-UI usage."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict, Optional

from .pipeline import GraphRAGSystem


def _parse_file_filter(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    return json.loads(raw)


async def run_once(query: str, setting: str, file_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run one pipeline request and return a serializable payload."""
    rag_system = GraphRAGSystem(setting_name=setting)
    result = await rag_system.query(query=query, file_filter=file_filter)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one GraphRAG request.")
    parser.add_argument("--query", required=True, help="User request/query to process.")
    parser.add_argument(
        "--setting",
        default="setting_1",
        choices=["setting_1", "setting_2", "setting_3", "setting_4", "setting_5"],
        help="Graph setting to use.",
    )
    parser.add_argument(
        "--file-filter",
        default=None,
        help="Optional JSON string passed to retrieval filtering.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    file_filter = _parse_file_filter(args.file_filter)
    payload = asyncio.run(run_once(args.query, args.setting, file_filter=file_filter))
    if args.pretty:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(json.dumps(payload, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

