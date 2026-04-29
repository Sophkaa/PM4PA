"""Ingestion use-case orchestration for document embedding."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .pipeline import GraphRAGSystem

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".html", ".htm"}


def find_documents(documents_dir: Path, extensions: Optional[Iterable[str]] = None) -> List[Path]:
    """Find all supported documents in a directory recursively."""
    suffixes = {ext.lower() for ext in (extensions or SUPPORTED_EXTENSIONS)}
    if not documents_dir.exists():
        return []
    documents: List[Path] = []
    for file_path in documents_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in suffixes:
            documents.append(file_path)
    return sorted(documents)


def _collect_source_dirs(project_root: Path) -> List[Tuple[str, Path]]:
    data_dir = project_root / "data"
    if data_dir.exists():
        return [("data", data_dir)]
    return []


async def ingest_all_documents(update_existing: bool = True) -> None:
    """Ingest all supported files from configured source directories."""
    print("Initializing Graph-based Agentic RAG System...")
    rag_system = GraphRAGSystem()
    project_root = Path(__file__).resolve().parents[2]
    source_dirs = _collect_source_dirs(project_root)
    documents: List[Tuple[Path, str]] = []

    for source_name, source_dir in source_dirs:
        source_docs = find_documents(source_dir)
        documents.extend((doc, source_name) for doc in source_docs)

    if not documents:
        print("\n⚠️  No supported files found in data/.")
        return

    print(f"\nFound {len(documents)} document(s).")
    total_chunks = 0
    successful = 0
    skipped = 0
    failed = 0

    for file_path, source_folder in documents:
        base_dir = project_root / source_folder
        relative_path_from_base = file_path.relative_to(base_dir)
        relative_path = f"{source_folder}/{relative_path_from_base}"
        subfolder = str(relative_path_from_base.parent) if relative_path_from_base.parent != Path(".") else None

        print(f"\nProcessing: {relative_path}")
        try:
            existing_docs = rag_system.vector_store.collection.get(where={"file_path": relative_path}, limit=1)
            has_existing = bool(existing_docs and existing_docs.get("ids"))
            if has_existing and not update_existing:
                print("  ⏭️  Already embedded, skipping.")
                skipped += 1
                continue
            if has_existing and update_existing:
                print("  ⚠️  Existing chunks found, deleting before re-ingestion...")
                rag_system.vector_store.delete(where={"file_path": relative_path})

            metadata = {
                "source": "file_ingestion",
                "file_name": file_path.name,
                "file_path": relative_path,
                "source_folder": source_folder,
            }
            if subfolder:
                metadata["subfolder"] = subfolder
                metadata["folder_path"] = str(relative_path_from_base.parent)

            chunk_ids = await rag_system.ingest_file(
                file_path=file_path,
                metadata=metadata,
                use_semantic_chunking=True,
            )
            successful += 1
            total_chunks += len(chunk_ids)
            print(f"  ✅ Chunks stored: {len(chunk_ids)}")
        except Exception as exc:
            failed += 1
            print(f"  ❌ Failed: {exc}")

    print("\n" + "=" * 60)
    print("Ingestion Summary")
    print("=" * 60)
    print(f"Total documents found: {len(documents)}")
    print(f"Successfully ingested: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Total chunks created: {total_chunks}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest documents into vector storage.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that are already embedded.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(ingest_all_documents(update_existing=not args.skip_existing))


if __name__ == "__main__":
    main()

