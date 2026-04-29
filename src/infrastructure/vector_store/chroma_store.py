"""ChromaDB vector store implementation."""

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from ...config import settings


class ChromaVectorStore:
    """ChromaDB-based vector store for RAG."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self.collection_name = collection_name or settings.chroma_collection_name
        self.db_path = db_path or settings.chroma_db_path

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add documents with embeddings to the vector store."""
        if metadatas is None:
            metadatas = [{}] * len(documents)

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 30,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query the vector store for similar documents."""
        if include is None:
            include = ["documents", "metadatas", "distances"]

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=include,
        )

        return results

    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None) -> None:
        """Delete documents from the vector store."""
        self.collection.delete(ids=ids, where=where)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "path": self.db_path,
        }


__all__ = ["ChromaVectorStore"]

