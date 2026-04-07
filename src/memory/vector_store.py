"""
Vector Store — Tier 3 of the 3-tier memory architecture.

Persistent, semantically-searchable long-term memory backed by
ChromaDB (local, no external infrastructure needed).  Findings
flushed from the episodic buffer are embedded and stored here.

On subsequent queries the agent performs a cosine-similarity
retrieval before falling back to live web search, avoiding
redundant API calls and reducing session cost.
"""
from __future__ import annotations

import hashlib
import os


class VectorStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    ChromaDB uses the all-MiniLM-L6-v2 ONNX model for embeddings
    locally — ro no external embedding API key required.
    """

    def __init__(
        self,
        collection_name: str = "research",
        persist_dir: str = "./.chroma",
    ) -> None:
        try:
            import chromadb  # noqa: F401 — lazy import to avoid hard dep at import time
        except ImportError as e:
            raise ImportError(
                "chromadb is required for VectorStore. "
                "Install it with: pip install chromadb"
            ) from e

        import chromadb as _chromadb

        self._persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self._client = _chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(self, query: str, finding: str, metadata: dict | None = None) -> None:
        """
        Upsert a (query, finding) pair.

        The document is stored as "Q: <query>\nA: <finding>" so the
        embedding captures both the question context and the answer.
        Using upsert means re-running the agent won't create duplicates.
        """
        doc_id = hashlib.md5(query.encode()).hexdigest()
        document = f"Q: {query}\nA: {finding}"
        meta = metadata or {"query": query}

        self._collection.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[meta],
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve the *top_k* most relevant past findings for *query*.

        Returns an empty string if the store is empty (no API cost).
        """
        if self._collection.count() == 0:
            return ""

        n = min(top_k, self._collection.count())
        results = self._collection.query(
            query_texts=[query],
            n_results=n,
        )

        docs: list[str] = results.get("documents", [[]])[0]
        return "\n\n".join(docs) if docs else ""

    def count(self) -> int:
        return self._collection.count()

    def __repr__(self) -> str:
        return f"VectorStore(entries={self.count()}, dir='{self._persist_dir}')"
