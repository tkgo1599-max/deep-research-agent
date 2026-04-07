"""
Episodic Buffer — Tier 2 of the 3-tier memory architecture.

A fixed-size FIFO ring of recent (sub-query, finding) episodes.
When the buffer is full the agent flushes it to the vector store
(Tier 3) and clears it, implementing a *summarization cascade*.

Constraint: max_size (default 5 episodes)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Episode:
    query: str
    finding: str
    timestamp: datetime = field(default_factory=datetime.now)


class EpisodicBuffer:
    """
    Short-term ring buffer of the most recent research episodes.

    Provides keyword-overlap similarity search so the agent can
    cheaply retrieve a relevant past finding without a vector lookup.
    When the buffer reaches capacity the caller should flush its
    contents to the vector store before clearing it.
    """

    def __init__(self, max_size: int = 5) -> None:
        self.max_size = max_size
        self._buffer: deque[Episode] = deque()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, query: str, finding: str) -> None:
        """Append a new episode (does NOT auto-flush; caller is responsible)."""
        self._buffer.append(Episode(query=query, finding=finding))

    def clear(self) -> None:
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def is_full(self) -> bool:
        return len(self._buffer) >= self.max_size

    def get_all(self) -> list[tuple[str, str]]:
        """Return all episodes as (query, finding) pairs."""
        return [(ep.query, ep.finding) for ep in self._buffer]

    def search_similar(self, query: str, top_k: int = 2) -> list[str]:
        """
        Lightweight keyword-overlap similarity (no embeddings required).

        Returns up to *top_k* finding strings ordered by descending overlap.
        Stops querying the LLM if a relevant answer already exists in the
        buffer, saving tokens and cost.
        """
        query_words = set(query.lower().split())
        scored: list[tuple[int, str]] = []

        for ep in self._buffer:
            ep_words = set(ep.query.lower().split())
            overlap = len(query_words & ep_words)
            if overlap > 0:
                scored.append((overlap, ep.finding))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [finding for _, finding in scored[:top_k]]

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"EpisodicBuffer(size={len(self._buffer)}/{self.max_size})"
