"""
Working Memory — Tier 1 of the 3-tier memory architecture.

A token-limited, in-process context window for a single sub-query.
All content is cleared between sub-queries to enforce the per-call
token constraint.

Constraint: max_tokens (default 2 000 tokens ≈ ~8 000 characters)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MemoryEntry:
    source: str
    content: str
    tokens: int


class WorkingMemory:
    """
    Token-limited scratch-pad for the current sub-query context.

    Enforces a hard ceiling on how many tokens can be held at once,
    preventing any single LLM call from exceeding the stated limit.
    Entries are added in priority order (caller decides priority by
    insertion order); once full, further additions are silently rejected
    and the caller must decide whether to truncate or skip.
    """

    def __init__(self, max_tokens: int = 2_000) -> None:
        self.max_tokens = max_tokens
        self._entries: list[MemoryEntry] = []
        self._used_tokens: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximate token count: 1 token ≈ 4 characters (conservative)."""
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, source: str, content: str) -> bool:
        """
        Attempt to add content labelled with *source*.

        Returns True if accepted, False if adding would exceed the limit.
        Callers should treat False as a signal to stop appending context.
        """
        label = f"[{source}]"
        full_text = f"{label}\n{content}"
        tokens = self.estimate_tokens(full_text)

        if self._used_tokens + tokens > self.max_tokens:
            return False

        self._entries.append(MemoryEntry(source=source, content=content, tokens=tokens))
        self._used_tokens += tokens
        return True

    def clear(self) -> None:
        """Wipe all entries — called at the start of each new sub-query."""
        self._entries.clear()
        self._used_tokens = 0

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_context(self) -> str:
        """Return a formatted string suitable for injection into an LLM prompt."""
        sections = [f"[{e.source}]\n{e.content}" for e in self._entries]
        return "\n\n".join(sections)

    def token_usage(self) -> Tuple[int, int]:
        """Return (used_tokens, max_tokens)."""
        return self._used_tokens, self.max_tokens

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(entries={len(self._entries)}, "
            f"tokens={self._used_tokens}/{self.max_tokens})"
        )
