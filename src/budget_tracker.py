"""
Budget Tracker — enforces hard token and cost constraints per session.

Self-defined session constraints (configurable via .env / config dict):
  - max_session_tokens : 50 000  (prevents runaway token usage)
  - max_session_cost   : $0.10   (hard spending ceiling)
  - max_api_calls      : 20      (prevents infinite retry loops)

These limits are checked BEFORE each LLM call so the agent degrades
gracefully (returns cached / partial answers) rather than silently
overspending.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------------
# Approximate per-token costs (USD) — update as pricing changes
# ---------------------------------------------------------------------------
COST_TABLE: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 0.00000080,   # $0.80 / 1M input tokens
        "output": 0.00000400,  # $4.00 / 1M output tokens
    },
    "claude-3-5-sonnet-20241022": {
        "input": 0.000003,
        "output": 0.000015,
    },
    "gpt-4o-mini": {
        "input": 0.00000015,
        "output": 0.00000060,
    },
}

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class CallRecord:
    timestamp: datetime
    call_index: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class BudgetTracker:
    """
    Tracks cumulative token usage and estimated cost for a session.

    Usage flow
    ----------
    1.  Call can_proceed() at the start of each agent step.
    2.  Call can_afford(estimated_tokens) before constructing a prompt.
    3.  Call record_usage(input, output) immediately after the API response.
    4.  Call get_summary() to retrieve a human-readable usage report.
    5.  Call reset() to start a fresh session without re-instantiating.
    """

    def __init__(
        self,
        max_tokens: int = 50_000,
        max_cost: float = 0.10,
        max_calls: int = 20,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.max_calls = max_calls
        self.default_model = model

        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._total_cost: float = 0.0
        self._call_count: int = 0
        self._records: list[CallRecord] = []

    # ------------------------------------------------------------------
    # Guard methods (call BEFORE making an API request)
    # ------------------------------------------------------------------

    def can_proceed(self) -> bool:
        """True if none of the three hard limits have been reached."""
        return (
            self.total_tokens < self.max_tokens
            and self._total_cost < self.max_cost
            and self._call_count < self.max_calls
        )

    def can_afford(self, estimated_input_tokens: int, model: str | None = None) -> bool:
        """
        True if the estimated cost of the next call fits within the
        remaining budget (tokens AND cost AND call count).
        """
        m = model or self.default_model
        rates = COST_TABLE.get(m, COST_TABLE[_DEFAULT_MODEL])
        estimated_cost = estimated_input_tokens * rates["input"]

        return (
            self.total_tokens + estimated_input_tokens <= self.max_tokens
            and self._total_cost + estimated_cost <= self.max_cost
            and self._call_count < self.max_calls
        )

    # ------------------------------------------------------------------
    # Recording (call AFTER receiving the API response)
    # ------------------------------------------------------------------

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> None:
        m = model or self.default_model
        rates = COST_TABLE.get(m, COST_TABLE[_DEFAULT_MODEL])
        cost = input_tokens * rates["input"] + output_tokens * rates["output"]

        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._total_cost += cost
        self._call_count += 1

        self._records.append(
            CallRecord(
                timestamp=datetime.now(),
                call_index=self._call_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Conservative 4-chars-per-token estimate (no external dependency)."""
        return max(1, len(text) // 4)

    @property
    def total_tokens(self) -> int:
        return self._input_tokens + self._output_tokens

    def get_summary(self) -> dict:
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "api_calls": self._call_count,
            "budget_used_pct": round(self._total_cost / self.max_cost * 100, 1)
            if self.max_cost > 0
            else 0,
            "tokens_used_pct": round(self.total_tokens / self.max_tokens * 100, 1)
            if self.max_tokens > 0
            else 0,
        }

    def reset(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0
        self._records.clear()

    def __repr__(self) -> str:
        s = self.get_summary()
        return (
            f"BudgetTracker(tokens={s['total_tokens']}/{self.max_tokens}, "
            f"cost=${s['total_cost_usd']:.4f}/${self.max_cost:.2f}, "
            f"calls={s['api_calls']}/{self.max_calls})"
        )
