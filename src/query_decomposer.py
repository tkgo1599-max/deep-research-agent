"""
Query Decomposer — breaks a complex question into focused sub-questions.

Decomposing first means each sub-query can be researched independently
with a small, bounded context window — the core mechanism that allows
the agent to answer multi-part questions without ever exceeding the
per-call token constraint.
"""
from __future__ import annotations

import json
import re

from anthropic import Anthropic

from .budget_tracker import BudgetTracker

_SYSTEM = """\
You are a research planning assistant. Your sole job is to decompose
a complex, multi-part question into 3-5 focused, independently
answerable sub-questions.

Rules
-----
1. Each sub-question must be independently researchable via a web search.
2. Avoid overlap — each sub-question should cover a distinct angle.
3. Order questions logically: foundational concepts first.
4. Keep each sub-question concise (≤ 20 words).
5. Respond with ONLY a valid JSON array of strings. No other text.

Example
-------
Input:  "How do LLMs work and which companies lead the market?"
Output: ["What is the transformer architecture underlying LLMs?",
         "How are large language models trained at scale?",
         "Which companies have the largest LLMs by parameter count?",
         "What metrics are used to benchmark LLM performance?"]
"""


class QueryDecomposer:
    """
    Uses a single LLM call to produce a structured research plan.

    Falls back to returning the original query as a single-item list
    if the LLM call fails or exceeds budget, so the agent never stalls.
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        budget_tracker: BudgetTracker,
        max_sub_queries: int = 5,
    ) -> None:
        self.client = client
        self.model = model
        self.budget = budget_tracker
        self.max_sub_queries = max_sub_queries

    def decompose(self, query: str) -> list[str]:
        """
        Return a list of sub-questions derived from *query*.

        Guarantees at least one item is returned even on failure.
        """
        if not self.budget.can_proceed():
            return [query]

        estimated = self.budget.estimate_tokens(_SYSTEM + query)
        if not self.budget.can_afford(estimated, self.model):
            return [query]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=_SYSTEM,
                messages=[{"role": "user", "content": f"Query: {query}"}],
            )

            self.budget.record_usage(
                response.usage.input_tokens,
                response.usage.output_tokens,
                self.model,
            )

            raw_text = response.content[0].text.strip()

            # Strip markdown code fences if present
            raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)

            sub_queries: list[str] = json.loads(raw_text)

            if not isinstance(sub_queries, list) or not sub_queries:
                return [query]

            return [str(q) for q in sub_queries[: self.max_sub_queries]]

        except Exception:
            return [query]
