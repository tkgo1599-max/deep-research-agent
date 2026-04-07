"""
Synthesizer — combines per-sub-query findings into a final coherent answer.

This is the only LLM call that sees ALL findings at once.  By keeping
individual sub-query answers short (≤ 300 words each) and only passing
them as a structured list, the synthesizer prompt stays well within the
working-memory limit even for five sub-questions.
"""
from __future__ import annotations

from anthropic import Anthropic

from .budget_tracker import BudgetTracker

_SYSTEM = """\
You are a senior research analyst. You receive a set of focused research
findings and must integrate them into a single, well-structured answer
to the original question.

Guidelines
----------
- Weave the findings together naturally — do not just concatenate them.
- Highlight connections, contrasts, and key takeaways.
- Flag any gaps or uncertainties in the research.
- Use Markdown headers (##) if the answer benefits from structure.
- Maximum length: 600 words.
"""


class Synthesizer:
    """
    Produces the final answer by synthesising all sub-query findings.

    Falls back to a simple concatenation if the budget is exhausted,
    ensuring the agent always returns something useful.
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        budget_tracker: BudgetTracker,
    ) -> None:
        self.client = client
        self.model = model
        self.budget = budget_tracker

    def synthesize(
        self,
        original_query: str,
        findings: list[tuple[str, str]],
    ) -> str:
        """
        Merge *findings* (list of (sub_query, answer) pairs) into a
        single answer to *original_query*.
        """
        if not findings:
            return "No research findings were collected."

        # Build the findings block
        findings_block = "\n\n".join(
            f"**Sub-question {i + 1}:** {q}\n**Finding:** {a}"
            for i, (q, a) in enumerate(findings)
        )

        user_msg = (
            f"**Original question:** {original_query}\n\n"
            f"**Research findings:**\n\n{findings_block}\n\n"
            "Please synthesise these into a comprehensive answer."
        )

        if not self.budget.can_proceed():
            # Graceful degradation — return raw findings
            return f"# Research Findings (Budget Exhausted — No Synthesis)\n\n{findings_block}"

        estimated = self.budget.estimate_tokens(_SYSTEM + user_msg)
        if not self.budget.can_afford(estimated, self.model):
            return f"# Research Findings (Budget Insufficient for Synthesis)\n\n{findings_block}"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1_500,
                system=_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )

            self.budget.record_usage(
                response.usage.input_tokens,
                response.usage.output_tokens,
                self.model,
            )

            return response.content[0].text

        except Exception as exc:
            return (
                f"Synthesis failed ({exc}). Raw findings:\n\n{findings_block}"
            )
