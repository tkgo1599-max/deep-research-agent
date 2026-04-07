"""
DeepResearchAgent — main orchestrator.

Architecture overview
---------------------

  Query
    │
    ▼
  QueryDecomposer  ──►  [sub_q1, sub_q2, ..., sub_qN]  (max 5)
                                     │
              ┌──────────────────────┤  for each sub-query
              │                      │
              ▼                      ▼
    ┌─────────────────────────────────────────────┐
    │              3-Tier Memory System            │
    │                                             │
    │  Tier 1: WorkingMemory  (max 2 000 tokens)  │
    │     ↓ overflow                              │
    │  Tier 2: EpisodicBuffer  (max 5 episodes)   │
    │     ↓ flush when full                       │
    │  Tier 3: VectorStore  (ChromaDB, on-disk)   │
    └─────────────────────────────────────────────┘
              │
              ▼
         LLM sub-answer  (Claude Haiku, max 512 output tokens)
              │
              ▼ (all sub-answers collected)
         Synthesizer  ──►  final Markdown answer
              │
              ▼
         BudgetTracker  ──►  session cost report

Self-defined constraints
------------------------
  max_working_tokens  : 2 000  (per-call context ceiling)
  max_episodic_size   : 5      (episodes before vector flush)
  max_session_tokens  : 50 000 (hard session token cap)
  max_session_cost    : $0.10  (hard spending ceiling)
  max_api_calls       : 20     (call count guard)
"""
from __future__ import annotations

from anthropic import Anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from .budget_tracker import BudgetTracker
from .memory.episodic_buffer import EpisodicBuffer
from .memory.vector_store import VectorStore
from .memory.working_memory import WorkingMemory
from .query_decomposer import QueryDecomposer
from .synthesizer import Synthesizer
from .tools.web_search import WebSearchTool

console = Console()

_SUB_QUERY_SYSTEM = """\
You are a concise research assistant. Answer the given question using
ONLY the context provided below. If the context is insufficient, state
what information is missing. Maximum 250 words.
"""


class DeepResearchAgent:
    """
    A memory-constrained research agent that answers complex multi-part
    questions by decomposing them, researching each part independently,
    and synthesising a final answer — all within a self-enforced budget.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        model = config.get("model", "claude-haiku-4-5-20251001")

        # LLM client
        self.client = Anthropic(api_key=config["anthropic_api_key"])

        # ── Memory layers ────────────────────────────────────────────────
        self.working_memory = WorkingMemory(
            max_tokens=config.get("max_working_tokens", 2_000)
        )
        self.episodic_buffer = EpisodicBuffer(
            max_size=config.get("max_episodic_size", 5)
        )
        self.vector_store = VectorStore(
            collection_name=config.get("collection_name", "research"),
            persist_dir=config.get("chroma_dir", "./.chroma"),
        )

        # ── Tools ────────────────────────────────────────────────────────
        self.search_tool = WebSearchTool(max_results=5, snippet_max_chars=400)

        # ── Orchestration ────────────────────────────────────────────────
        self.budget = BudgetTracker(
            max_tokens=config.get("max_session_tokens", 50_000),
            max_cost=config.get("max_session_cost", 0.10),
            max_calls=config.get("max_api_calls", 20),
            model=model,
        )
        self.decomposer = QueryDecomposer(self.client, model, self.budget)
        self.synthesizer = Synthesizer(self.client, model, self.budget)
        self._model = model

    # ──────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────

    def research(self, query: str, verbose: bool = True) -> dict:
        """
        Conduct deep research on *query*.

        Returns
        -------
        dict with keys:
            answer       : str   — final synthesised Markdown answer
            sub_queries  : list  — decomposed sub-questions
            findings     : list  — [(sub_q, answer), ...]
            budget       : dict  — session usage summary
            memory_stats : dict  — memory tier occupancy
        """
        if verbose:
            console.print(
                Panel(
                    f"[bold blue]{query}[/bold blue]",
                    title="🔬 Deep Research Agent",
                    subtitle="G3 — Memory-Constrained Research",
                )
            )

        if not self.budget.can_proceed():
            return self._budget_exhausted_response()

        # ── Step 1: Decompose ─────────────────────────────────────────
        if verbose:
            console.print("\n[yellow]📋 Decomposing query...[/yellow]")

        sub_queries = self.decomposer.decompose(query)

        if verbose:
            for idx, sq in enumerate(sub_queries, 1):
                console.print(f"  {idx}. {sq}")

        # ── Step 2: Research each sub-query ───────────────────────────
        findings: list[tuple[str, str]] = []

        for idx, sub_query in enumerate(sub_queries):
            if not self.budget.can_proceed():
                if verbose:
                    console.print("\n[red]⚠  Budget limit reached — stopping early.[/red]")
                break

            if verbose:
                console.print(
                    f"\n[cyan]🔍 [{idx + 1}/{len(sub_queries)}][/cyan] {sub_query}"
                )

            finding = self._research_sub_query(sub_query, verbose=verbose)
            findings.append((sub_query, finding))

            # Update episodic buffer
            self.episodic_buffer.add(sub_query, finding)

            # Cascade flush when buffer is full
            if self.episodic_buffer.is_full():
                if verbose:
                    console.print(
                        "  [dim]↳ Episodic buffer full — flushing to vector store[/dim]"
                    )
                self._flush_episodic_to_vector()

        # ── Step 3: Synthesise ────────────────────────────────────────
        if verbose:
            console.print("\n[yellow]✨ Synthesising final answer...[/yellow]")

        answer = self.synthesizer.synthesize(query, findings)
        summary = self.budget.get_summary()

        if verbose:
            self._print_budget_table(summary)

        return {
            "answer": answer,
            "sub_queries": sub_queries,
            "findings": findings,
            "budget": summary,
            "memory_stats": {
                "working_memory_tokens": self.working_memory.token_usage()[0],
                "working_memory_max_tokens": self.working_memory.token_usage()[1],
                "episodic_buffer_size": len(self.episodic_buffer),
                "episodic_buffer_max_size": self.episodic_buffer.max_size,
                "vector_store_entries": self.vector_store.count(),
            },
        }

    def reset(self) -> None:
        """Reset session state (memory + budget) without re-instantiating."""
        self.working_memory.clear()
        self.episodic_buffer.clear()
        self.budget.reset()

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _research_sub_query(self, sub_query: str, verbose: bool = False) -> str:
        """Research one sub-question using the 3-tier memory pipeline."""
        # Always start with a clean working memory for this sub-query
        self.working_memory.clear()

        # 1. Pull relevant context from Tier 3 (vector store)
        if self.vector_store.count() > 0:
            past = self.vector_store.retrieve(sub_query, top_k=2)
            if past:
                added = self.working_memory.add("Past Research (Vector Store)", past)
                if verbose and added:
                    console.print("  [dim]↳ Retrieved context from vector store[/dim]")

        # 2. Pull relevant context from Tier 2 (episodic buffer)
        similar = self.episodic_buffer.search_similar(sub_query, top_k=1)
        for s in similar:
            self.working_memory.add("Recent Finding (Episodic Buffer)", s)

        # 3. Web search → Tier 1 (working memory), respecting token ceiling
        search_results = self.search_tool.search(sub_query, max_results=5)
        added_count = 0
        for result in search_results:
            snippet = f"{result['title']}: {result['snippet']}"
            if self.working_memory.add("Web Search", snippet):
                added_count += 1
            else:
                break  # Memory ceiling reached — enforce constraint

        if verbose:
            used, cap = self.working_memory.token_usage()
            console.print(
                f"  [dim]↳ {added_count} snippets added | "
                f"Working memory: {used}/{cap} tokens[/dim]"
            )

        # 4. Generate sub-answer with bounded context
        context = self.working_memory.get_context()

        if not context.strip():
            return "No relevant information found for this sub-question."

        user_msg = f"Context:\n{context}\n\nQuestion: {sub_query}"
        estimated = self.budget.estimate_tokens(_SUB_QUERY_SYSTEM + user_msg)

        if not self.budget.can_afford(estimated, self._model):
            return "Remaining budget insufficient to answer this sub-question."

        try:
            response = self.client.messages.create(
                model=self._model,
                max_tokens=512,
                system=_SUB_QUERY_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            self.budget.record_usage(
                response.usage.input_tokens,
                response.usage.output_tokens,
                self._model,
            )
            return response.content[0].text

        except Exception as exc:
            return f"LLM call failed: {exc}"

    def _flush_episodic_to_vector(self) -> None:
        """Move all episodic buffer entries to the vector store and clear the buffer."""
        for q, finding in self.episodic_buffer.get_all():
            self.vector_store.store(q, finding)
        self.episodic_buffer.clear()

    def _budget_exhausted_response(self) -> dict:
        return {
            "answer": "Session budget exhausted. Call agent.reset() to start a new session.",
            "sub_queries": [],
            "findings": [],
            "budget": self.budget.get_summary(),
            "memory_stats": {},
        }

    def _print_budget_table(self, summary: dict) -> None:
        table = Table(title="💰 Session Budget", show_lines=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Used", style="green", justify="right")
        table.add_column("Limit", style="yellow", justify="right")
        table.add_column("%", justify="right")

        table.add_row(
            "Total tokens",
            str(summary["total_tokens"]),
            str(self.budget.max_tokens),
            f"{summary['tokens_used_pct']}%",
        )
        table.add_row(
            "Estimated cost",
            f"${summary['total_cost_usd']:.4f}",
            f"${self.budget.max_cost:.2f}",
            f"{summary['budget_used_pct']}%",
        )
        table.add_row(
            "API calls",
            str(summary["api_calls"]),
            str(self.budget.max_calls),
            f"{round(summary['api_calls'] / self.budget.max_calls * 100, 1)}%",
        )
        console.print("\n")
        console.print(table)
