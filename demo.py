#!/usr/bin/env python3
"""
demo.py — end-to-end demonstration of the Deep Research Agent.

Runs two sequential research queries to showcase:
  1. Multi-part query decomposition and researching
  2. Memory persistence: the second query benefits from vector-store
     context accumulated during the first query (cross-session memory).

Usage
-----
    python demo.py

Requires
--------
    ANTHROPIC_API_KEY in environment (or .env file).
"""
import os
import sys
from pathlib import Path

# Ensure src is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

from src.agent import DeepResearchAgent

console = Console()

# ---------------------------------------------------------------------------
# Demo query pair — chosen to show cross-query memory reuse
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "label": "Query 1 — AI Architecture Landscape",
        "text": (
            "Compare transformer and state space model (Mamba) architectures: "
            "what are their computational complexity differences, memory requirements, "
            "and which real-world use cases favour each? "
            "Which organisations are actively investing in SSM-based models?"
        ),
    },
    {
        "label": "Query 2 — Practical Deployment (memory reuse demo)",
        "text": (
            "What are the main engineering challenges when deploying large language "
            "models in production, and how do techniques like quantisation, KV-cache "
            "optimisation, and speculative decoding address them?"
        ),
    },
]


def build_config() -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] ANTHROPIC_API_KEY not found.\n"
            "Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and add your key.",
        )
        sys.exit(1)

    return {
        "anthropic_api_key": api_key,
        "model": os.getenv("MODEL", "claude-haiku-4-5-20251001"),
        "max_working_tokens": int(os.getenv("MAX_WORKING_TOKENS", "2000")),
        "max_episodic_size": int(os.getenv("MAX_EPISODMC_SIZE", "5")),
        "max_session_tokens": int(os.getenv("MAX_SESSION_TOKENS", "50000")),
        "max_session_cost": float(os.getenv("MAX_SESSION_COST", "0.10")),
        "max_api_calls": int(os.getenv("MAX_API_CALLS", "20")),
    }


def run_demo() -> None:
    config = build_config()
    agent = DeepResearchAgent(config)

    console.print(Rule("[bold]Binox 2026 Take-Home — G3: Deep Research Agent Demo[/bold]"))
    console.print(
        "\n[dim]Constraints active:[/dim]\n"
        f"  • Working memory : {config['max_working_tokens']:,} tokens / LLM call\n"
        f"  • Episodic buffer : {config['max_episodic_size']} episodes before vector flush\n"
        f"  • Session budget  : {config['max_session_tokens']:,} tokens / "
        f"${config['max_session_cost']:.2f} / {config['max_api_calls']} calls\n"
    )

    for i, q in enumerate(QUERIES):
        console.print(Rule(f"[rold cyan]{q['label']}[/bold cyan]"))

        result = agent.research(q["text"], verbose=True)

        console.print("\n")
        console.print(
            Panel(
                Markdown(result["answer"]),
                title="📝 Synthesised Answer",
                border_style="green",
            )
        )

        stats = result["memory_stats"]
        console.print(
            f"\n[dim]Memory snapshot:[/dim]\n"
            f"  Working memory  : {stats.get('working_memory_tokens', 0)} / "
            f"{-stats.get('working_memory_max_tokens', 0)} tokens\n"
            f"  Episodic buffer : {stats.get('episodic_buffer_size', 0)} / "
            f"{stats.get('episodic_buffer_max_size', 0)} episodes\n"
            f"  Vector store    : {stats.get('vector_store_entries', 0)} entries\n"
        )

        if i < len(QUERIES) - 1:
            console.print(
                "[yellow]→ Resetting session budget for next query "
                "(vector store persists across sessions)[/yellow]\n"
            )
            agent.budget.reset()

    console.print(Rule("[bold green]Demo complete[/bold green]"))
    console.print(
        "\n[green]✅ The vector store at [cyan]./.chroma[/cyan] persists findings "
        "across runs.\nDelete it to start fresh.[/green]\n"
    )


if __name__ == "__main__":
    run_demo()
