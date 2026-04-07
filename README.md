# Deep Research Agent — Binox 2026 Take-Home (G3)

A memory-constrained research agent that answers complex multi-part questions
by decomposing them, researching each part independently within hard token and
cost budgets, and synthesising a final Markdown answer.

---

## Architecture

```
  ┌────────────────────────────────────────────────────────────────┐
  │                     Deep Research Agent                         │
  │                                                                 │
  │  Query ──► QueryDecomposer ──► [sub_q1, sub_q2, … sub_qN]      │
  │                                         │                       │
  │              for each sub-query:        │                       │
  │              ┌──────────────────────────┤                       │
  │              │                          │                       │
  │              ▼                          ▼                       │
  │   ┌─────────────────────────────────────────────────┐          │
  │   │           3-Tier Memory System                  │          │
  │   │                                                 │          │
  │   │  Tier 1 ▶ WorkingMemory   (max 2 000 tokens)   │          │
  │   │              │ full?                            │          │
  │   │              ▼                                  │          │
  │   │  Tier 2 ▶ EpisodicBuffer  (max 5 episodes)     │          │
  │   │              │ full?                            │          │
  │   │              ▼                                  │          │
  │   │  Tier 3 ▶ VectorStore     (ChromaDB, on-disk)  │          │
  │   └─────────────────────────────────────────────────┘          │
  │              │                                                  │
  │              ▼                                                  │
  │         LLM sub-answer  (Haiku, ≤ 512 output tokens)           │
  │              │                                                  │
  │   (all sub-answers collected)                                   │
  │              ▼                                                  │
  │         Synthesizer ──► final Markdown answer                  │
  │                                                                 │
  │         BudgetTracker ── guards every LLM call ─────────────── │
  └─────────────────────────────────────────────────────────────────┘
```

### Memory tiers explained

| Tier | Component | Capacity | Eviction strategy | Purpose |
|------|-----------|----------|-------------------|---------|
| 1 | `WorkingMemory` | 2 000 tokens | Hard ceiling — reject overflow | Per-call context window |
| 2 | `EpisodicBuffer` | 5 episodes | FIFO flush to Tier 3 | Short-term recency cache |
| 3 | `VectorStore` (ChromaDB) | Unlimited (disk) | None | Long-term semantic retrieval |

### Self-defined constraints

| Constraint | Default | Env var |
|------------|---------|---------|
| Tokens per LLM call | 2 000 | `MAX_WORKING_TOKENS` |
| Episodic buffer size | 5 | `MAX_EPISODIC_SIZE` |
| Session token cap | 50 000 | `MAX_SESSION_TOKENS` |
| Session cost cap | $0.10 | `MAX_SESSION_COST` |
| Max API calls | 20 | `MAX_API_CALLS` |

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/deep-research-agent.git
cd deep-research-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** ChromaDB downloads the `all-MiniLM-L6-v2` ONNX embedding model
> (~90 MB) on first run. This only happens once.

### 2. Configure

```bash
cp .env.example .env
# Edit .env and paste your Anthropic API key
```

### 3. Run the demo

```bash
python demo.py
```

The demo runs two sequential research queries.  The second query automatically
retrieves context from the vector store populated by the first, demonstrating
cross-session memory persistence.

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Project structure

```
deep-research-agent/
├── src/
│   ├── agent.py              # Main orchestrator
│   ├── budget_tracker.py     # Token + cost enforcement
│   ├── query_decomposer.py   # LLM-based query decomposition
│   ├── synthesizer.py        # Final answer synthesis
│   ├── memory/
│   │   ├── working_memory.py # Tier 1 — token-limited scratch-pad
│   │   ├── episodic_buffer.py# Tier 2 — FIFO recency cache
│   │   └── vector_store.py   # Tier 3 — ChromaDB semantic store
│   └── tools/
│       ├── web_search.py     # DuckDuckGo (no API key)
│       └── web_fetch.py      # Optional full-page extraction
├── tests/
│   └── test_memory.py        # Unit tests for all memory tiers
├── demo.py                   # End-to-end demonstration script
├── evaluation.md             # Architecture trade-off analysis
├── requirements.txt
├── .env.example
└── README.md
```

---

## Using the agent programmatically

```python
from src.agent import DeepResearchAgent

agent = DeepResearchAgent({
    "anthropic_api_key": "sk-ant-...",
    "model": "claude-haiku-4-5-20251001",
    "max_working_tokens": 2000,
    "max_session_cost": 0.05,   # tighter budget
})

result = agent.research(
    "What are the key differences between RAG and fine-tuning for LLM adaptation, "
    "and which enterprises are adopting each approach"
)

print(result["answer"])
print(result["budget"])   # token/cost summary
```

---

## Tech stack

| Component | Library | Reason |
|-----------|---------|--------|
| LLM | `anthropic` (Claude Haiku) | Cheapest model per token; sufficient for factual synthesis |
| Vector store | `chromadb` | Local, no infrastructure, cosine similarity built-in |
| Web search | `duckduckgo-search` | No API key; adequate recall for research snippets |
| HTTP fetch | `httpx` + `beautifulsoup4` | Async-ready; clean text extraction |
| CLI output | `rich` | Human-readable progress and budget tables |

---

## Self-assessment

**What works well**
- The 3-tier cascade cleanly separates recency (episodic) from semantic depth
  (vector) mirroring how human working memory operates.
- Budget enforcement is pre-flight (checked _before_ each LLM call), so the
  agent degrades gracefully rather than crashing mid-session.
- Every component is independently testable; the agent accepts a config dict so
  constraints can be tuned without code changes.

**Known limitations**
- DuckDuckGo can be rate-limited; a Tavily or SerpAPI key would give more
  reliable results in production.
- The per-token cost estimates are approximations; actual Anthropic billing may
  vary slightly.
- ChromaDB's local ONNX embeddings are fast but lower quality than
  `text-embedding-3-small`; swapping is one line change.

See `evaluation.md` for the full trade-off analysis.
