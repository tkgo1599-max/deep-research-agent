# Evaluation — Architecture Trade-offs & Design Rationale

*Binox 2026 Take-Home — G3: Deep Research Agent with Memory Constraints*

---

## 1. Memory architecture choice: Why 3 tiers?

A single, unbounded context window is the simplest approach but fails
to meet the success criterion of *demonstrating retrieval without
exceeding stated limits*.  Three tiers were chosen to mirror how
effective human research actually works:

| Human analogue | Agent tier | Constraint |
|----------------|-----------|------------|
| Working memory (current thought) | `WorkingMemory` | 2 000 tokens / call |
| Short-term memory (last few notes) | `EpisodicBuffer` | 5 episodes |
| Long-term memory (filed notes) | `VectorStore` | Unlimited (disk) |

### Why not just one tier?

**Single vector store only** — Every sub-query would require a ChromaDB
cosine search, adding latency on every call and requiring embedding
computation even for trivially similar questions from the same session.
The episodic buffer provides O(1) keyword similarity for very recent
findings at zero cost.

**Single in-memory buffer only** — Findings from earlier sessions would
be permanently lost; cross-session knowledge accumulation (one of the
more compelling agent features) would be impossible.

**The 3-tier cascade** is the only design that provides both low-latency
recency access _and_ persistent, semantically-indexed long-term storage,
all within a bounded per-call context.

---

## 2. Token constraint: 2 000 tokens per LLM call

### Rationale

- Claude Haiku has a 200 000-token context window, so 2 000 tokens is
  a deliberately tight self-imposed constraint — not a platform limit.
- Forcing the agent to stay under 2 000 tokens means each LLM call
  focuses on one well-scoped sub-question rather than attempting to
  answer everything at once (which degrades output quality on complex
  queries).
- At ~$0.80/M input tokens (Haiku), a 2 000-token call costs ≈ $0.0016,
  meaning the $0.10 session budget comfortably covers 60+ sub-query
  calls — more than enough for five sub-questions with synthesis.

### Trade-off

A 2 000-token limit means some web search snippets are dropped when the
working memory is already partially filled by vector store context.
This is intentional: **past knowledge should take priority over new
search results** when it is already available (avoids re-fetching things
the agent already knows).  The insertion order in `_research_sub_query`
reflects this priority:

1. Vector store context (Tier 3) — highest priority
2. Episodic buffer hits (Tier 2) — medium priority
3. Fresh web search snippets (Tier 1) — lowest priority, fills remainder

---

## 3. Episodic buffer size: 5 episodes

### Rationale

Five episodes covers the maximum number of sub-queries generated per
session (also capped at 5 in `QueryDecomposer`).  This means the
buffer _just_ fills after one full research session, triggering exactly
one vector flush per session — a clean, predictable lifecycle.

### Trade-off

If `max_sub_queries` is raised above `max_episodic_size`, the buffer
will flush mid-session, meaning the _first_ findings become retrievable
via vector search while the buffer continues accumulating newer ones.
This is actually desirable behaviour (and tested implicitly by the
demo's second query), but requires the caller to understand the flush
boundary.  An alternative would be to set `max_episodic_size` to
`infinity` and flush only at session end, at the cost of unbounded
memory growth.

---

## 4. Query decomposition via LLM vs rule-based

### Choice: LLM-based

The `QueryDecomposer` uses a small Haiku call (≤ 512 tokens output) to
produce a JSON list of sub-questions.  Rule-based alternatives
(splitting on "and", "also", "?" etc.) were considered but rejected
because:

- They fail on nested conjunctions ("A, B, and C all of which relate
  to D").
- They cannot infer *implied* sub-questions ("compare X and Y" implies
  questions about X, about Y, and about their comparison).

Cost: one extra Haiku call at ~$0.001 per decomposition — negligible.

### Fallback

If the decomposition call fails (budget, network, JSON parse error) the
agent falls back to treating the whole query as a single sub-question,
so robustness is preserved.

---

## 5. Web search: DuckDuckGo vs Tavily

| | DuckDuckGo | Tavily |
|-|------------|--------|
| API key required | No | Yes ($) |
| Result quality | Good for general topics | Excellent, research-tuned |
| Rate limits | Soft, occasional blocks | Generous with API key |
| Latency | ~1–2 s | ~1–2 s |

DuckDuckGo was chosen as the default to maximise zero-friction setup
(no account, no billing).  The `WebSearchTool` interface is provider-
agnostic; swapping to Tavily requires changing only the `_search_ddg`
implementation.

---

## 6. Vector embeddings: local ONNX vs API

ChromaDB's default uses `all-MiniLM-L6-v2` via ONNX Runtime locally.

| | Local ONNX | OpenAI `text-embedding-3-small` |
|-|------------|--------------------------------|
| API key required | No | Yes ($) |
| Embedding quality | Good (MTEB score ~56) | Excellent (MTEB score ~62) |
| Latency | ~20–50 ms | ~200–400 ms (network) |
| Cost | $0.00 | ~$0.00002 / 1K tokens |

For a demo with tens of documents local ONNX is the right choice.
For production with millions of documents, an API embedding service
with a caching layer (Redis) would be preferable.

---

## 7. Business impact reasoning

**Why does this matter to enterprise clients?**

Enterprise knowledge workers spend significant time researching topics
that partially overlap with previous research.  A memory-constrained
agent that:

1. Never exceeds a defined cost ceiling (predictable billing)
2. Reuses prior findings rather than re-fetching (faster, cheaper)
3. Produces structured, cited answers (audit trail)

…directly reduces research time and cost.

**Quantified example:** An analyst team running 20 research queries/day
averaging $0.05 each would spend ~$1/day.  If 30% of sub-questions are
answered from the vector store (no LLM call needed), cost drops to ~$0.70/day —
a 30% saving that compounds as the vector store grows.

**Deployment path for an FDE:**

| Phase | Action |
|-------|--------|
| Week 1 | Deploy with client's existing Anthropic API key; configure budget limits per team/user |
| Week 2–4 | Seed vector store with client's internal documents via batch ingest |
| Month 2 | Connect to client's Slack/Confluence/Notion via MCP for automatic knowledge ingestion |
| Month 3 | Add evaluation loop: track which sub-answers were overridden by human analysts and use as fine-tuning signal |

---

## 8. What I would improve with more time

1. **Streaming output** — pipe Haiku responses as they arrive using `stream=True`
   so users see progress in real time.
2. **Confidence scoring** — attach a relevance score to each finding and skip
   LLM synthesis for high-confidence vector hits.
3. **Multi-source citation** — track which URL each snippet came from and include
   inline citations in the final answer.
4. **Async parallelism** — research sub-questions concurrently with
   `asyncio.gather` (currently serial to keep the demo readable).
5. **Evaluation harness** — benchmark answer quality on a fixed question set
   before and after vector store warm-up to quantify memory benefit.
