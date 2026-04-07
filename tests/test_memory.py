"""
Unit tests for the 3-tier memory system.

Run with:  pytest tests/ -v
"""
import pytest

from src.memory.working_memory import WorkingMemory
from src.memory.episodic_buffer import EpisodicBuffer


# ──────────────────────────────────────────────────────────────────────
# WorkingMemory
# ──────────────────────────────────────────────────────────────────────

class TestWorkingMemory:

    def test_add_within_limit(self):
        wm = WorkingMemory(max_tokens=500)
        assert wm.add("Source", "Short text") is True
        assert len(wm) == 1

    def test_add_exceeds_limit_returns_false(self):
        wm = WorkingMemory(max_tokens=5)  # tiny limit
        # First add might succeed or fail depending on text length
        result = wm.add("S", "This text is definitely longer than five tokens xxxxxx")
        # Either the first or second add must fail at some point
        wm2 = WorkingMemory(max_tokens=5)
        wm2.add("A", "hello")  # ~1 token
        # Second add to an almost-full memory should fail
        result2 = wm2.add("B", "x" * 100)
        assert result2 is False

    def test_clear_resets_state(self):
        wm = WorkingMemory(max_tokens=1000)
        wm.add("Src", "Some content")
        wm.clear()
        assert len(wm) == 0
        assert wm.token_usage()[0] == 0

    def test_get_context_contains_source_label(self):
        wm = WorkingMemory(max_tokens=1000)
        wm.add("WebSearch", "Paris is the capital of France.")
        ctx = wm.get_context()
        assert "WebSearch" in ctx
        assert "Paris" in ctx

    def test_token_usage_increases_with_adds(self):
        wm = WorkingMemory(max_tokens=10_000)
        before = wm.token_usage()[0]
        wm.add("X", "The quick brown fox jumps over the lazy dog.")
        after = wm.token_usage()[0]
        assert after > before

    def test_estimate_tokens_positive(self):
        assert WorkingMemory.estimate_tokens("hello") > 0
        assert WorkingMemory.estimate_tokens("") == 1  # min 1

    def test_is_empty_initially(self):
        wm = WorkingMemory()
        assert wm.is_empty()


# ──────────────────────────────────────────────────────────────────────
# EpisodicBuffer
# ──────────────────────────────────────────────────────────────────────

class TestEpisodicBuffer:

    def test_add_and_length(self):
        eb = EpisodicBuffer(max_size=3)
        eb.add("q1", "finding 1")
        assert len(eb) == 1

    def test_is_full(self):
        eb = EpisodicBuffer(max_size=2)
        assert not eb.is_full()
        eb.add("q1", "f1")
        eb.add("q2", "f2")
        assert eb.is_full()

    def test_get_all_returns_tuples(self):
        eb = EpisodicBuffer(max_size=5)
        eb.add("What is X?", "X is Y.")
        pairs = eb.get_all()
        assert pairs == [("What is X?", "X is Y.")]

    def test_clear_empties_buffer(self):
        eb = EpisodicBuffer(max_size=5)
        eb.add("q", "f")
        eb.clear()
        assert len(eb) == 0
        assert eb.get_all() == []

    def test_search_similar_keyword_overlap(self):
        eb = EpisodicBuffer(max_size=5)
        eb.add("transformer architecture overview", "Transformers use attention.")
        eb.add("python web scraping", "Use BeautifulSoup.")

        # Query related to transformers should return transformer finding
        results = eb.search_similar("transformer model architecture", top_k=1)
        assert len(results) == 1
        assert "attention" in results[0].lower() or "transformer" in results[0].lower()

    def test_search_similar_no_overlap_returns_empty(self):
        eb = EpisodicBuffer(max_size=5)
        eb.add("database indexing", "B-trees are used for indexing.")
        results = eb.search_similar("quantum physics", top_k=2)
        assert results == []

    def test_max_size_respected_via_deque(self):
        eb = EpisodicBuffer(max_size=3)
        for i in range(10):
            eb.add(f"q{i}", f"f{i}")
        # deque(maxlen=3) keeps only last 3
        assert len(eb) == 3


# ──────────────────────────────────────────────────────────────────────
# BudgetTracker
# ──────────────────────────────────────────────────────────────────────

class TestBudgetTracker:

    def _make_tracker(self, **kwargs):
        from src.budget_tracker import BudgetTracker
        return BudgetTracker(**kwargs)

    def test_can_proceed_initially(self):
        bt = self._make_tracker(max_tokens=1000, max_cost=1.0, max_calls=10)
        assert bt.can_proceed() is True

    def test_can_proceed_false_after_call_limit(self):
        bt = self._make_tracker(max_tokens=100_000, max_cost=100.0, max_calls=1)
        bt.record_usage(10, 10)
        assert bt.can_proceed() is False

    def test_record_usage_increments_totals(self):
        bt = self._make_tracker(max_tokens=100_000, max_cost=10.0, max_calls=100)
        bt.record_usage(100, 50)
        assert bt.total_tokens == 150

    def test_reset_clears_all(self):
        bt = self._make_tracker(max_tokens=100_000, max_cost=10.0, max_calls=100)
        bt.record_usage(500, 200)
        bt.reset()
        assert bt.total_tokens == 0
        assert bt._call_count == 0

    def test_get_summary_keys(self):
        bt = self._make_tracker()
        summary = bt.get_summary()
        for key in ("total_tokens", "total_cost_usd", "api_calls", "budget_used_pct"):
            assert key in summary
