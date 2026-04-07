"""
Web Search Tool — thin wrapper around DuckDuckGo.

DuckDuckGo requires no API key and has generous rate limits for
lightweight use. Results are normalised into a consistent dict schema
so the agent code is decoupled from the underlying provider.

To swap in Tavily (higher quality, requires API key) simply replace
the _search_ddg implementation and keep the same return schema.
"""
from __future__ import annotations

import time


class WebSearchTool:
    """
    Fetch web search results without requiring an API key.

    Parameters
    ----------
    max_results:
        Default number of results to retrieve per query.
    snippet_max_chars:
        Truncation limit per result snippet — keeps token usage bounded.
    retry_delay:
        Seconds to wait before a single retry on transient failures.
    """

    def __init__(
        self,
        max_results: int = 5,
        snippet_max_chars: int = 400,
        retry_delay: float = 2.0,
    ) -> None:
        self.max_results = max_results
        self.snippet_max_chars = snippet_max_chars
        self.retry_delay = retry_delay

    def search(self, query: str, max_results: int | None = None) -> list[dict]:
        """
        Search and return a list of result dicts:
            {"title": str, "url": str, "snippet": str}

        Never raises — returns an error entry on failure so the agent
        can continue gracefully without crashing mid-session.
        """
        n = max_results or self.max_results
        return self._search_ddg(query, n)

    def _search_ddg(self, query: str, n: int) -> list[dict]:
        def _parse(raw: list[dict]) -> list[dict]:
            return [
                {
                    "title": r.get("title", "No title"),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")[: self.snippet_max_chars],
                }
                for r in raw
            ]

        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                return _parse(list(ddgs.text(query, max_results=n)))

        except Exception:
            time.sleep(self.retry_delay)
            try:
                from duckduckgo_search import DDGS

                with DDGS() as ddgs:
                    return _parse(list(ddgs.text(query, max_results=n)))
            except Exception as exc:
                return [
                    {
                        "title": "Search Unavailable",
                        "url": "",
                        "snippet": f"DuckDuckGo search failed: {exc}",
                    }
                ]
