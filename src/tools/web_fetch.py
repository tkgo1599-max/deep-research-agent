"""
Web Fetch Tool — retrieve and clean body text from a URL.

Used as an optional depth step: if search snippets are insufficient
the agent can fetch the top result's full page and extract its text,
staying within the working-memory token limit by truncating early.
"""
from __future__ import annotations


class WebFetchTool:
    """
    Fetch and extract plain text from a URL.

    Parameters
    ----------
    timeout:
        HTTP request timeout in seconds.
    max_content_chars:
        Maximum characters to return — prevents flooding working memory.
    """

    def __init__(self, timeout: int = 10, max_content_chars: int = 2_000) -> None:
        self.timeout = timeout
        self.max_content_chars = max_content_chars

    def fetch(self, url: str) -> str:
        """
        Return cleaned page text, truncated to max_content_chars.
        Returns an empty string on any error (agent continues gracefully).
        """
        if not url:
            return ""

        try:
            import httpx
            from bs4 import BeautifulSoup

            headers = {"User-Agent": "Mozilla/5.0 (compatible; DeepResearchBot/1.0)"}
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            lines = [
                ln.strip()
                for ln in soup.get_text(separator="\n").splitlines()
                if ln.strip()
            ]
            return "\n".join(lines)[: self.max_content_chars]

        except Exception:
            return ""
