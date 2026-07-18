"""
web_search.py
=============
Web search integration using DuckDuckGo (no API key required).
Provides real-time web search results to augment the chatbot's knowledge.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("chatbot.web_search")

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False
    logger.warning("[WebSearch] duckduckgo_search not installed. Web search disabled.")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "web"


@dataclass
class WebSearchResult:
    query: str
    results: List[SearchResult] = field(default_factory=list)
    answer: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


class WebSearchEngine:
    """
    Web search engine using DuckDuckGo.
    No API key required - uses the public DuckDuckGo search API.
    """

    def __init__(
        self,
        max_results: int = 5,
        timeout: float = 10.0,
        min_query_length: int = 3,
    ):
        self.max_results = max_results
        self.timeout = timeout
        self.min_query_length = min_query_length
        self._available = HAS_DDG

        # Patterns that indicate the user wants current/real-time information
        self._search_patterns = [
            r"\b(latest|recent|current|today|now|this week|this month|this year)\b",
            r"\b(news|update|announcement|release|launch)\b",
            r"\b(price|stock|market|rate|exchange)\b",
            r"\b(weather|temperature|forecast)\b",
            r"\b(score|result|winner|champion)\b",
            r"\b(release date|when does|when did|when will)\b",
            r"\b(vs|versus|compared to|comparison)\b.*\b(2024|2025|2026)\b",
            r"\b(best|top|recommended)\b.*\b(2024|2025|2026)\b",
            r"\b(who is|what is|tell me about)\b.*(right now|currently|today|now)\b",
        ]

    def should_search(self, query: str) -> bool:
        """Determine if a query would benefit from web search."""
        if not self._available:
            return False

        query_lower = query.lower().strip()

        # Don't search for very short queries
        if len(query_lower) < self.min_query_length:
            return False

        # Don't search for code-related queries
        code_patterns = [
            r"\b(def |class |import |function |const |let |var )\b",
            r"```",
            r"\b(python|javascript|java|c\+\+|rust|go)\b.*\b(code|function|class|implement)\b",
        ]
        if any(re.search(p, query_lower) for p in code_patterns):
            return False

        # Don't search for simple factual queries the model should know
        simple_facts = [
            r"\bwhat is (?:a |an |the )?\w+\b$",
            r"\bhow (?:do|does|to) \w+\b$",
            r"\b(define|explain)\b",
        ]
        if any(re.search(p, query_lower) for p in simple_facts):
            return False

        # Search if any search pattern matches
        return any(re.search(p, query_lower) for p in self._search_patterns)

    def search(self, query: str) -> WebSearchResult:
        """Perform a web search and return results."""
        t0 = time.perf_counter()

        if not self._available:
            return WebSearchResult(
                query=query,
                success=False,
                error="duckduckgo_search not installed",
            )

        try:
            with DDGS() as ddgs:
                # Try to get an instant answer first
                answer = ""
                try:
                    answers = ddgs.answers(query)
                    if answers and len(answers) > 0:
                        answer = answers[0].get("text", "")
                except Exception:
                    pass

                # Get web search results
                raw_results = list(ddgs.text(query, max_results=self.max_results))

                results = []
                for r in raw_results:
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source="web",
                    ))

                latency = round((time.perf_counter() - t0) * 1000, 2)
                logger.info(
                    "[WebSearch] Query='%s' | Results=%d | Answer=%s | Latency=%.1fms",
                    query[:50], len(results), bool(answer), latency,
                )

                return WebSearchResult(
                    query=query,
                    results=results,
                    answer=answer,
                    latency_ms=latency,
                    success=True,
                )

        except Exception as e:
            latency = round((time.perf_counter() - t0) * 1000, 2)
            logger.error("[WebSearch] Search failed: %s", e)
            return WebSearchResult(
                query=query,
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    def format_results_for_context(self, result: WebSearchResult) -> str:
        """Format search results as context block for the LLM prompt."""
        if not result.success or not result.results:
            return ""

        lines = ["### Web Search Results (Real-time)"]

        if result.answer:
            lines.append(f"**Quick Answer:** {result.answer}")
            lines.append("")

        for i, r in enumerate(result.results, 1):
            lines.append(f"[{i}] **{r.title}**")
            lines.append(f"    Source: {r.url}")
            lines.append(f"    {r.snippet}")
            lines.append("")

        lines.append("Instructions: Use these real-time search results to answer the query. "
                      "Cite sources when possible. If the search results don't contain the answer, "
                      "say so and provide your best knowledge.")

        return "\n".join(lines)


# Global instance
web_search_engine = WebSearchEngine()
