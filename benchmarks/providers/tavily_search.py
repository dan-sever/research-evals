"""Tavily search provider: /search endpoint, hits + snippets only.

This is the search-tier counterpart to `TavilyProvider` (which uses the
research endpoint). Much cheaper, much faster, intended for a strict
apples-to-apples comparison against EXA's and Parallel's search APIs.

Each result's `content` field carries Tavily's relevance-ranked snippet
("Short description of the search result"). We deliberately *do not* use
`include_answer` — that would let Tavily's own LLM synthesize the final
answer, while EXA and Parallel only return snippets. To keep the
comparison fair, all three providers funnel through the same Claude
Haiku synthesizer (`_synth.synthesize_answer`).

`model` maps to Tavily's `search_depth`.
"""

from __future__ import annotations

import time
from typing import Any

from tavily import TavilyClient

from ._synth import synthesize_answer
from .base import ProviderResult, ResearchProvider


class TavilySearchProvider(ResearchProvider):
    name = "tavily_search"
    default_model = "advanced"
    available_models = ("basic", "advanced", "fast", "ultra-fast")
    env_var = "TAVILY_API_KEY"

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = TavilyClient(api_key=api_key)

    def run(
        self,
        question: str,
        model: str | None = None,
        *,
        max_results: int = 10,
        **_: Any,
    ) -> ProviderResult:
        started = time.monotonic()
        resp = self._client.search(
            query=question,
            search_depth=model or self.default_model,
            max_results=max_results,
        )

        excerpts: list[dict] = []
        for r in (resp.get("results") or []):
            excerpts.append({
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "content": r.get("content") or "",
            })

        content = synthesize_answer(question, excerpts)
        sources = [{"title": e["title"], "url": e["url"]} for e in excerpts]
        return ProviderResult(
            content=content,
            sources=sources,
            request_id=resp.get("request_id") or resp.get("response_id"),
            status="completed",
            duration_seconds=time.monotonic() - started,
            raw=resp,
        )
