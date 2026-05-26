"""Parallel search-tier provider: `client.search(...)` + Claude synthesis.

Different from `ParallelProvider` which uses the task-runner research
processors (`lite`/`base`/`core`/`pro`/`ultra`). This provider hits the
regular `/search` endpoint, which returns web hits with LLM-optimized
`excerpts`. The same Claude Haiku synthesizer used by `tavily_search`
and `exa_search` turns those excerpts into a gradeable answer, so all
three providers share an identical synthesis layer.

`model` maps to Parallel's `mode` parameter: `basic` (fastest, 2-3 short
queries) or `advanced` (more aggressive retrieval).

The benchmark question is passed verbatim as the single search_query and
as the objective. Parallel's docs recommend 3-6 word keyword queries, so
longer benchmark questions are slightly off-spec; this is a known
trade-off for keeping the comparison apples-to-apples.
"""

from __future__ import annotations

import time
from typing import Any

from parallel import Parallel

from ._synth import synthesize_answer
from .base import ProviderResult, ResearchProvider


class ParallelSearchProvider(ResearchProvider):
    name = "parallel_search"
    default_model = "advanced"
    available_models = ("basic", "advanced")
    env_var = "PARALLEL_API_KEY"

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = Parallel(api_key=api_key)

    def run(
        self,
        question: str,
        model: str | None = None,
        **_: Any,
    ) -> ProviderResult:
        started = time.monotonic()
        result = self._client.search(
            search_queries=[question],
            objective=question,
            mode=model or self.default_model,
        )

        excerpts: list[dict] = []
        for r in (getattr(result, "results", None) or []):
            ex_list = list(getattr(r, "excerpts", None) or [])
            excerpts.append({
                "title": getattr(r, "title", "") or "",
                "url": getattr(r, "url", "") or "",
                "content": "\n\n".join(ex_list),
            })

        content = synthesize_answer(question, excerpts)
        sources = [{"title": e["title"], "url": e["url"]} for e in excerpts]
        return ProviderResult(
            content=content,
            sources=sources,
            request_id=getattr(result, "search_id", None),
            status="completed",
            duration_seconds=time.monotonic() - started,
            raw=_to_dict(result),
        )


def _to_dict(obj: Any) -> dict:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    return {"_repr": repr(obj)[:4000]}
