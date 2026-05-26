"""EXA search-tier provider: `client.search(type=...)` + Claude synthesis.

Different from `ExaProvider` which uses the deep research types (`deep`,
`deep-lite`, `deep-reasoning`). This provider hits the regular search
types (`auto`, `neural`, `keyword`, `fast`) which return hits + content.

For apples-to-apples comparison against Tavily and Parallel we request
`contents={"highlights": True}` — EXA's "LLM-identified relevant
snippets" are the closest analog to Tavily's `content` field and
Parallel's `excerpts`. Falling back to `text` only when highlights are
empty keeps the snippet shape comparable. The same Claude Haiku
synthesizer is used across all three providers.

`model` maps to EXA's `type` argument.
"""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any

from exa_py import Exa

from ._synth import synthesize_answer
from .base import ProviderResult, ResearchProvider


class ExaSearchProvider(ResearchProvider):
    name = "exa_search"
    default_model = "auto"
    available_models = ("auto", "neural", "keyword", "fast")
    env_var = "EXA_API_KEY"

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = Exa(api_key)

    def run(
        self,
        question: str,
        model: str | None = None,
        *,
        num_results: int = 10,
        **_: Any,
    ) -> ProviderResult:
        started = time.monotonic()
        result = self._client.search(
            question,
            num_results=num_results,
            type=model or self.default_model,
            contents={"highlights": True, "text": True},
        )

        excerpts: list[dict] = []
        for r in (getattr(result, "results", None) or []):
            highlights = list(getattr(r, "highlights", None) or [])
            # Prefer EXA's LLM-extracted highlights for parity with Tavily's
            # `content` and Parallel's `excerpts`. Fall back to truncated
            # rendered text only when highlights are empty.
            snippet = "\n\n".join(highlights) if highlights else (
                getattr(r, "text", "") or ""
            )
            excerpts.append({
                "title": getattr(r, "title", "") or "",
                "url": getattr(r, "url", "") or "",
                "content": snippet,
            })

        content = synthesize_answer(question, excerpts)
        sources = [{"title": e["title"], "url": e["url"]} for e in excerpts]
        return ProviderResult(
            content=content,
            sources=sources,
            request_id=None,
            status="completed",
            duration_seconds=time.monotonic() - started,
            raw=_to_dict(result),
        )


def _to_dict(obj: Any) -> dict:
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            pass
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    return {"_repr": repr(obj)[:4000]}
