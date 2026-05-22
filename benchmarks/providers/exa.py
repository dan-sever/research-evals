"""EXA research provider via the official `exa_py` SDK.

EXA's `search` endpoint combines retrieval + synthesis when called with a
`deep` type. We expose the three research-grade types as `available_models`:

  - `deep-lite`     — cheapest, fastest research mode
  - `deep`          — standard deep research
  - `deep-reasoning` — adds extended reasoning over retrieved content

Calling `search(..., output_schema={"type": "text"}, ...)` returns a
synthesized text answer in `result.output.content` plus the underlying
search hits in `result.results`. We use the former as the report and the
latter as the source list.
"""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any

from exa_py import Exa

from .base import ProviderResult, ResearchProvider


class ExaProvider(ResearchProvider):
    name = "exa"
    default_model = "deep-lite"
    available_models = ("deep-lite", "deep", "deep-reasoning")
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
            output_schema={"type": "text"},
            contents={"highlights": True},
        )

        # Synthesized answer
        content = ""
        out = getattr(result, "output", None)
        if out is not None:
            raw_content = getattr(out, "content", "")
            content = raw_content if isinstance(raw_content, str) else str(raw_content)

        # Sources
        sources: list[dict] = []
        for r in (getattr(result, "results", None) or []):
            sources.append({
                "title": getattr(r, "title", "") or "",
                "url": getattr(r, "url", "") or "",
            })

        return ProviderResult(
            content=content,
            sources=sources,
            request_id=None,  # EXA SDK does not expose a per-call id
            status="completed",
            duration_seconds=time.monotonic() - started,
            raw=_to_dict(result),
        )


def _to_dict(obj: Any) -> dict:
    """Best-effort serialization of the SDK dataclass response for the raw blob."""
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
