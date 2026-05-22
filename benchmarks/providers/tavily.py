"""Tavily research provider: submit a job, poll get_research until complete."""

from __future__ import annotations

import time
from typing import Any

from tavily import TavilyClient

from .base import ProviderResult, ResearchProvider, normalize_sources


class TavilyProvider(ResearchProvider):
    name = "tavily"
    default_model = "mini"
    available_models = ("mini", "pro", "auto")
    env_var = "TAVILY_API_KEY"

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = TavilyClient(api_key=api_key)

    def run(
        self,
        question: str,
        model: str | None = None,
        *,
        citation_format: str = "numbered",
        poll_interval: float = 2.0,
        poll_timeout: float = 600.0,
        **_: Any,
    ) -> ProviderResult:
        started = time.monotonic()
        submission = self._client.research(
            input=question,
            model=model or self.default_model,
            citation_format=citation_format,
        )
        request_id = submission.get("request_id")
        if not request_id:
            raise RuntimeError(f"Tavily returned no request_id: {submission}")

        deadline = started + poll_timeout
        last: dict = submission
        while True:
            last = self._client.get_research(request_id)
            status = (last.get("status") or "").lower()
            if status in {"completed", "failed", "error", "cancelled"}:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Tavily research {request_id} did not finish in "
                    f"{poll_timeout}s (last status={status!r})"
                )
            time.sleep(poll_interval)

        return ProviderResult(
            content=last.get("content") or last.get("answer") or "",
            sources=normalize_sources(last.get("sources")),
            request_id=request_id,
            status=last.get("status") or "unknown",
            duration_seconds=time.monotonic() - started,
            raw=last,
        )
