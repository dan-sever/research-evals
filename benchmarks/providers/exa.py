"""EXA `/answer` provider.

EXA exposes an `/answer` endpoint that returns an LLM-generated answer plus
the source citations it used. Sync, single call.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .base import ProviderResult, ResearchProvider, normalize_sources


class ExaProvider(ResearchProvider):
    name = "exa"
    default_model = "exa"
    available_models = ("exa",)
    env_var = "EXA_API_KEY"

    BASE_URL = "https://api.exa.ai/answer"

    def run(
        self,
        question: str,
        model: str | None = None,
        *,
        timeout: float = 180.0,
        include_text: bool = False,
        **_: Any,
    ) -> ProviderResult:
        started = time.monotonic()
        resp = httpx.post(
            self.BASE_URL,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            json={"query": question, "text": include_text},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("answer") or ""
        citations = data.get("citations") or data.get("sources") or []
        sources = normalize_sources(citations)

        return ProviderResult(
            content=answer,
            sources=sources,
            request_id=data.get("requestId") or data.get("id"),
            status="completed",
            duration_seconds=time.monotonic() - started,
            raw=data,
        )
