"""Perplexity Sonar API provider.

Sync chat-completions style endpoint. No polling. Citations come back as a
top-level `citations` list of URLs alongside the assistant message.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .base import ProviderResult, ResearchProvider, normalize_sources


class PerplexityProvider(ResearchProvider):
    name = "perplexity"
    default_model = "sonar-reasoning-pro"
    available_models = ("sonar-reasoning-pro", "sonar-deep-research")
    env_var = "PERPLEXITY_API_KEY"

    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def run(
        self,
        question: str,
        model: str | None = None,
        *,
        timeout: float = 180.0,
        **_: Any,
    ) -> ProviderResult:
        started = time.monotonic()
        resp = httpx.post(
            self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model or self.default_model,
                "messages": [{"role": "user", "content": question}],
                "return_citations": True,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        content = ""
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""

        citations = data.get("citations") or []
        sources = normalize_sources(citations)

        return ProviderResult(
            content=content,
            sources=sources,
            request_id=data.get("id"),
            status="completed",
            duration_seconds=time.monotonic() - started,
            raw=data,
        )
