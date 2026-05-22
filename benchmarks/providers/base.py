"""Shared interface every research provider implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResult:
    """Normalized result returned by every provider."""
    content: str
    sources: list[dict] = field(default_factory=list)
    request_id: str | None = None
    status: str = "completed"
    duration_seconds: float = 0.0
    raw: dict = field(default_factory=dict)


class ResearchProvider(ABC):
    """Each vendor implements run(question, model) -> ProviderResult.

    The provider is responsible for whatever vendor-specific dance is needed
    (submit + poll, single sync call, etc.) and only returns when the answer
    is final. The runner does not know or care.
    """

    name: str = "base"
    default_model: str = ""
    available_models: tuple[str, ...] = ()
    env_var: str = ""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(f"{self.env_var} missing from environment")
        self.api_key = api_key

    @abstractmethod
    def run(
        self,
        question: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> ProviderResult:
        """Execute one research call. Block until complete."""


def normalize_sources(raw: Any) -> list[dict]:
    """Coerce a provider's source list into [{title, url}, ...]."""
    if not raw:
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            out.append({
                "title": item.get("title") or item.get("name") or item.get("url") or "",
                "url": item.get("url") or item.get("link") or "",
            })
        elif isinstance(item, str):
            out.append({
                "title": item,
                "url": item if item.startswith(("http://", "https://")) else "",
            })
    return out
