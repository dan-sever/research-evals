"""Registry of research providers.

Add a new provider by implementing benchmarks/providers/base.ResearchProvider
and registering the class in PROVIDERS below.
"""

from __future__ import annotations

from .base import ProviderResult, ResearchProvider, normalize_sources
from .exa import ExaProvider
from .exa_search import ExaSearchProvider
from .parallel import ParallelProvider
from .parallel_search import ParallelSearchProvider
from .perplexity import PerplexityProvider
from .tavily import TavilyProvider
from .tavily_search import TavilySearchProvider

PROVIDERS: dict[str, type[ResearchProvider]] = {
    "tavily": TavilyProvider,
    "tavily_search": TavilySearchProvider,
    "perplexity": PerplexityProvider,
    "exa": ExaProvider,
    "exa_search": ExaSearchProvider,
    "parallel": ParallelProvider,
    "parallel_search": ParallelSearchProvider,
}


def build(name: str, api_key: str) -> ResearchProvider:
    if name not in PROVIDERS:
        raise KeyError(f"Unknown provider {name!r}. Known: {list(PROVIDERS)}")
    return PROVIDERS[name](api_key=api_key)


def env_var(name: str) -> str:
    return PROVIDERS[name].env_var


def default_model(name: str) -> str:
    return PROVIDERS[name].default_model


__all__ = [
    "PROVIDERS",
    "ProviderResult",
    "ResearchProvider",
    "build",
    "default_model",
    "env_var",
    "normalize_sources",
]
