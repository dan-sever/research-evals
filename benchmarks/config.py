"""Run configuration: which provider/benchmark/model, sample size, concurrency."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class RunConfig:
    benchmark: str
    provider: str = "tavily"
    model: str = "mini"
    limit: Optional[int] = None
    workers: int = 4
    sample_seed: Optional[int] = None
    citation_format: str = "numbered"
    judge_model: str = "claude-haiku-4-5"
    poll_interval_seconds: float = 2.0
    poll_timeout_seconds: float = 600.0
    note: str = ""
    comparison_set: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_env() -> dict:
    """Read API keys from environment. Caller decides what's required."""
    from dotenv import load_dotenv

    load_dotenv()
    return {
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY"),
        "EXA_API_KEY": os.getenv("EXA_API_KEY"),
        "PARALLEL_API_KEY": os.getenv("PARALLEL_API_KEY"),
    }
