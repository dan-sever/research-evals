"""Parallel.ai research provider via the official `parallel-web` SDK.

Uses the Task Run API:

  task_run = client.task_run.create(input=..., processor=..., task_spec=...)
  result  = client.task_run.result(task_run.run_id, api_timeout=...)
  result.output.content   # synthesized answer (str)
  result.output.basis     # grounding/citations

The `processor` parameter is what we expose as the model. Parallel's
research-grade processors are `lite`, `base`, `core`, `core-fast`, `pro`,
and `ultra`. We expose all of them.
"""

from __future__ import annotations

import time
from typing import Any

from parallel import Parallel

from .base import ProviderResult, ResearchProvider


_TASK_SPEC = {
    "input_schema": {
        "type": "text",
        "description": "A research question to investigate.",
    },
    "output_schema": {
        "type": "text",
        "description": (
            "Return a helpful final answer in clear markdown that addresses "
            "the user question, with citations to the sources used."
        ),
    },
}


class ParallelProvider(ResearchProvider):
    name = "parallel"
    default_model = "core"
    available_models = ("lite", "base", "core", "core-fast", "pro", "ultra")
    env_var = "PARALLEL_API_KEY"

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = Parallel(api_key=api_key)

    def run(
        self,
        question: str,
        model: str | None = None,
        *,
        api_timeout: int = 3600,
        **_: Any,
    ) -> ProviderResult:
        started = time.monotonic()
        task_run = self._client.task_run.create(
            input=question,
            processor=model or self.default_model,
            task_spec=_TASK_SPEC,
        )
        result = self._client.task_run.result(
            task_run.run_id,
            api_timeout=api_timeout,
        )

        output = getattr(result, "output", None)
        content = ""
        sources: list[dict] = []
        if output is not None:
            raw_content = getattr(output, "content", "")
            content = raw_content if isinstance(raw_content, str) else str(raw_content)
            # Flatten basis citations across all fields
            for basis in (getattr(output, "basis", None) or []):
                for cite in (getattr(basis, "citations", None) or []):
                    sources.append({
                        "title": getattr(cite, "title", "") or "",
                        "url": getattr(cite, "url", "") or "",
                    })

        return ProviderResult(
            content=content,
            sources=sources,
            request_id=task_run.run_id,
            status="completed",
            duration_seconds=time.monotonic() - started,
            raw=_to_dict(result),
        )


def _to_dict(obj: Any) -> dict:
    """Best-effort pydantic serialization for the raw blob."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    return {"_repr": repr(obj)[:4000]}
