"""Parallel.ai task-runner provider.

Submit a task, poll until status is terminal, fetch the result. Models map
to Parallel processors (`base`, `core`, `pro`, `ultra`). Verify endpoint
shapes against https://docs.parallel.ai when the API key first arrives, as
this implementation reflects the public docs at time of writing.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .base import ProviderResult, ResearchProvider, normalize_sources


class ParallelProvider(ResearchProvider):
    name = "parallel"
    default_model = "core"
    available_models = ("base", "core", "pro", "ultra")
    env_var = "PARALLEL_API_KEY"

    BASE_URL = "https://api.parallel.ai/v1"

    def run(
        self,
        question: str,
        model: str | None = None,
        *,
        poll_interval: float = 3.0,
        poll_timeout: float = 600.0,
        **_: Any,
    ) -> ProviderResult:
        started = time.monotonic()
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        submit = httpx.post(
            f"{self.BASE_URL}/tasks/runs",
            headers=headers,
            json={
                "input": question,
                "processor": model or self.default_model,
            },
            timeout=60.0,
        )
        submit.raise_for_status()
        sub_data = submit.json()
        run_id = sub_data.get("run_id") or sub_data.get("id")
        if not run_id:
            raise RuntimeError(f"Parallel returned no run_id: {sub_data}")

        deadline = started + poll_timeout
        data: dict = sub_data
        while True:
            r = httpx.get(
                f"{self.BASE_URL}/tasks/runs/{run_id}",
                headers=headers,
                timeout=30.0,
            )
            r.raise_for_status()
            data = r.json()
            status = (data.get("status") or "").lower()
            if status in {"completed", "succeeded", "failed", "error", "cancelled"}:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Parallel task {run_id} did not finish in "
                    f"{poll_timeout}s (last status={status!r})"
                )
            time.sleep(poll_interval)

        output = data.get("output") or {}
        if not output and status in {"completed", "succeeded"}:
            out_r = httpx.get(
                f"{self.BASE_URL}/tasks/runs/{run_id}/result",
                headers=headers,
                timeout=30.0,
            )
            if out_r.status_code == 200:
                output = out_r.json()
                data = {**data, "result": output}

        if isinstance(output, str):
            content = output
            citations: list = []
        else:
            content = (
                output.get("content")
                or output.get("text")
                or output.get("answer")
                or ""
            )
            citations = output.get("citations") or output.get("sources") or []

        return ProviderResult(
            content=str(content),
            sources=normalize_sources(citations),
            request_id=run_id,
            status=status or "completed",
            duration_seconds=time.monotonic() - started,
            raw=data,
        )
