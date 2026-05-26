"""Orchestrates: load dataset -> provider research -> judge -> store."""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from . import datasets, providers, storage
from .config import RunConfig, load_env
from .judge import Judge


def _search_query(q: datasets.Question) -> str:
    """Question text as sent to the provider, with `company` context
    prepended when the question itself omits it.

    `financeqa` rows ask things like "What is Gross Profit in the year
    ending 2024?" — the company ("Costco") lives in a separate column.
    A bare search engine has no way to resolve that, so we inject the
    company name here. `financebench` rows already mention the company
    in the prompt, so the substring check makes this a no-op for them.

    The original `q.question` is preserved everywhere else (storage,
    judge input) — only what we send to the provider gets augmented.
    """
    company = (q.extras or {}).get("company")
    if not company:
        return q.question
    if str(company).lower() in q.question.lower():
        return q.question
    return f"{company}: {q.question}"


def run(config: RunConfig) -> int:
    env = load_env()

    provider_var = providers.env_var(config.provider)
    provider_key = env.get(provider_var)
    if not provider_key:
        raise RuntimeError(
            f"{provider_var} missing. Add it to .env to run provider "
            f"{config.provider!r}."
        )
    if not env.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing. The judge needs it.")

    provider = providers.build(config.provider, api_key=provider_key)
    judge = Judge(api_key=env["ANTHROPIC_API_KEY"], model=config.judge_model)

    storage.init_db()
    run_id = storage.create_run(config.to_dict())

    questions = list(
        datasets.load(
            config.benchmark,
            limit=config.limit,
            seed=config.sample_seed,
            offset=config.offset,
            q_indices=list(config.q_indices) if config.q_indices else None,
        )
    )
    if not questions:
        storage.finish_run(run_id)
        print(f"Run {run_id}: no questions to process.")
        return run_id

    offset_part = f" | offset={config.offset}" if config.offset else ""
    print(
        f"Run {run_id}: provider={config.provider} model={config.model} "
        f"| {config.benchmark} | n={len(questions)}{offset_part} "
        f"| workers={config.workers}"
    )

    def process(q):
        try:
            res = provider.run(
                question=_search_query(q),
                model=config.model,
                citation_format=config.citation_format,
                poll_interval=config.poll_interval_seconds,
                poll_timeout=config.poll_timeout_seconds,
            )
        except Exception as e:
            storage.insert_result(
                run_id=run_id,
                q_index=q.index,
                question=q.question,
                expected_answer=q.expected_answer,
                error=f"research_failed: {e}\n{traceback.format_exc(limit=2)}",
            )
            return q.index, None

        try:
            grade = judge.grade(
                question=q.question,
                expected_answer=q.expected_answer,
                research_content=res.content,
            )
        except Exception as e:
            storage.insert_result(
                run_id=run_id,
                q_index=q.index,
                question=q.question,
                expected_answer=q.expected_answer,
                research_status=res.status,
                research_content=res.content,
                research_sources=res.sources,
                research_request_id=res.request_id,
                research_duration_seconds=res.duration_seconds,
                error=f"judge_failed: {e}\n{traceback.format_exc(limit=2)}",
            )
            return q.index, None

        storage.insert_result(
            run_id=run_id,
            q_index=q.index,
            question=q.question,
            expected_answer=q.expected_answer,
            research_status=res.status,
            research_content=res.content,
            research_sources=res.sources,
            research_request_id=res.request_id,
            research_duration_seconds=res.duration_seconds,
            extracted_answer=grade.extracted_answer,
            is_correct=grade.is_correct,
            confidence=grade.confidence,
            reasoning=grade.reasoning,
        )
        return q.index, grade.is_correct

    correct = 0
    total = 0
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = [pool.submit(process, q) for q in questions]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{config.provider}:{config.model}",
        ):
            _, is_correct = fut.result()
            total += 1
            if is_correct:
                correct += 1

    storage.finish_run(run_id)
    pct = (correct / total * 100) if total else 0
    print(
        f"\nRun {run_id} done ({config.provider}/{config.model}). "
        f"Correct {correct}/{total} ({pct:.1f}%)."
    )
    return run_id
