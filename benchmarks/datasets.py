"""Read-only loaders for downloaded HuggingFace parquet files.

Each benchmark has a different schema. The registry normalizes them to a
standard pair of columns: `question` and `expected_answer`. The original
parquet is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    parquet: str
    question_col: str
    answer_col: str
    extra_cols: tuple[str, ...] = ()


REGISTRY: dict[str, BenchmarkSpec] = {
    "sealqa_seal0": BenchmarkSpec(
        name="sealqa_seal0",
        parquet="sealqa_seal0.parquet",
        question_col="question",
        answer_col="answer",
        extra_cols=("topic", "freshness", "question_types"),
    ),
    "sealqa_seal_hard": BenchmarkSpec(
        name="sealqa_seal_hard",
        parquet="sealqa_seal_hard.parquet",
        question_col="question",
        answer_col="answer",
        extra_cols=("topic", "freshness", "question_types"),
    ),
    "sealqa_longseal": BenchmarkSpec(
        name="sealqa_longseal",
        parquet="sealqa_longseal.parquet",
        question_col="question",
        answer_col="answer",
        extra_cols=("topic", "freshness"),
    ),
    "finsearchcomp": BenchmarkSpec(
        name="finsearchcomp",
        parquet="finsearchcomp.parquet",
        # `ground_truth` is mostly null; `response_reference` is the
        # canonical answer string used by the dataset's own judge prompt.
        question_col="prompt",
        answer_col="response_reference",
        extra_cols=("prompt_id", "label", "time", "ground_truth"),
    ),
    "deepsearchqa": BenchmarkSpec(
        name="deepsearchqa",
        parquet="deepsearchqa.parquet",
        question_col="problem",
        answer_col="answer",
        extra_cols=("problem_category", "answer_type"),
    ),
    "financebench": BenchmarkSpec(
        name="financebench",
        parquet="financebench.parquet",
        question_col="question",
        answer_col="answer",
        extra_cols=(
            "company", "doc_name", "question_type", "question_reasoning",
            "gics_sector", "doc_type", "doc_period",
        ),
    ),
    "financeqa": BenchmarkSpec(
        name="financeqa",
        parquet="financeqa.parquet",
        question_col="question",
        answer_col="answer",
        extra_cols=("company", "file_name", "question_type"),
    ),
}


@dataclass(frozen=True)
class Question:
    index: int
    question: str
    expected_answer: str
    extras: dict


def list_benchmarks() -> list[str]:
    return list(REGISTRY)


def load(
    name: str,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    offset: int = 0,
    q_indices: Optional[list[int]] = None,
) -> Iterator[Question]:
    """Stream questions from a benchmark.

    Two selection modes:

    * Range mode (`offset` + `limit`): skip the first `offset` rows of the
      (possibly shuffled) dataset, then return at most `limit`.
    * Cherry-pick mode (`q_indices`): return exactly those q_indices in
      the given order. Invalid indices are silently skipped. When set,
      this overrides `offset` and `limit`.

    With the same `seed`, calls with disjoint ranges or disjoint
    `q_indices` produce non-overlapping question sets.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown benchmark {name!r}. Known: {list_benchmarks()}")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    spec = REGISTRY[name]
    path = DATA_DIR / spec.parquet
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `python load-datasets.py` first."
        )

    df = pd.read_parquet(path)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if q_indices is not None:
        valid = [qi for qi in q_indices if qi in df.index]
        df = df.loc[valid]
    else:
        if offset:
            df = df.iloc[offset:]
        if limit is not None:
            df = df.head(limit)

    for i, row in df.iterrows():
        extras = {c: row[c] for c in spec.extra_cols if c in df.columns}
        # numpy/pandas types do not serialize cleanly; coerce now
        extras = {k: _to_python(v) for k, v in extras.items()}
        yield Question(
            index=int(i),
            question=str(row[spec.question_col]),
            expected_answer=str(row[spec.answer_col]),
            extras=extras,
        )


def _to_python(v):
    import numpy as np
    if isinstance(v, (list, tuple)):
        return [_to_python(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_to_python(x) for x in v.tolist()]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v
