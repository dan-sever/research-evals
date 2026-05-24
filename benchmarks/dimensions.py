"""Shared dimension + matrix helpers.

These were duplicated across `ui/tabs/dashboard.py`, `ui/tabs/export.py`,
and `benchmarks/insights.py`. Centralizing them means a fix to the
finsearchcomp tier parser or the latest-wins matrix builder propagates
everywhere without copy-paste drift.

All functions here are pure: no Streamlit dependency, no DB writes. Tabs
that want memoization wrap these in `@st.cache_data` themselves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from benchmarks import datasets, storage

REPO_ROOT = Path(__file__).resolve().parent.parent
TAGS_DIR = REPO_ROOT / "docs" / "tags"

_FINSEARCH_TIER_MAP = {
    "Time-Sensitive_Data_Fetching": "T1",
    "Simple_Historical_Lookup": "T2",
    "Complex_Historical_Investigation": "T3",
}


# --- FinSearchComp label parsing --------------------------------------------

def split_finsearchcomp_label(label: str) -> tuple[str, str]:
    """`Time-Sensitive_Data_Fetching(Greater China)` -> ('T1', 'Greater China').

    Tier numbering follows the dataset's own prompt_id prefixes. Unknown
    heads pass through unchanged; missing region becomes '?'.
    """
    if not isinstance(label, str):
        return ("?", "?")
    if "(" in label and label.endswith(")"):
        head, _, tail = label.rpartition("(")
        region = tail[:-1]
    else:
        head, region = label, "?"
    return (_FINSEARCH_TIER_MAP.get(head, head), region)


def finsearchcomp_dims(seed: Optional[int]) -> pd.DataFrame:
    """q_index -> (tier, region, label, prompt_id) at the given seed."""
    spec = datasets.REGISTRY["finsearchcomp"]
    df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.reset_index(drop=False).rename(columns={"index": "q_index"})
    tr = df["label"].astype(str).apply(split_finsearchcomp_label)
    df["tier"] = [t[0] for t in tr]
    df["region"] = [t[1] for t in tr]
    keep = ["q_index", "tier", "region", "label"]
    if "prompt_id" in df.columns:
        df["prompt_id"] = df["prompt_id"].astype(str)
        keep.append("prompt_id")
    return df[keep].copy()


# --- SealQA dims ------------------------------------------------------------

def sealqa_tags(benchmark: str) -> pd.DataFrame:
    """Taxonomy CSV at docs/tags/{benchmark}.csv. Anchored to the parquet's
    natural order. Currently only sealqa_seal0 ships one; the other seal
    variants return empty so callers gracefully omit taxonomy slices."""
    path = TAGS_DIR / f"{benchmark}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["q_index", "reasoning", "retrieval", "notes"])
    df = pd.read_csv(path)
    keep = [c for c in ("q_index", "reasoning", "retrieval", "notes") if c in df.columns]
    return df[keep].copy()


def sealqa_native_dims(benchmark: str, seed: Optional[int]) -> pd.DataFrame:
    """q_index -> (topic, freshness, question_types) at the given seed.

    `question_types` is kept as a list so callers can `.explode()` it for
    per-tag slicing. Missing columns are silently skipped, which is what
    we want — longseal's REGISTRY entry omits question_types even though
    the parquet has it; the dynamic check protects against either case.
    """
    spec = datasets.REGISTRY[benchmark]
    df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.reset_index(drop=False).rename(columns={"index": "q_index"})
    keep = ["q_index"]
    for c in ("topic", "freshness", "question_types"):
        if c in df.columns:
            keep.append(c)
    out = df[keep].copy()
    if "question_types" in out.columns:
        out["question_types"] = out["question_types"].apply(
            lambda v: list(v) if v is not None and not isinstance(v, float) else []
        )
    return out


# --- Latest-run-wins matrix -------------------------------------------------

def latest_results_matrix(benchmark: str, seed: Optional[int]) -> pd.DataFrame:
    """One row per (provider, model, q_index) for `benchmark` at `seed`,
    keeping the most recent run per question.

    `storage.get_question_status()` returns every attempt sorted by run_id
    ASC, so `drop_duplicates(keep="last")` selects the latest answer per
    (provider, model, q_index). Question/expected_answer are joined from
    the parquet so they're available even if a given run's results lacked
    them.

    Columns: provider, model, q_index, question, expected_answer,
    is_correct, extracted_answer, research_duration_seconds, error,
    plus run_id.
    """
    rows = storage.get_question_status(benchmark)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["seed"].apply(lambda s: s == seed)].copy()
    if df.empty:
        return df
    df = df.drop_duplicates(
        subset=["provider", "model", "q_index"], keep="last"
    ).reset_index(drop=True)

    spec = datasets.REGISTRY[benchmark]
    dataset_df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if seed is not None:
        dataset_df = dataset_df.sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)
    dataset_df = dataset_df.reset_index(drop=False).rename(
        columns={"index": "q_index"}
    )
    dataset_df["question"] = dataset_df[spec.question_col].astype(str)
    dataset_df["expected_answer"] = dataset_df[spec.answer_col].astype(str)
    df = df.merge(
        dataset_df[["q_index", "question", "expected_answer"]],
        on="q_index", how="left", suffixes=("", "_ds"),
    )
    return df
