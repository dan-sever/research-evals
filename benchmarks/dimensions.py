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
    natural order. SealQA / DeepSearchQA carry `reasoning` + `retrieval`
    (Schemes A & B); finance benchmarks (`financebench`, `financeqa`) carry
    `query_type` (Scheme C). All columns are optional — callers pick which
    dimension to slice on. Benchmarks without a CSV return empty so callers
    gracefully omit taxonomy slices.

    Regenerate the finance CSVs via
        python tools/classify_finance_questions.py {financebench|financeqa}
    and SealQA-style CSVs via
        python tools/classify_questions.py {benchmark}.
    """
    path = TAGS_DIR / f"{benchmark}.csv"
    candidate = ("q_index", "reasoning", "retrieval", "query_type", "notes")
    if not path.exists():
        return pd.DataFrame(columns=list(candidate))
    df = pd.read_csv(path)
    keep = [c for c in candidate if c in df.columns]
    return df[keep].copy()


# Generic alias — sealqa_tags is already benchmark-parameterized; the name
# stuck because sealqa was the first user. New callers should prefer this.
taxonomy_tags = sealqa_tags


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


# --- DeepSearchQA dims ------------------------------------------------------

def finance_native_dims(benchmark: str, seed: Optional[int]) -> pd.DataFrame:
    """q_index -> finance-native columns at the given seed.

    Pulls the columns that each finance parquet ships with:
      - financebench: question_type, question_reasoning, gics_sector, doc_type, doc_period, company
      - financeqa:    question_type, company

    Missing columns are silently skipped so the helper is robust to either
    benchmark. Same shape as `sealqa_native_dims` — `q_index` plus whatever
    extras the parquet carries — so callers can pass it straight to
    `dc.slice_accuracy(matrix, native, col, ...)`.
    """
    spec = datasets.REGISTRY[benchmark]
    df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.reset_index(drop=False).rename(columns={"index": "q_index"})
    keep = ["q_index"]
    for c in (
        "question_type", "question_reasoning", "gics_sector",
        "doc_type", "doc_period", "company",
    ):
        if c in df.columns:
            df[c] = df[c].astype(str)
            keep.append(c)
    return df[keep].copy()


def deepsearchqa_native_dims(seed: Optional[int]) -> pd.DataFrame:
    """q_index -> (problem_category, answer_type) at the given seed.

    Both columns ship in the parquet, so no taxonomy CSV is needed. Missing
    columns are silently skipped (same shape as `sealqa_native_dims`)."""
    spec = datasets.REGISTRY["deepsearchqa"]
    df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.reset_index(drop=False).rename(columns={"index": "q_index"})
    keep = ["q_index"]
    for c in ("problem_category", "answer_type"):
        if c in df.columns:
            df[c] = df[c].astype(str)
            keep.append(c)
    return df[keep].copy()


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
    # `fillna("")` before `astype(str)` because pandas's `.astype(str)` on an
    # object-dtype Series silently preserves float('nan') values (the dtype
    # label lies). deepsearchqa has 4 questions with null answers and that
    # NaN was sneaking into downstream code that did `(val or "")[:240]`,
    # which raises "'float' object is not subscriptable".
    dataset_df["question"] = dataset_df[spec.question_col].fillna("").astype(str)
    dataset_df["expected_answer"] = dataset_df[spec.answer_col].fillna("").astype(str)
    df = df.merge(
        dataset_df[["q_index", "question", "expected_answer"]],
        on="q_index", how="left", suffixes=("", "_ds"),
    )
    return df
