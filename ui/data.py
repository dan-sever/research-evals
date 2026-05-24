"""Cached parquet readers used by Launch, Inspect, and Compare tabs.

Centralized here so the Streamlit cache layer is the same across tabs
(one `dataset_df` cache key per `(name, seed)`, not three).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from benchmarks import datasets


@st.cache_data
def dataset_size(name: str) -> int:
    """Total questions in the parquet, ignoring limit/offset."""
    spec = datasets.REGISTRY[name]
    import pyarrow.parquet as pq
    return pq.ParquetFile(datasets.DATA_DIR / spec.parquet).metadata.num_rows


@st.cache_data
def load_dataset_df(name: str, seed) -> pd.DataFrame:
    """Return the dataset as `(q_index, question, expected_answer)` in
    whatever order matches `seed`. q_index is the row's position after
    shuffling, so it lines up exactly with what `datasets.load(... seed=...)`
    produces. If the parquet has a `prompt_id` column (e.g. finsearchcomp),
    it is surfaced as an extra column between q_index and question."""
    spec = datasets.REGISTRY[name]
    df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.reset_index(drop=False).rename(columns={"index": "q_index"})
    df["question"] = df[spec.question_col].astype(str)
    df["expected_answer"] = df[spec.answer_col].astype(str)
    cols = ["q_index", "question", "expected_answer"]
    if "prompt_id" in df.columns:
        df["prompt_id"] = df["prompt_id"].astype(str)
        cols.insert(1, "prompt_id")
    return df[cols].copy()


@st.cache_data
def prompt_ids(name: str, seed) -> dict[int, str]:
    """q_index -> parquet `prompt_id` mapping for benchmarks that ship one
    (currently only finsearchcomp). Returns `{}` for datasets without the
    column. Seed-aware so it matches `load_dataset_df`'s shuffled ordering,
    which is also how `q_index` is assigned in `runs.seed` rows."""
    spec = datasets.REGISTRY[name]
    df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if "prompt_id" not in df.columns:
        return {}
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.reset_index(drop=False).rename(columns={"index": "q_index"})
    return dict(zip(df["q_index"].astype(int), df["prompt_id"].astype(str)))
