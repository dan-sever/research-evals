"""Streamlit-side caches for hot DB / parquet reads.

`storage.get_question_status` is called by the Launch coverage table,
the Tier roster + matrix, the Dashboard seed picker, and the Insights
seed picker — each on every widget interaction. The actual data only
changes when a run completes, so a 10s TTL collapses the per-toggle DB
roundtrip without hiding completed re-runs from the user. The In-flight
runs panel polls a separate query (`list_in_progress_runs`) every 3s so
live progress is unaffected.

Keep Streamlit-specific caching here, not in `benchmarks/dimensions.py`
(which stays Streamlit-free so CLI tools can import it).
"""
from __future__ import annotations

import streamlit as st

from benchmarks import storage


@st.cache_data(ttl=10)
def question_status(benchmark: str) -> list[dict]:
    return storage.get_question_status(benchmark)
