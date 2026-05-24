"""Per-call cost lookup for the Launch tab cost preview.

`model_costs.json` is optional and hot-editable. Missing file -> empty
config -> Launch tab hides the `Est. cost` metric and falls back to
call counts only. Numbers are list-price approximations, not contract
prices.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

COSTS_PATH = Path("model_costs.json")


def load_costs() -> dict:
    """Cached per-call USD lookup. Returns `{}` if the file is missing or
    malformed — callers then hide the $ estimate gracefully."""
    if "_costs_cache" in st.session_state:
        return st.session_state["_costs_cache"]
    out: dict = {}
    if COSTS_PATH.exists():
        try:
            raw = json.loads(COSTS_PATH.read_text())
            if isinstance(raw, dict):
                out = raw
        except Exception:
            out = {}
    st.session_state["_costs_cache"] = out
    return out


def estimate_cost(
    provider_models: dict[str, list[str]], n_questions: int,
) -> tuple[float, list[tuple[str, str]]]:
    """Sum estimated USD over every (provider, model, question). Returns
    `(total, missing)` where `missing` lists (provider, model) pairs that
    had no cost entry. Includes one judge call per question per run."""
    costs = load_costs()
    if not costs or n_questions == 0:
        return 0.0, []
    judge_per_call = float(costs.get("judge_per_call") or 0)
    provider_costs = costs.get("providers") or {}
    total = 0.0
    missing: list[tuple[str, str]] = []
    for p, models in provider_models.items():
        for m in models:
            per_call = (
                provider_costs.get(p, {}).get(m)
                if isinstance(provider_costs.get(p), dict)
                else None
            )
            if per_call is None:
                missing.append((p, m))
                per_call = 0.0
            total += (float(per_call) + judge_per_call) * n_questions
    return total, missing
