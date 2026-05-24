"""Model tier overlay loaded from `model_tiers.json`.

Pure UI overlay. Never persisted to results.db, never affects which runs
are launched. Just groups `(provider, model)` pairs for the Tier analysis
tab and the Launch tab's column ordering / filter.

Missing or malformed file -> empty dict -> tier-aware controls hide
silently. The rest of the app keeps working.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from benchmarks import storage

TIERS_PATH = Path("model_tiers.json")


def load_tiers() -> dict[str, list[tuple[str, str]]]:
    """Cached tier definitions. Returns {} if the file is missing or
    malformed. Normalizes members to (provider, model) tuples and drops
    anything that isn't a 2-element list of strings."""
    if "_tiers_cache" in st.session_state:
        return st.session_state["_tiers_cache"]
    out: dict[str, list[tuple[str, str]]] = {}
    if TIERS_PATH.exists():
        try:
            raw = json.loads(TIERS_PATH.read_text())
            if isinstance(raw, dict):
                for name, members in raw.items():
                    if not isinstance(name, str) or not isinstance(members, list):
                        continue
                    cleaned: list[tuple[str, str]] = []
                    for m in members:
                        if (
                            isinstance(m, (list, tuple))
                            and len(m) == 2
                            and all(isinstance(x, str) for x in m)
                        ):
                            cleaned.append((m[0], m[1]))
                    if cleaned:
                        out[name] = cleaned
        except Exception:
            out = {}
    st.session_state["_tiers_cache"] = out
    return out


def tier_of(provider: str, model: str, tiers: dict) -> str | None:
    """First tier (in JSON-key order) that contains (provider, model)."""
    for name, members in tiers.items():
        if (provider, model) in members:
            return name
    return None


def tier_members(tier_name: str, tiers: dict) -> list[tuple[str, str]]:
    return list(tiers.get(tier_name, []))


def tier_run_data(
    benchmark: str,
    tier_member_list: list[tuple[str, str]],
    seed,
) -> tuple[pd.DataFrame, dict[tuple[str, str], int]]:
    """For each (provider, model) in `tier_member_list`, collect results
    from EVERY run matching (benchmark, seed, provider, model), then
    dedupe by (provider, model, q_index) keeping the row from the most
    recent run. Mirrors the Launch tab coverage convention of
    latest-run-wins per question, so partial runs spread across many
    batches still show their full cumulative coverage in the matrix.

    Returns `(data, most_recent_run)` where `data` has one row per
    (provider, model, q_index) and `most_recent_run` maps each present
    member to its latest run_id (for roster help text).
    """
    wanted = set(tier_member_list)
    runs_for_member: dict[tuple[str, str], list[int]] = {}
    most_recent: dict[tuple[str, str], int] = {}
    for r in storage.list_runs():
        if r["benchmark"] != benchmark or r["seed"] != seed:
            continue
        key = (r["provider"], r["model"])
        if key not in wanted:
            continue
        rid = int(r["id"])
        runs_for_member.setdefault(key, []).append(rid)
        if key not in most_recent or rid > most_recent[key]:
            most_recent[key] = rid

    frames: list[pd.DataFrame] = []
    for (p, m), run_ids in runs_for_member.items():
        # Iterate ASC so drop_duplicates(keep="last") picks the latest row
        # per q_index for this member.
        for rid in sorted(run_ids):
            rows = storage.get_results(rid)
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["provider"] = p
            df["model_used"] = m
            df["_run_id"] = rid
            frames.append(df)

    if not frames:
        return pd.DataFrame(), most_recent

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(
        subset=["provider", "model_used", "q_index"], keep="last"
    ).reset_index(drop=True)
    return data, most_recent
