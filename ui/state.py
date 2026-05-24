"""Streamlit widget state persistence to `.ui_state.json`.

`.ui_state.json` lives at the project root, is gitignored, and only holds
widget keys (toggles, last-used providers, etc). It never touches the
eval database. Safe to delete at any time — the next interaction
recreates it.

Tab modules that want a persistent widget:
1. Use a key matching one of `UI_PERSIST_PREFIXES` (extend the tuple
   below if you're introducing a new prefix family).
2. Read the saved default via `persisted(key, default)`.
3. Pass `on_change=save_ui_state` to the widget so edits flush to disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

UI_STATE_PATH = Path(".ui_state.json")
UI_PERSIST_PREFIXES = (
    "launch_show_details",
    "launch_english_only",
    "launch_bench",
    "launch_seed",
    "launch_count",
    "launch_workers",
    "launch_chk_",
    "launch_tier_filter",
    "tier_",
)


def load_ui_state() -> dict:
    if "_ui_state_cache" in st.session_state:
        return st.session_state["_ui_state_cache"]
    data: dict = {}
    if UI_STATE_PATH.exists():
        try:
            data = json.loads(UI_STATE_PATH.read_text())
        except Exception:
            data = {}
    st.session_state["_ui_state_cache"] = data
    return data


def persisted(key: str, default):
    """Return persisted value for `key` if present, else `default`."""
    return load_ui_state().get(key, default)


def save_ui_state() -> None:
    """Snapshot tracked widget keys from session_state to disk."""
    payload = {
        k: st.session_state[k]
        for k in list(st.session_state.keys())
        if isinstance(k, str) and k.startswith(UI_PERSIST_PREFIXES)
    }
    try:
        old = json.loads(UI_STATE_PATH.read_text()) if UI_STATE_PATH.exists() else {}
    except Exception:
        old = {}
    if payload != old:
        UI_STATE_PATH.write_text(json.dumps(payload, indent=2, default=str))
        st.session_state["_ui_state_cache"] = payload
