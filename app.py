"""Streamlit dashboard for browsing eval runs and comparing providers.

    streamlit run app.py
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from benchmarks import datasets, launcher, providers, storage
from benchmarks.config import load_env

st.set_page_config(page_title="Research Benchmarks", layout="wide")
st.title("Research benchmark runs")

storage.init_db()


# ---------- UI state persistence (separate from results.db) ----------
# `.ui_state.json` lives in the project root and only holds widget keys
# (toggles, checkboxes, last-used providers, etc). It never touches the
# eval database. Safe to delete at any time.
_UI_STATE_PATH = Path(".ui_state.json")
_UI_PERSIST_PREFIXES = (
    "launch_show_details",
    "launch_english_only",
    "launch_bench",
    "launch_seed",
    "launch_count",
    "launch_workers",
    "launch_chk_",
)


def _load_ui_state() -> dict:
    if "_ui_state_cache" in st.session_state:
        return st.session_state["_ui_state_cache"]
    data: dict = {}
    if _UI_STATE_PATH.exists():
        try:
            data = json.loads(_UI_STATE_PATH.read_text())
        except Exception:
            data = {}
    st.session_state["_ui_state_cache"] = data
    return data


def _persisted(key: str, default):
    """Return persisted value for `key` if present, else `default`."""
    return _load_ui_state().get(key, default)


def _save_ui_state() -> None:
    """Snapshot tracked `launch_*` keys from session_state to disk."""
    payload = {
        k: st.session_state[k]
        for k in list(st.session_state.keys())
        if isinstance(k, str) and k.startswith(_UI_PERSIST_PREFIXES)
    }
    try:
        old = json.loads(_UI_STATE_PATH.read_text()) if _UI_STATE_PATH.exists() else {}
    except Exception:
        old = {}
    if payload != old:
        _UI_STATE_PATH.write_text(json.dumps(payload, indent=2, default=str))
        st.session_state["_ui_state_cache"] = payload


def _elapsed_human(started_at: str | None) -> str:
    """Human-readable elapsed since an ISO timestamp from SQLite."""
    if not started_at:
        return "—"
    try:
        t0 = datetime.fromisoformat(started_at)
        t1 = datetime.now(t0.tzinfo) if t0.tzinfo else datetime.now()
        secs = (t1 - t0).total_seconds()
        if secs < 0:
            return "—"
        if secs < 60:
            return f"{secs:.0f}s"
        if secs < 3600:
            return f"{secs / 60:.1f}m"
        return f"{secs / 3600:.1f}h"
    except Exception:
        return "—"


# CJK ideographs, Hiragana/Katakana, Hangul, fullwidth forms. Any hit
# means the string is almost certainly not English. Benchmarks here are
# either entirely English or entirely a CJK language per row, so even
# a stray glyph is a strong signal.
_NON_ENGLISH_RE = re.compile(
    "["
    "　-鿿"   # CJK Symbols, Hiragana, Katakana, CJK Unified Ideographs
    "가-힯"   # Hangul Syllables
    "＀-￯"   # Halfwidth/fullwidth forms
    "]"
)


def _looks_non_english(text: str) -> bool:
    return bool(_NON_ENGLISH_RE.search(text or ""))


def _fmt_duration(seconds) -> str:
    """`45.3s` under a minute, `128.0s (2.1 min)` above."""
    if seconds is None or pd.isna(seconds) or seconds == 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds:.1f}s ({seconds / 60:.1f} min)"


def _fmt_duration_short(seconds) -> str:
    """Compact form for dense table cells: `12.3s` or `2.1m`."""
    if seconds is None or pd.isna(seconds) or seconds == 0:
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def _quote(text: str | None) -> str:
    """Render `text` as a markdown blockquote, preserving line breaks."""
    if not text:
        return "> —"
    return "\n".join(f"> {ln}" for ln in str(text).splitlines()) or "> —"


def _ranges(sorted_ints: list[int]) -> str:
    """`[0,1,2,4,7,8,9]` -> `0-2, 4, 7-9`."""
    if not sorted_ints:
        return "—"
    parts: list[str] = []
    start = prev = sorted_ints[0]
    for n in sorted_ints[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = n
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


@st.cache_data
def _dataset_size(name: str) -> int:
    """Total questions in the parquet, ignoring limit/offset."""
    spec = datasets.REGISTRY[name]
    import pyarrow.parquet as pq
    return pq.ParquetFile(datasets.DATA_DIR / spec.parquet).metadata.num_rows


@st.cache_data
def _load_dataset_df(name: str, seed) -> pd.DataFrame:
    """Return the dataset as `(q_index, question, expected_answer)` in
    whatever order matches `seed`. q_index is the row's position after
    shuffling, so it lines up exactly with what `datasets.load(... seed=...)`
    produces."""
    spec = datasets.REGISTRY[name]
    df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.reset_index(drop=False).rename(columns={"index": "q_index"})
    df["question"] = df[spec.question_col].astype(str)
    df["expected_answer"] = df[spec.answer_col].astype(str)
    return df[["q_index", "question", "expected_answer"]].copy()


tab_launch, tab_inspect, tab_compare = st.tabs(
    ["Launch run", "Single run inspector", "Provider comparison"]
)


# ============================================================================
# Tab 0 — Launch run
# ============================================================================
with tab_launch:
    st.subheader("Configure a new run")

    _benchmarks = datasets.list_benchmarks()
    bench = st.segmented_control(
        "Benchmark",
        _benchmarks,
        default=_benchmarks[0],
        key="launch_bench",
    )
    if not bench:
        bench = _benchmarks[0]
    total_q = _dataset_size(bench)
    st.caption(f"{bench}: {total_q} questions total")

    coverage = storage.get_coverage(bench)

    with st.expander("Run options", expanded=False):
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            seed_str = st.text_input(
                "Seed (blank = original order)",
                value=_persisted("launch_seed", ""),
                key="launch_seed",
                on_change=_save_ui_state,
            )
        with oc2:
            count = st.number_input(
                "Max per batch (cap)",
                min_value=1, max_value=50,
                value=_persisted("launch_count", 5), step=1,
                help="Hard cap on how many rows you can select at once.",
                key="launch_count",
                on_change=_save_ui_state,
            )
        with oc3:
            workers = st.number_input(
                "Workers",
                min_value=1, max_value=16,
                value=_persisted("launch_workers", 4), step=1,
                key="launch_workers",
                on_change=_save_ui_state,
            )
        note = st.text_input("Note (saved with every run)", value="", key="launch_note")

    try:
        seed_int = int(seed_str) if seed_str.strip() else None
    except ValueError:
        st.error("Seed must be an integer or empty.")
        seed_int = None

    # ----- Dataset table with coverage marks -----
    st.subheader("Questions to run")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        show_details = st.toggle(
            "Show answer and duration in coverage cells",
            value=_persisted("launch_show_details", True),
            key="launch_show_details",
            on_change=_save_ui_state,
            help="Off = just ✅/❌/⚠ symbols (fits more providers on screen). "
                 "On = symbol plus extracted answer and run duration.",
        )
    with tcol2:
        english_only = st.toggle(
            "English only (hide CJK questions)",
            value=_persisted("launch_english_only", False),
            key="launch_english_only",
            on_change=_save_ui_state,
            help="Hides questions containing Chinese/Japanese/Korean "
                 "characters. Useful for finsearchcomp which mixes English "
                 "and Chinese prompts. q_index numbering is preserved.",
        )
    st.caption(
        "Click row checkboxes to pick which questions to run. The right-hand "
        "columns show every provider:model that has ever run this benchmark "
        "for the selected seed (✅ correct, ❌ incorrect, ⚠ error, blank = "
        "not run). Toggle above adds the answer and duration. Hover a cell "
        "to read it in full if truncated."
    )

    ds_df = _load_dataset_df(bench, seed_int).copy()
    if english_only:
        before = len(ds_df)
        ds_df = ds_df[~ds_df["question"].map(_looks_non_english)].reset_index(drop=True)
        hidden = before - len(ds_df)
        if hidden:
            st.caption(f"Hiding {hidden} non-English question(s).")
    # Truncate long strings for display
    ds_df["question"] = ds_df["question"].str.slice(0, 140)
    ds_df["expected_answer"] = ds_df["expected_answer"].str.slice(0, 80)

    # Build per-(provider, model) status maps from DB. Iterate ascending by
    # run_id so latest assignment wins for any question re-run. With the
    # toggle on, each cell is "<symbol> <extracted_answer> · <duration>" so
    # the table doubles as a quick comparison view. With it off, cells are
    # just the symbol so more provider columns fit on screen at once.
    # Only count and display status for q_indices that survived the filter
    # so the per-column score matches what the user actually sees.
    visible_q_indices = set(ds_df["q_index"].astype(int).tolist())
    status_rows = storage.get_question_status(bench)
    combo_status: dict[tuple[str, str], dict[int, str]] = {}
    combo_score: dict[tuple[str, str], dict[str, int]] = {}
    for s in status_rows:
        if s["seed"] != seed_int:
            continue
        if int(s["q_index"]) not in visible_q_indices:
            continue
        if s["error"]:
            symbol = "⚠"
        elif s["is_correct"] == 1:
            symbol = "✅"
        elif s["is_correct"] == 0:
            symbol = "❌"
        else:
            symbol = "·"
        if show_details:
            ans = (s.get("extracted_answer") or "").strip()
            if len(ans) > 40:
                ans = ans[:40] + "…"
            head = "⚠ error" if s["error"] else (
                f"{symbol} {ans}" if ans else symbol
            )
            dur = _fmt_duration_short(s.get("research_duration_seconds"))
            cell = f"{head} · {dur}" if dur else head
        else:
            cell = symbol
        key = (s["provider"], s["model"])
        combo_status.setdefault(key, {})[int(s["q_index"])] = cell
        score = combo_score.setdefault(key, {"correct": 0, "total": 0})
        score["total"] += 1
        if s["is_correct"] == 1:
            score["correct"] += 1

    def _combo_key(combo):
        # Tavily first since this is your home turf, then alphabetical.
        p, m = combo
        return (0 if p == "tavily" else 1, p, m)

    all_combos = sorted(combo_status.keys(), key=_combo_key)

    coverage_cols: list[str] = []
    for p, m in all_combos:
        col = f"{p}:{m}"
        coverage_cols.append(col)
        status_map = combo_status[(p, m)]
        ds_df[col] = ds_df["q_index"].map(
            lambda qi, _s=status_map: _s.get(int(qi), "")
        )

    table_cols = ["q_index", "question", "expected_answer"] + coverage_cols
    column_config = {
        "q_index": st.column_config.NumberColumn("#", width=60, pinned=True),
        "question": st.column_config.TextColumn("Question", width="large", pinned=True),
        "expected_answer": st.column_config.TextColumn("Expected", width=140),
    }
    coverage_col_width = 220 if show_details else 130
    for p, m in all_combos:
        col = f"{p}:{m}"
        sc = combo_score.get((p, m), {"correct": 0, "total": 0})
        pct = (sc["correct"] / sc["total"] * 100) if sc["total"] else 0
        label = f"{col} ({sc['correct']}/{sc['total']}, {pct:.0f}%)"
        column_config[col] = st.column_config.TextColumn(label, width=coverage_col_width)

    table_height = min(720, max(300, 90 + 38 * len(ds_df)))
    sel = st.dataframe(
        ds_df[table_cols],
        on_select="rerun",
        selection_mode="multi-row",
        column_config=column_config,
        width="stretch",
        hide_index=True,
        height=table_height,
        key=f"launch_table_{bench}_{seed_int}",
    )

    selected_positions: list[int] = list(sel.selection.rows or [])
    over_cap = len(selected_positions) > int(count)
    if over_cap:
        st.error(
            f"You picked {len(selected_positions)} rows but the cap is "
            f"{int(count)}. Reduce the selection or raise the cap above."
        )

    selected_q_indices: list[int] = [
        int(ds_df.iloc[pos]["q_index"]) for pos in selected_positions
    ]

    if not selected_q_indices:
        st.info(
            "Pick rows in the table above to choose which questions to run. "
            "Click row checkboxes to multi-select."
        )
    else:
        sel_c1, sel_c2 = st.columns([1, 4])
        sel_c1.metric("Selected", f"{len(selected_q_indices)}")
        with sel_c2:
            st.caption("q_index ranges")
            st.code(_ranges(sorted(selected_q_indices)), language=None)

    # ----- Provider and model matrix -----
    st.subheader("Providers and models")
    st.caption("Tick a model to include it. Each ticked model becomes one run.")
    prov_names = list(providers.PROVIDERS)
    prov_cols = st.columns(len(prov_names))
    provider_models: dict[str, list[str]] = {}
    selected_providers: list[str] = []
    for col, p in zip(prov_cols, prov_names):
        with col:
            st.markdown(f"**{p}**")
            default_model = providers.PROVIDERS[p].default_model
            chosen: list[str] = []
            for m in providers.PROVIDERS[p].available_models:
                ck_key = f"launch_chk_{p}_{m}"
                initial = _persisted(
                    ck_key, p == "tavily" and m == default_model
                )
                if st.checkbox(
                    m, value=initial, key=ck_key, on_change=_save_ui_state
                ):
                    chosen.append(m)
            if chosen:
                provider_models[p] = chosen
                selected_providers.append(p)

    # ----- Overlap warning for the selected rows -----
    if selected_q_indices and selected_providers and any(provider_models.values()):
        picked_set = set(selected_q_indices)
        already_covered: list[tuple[str, str, str]] = []
        for p, models in provider_models.items():
            for m in models:
                match = next(
                    (
                        c for c in coverage
                        if c["provider"] == p and c["model"] == m
                        and c["seed"] == seed_int
                    ),
                    None,
                )
                if not match:
                    continue
                overlap = sorted(picked_set & set(match["q_indices"]))
                if overlap:
                    already_covered.append((p, m, _ranges(overlap)))
        if already_covered:
            lines = "\n".join(
                f"- `{p}:{m}` already covers q_index {r}"
                for p, m, r in already_covered
            )
            st.warning(
                "Some of these questions have already been run (same seed). "
                "Launching again will re-bill them:\n\n" + lines
            )

    # ----- Cost preview as metric strip -----
    runs_planned = sum(len(ms) for ms in provider_models.values())
    calls_planned = runs_planned * len(selected_q_indices)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Runs", runs_planned)
    mc2.metric("Research calls", calls_planned)
    mc3.metric("Judge calls", calls_planned)

    # ----- Env key check -----
    env = load_env()
    missing: list[str] = []
    for p in selected_providers:
        if not provider_models.get(p):
            continue
        var = providers.PROVIDERS[p].env_var
        if not env.get(var):
            missing.append(f"{p} ({var})")
    if not env.get("ANTHROPIC_API_KEY"):
        missing.append("judge (ANTHROPIC_API_KEY)")

    if missing:
        st.error("Missing API keys in `.env`: " + ", ".join(missing))

    @st.dialog("Confirm launch")
    def _confirm_launch_dialog(
        bench_arg, provider_models_arg, selected_q_indices_arg,
        seed_int_arg, workers_arg, note_arg,
        runs_planned_arg, calls_planned_arg, missing_arg, already_covered_arg,
    ):
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Runs", runs_planned_arg)
        dc2.metric("Research calls", calls_planned_arg)
        dc3.metric("Judge calls", calls_planned_arg)

        if missing_arg:
            st.error("❌ Missing API keys: " + ", ".join(missing_arg))
        if already_covered_arg:
            lines = "\n".join(
                f"- `{p}:{m}` already covers q_index {r}"
                for p, m, r in already_covered_arg
            )
            st.warning(
                "⚠ Some of these questions have already been run (same "
                "seed). Launching again will re-bill them:\n\n" + lines
            )

        st.caption("This will spend API credits.")

        bcol1, bcol2 = st.columns(2)
        if bcol1.button("Cancel", width="stretch"):
            st.rerun()
        if bcol2.button(
            "Confirm launch",
            type="primary",
            width="stretch",
            disabled=bool(missing_arg),
        ):
            comparison_set = str(uuid.uuid4()) if runs_planned_arg > 1 else None
            launched: list[tuple[str, str, int, Path]] = []
            for p, models in provider_models_arg.items():
                for m in models:
                    pid, log_path = launcher.launch_run(
                        benchmark=bench_arg,
                        provider=p,
                        model=m,
                        q_indices=selected_q_indices_arg,
                        seed=seed_int_arg,
                        workers=int(workers_arg),
                        note=note_arg,
                        comparison_set=comparison_set,
                    )
                    launched.append((p, m, pid, log_path))
            st.session_state["_last_launch"] = {
                "launched": [
                    (p, m, pid, str(log_path)) for p, m, pid, log_path in launched
                ],
                "comparison_set": comparison_set,
            }
            st.rerun()

    launch_disabled = (
        runs_planned == 0
        or not selected_q_indices
        or over_cap
    )
    if st.button("Launch", type="primary", disabled=launch_disabled):
        # Recompute overlap for the dialog so the warning is fresh.
        picked_set = set(selected_q_indices)
        already_covered_now: list[tuple[str, str, str]] = []
        for p, models in provider_models.items():
            for m in models:
                match = next(
                    (
                        c for c in coverage
                        if c["provider"] == p and c["model"] == m
                        and c["seed"] == seed_int
                    ),
                    None,
                )
                if not match:
                    continue
                overlap = sorted(picked_set & set(match["q_indices"]))
                if overlap:
                    already_covered_now.append((p, m, _ranges(overlap)))
        _confirm_launch_dialog(
            bench, provider_models, selected_q_indices,
            seed_int, workers, note,
            runs_planned, calls_planned, missing, already_covered_now,
        )

    last = st.session_state.get("_last_launch")
    if last:
        launched = last["launched"]
        comparison_set = last["comparison_set"]
        st.success(
            f"✅ Launched {len(launched)} run(s). They run in the background."
        )
        for p, m, pid, log_name in launched:
            st.caption(f"  `{p}:{m}`  pid `{pid}`  log `{Path(log_name).name}`")
        if comparison_set:
            st.caption(f"comparison_set `{comparison_set[:8]}…`")

    # ----- In-flight runs (auto-refreshing) -----
    st.subheader("In-flight runs")

    @st.fragment(run_every=3)
    def _render_in_flight():
        in_flight = storage.list_in_progress_runs()
        if not in_flight:
            st.caption("Nothing running.")
            return
        flight_df = pd.DataFrame([
            {
                "run_id": r["id"],
                "provider": r["provider"],
                "model": r["model"],
                "benchmark": r["benchmark"],
                "progress": (
                    f"{r['rows_so_far']} / {r['limit_n']}"
                    if r["limit_n"] is not None
                    else f"{r['rows_so_far']}"
                ),
                "correct": r["correct_so_far"] or 0,
                "errors": r["errors_so_far"] or 0,
                "elapsed": _elapsed_human(r["started_at"]),
                "started": (r["started_at"] or "")[:16].replace("T", " "),
                "note": r["note"] or "",
            }
            for r in in_flight
        ])
        st.dataframe(flight_df, width="stretch", hide_index=True)

    _render_in_flight()


# ============================================================================
# Tab 1 — Single run inspector
# ============================================================================
with tab_inspect:
    runs = storage.list_runs()

    if not runs:
        st.info(
            "No runs yet. Launch one with "
            "`python run.py --provider tavily --benchmark sealqa_seal0 --limit 5`."
        )
        st.stop()

    runs_df = pd.DataFrame(runs)
    runs_df["accuracy"] = runs_df.apply(
        lambda r: (r["correct"] or 0) / r["total"] if r["total"] else None,
        axis=1,
    )

    st.subheader("All runs")
    st.caption("Click a row to inspect that run.")
    overview = runs_df.assign(
        accuracy_pct=runs_df["accuracy"].map(
            lambda x: f"{x:.1%}" if x is not None else "—"
        ),
        started=runs_df["started_at"].str.slice(0, 16).str.replace("T", " "),
    )[
        [
            "id", "provider", "benchmark", "model", "limit_n", "workers",
            "judge_model", "total", "correct", "errors",
            "accuracy_pct", "note", "started",
        ]
    ].rename(columns={"accuracy_pct": "accuracy", "limit_n": "limit"})

    run_sel = st.dataframe(
        overview,
        on_select="rerun",
        selection_mode="single-row",
        width="stretch",
        hide_index=True,
        key="inspect_runs_table",
    )
    run_positions = list(run_sel.selection.rows or [])
    if run_positions:
        selected_run = int(overview.iloc[run_positions[0]]["id"])
    elif st.session_state.get("inspect_run_id"):
        # Cross-tab handoff from the comparison tab.
        selected_run = int(st.session_state.pop("inspect_run_id"))
    else:
        selected_run = int(runs[0]["id"])

    st.caption(f"Inspecting run #{selected_run}")
    run = storage.get_run(selected_run)
    results = storage.get_results(selected_run)
    if not results:
        st.warning("⚠ This run has no results recorded.")
    else:
        df = pd.DataFrame(results)
        df["is_correct_bool"] = df["is_correct"].map({1: True, 0: False})

        total = len(df)
        correct = int(df["is_correct"].fillna(0).sum())
        errors = int(df["error"].notna().sum())
        accuracy = correct / total if total else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Questions", total)
        col2.metric("Correct", correct)
        col3.metric("Errors", errors)
        col4.metric("Accuracy", f"{accuracy:.1%}")

        with st.expander("Run config"):
            st.json(json.loads(run["config_json"]))

        st.download_button(
            "Download results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"run_{selected_run}_{run['benchmark']}_"
                f"{run['provider']}_{run['model']}.csv"
            ),
            mime="text/csv",
            key=f"dl_run_{selected_run}",
        )

        st.subheader("Per-question results")
        st.caption("Click a row to drill into one question.")
        fcol, scol = st.columns([1, 2])
        with fcol:
            filter_choice = st.segmented_control(
                "Filter",
                ["all", "correct", "incorrect", "errors"],
                default="all",
                key="inspect_filter",
            )
            if not filter_choice:
                filter_choice = "all"
        with scol:
            search = st.text_input("Search question text", "", key="inspect_search")

        view = df.copy()
        if filter_choice == "correct":
            view = view[view["is_correct"] == 1]
        elif filter_choice == "incorrect":
            view = view[view["is_correct"] == 0]
        elif filter_choice == "errors":
            view = view[view["error"].notna()]
        if search:
            view = view[view["question"].str.contains(search, case=False, na=False)]
        view = view.reset_index(drop=True)

        if view.empty:
            st.info("No questions match the current filter.")
        else:
            view = view.assign(
                duration=view["research_duration_seconds"].map(_fmt_duration)
            )
            display_cols = [
                "q_index", "question", "expected_answer", "extracted_answer",
                "is_correct_bool", "confidence", "duration",
                "research_status", "error",
            ]
            display_cols = [c for c in display_cols if c in view.columns]
            results_sel = st.dataframe(
                view[display_cols],
                on_select="rerun",
                selection_mode="single-row",
                width="stretch",
                hide_index=True,
                key=f"inspect_results_{selected_run}_{filter_choice}",
            )
            drill_positions = list(results_sel.selection.rows or [])
            if not drill_positions:
                st.caption("Click a row above to drill into one question.")
                row = None
            else:
                row = view.iloc[drill_positions[0]]

            if row is not None:
                st.subheader("Drill into one question")

                if row["is_correct"] == 1:
                    st.success(
                        f"✅ Correct  ·  confidence {row['confidence']:.2f}"
                    )
                elif row["is_correct"] == 0:
                    st.error(
                        f"❌ Incorrect  ·  confidence {row['confidence']:.2f}"
                    )
                else:
                    st.warning("⚠ No grade recorded")
                if row["reasoning"]:
                    st.caption(f"Judge: {row['reasoning']}")

                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown("**Expected answer**")
                    st.markdown(_quote(row["expected_answer"]))
                with ec2:
                    st.markdown("**Extracted answer**")
                    st.markdown(_quote(row["extracted_answer"]))

                st.markdown("**Question**")
                st.write(row["question"])

                if row["error"]:
                    st.error(row["error"])

                with st.expander("Research report"):
                    st.markdown(row["research_content"] or "_(empty)_")

                raw_sources = row["research_sources_json"]
                if isinstance(raw_sources, str) and raw_sources:
                    sources = json.loads(raw_sources)
                    with st.expander(f"Sources ({len(sources)})"):
                        for i, s in enumerate(sources, 1):
                            if isinstance(s, dict):
                                title = s.get("title") or s.get("url") or f"source {i}"
                                url = s.get("url")
                                if url:
                                    st.markdown(f"{i}. [{title}]({url})")
                                else:
                                    st.markdown(f"{i}. {title}")
                            else:
                                st.markdown(f"{i}. {s}")

                st.caption(
                    f"q_index {row['q_index']}  ·  "
                    f"research {_fmt_duration(row['research_duration_seconds'])}  ·  "
                    f"status {row['research_status']}"
                )


# ============================================================================
# Tab 2 — Provider comparison
# ============================================================================
with tab_compare:
    sets = storage.list_comparison_sets()
    if not sets:
        st.info(
            "No comparison sets yet. Launch one with "
            "`python compare.py --benchmark sealqa_seal0 --limit 10 "
            "--providers tavily:mini,perplexity:sonar-reasoning-pro`."
        )
    else:
        def _fmt_set(cs: str) -> str:
            row = next(s for s in sets if s["comparison_set"] == cs)
            short = (cs or "")[:8]
            when = (row.get("started_at") or "")[:16].replace("T", " ")
            return (
                f"{short}…  ·  {row['benchmark']}  ·  "
                f"{row.get('providers') or '?'}  ·  "
                f"limit={row.get('limit_n')}  ·  "
                f"{when}"
            )

        selected_set = st.selectbox(
            "Comparison set",
            [s["comparison_set"] for s in sets],
            format_func=_fmt_set,
            key="cmp_set_picker",
        )

        set_runs = storage.get_runs_in_set(selected_set)
        if not set_runs:
            st.warning("This comparison set has no runs.")
            st.stop()

        meta = set_runs[0]
        st.markdown(
            f"**Benchmark:** {meta['benchmark']}  ·  "
            f"**Seed:** {meta.get('seed')}  ·  "
            f"**Limit:** {meta.get('limit_n')}  ·  "
            f"**{len(set_runs)} providers**"
        )

        # ----- Provider accuracy summary -----
        summary_rows = []
        for r in set_runs:
            total = r["total"] or 0
            correct = r["correct"] or 0
            summary_rows.append({
                "provider": r["provider"],
                "model": r["model"],
                "total": total,
                "correct": correct,
                "errors": r["errors"] or 0,
                "accuracy_raw": correct / total if total else 0,
                "avg_seconds_raw": r["avg_seconds"] or 0,
                "run_id": r["id"],
            })
        summary_df = pd.DataFrame(summary_rows).sort_values(
            "accuracy_raw", ascending=False
        )
        display_summary = summary_df.assign(
            accuracy=summary_df["accuracy_raw"].map(lambda x: f"{x:.1%}"),
            avg_duration=summary_df["avg_seconds_raw"].map(_fmt_duration),
        )[["provider", "model", "total", "correct", "errors", "accuracy", "avg_duration", "run_id"]].reset_index(drop=True)
        st.caption("Click a row, then 'Open in inspector' to drill in.")
        sum_sel = st.dataframe(
            display_summary,
            on_select="rerun",
            selection_mode="single-row",
            width="stretch",
            hide_index=True,
            key=f"cmp_summary_{selected_set}",
        )
        sum_positions = list(sum_sel.selection.rows or [])

        sum_btn_col, sum_dl_col = st.columns([2, 1])
        with sum_btn_col:
            if sum_positions:
                target_run_id = int(
                    display_summary.iloc[sum_positions[0]]["run_id"]
                )
                if st.button(
                    f"Open run #{target_run_id} in Single run inspector",
                    key=f"open_inspector_{selected_set}_{target_run_id}",
                ):
                    st.session_state["inspect_run_id"] = target_run_id
                    st.toast(
                        f"Run #{target_run_id} preselected. Click "
                        f"'Single run inspector' above to view it.",
                        icon="🔍",
                    )
        with sum_dl_col:
            st.download_button(
                "Download summary CSV",
                data=display_summary.to_csv(index=False).encode("utf-8"),
                file_name=f"comparison_{selected_set[:8]}_summary.csv",
                mime="text/csv",
                key=f"dl_cmp_sum_{selected_set}",
            )

        # ----- Build cross-provider data -----
        all_results: list[pd.DataFrame] = []
        for r in set_runs:
            rows = storage.get_results(r["id"])
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["provider"] = r["provider"]
            df["model_used"] = r["model"]
            all_results.append(df)

        if not all_results:
            st.warning("No per-question results recorded yet.")
            st.stop()

        data = pd.concat(all_results, ignore_index=True)
        providers_in_set = sorted(data["provider"].unique())

        # Pre-compute lookups for filters and rendering
        qi_to_question = (
            data.drop_duplicates("q_index").set_index("q_index")["question"].to_dict()
        )
        qi_to_expected = (
            data.drop_duplicates("q_index").set_index("q_index")["expected_answer"].to_dict()
        )
        qi_to_correctness: dict[int, dict[str, int | None]] = {}
        qi_to_extracted: dict[int, dict[str, str | None]] = {}
        for _, r in data.iterrows():
            qi = int(r["q_index"])
            qi_to_correctness.setdefault(qi, {})[r["provider"]] = (
                int(r["is_correct"]) if pd.notna(r["is_correct"]) else None
            )
            qi_to_extracted.setdefault(qi, {})[r["provider"]] = r.get("extracted_answer")

        # ----- Matrix table -----
        st.subheader("Per-question matrix")
        st.caption(
            "One row per question, one column per provider. ✅ correct, ❌ wrong, "
            "— no result. Hover or scroll right for the extracted answer."
        )

        all_qis = sorted(qi_to_correctness.keys())

        # Filters — two independent controls: row-class on the left, an
        # optional Tavily pivot on the right (only when Tavily is in this set).
        fc1, fc2 = st.columns([2, 2])
        with fc1:
            cmp_filter = st.segmented_control(
                "Filter",
                ["all", "any disagreement", "all correct", "all wrong"],
                default="all",
                key="cmp_filter",
            )
            if not cmp_filter:
                cmp_filter = "all"
            cmp_search = st.text_input(
                "Search question text",
                "",
                key="cmp_search",
            )
        with fc2:
            has_tavily = "tavily" in providers_in_set
            if has_tavily:
                tav_pivot = st.toggle(
                    "Tavily pivot",
                    value=False,
                    key="cmp_tavily_pivot",
                    help="When on, show only questions where Tavily is "
                         "uniquely right or uniquely wrong vs every other "
                         "provider in this set. Overrides the filter on the left.",
                )
                if tav_pivot:
                    tav_side = st.segmented_control(
                        "Tavily side",
                        ["wins (unique)", "loses (unique)"],
                        default="wins (unique)",
                        key="cmp_tavily_side",
                    )
                    if not tav_side:
                        tav_side = "wins (unique)"
                else:
                    tav_side = None
            else:
                tav_pivot = False
                tav_side = None

        def _passes(qi: int) -> bool:
            c = qi_to_correctness.get(qi, {})
            vals = [v for v in c.values() if v is not None]
            if tav_pivot and has_tavily:
                if not vals:
                    return False
                tav = c.get("tavily")
                others = [
                    v for k, v in c.items() if k != "tavily" and v is not None
                ]
                if tav_side == "wins (unique)":
                    return tav == 1 and bool(others) and all(v == 0 for v in others)
                if tav_side == "loses (unique)":
                    return tav == 0 and bool(others) and all(v == 1 for v in others)
                return True
            if not vals:
                return cmp_filter == "all"
            if cmp_filter == "all":
                return True
            if cmp_filter == "any disagreement":
                return len(set(vals)) > 1
            if cmp_filter == "all correct":
                return all(v == 1 for v in vals)
            if cmp_filter == "all wrong":
                return all(v == 0 for v in vals)
            return True

        filtered_qis = [qi for qi in all_qis if _passes(qi)]
        if cmp_search:
            needle = cmp_search.lower()
            filtered_qis = [
                qi for qi in filtered_qis
                if needle in (qi_to_question.get(qi) or "").lower()
            ]

        st.markdown(f"**{len(filtered_qis)} of {len(all_qis)} questions**")

        if not filtered_qis:
            st.info("No questions match this filter.")
        else:
            matrix_rows = []
            for qi in filtered_qis:
                row = {
                    "q_index": qi,
                    "question": qi_to_question.get(qi) or "",
                    "expected": qi_to_expected.get(qi) or "",
                }
                for p in providers_in_set:
                    c = qi_to_correctness.get(qi, {}).get(p)
                    ext = qi_to_extracted.get(qi, {}).get(p) or ""
                    badge = "✅" if c == 1 else ("❌" if c == 0 else "—")
                    row[p] = f"{badge} {ext}".strip()
                matrix_rows.append(row)
            matrix_df = pd.DataFrame(matrix_rows)
            matrix_column_config = {
                "q_index": st.column_config.NumberColumn("#", width=60, pinned=True),
                "question": st.column_config.TextColumn("Question", width="large", pinned=True),
                "expected": st.column_config.TextColumn("Expected", width="medium"),
            }
            for p in providers_in_set:
                matrix_column_config[p] = st.column_config.TextColumn(
                    p, width="medium"
                )
            st.dataframe(
                matrix_df,
                width="stretch",
                hide_index=True,
                column_config=matrix_column_config,
            )
            st.download_button(
                "Download matrix CSV",
                data=matrix_df.to_csv(index=False).encode("utf-8"),
                file_name=f"comparison_{selected_set[:8]}_matrix.csv",
                mime="text/csv",
                key=f"dl_cmp_mat_{selected_set}",
            )

            # ----- Drill into one question, side-by-side providers -----
            st.subheader("Drill into one question")

            def _fmt_cmp_q(qi: int) -> str:
                q = qi_to_question.get(qi) or ""
                truncated = q if len(q) <= 90 else q[:90] + "…"
                c = qi_to_correctness.get(qi, {})
                badges = "".join(
                    "✅" if c.get(p) == 1
                    else ("❌" if c.get(p) == 0 else "·")
                    for p in providers_in_set
                )
                return f"{badges}  Q{qi}  ·  {truncated}"

            drill_qi = st.selectbox(
                "Pick a question",
                filtered_qis,
                format_func=_fmt_cmp_q,
                key=f"cmp_drill_{selected_set}",
            )

            st.markdown(f"**Question:** {qi_to_question.get(drill_qi, '')}")
            st.markdown(
                f"**Expected answer:** `{qi_to_expected.get(drill_qi, '')}`"
            )

            cols = st.columns(len(providers_in_set))
            for col, p in zip(cols, providers_in_set):
                sub = data[(data["q_index"] == drill_qi) & (data["provider"] == p)]
                with col:
                    if sub.empty:
                        st.markdown(f"### · {p}")
                        st.caption("no result")
                        continue
                    r = sub.iloc[0]
                    is_c = r["is_correct"]
                    badge = "✅" if is_c == 1 else ("❌" if is_c == 0 else "—")
                    st.markdown(f"### {badge} {p}")
                    st.caption(
                        f"model={r['model_used']}  ·  "
                        f"{_fmt_duration(r['research_duration_seconds'])}"
                    )
                    st.markdown("**Extracted**")
                    st.markdown(_quote(r["extracted_answer"]))
                    if pd.notna(r["confidence"]):
                        st.caption(f"confidence {r['confidence']:.2f}")
                    if r["reasoning"]:
                        st.caption(r["reasoning"])
                    if r["error"]:
                        st.error(r["error"])
                    with st.expander("Report"):
                        st.markdown(r["research_content"] or "_(empty)_")
                    raw_sources = r["research_sources_json"]
                    if isinstance(raw_sources, str) and raw_sources:
                        srcs = json.loads(raw_sources)
                        with st.expander(f"Sources ({len(srcs)})"):
                            for i, s in enumerate(srcs, 1):
                                if isinstance(s, dict):
                                    title = s.get("title") or s.get("url") or f"source {i}"
                                    url = s.get("url")
                                    if url:
                                        st.markdown(f"{i}. [{title}]({url})")
                                    else:
                                        st.markdown(f"{i}. {title}")
                                else:
                                    st.markdown(f"{i}. {s}")
