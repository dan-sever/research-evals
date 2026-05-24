"""Streamlit dashboard for browsing eval runs and comparing providers.

    streamlit run app.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from benchmarks import datasets, launcher, providers, storage
from benchmarks.config import load_env
from ui import cache as ui_cache
from ui import costs as ui_costs
from ui import data as ui_data
from ui import format as fmt
from ui import state as ui_state
from ui import tiers as ui_tiers
from ui.tabs import dashboard as dashboard_tab
from ui.tabs import export as export_tab
from ui.tabs import insights as insights_tab
from ui.tabs import inspect as inspect_tab

st.set_page_config(page_title="Research Benchmarks", layout="wide")
st.title("Research benchmark runs")

storage.init_db()

# One-shot log prune at app boot. Keeps last 100 logs, skips anything
# touched in the last hour so in-flight runs are never disturbed.
if "_logs_pruned" not in st.session_state:
    try:
        launcher.prune_logs(keep_n=100, min_age_seconds=3600)
    except Exception:
        pass
    st.session_state["_logs_pruned"] = True


# Helpers (UI state, formatting, tiers, parquet readers, cost lookup)
# live in dedicated modules under ui/. Local aliases keep the tab bodies
# below unchanged while we incrementally move them into ui/tabs/.
_persisted = ui_state.persisted
_save_ui_state = ui_state.save_ui_state
_elapsed_human = fmt.elapsed_human
_fmt_duration = fmt.fmt_duration
_fmt_duration_short = fmt.fmt_duration_short
_quote = fmt.quote
_ranges = fmt.ranges
_looks_non_english = fmt.looks_non_english
_load_tiers = ui_tiers.load_tiers
_tier_members = ui_tiers.tier_members
_tier_run_data = ui_tiers.tier_run_data
_dataset_size = ui_data.dataset_size
_load_dataset_df = ui_data.load_dataset_df
_prompt_ids = ui_data.prompt_ids
_load_costs = ui_costs.load_costs
_estimate_cost = ui_costs.estimate_cost


(tab_launch, tab_inspect, tab_compare, tab_tier, tab_dashboard,
 tab_insights, tab_export) = st.tabs(
    ["Launch run", "Single run inspector", "Provider comparison",
     "Tier analysis", "Dashboard", "Insights", "Export data"]
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
    _launch_tiers = _load_tiers()
    _tier_names = list(_launch_tiers.keys())
    if _tier_names:
        tcol1, tcol2, tcol3 = st.columns(3)
    else:
        tcol1, tcol2 = st.columns(2)
        tcol3 = None
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
    if tcol3 is not None:
        with tcol3:
            _tier_filter_options = ["all"] + _tier_names
            _saved_tier_filter = _persisted("launch_tier_filter", "all")
            if _saved_tier_filter not in _tier_filter_options:
                _saved_tier_filter = "all"
            tier_filter = st.segmented_control(
                "Tier columns",
                _tier_filter_options,
                default=_saved_tier_filter,
                key="launch_tier_filter",
                on_change=_save_ui_state,
                help="Limit coverage columns to one tier's members "
                     "(from model_tiers.json).",
            )
            if not tier_filter:
                tier_filter = "all"
    else:
        tier_filter = "all"
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
    #
    # `get_question_status` returns every attempt (one row per run_id, q_index),
    # sorted by run_id ascending. Dedupe first by overwriting per
    # (provider, model, q_index) so retries collapse to the latest attempt,
    # then derive cell symbols and per-column scores from the deduped map.
    # Same latest-wins convention as the Tier tab roster (see _tier_run_data).
    # Counting before dedupe would inflate `total` by every retry — e.g.
    # Perplexity heavy with many 429/401 retries on the same q_indices would
    # show e.g. (1/34, 3%) in the header while the body shows only 15 cells.
    visible_q_indices = set(ds_df["q_index"].astype(int).tolist())
    status_rows = ui_cache.question_status(bench)
    combo_latest: dict[tuple[str, str], dict[int, dict]] = {}
    for s in status_rows:
        if s["seed"] != seed_int:
            continue
        if int(s["q_index"]) not in visible_q_indices:
            continue
        key = (s["provider"], s["model"])
        combo_latest.setdefault(key, {})[int(s["q_index"])] = s

    combo_status: dict[tuple[str, str], dict[int, str]] = {}
    combo_score: dict[tuple[str, str], dict[str, int]] = {}
    for key, qi_to_row in combo_latest.items():
        # `graded` excludes errored / ungraded rows from the accuracy
        # denominator so a flaky API doesn't artificially drag accuracy down.
        score = combo_score.setdefault(
            key, {"correct": 0, "graded": 0, "total": 0}
        )
        status_map = combo_status.setdefault(key, {})
        for qi, s in qi_to_row.items():
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
            status_map[qi] = cell
            score["total"] += 1
            if s["is_correct"] is not None:
                score["graded"] += 1
            if s["is_correct"] == 1:
                score["correct"] += 1

    def _combo_key(combo):
        # When tiers are defined, sort by (tier order, position within tier,
        # provider, model). Members of the first tier come first in their
        # declared order, then second tier, etc; anything outside every tier
        # falls to the end alphabetically. Without tiers, fall back to the
        # original tavily-first-then-alphabetical ordering.
        p, m = combo
        if _launch_tiers:
            for ti, (_tname, members) in enumerate(_launch_tiers.items()):
                if (p, m) in members:
                    return (ti, members.index((p, m)), p, m)
            return (len(_launch_tiers), 0, p, m)
        return (0 if p == "tavily" else 1, 0, p, m)

    all_combos = sorted(combo_status.keys(), key=_combo_key)
    if tier_filter != "all":
        _members = set(_tier_members(tier_filter, _launch_tiers))
        all_combos = [c for c in all_combos if c in _members]

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
    if "prompt_id" in ds_df.columns:
        table_cols.insert(1, "prompt_id")
        column_config["prompt_id"] = st.column_config.TextColumn(
            "Prompt ID", width=220, pinned=True,
            help="Source dataset's prompt identifier (e.g. "
                 "`(T2)Simple_Historical_Lookup_001`).",
        )
    coverage_col_width = 220 if show_details else 130
    for p, m in all_combos:
        col = f"{p}:{m}"
        sc = combo_score.get((p, m), {"correct": 0, "graded": 0, "total": 0})
        graded = sc.get("graded", 0)
        pct = (sc["correct"] / graded * 100) if graded else 0
        pct_label = f"{pct:.0f}%" if graded else "—"
        label = f"{col} ({sc['correct']}/{graded}, {pct_label})"
        column_config[col] = st.column_config.TextColumn(label, width=coverage_col_width)

    table_height = min(720, max(300, 90 + 38 * len(ds_df)))
    # Bumping launch_table_version forces Streamlit to mount a fresh
    # dataframe widget, which is the reliable way to clear a multi-row
    # selection. The Deselect all button below bumps it.
    sel_version = st.session_state.get("launch_table_version", 0)
    sel = st.dataframe(
        ds_df[table_cols],
        on_select="rerun",
        selection_mode="multi-row",
        column_config=column_config,
        width="stretch",
        hide_index=True,
        height=table_height,
        key=f"launch_table_{bench}_{seed_int}_v{sel_version}",
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
        sel_c1, sel_c2, sel_c3 = st.columns([1, 3, 1])
        sel_c1.metric("Selected", f"{len(selected_q_indices)}")
        with sel_c2:
            st.caption("q_index ranges")
            st.code(_ranges(sorted(selected_q_indices)), language=None)
        with sel_c3:
            if st.button(
                "Deselect all",
                key="launch_deselect_all",
                width="stretch",
                help="Clear the current row selection. Does not delete any data.",
            ):
                st.session_state["launch_table_version"] = sel_version + 1
                st.rerun()

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
    est_cost, missing_costs = _estimate_cost(
        provider_models, len(selected_q_indices),
    )
    if _load_costs():
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Runs", runs_planned)
        mc2.metric("Research calls", calls_planned)
        mc3.metric("Judge calls", calls_planned)
        # Hard to predict the actual bill; this is a rough order-of-magnitude
        # from list prices in model_costs.json. Useful as a sanity check
        # before clicking Launch, not as a finance number.
        cost_label = f"≈ ${est_cost:.2f}" if est_cost else "—"
        help_text = "Rough estimate from model_costs.json. Includes judge calls."
        if missing_costs:
            help_text += (
                "  Missing prices for: "
                + ", ".join(f"{p}:{m}" for p, m in missing_costs)
                + ". Those rows counted as $0 — edit model_costs.json to fill them in."
            )
        mc4.metric("Est. cost", cost_label, help=help_text)
    else:
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
        est_cost_arg, _missing_costs_arg = _estimate_cost(
            provider_models_arg, len(selected_q_indices_arg),
        )
        if _load_costs():
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Runs", runs_planned_arg)
            dc2.metric("Research calls", calls_planned_arg)
            dc3.metric("Judge calls", calls_planned_arg)
            dc4.metric(
                "Est. cost",
                f"≈ ${est_cost_arg:.2f}" if est_cost_arg else "—",
                help="Rough estimate from model_costs.json.",
            )
        else:
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
# Tab 1 — Single run inspector (delegated to ui/tabs/inspect.py)
# ============================================================================
with tab_inspect:
    inspect_tab.render()


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
        # accuracy = correct / graded so errored / ungraded rows do not drag
        # the rate down. `total` and `errors` stay visible as separate cols.
        summary_rows = []
        for r in set_runs:
            total = r["total"] or 0
            correct = r["correct"] or 0
            graded = r.get("graded") or 0
            summary_rows.append({
                "provider": r["provider"],
                "model": r["model"],
                "total": total,
                "graded": graded,
                "correct": correct,
                "errors": r["errors"] or 0,
                "accuracy_raw": correct / graded if graded else None,
                "avg_seconds_raw": r["avg_seconds"] or 0,
                "run_id": r["id"],
            })
        summary_df = pd.DataFrame(summary_rows).sort_values(
            "accuracy_raw", ascending=False, na_position="last"
        )
        display_summary = summary_df.assign(
            accuracy=summary_df["accuracy_raw"].map(
                lambda x: f"{x:.1%}" if x is not None else "—"
            ),
            avg_duration=summary_df["avg_seconds_raw"].map(_fmt_duration),
        )[["provider", "model", "total", "graded", "correct", "errors",
           "accuracy", "avg_duration", "run_id"]].reset_index(drop=True)
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
        # Key everything downstream by (provider, model) tuples so a
        # comparison_set with multiple models from the same provider
        # (e.g. tavily:mini + tavily:pro) doesn't collapse into one column.
        # Tavily entries come first, then everyone else alphabetical.
        combos_in_set = sorted(
            {(p, m) for p, m in zip(data["provider"], data["model_used"])},
            key=lambda c: (0 if c[0] == "tavily" else 1, c[0], c[1]),
        )
        combo_labels = [f"{p}:{m}" for p, m in combos_in_set]

        # Pre-compute lookups for filters and rendering
        qi_to_question = (
            data.drop_duplicates("q_index").set_index("q_index")["question"].to_dict()
        )
        qi_to_expected = (
            data.drop_duplicates("q_index").set_index("q_index")["expected_answer"].to_dict()
        )
        qi_to_correctness: dict[int, dict[tuple[str, str], int | None]] = {}
        qi_to_extracted: dict[int, dict[tuple[str, str], str | None]] = {}
        for _, r in data.iterrows():
            qi = int(r["q_index"])
            combo = (r["provider"], r["model_used"])
            qi_to_correctness.setdefault(qi, {})[combo] = (
                int(r["is_correct"]) if pd.notna(r["is_correct"]) else None
            )
            qi_to_extracted.setdefault(qi, {})[combo] = r.get("extracted_answer")

        # ----- Matrix table -----
        st.subheader("Per-question matrix")
        st.caption(
            "One row per question, one column per provider:model. ✅ correct, "
            "❌ wrong, — no result. Hover or scroll right for the extracted answer."
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
            has_tavily = any(p == "tavily" for p, _ in combos_in_set)
            if has_tavily:
                tav_pivot = st.toggle(
                    "Tavily pivot",
                    value=False,
                    key="cmp_tavily_pivot",
                    help="When on, show only questions where every Tavily "
                         "entry in this set is right (or wrong) and every "
                         "non-Tavily entry disagrees. Overrides the filter "
                         "on the left.",
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
                tav = [
                    v for combo, v in c.items()
                    if combo[0] == "tavily" and v is not None
                ]
                oth = [
                    v for combo, v in c.items()
                    if combo[0] != "tavily" and v is not None
                ]
                if not tav or not oth:
                    return False
                if tav_side == "wins (unique)":
                    return all(v == 1 for v in tav) and all(v == 0 for v in oth)
                if tav_side == "loses (unique)":
                    return all(v == 0 for v in tav) and all(v == 1 for v in oth)
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
            _cmp_pid_map = _prompt_ids(meta["benchmark"], meta.get("seed"))
            matrix_rows = []
            for qi in filtered_qis:
                row = {
                    "q_index": qi,
                    "question": qi_to_question.get(qi) or "",
                    "expected": qi_to_expected.get(qi) or "",
                }
                if _cmp_pid_map:
                    row["prompt_id"] = _cmp_pid_map.get(int(qi), "")
                for combo, label in zip(combos_in_set, combo_labels):
                    c = qi_to_correctness.get(qi, {}).get(combo)
                    ext = qi_to_extracted.get(qi, {}).get(combo) or ""
                    badge = "✅" if c == 1 else ("❌" if c == 0 else "—")
                    row[label] = f"{badge} {ext}".strip()
                matrix_rows.append(row)
            matrix_df = pd.DataFrame(matrix_rows)
            matrix_column_config = {
                "q_index": st.column_config.NumberColumn("#", width=60, pinned=True),
                "question": st.column_config.TextColumn("Question", width="large", pinned=True),
                "expected": st.column_config.TextColumn("Expected", width="medium"),
            }
            if _cmp_pid_map:
                matrix_column_config["prompt_id"] = st.column_config.TextColumn(
                    "Prompt ID", width=220, pinned=True,
                )
            for label in combo_labels:
                matrix_column_config[label] = st.column_config.TextColumn(
                    label, width="medium"
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
                    "✅" if c.get(combo) == 1
                    else ("❌" if c.get(combo) == 0 else "·")
                    for combo in combos_in_set
                )
                return f"{badges}  Q{qi}  ·  {truncated}"

            drill_qi = st.selectbox(
                "Pick a question",
                filtered_qis,
                format_func=_fmt_cmp_q,
                key=f"cmp_drill_{selected_set}",
            )

            if _cmp_pid_map and _cmp_pid_map.get(int(drill_qi)):
                st.caption(f"prompt_id: `{_cmp_pid_map[int(drill_qi)]}`")
            st.markdown(f"**Question:** {qi_to_question.get(drill_qi, '')}")
            st.markdown(
                f"**Expected answer:** `{qi_to_expected.get(drill_qi, '')}`"
            )

            cols = st.columns(len(combos_in_set))
            for col, (provider, model) in zip(cols, combos_in_set):
                label = f"{provider}:{model}"
                sub = data[
                    (data["q_index"] == drill_qi)
                    & (data["provider"] == provider)
                    & (data["model_used"] == model)
                ]
                with col:
                    if sub.empty:
                        st.markdown(f"### · {label}")
                        st.caption("no result")
                        continue
                    r = sub.iloc[0]
                    is_c = r["is_correct"]
                    badge = "✅" if is_c == 1 else ("❌" if is_c == 0 else "—")
                    st.markdown(f"### {badge} {label}")
                    st.caption(
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


# ============================================================================
# Tab 3 — Tier analysis
# ============================================================================
with tab_tier:
    _tiers_def = _load_tiers()
    if not _tiers_def:
        st.info(
            "No tiers defined. Create `model_tiers.json` at the project "
            "root with a shape like:\n\n"
            "```json\n"
            "{\n"
            "  \"fast\":  [[\"tavily\", \"mini\"], [\"perplexity\", \"sonar-reasoning-pro\"]],\n"
            "  \"heavy\": [[\"tavily\", \"pro\"],  [\"perplexity\", \"sonar-deep-research\"]]\n"
            "}\n"
            "```"
        )
    else:
        # ----- Picker row: benchmark, tier (seed only when ambiguous) -----
        pc1, pc2 = st.columns(2)
        with pc1:
            _benchmarks = datasets.list_benchmarks()
            _saved_bench = _persisted("tier_bench", _benchmarks[0])
            if _saved_bench not in _benchmarks:
                _saved_bench = _benchmarks[0]
            tier_bench = st.segmented_control(
                "Benchmark",
                _benchmarks,
                default=_saved_bench,
                key="tier_bench",
                on_change=_save_ui_state,
            )
            if not tier_bench:
                tier_bench = _benchmarks[0]
        with pc2:
            _tier_keys = ["all"] + list(_tiers_def.keys())
            _saved_tier = _persisted("tier_name", _tier_keys[0])
            if _saved_tier not in _tier_keys:
                _saved_tier = _tier_keys[0]
            tier_name = st.segmented_control(
                "Tier",
                _tier_keys,
                default=_saved_tier,
                key="tier_name",
                on_change=_save_ui_state,
            )
            if not tier_name:
                tier_name = _tier_keys[0]

        if tier_name == "all":
            # Union of every tier's members, preserving first-seen order.
            seen: set[tuple[str, str]] = set()
            tier_member_list = []
            for _tname, _members in _tiers_def.items():
                for _pair in _members:
                    _key = tuple(_pair)
                    if _key not in seen:
                        seen.add(_key)
                        tier_member_list.append(_key)
        else:
            tier_member_list = _tier_members(tier_name, _tiers_def)
        _member_set = set(tier_member_list)

        # Distinct seeds with data for this tier on this benchmark.
        _status_rows = ui_cache.question_status(tier_bench)
        _seeds = sorted(
            {
                s["seed"] for s in _status_rows
                if (s["provider"], s["model"]) in _member_set
            },
            key=lambda x: (x is None, x),
        )

        # Only surface a seed picker when there's actual ambiguity. When all
        # data is at a single seed (the common case for UI-launched runs),
        # we silently pick it and keep the picker out of sight.
        if not _seeds:
            tier_seed = None
        elif len(_seeds) == 1:
            tier_seed = _seeds[0]
        else:
            _saved_seed = _persisted("tier_seed", _seeds[0])
            if _saved_seed not in _seeds:
                _saved_seed = _seeds[0]
            tier_seed = st.selectbox(
                "Seed (multiple detected — pick which slice to compare)",
                _seeds,
                index=_seeds.index(_saved_seed),
                format_func=lambda s: "(no seed)" if s is None else str(s),
                key="tier_seed",
                on_change=_save_ui_state,
                help="Different seeds shuffle the dataset differently, so "
                     "q_index 5 means a different question across seeds. "
                     "Pick one slice to keep comparisons honest.",
            )

        if not _seeds:
            st.info(
                f"No runs yet for any member of the `{tier_name}` tier on "
                f"`{tier_bench}`. Once a tier member has run this benchmark, "
                "results will appear here."
            )
        else:
            data, run_ids = _tier_run_data(tier_bench, tier_member_list, tier_seed)

            # ----- Tier roster -----
            st.subheader(f"`{tier_name}` tier roster")
            roster_cols = st.columns(len(tier_member_list))
            for col, (p, m) in zip(roster_cols, tier_member_list):
                with col:
                    if (p, m) not in run_ids:
                        st.metric(f"{p}:{m}", "—", help="No run yet")
                    else:
                        sub = data[
                            (data["provider"] == p)
                            & (data["model_used"] == m)
                        ]
                        total = len(sub)
                        correct = int((sub["is_correct"] == 1).sum())
                        graded = int(sub["is_correct"].notna().sum())
                        errored = int(sub.get("error", pd.Series([])).notna().sum())
                        accuracy_str = f"{correct / graded:.1%}" if graded else "—"
                        st.metric(
                            f"{p}:{m}",
                            accuracy_str,
                            help=(
                                f"latest run #{run_ids[(p, m)]}  ·  "
                                f"{correct}/{graded} graded correct  ·  "
                                f"{errored} errored, {total - graded - errored} ungraded "
                                "(latest answer per q_index wins; errors excluded "
                                "from the accuracy denominator)"
                            ),
                        )

            if data.empty:
                st.info("No per-question data for this tier yet.")
            else:
                providers_in_tier = [
                    (p, m) for (p, m) in tier_member_list if (p, m) in run_ids
                ]

                qi_to_question = (
                    data.drop_duplicates("q_index")
                    .set_index("q_index")["question"].to_dict()
                )
                qi_to_expected = (
                    data.drop_duplicates("q_index")
                    .set_index("q_index")["expected_answer"].to_dict()
                )
                qi_to_correctness: dict[int, dict[tuple[str, str], int | None]] = {}
                qi_to_extracted: dict[int, dict[tuple[str, str], str | None]] = {}
                for _, row in data.iterrows():
                    qi = int(row["q_index"])
                    key = (row["provider"], row["model_used"])
                    qi_to_correctness.setdefault(qi, {})[key] = (
                        int(row["is_correct"]) if pd.notna(row["is_correct"]) else None
                    )
                    qi_to_extracted.setdefault(qi, {})[key] = row.get("extracted_answer")

                all_qis = sorted(qi_to_correctness.keys())

                # ----- Filters -----
                fc1, fc2 = st.columns([2, 2])
                with fc1:
                    _filter_opts = ["all", "any disagreement", "all correct", "all wrong"]
                    _saved_filter = _persisted("tier_filter", "all")
                    if _saved_filter not in _filter_opts:
                        _saved_filter = "all"
                    tier_filter_val = st.segmented_control(
                        "Filter",
                        _filter_opts,
                        default=_saved_filter,
                        key="tier_filter",
                        on_change=_save_ui_state,
                    )
                    if not tier_filter_val:
                        tier_filter_val = "all"
                    tier_search = st.text_input(
                        "Search question text",
                        "",
                        key="tier_search",
                    )
                with fc2:
                    _has_tavily = any(p == "tavily" for (p, _m) in providers_in_tier)
                    if _has_tavily:
                        tier_pivot = st.toggle(
                            "Tavily pivot",
                            value=_persisted("tier_pivot", False),
                            key="tier_pivot",
                            on_change=_save_ui_state,
                            help="Show only questions where tavily members are "
                                 "uniquely right or wrong vs every other tier "
                                 "member. Overrides the filter on the left.",
                        )
                        if tier_pivot:
                            _side_opts = ["wins (unique)", "loses (unique)"]
                            _saved_side = _persisted("tier_pivot_side", "wins (unique)")
                            if _saved_side not in _side_opts:
                                _saved_side = "wins (unique)"
                            tier_pivot_side = st.segmented_control(
                                "Tavily side",
                                _side_opts,
                                default=_saved_side,
                                key="tier_pivot_side",
                                on_change=_save_ui_state,
                            )
                            if not tier_pivot_side:
                                tier_pivot_side = "wins (unique)"
                        else:
                            tier_pivot_side = None
                    else:
                        tier_pivot = False
                        tier_pivot_side = None

                def _tier_passes(qi: int) -> bool:
                    c = qi_to_correctness.get(qi, {})
                    vals = [v for v in c.values() if v is not None]
                    if tier_pivot and _has_tavily:
                        if not vals:
                            return False
                        tav = [v for k, v in c.items() if k[0] == "tavily" and v is not None]
                        oth = [v for k, v in c.items() if k[0] != "tavily" and v is not None]
                        if not tav or not oth:
                            return False
                        if tier_pivot_side == "wins (unique)":
                            return all(v == 1 for v in tav) and all(v == 0 for v in oth)
                        if tier_pivot_side == "loses (unique)":
                            return all(v == 0 for v in tav) and all(v == 1 for v in oth)
                        return True
                    if not vals:
                        return tier_filter_val == "all"
                    if tier_filter_val == "all":
                        return True
                    if tier_filter_val == "any disagreement":
                        return len(set(vals)) > 1
                    if tier_filter_val == "all correct":
                        return all(v == 1 for v in vals)
                    if tier_filter_val == "all wrong":
                        return all(v == 0 for v in vals)
                    return True

                filtered_qis = [qi for qi in all_qis if _tier_passes(qi)]
                if tier_search:
                    needle = tier_search.lower()
                    filtered_qis = [
                        qi for qi in filtered_qis
                        if needle in (qi_to_question.get(qi) or "").lower()
                    ]

                st.markdown(f"**{len(filtered_qis)} of {len(all_qis)} questions**")

                if not filtered_qis:
                    st.info("No questions match this filter.")
                else:
                    # ----- Matrix -----
                    _tier_pid_map = _prompt_ids(tier_bench, tier_seed)
                    matrix_rows = []
                    for qi in filtered_qis:
                        row = {
                            "q_index": qi,
                            "question": qi_to_question.get(qi) or "",
                            "expected": qi_to_expected.get(qi) or "",
                        }
                        if _tier_pid_map:
                            row["prompt_id"] = _tier_pid_map.get(int(qi), "")
                        for (p, m) in providers_in_tier:
                            c = qi_to_correctness.get(qi, {}).get((p, m))
                            ext = qi_to_extracted.get(qi, {}).get((p, m)) or ""
                            badge = "✅" if c == 1 else ("❌" if c == 0 else "—")
                            row[f"{p}:{m}"] = f"{badge} {ext}".strip()
                        matrix_rows.append(row)
                    matrix_df = pd.DataFrame(matrix_rows)

                    mcc = {
                        "q_index": st.column_config.NumberColumn(
                            "#", width=60, pinned=True
                        ),
                        "question": st.column_config.TextColumn(
                            "Question", width="large", pinned=True
                        ),
                        "expected": st.column_config.TextColumn(
                            "Expected", width="medium"
                        ),
                    }
                    if _tier_pid_map:
                        mcc["prompt_id"] = st.column_config.TextColumn(
                            "Prompt ID", width=220, pinned=True,
                        )
                    for (p, m) in providers_in_tier:
                        col_name = f"{p}:{m}"
                        mcc[col_name] = st.column_config.TextColumn(
                            col_name, width="medium"
                        )

                    st.caption("Click a row to drill into one question.")
                    seed_tag = "noseed" if tier_seed is None else str(tier_seed)
                    mat_sel = st.dataframe(
                        matrix_df,
                        on_select="rerun",
                        selection_mode="single-row",
                        width="stretch",
                        hide_index=True,
                        column_config=mcc,
                        key=f"tier_matrix_{tier_bench}_{tier_name}_{seed_tag}",
                    )

                    # CSV downloads
                    roster_rows = []
                    for (p, m) in tier_member_list:
                        if (p, m) not in run_ids:
                            roster_rows.append({
                                "provider": p, "model": m,
                                "total": 0, "graded": 0, "correct": 0, "errored": 0,
                                "accuracy": "—", "run_id": None,
                            })
                        else:
                            sub = data[
                                (data["provider"] == p)
                                & (data["model_used"] == m)
                            ]
                            total = len(sub)
                            correct = int((sub["is_correct"] == 1).sum())
                            graded = int(sub["is_correct"].notna().sum())
                            errored = int(sub.get("error", pd.Series([])).notna().sum())
                            acc_str = f"{correct / graded:.1%}" if graded else "—"
                            roster_rows.append({
                                "provider": p, "model": m,
                                "total": total, "graded": graded,
                                "correct": correct, "errored": errored,
                                "accuracy": acc_str,
                                "run_id": run_ids[(p, m)],
                            })
                    roster_df = pd.DataFrame(roster_rows)

                    dl_c1, dl_c2 = st.columns(2)
                    with dl_c1:
                        st.download_button(
                            "Download roster CSV",
                            data=roster_df.to_csv(index=False).encode("utf-8"),
                            file_name=(
                                f"tier_{tier_name}_{tier_bench}_seed{seed_tag}_roster.csv"
                            ),
                            mime="text/csv",
                            key=f"dl_tier_r_{tier_name}_{tier_bench}_{seed_tag}",
                        )
                    with dl_c2:
                        st.download_button(
                            "Download matrix CSV",
                            data=matrix_df.to_csv(index=False).encode("utf-8"),
                            file_name=(
                                f"tier_{tier_name}_{tier_bench}_seed{seed_tag}_matrix.csv"
                            ),
                            mime="text/csv",
                            key=f"dl_tier_m_{tier_name}_{tier_bench}_{seed_tag}",
                        )

                    # ----- Drill view -----
                    drill_positions = list(mat_sel.selection.rows or [])
                    if not drill_positions:
                        st.caption("Click a matrix row to see side-by-side drill detail.")
                    else:
                        drill_qi = int(matrix_df.iloc[drill_positions[0]]["q_index"])
                        st.subheader("Drill into one question")
                        if _tier_pid_map and _tier_pid_map.get(drill_qi):
                            st.caption(f"prompt_id: `{_tier_pid_map[drill_qi]}`")
                        st.markdown(f"**Question:** {qi_to_question.get(drill_qi, '')}")
                        st.markdown(
                            f"**Expected answer:** `{qi_to_expected.get(drill_qi, '')}`"
                        )

                        d_cols = st.columns(len(providers_in_tier))
                        for col, (p, m) in zip(d_cols, providers_in_tier):
                            sub = data[
                                (data["q_index"] == drill_qi)
                                & (data["provider"] == p)
                                & (data["model_used"] == m)
                            ]
                            with col:
                                if sub.empty:
                                    st.markdown(f"### · {p}:{m}")
                                    st.caption("no result")
                                    continue
                                r = sub.iloc[0]
                                is_c = r["is_correct"]
                                badge = "✅" if is_c == 1 else ("❌" if is_c == 0 else "—")
                                st.markdown(f"### {badge} {p}:{m}")
                                st.caption(
                                    f"{_fmt_duration(r['research_duration_seconds'])}"
                                )
                                st.markdown("**Extracted**")
                                st.markdown(_quote(r["extracted_answer"]))
                                if pd.notna(r.get("confidence")):
                                    st.caption(f"confidence {r['confidence']:.2f}")
                                if r.get("reasoning"):
                                    st.caption(r["reasoning"])
                                if r.get("error"):
                                    st.error(r["error"])
                                with st.expander("Report"):
                                    st.markdown(r["research_content"] or "_(empty)_")
                                raw_sources = r.get("research_sources_json")
                                if isinstance(raw_sources, str) and raw_sources:
                                    srcs = json.loads(raw_sources)
                                    with st.expander(f"Sources ({len(srcs)})"):
                                        for i, s in enumerate(srcs, 1):
                                            if isinstance(s, dict):
                                                title = (
                                                    s.get("title")
                                                    or s.get("url")
                                                    or f"source {i}"
                                                )
                                                url = s.get("url")
                                                if url:
                                                    st.markdown(f"{i}. [{title}]({url})")
                                                else:
                                                    st.markdown(f"{i}. {title}")
                                            else:
                                                st.markdown(f"{i}. {s}")



with tab_dashboard:
    dashboard_tab.render()


with tab_insights:
    insights_tab.render()


with tab_export:
    export_tab.render()
