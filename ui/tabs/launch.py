"""Launch run tab.

Cherry-pick mode for new evals. Pick a benchmark, set seed / cap /
workers, tick provider:model combinations, click rows in the dataset
table, see live cost preview + overlap warning, confirm launch, then
watch detached subprocesses progress in the in-flight panel.

UI state (seed, last-used providers, toggles, tier filter) persists
across reloads through ui/state.py.
"""
from __future__ import annotations

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


def render() -> None:
    st.subheader("Configure a new run")

    benchmarks = datasets.list_benchmarks()
    bench = st.segmented_control(
        "Benchmark",
        benchmarks,
        default=benchmarks[0],
        key="launch_bench",
    )
    if not bench:
        bench = benchmarks[0]
    total_q = ui_data.dataset_size(bench)
    st.caption(f"{bench}: {total_q} questions total")

    coverage = storage.get_coverage(bench)

    with st.expander("Run options", expanded=False):
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            seed_str = st.text_input(
                "Seed (blank = original order)",
                value=ui_state.persisted("launch_seed", ""),
                key="launch_seed",
                on_change=ui_state.save_ui_state,
            )
        with oc2:
            count = st.number_input(
                "Max per batch (cap)",
                min_value=1, max_value=50,
                value=ui_state.persisted("launch_count", 5), step=1,
                help="Hard cap on how many rows you can select at once.",
                key="launch_count",
                on_change=ui_state.save_ui_state,
            )
        with oc3:
            workers = st.number_input(
                "Workers",
                min_value=1, max_value=16,
                value=ui_state.persisted("launch_workers", 4), step=1,
                key="launch_workers",
                on_change=ui_state.save_ui_state,
            )
        note = st.text_input("Note (saved with every run)", value="", key="launch_note")

    try:
        seed_int = int(seed_str) if seed_str.strip() else None
    except ValueError:
        st.error("Seed must be an integer or empty.")
        seed_int = None

    # ----- Dataset table with coverage marks -----
    st.subheader("Questions to run")
    launch_tiers = ui_tiers.load_tiers()
    tier_names = list(launch_tiers.keys())
    if tier_names:
        tcol1, tcol2, tcol3 = st.columns(3)
    else:
        tcol1, tcol2 = st.columns(2)
        tcol3 = None
    with tcol1:
        show_details = st.toggle(
            "Show answer and duration in coverage cells",
            value=ui_state.persisted("launch_show_details", True),
            key="launch_show_details",
            on_change=ui_state.save_ui_state,
            help="Off = just ✅/❌/⚠ symbols (fits more providers on screen). "
                 "On = symbol plus extracted answer and run duration.",
        )
    with tcol2:
        english_only = st.toggle(
            "English only (hide CJK questions)",
            value=ui_state.persisted("launch_english_only", False),
            key="launch_english_only",
            on_change=ui_state.save_ui_state,
            help="Hides questions containing Chinese/Japanese/Korean "
                 "characters. Useful for finsearchcomp which mixes English "
                 "and Chinese prompts. q_index numbering is preserved.",
        )
    if tcol3 is not None:
        with tcol3:
            tier_filter_options = ["all"] + tier_names
            saved_tier_filter = ui_state.persisted("launch_tier_filter", "all")
            if saved_tier_filter not in tier_filter_options:
                saved_tier_filter = "all"
            tier_filter = st.segmented_control(
                "Tier columns",
                tier_filter_options,
                default=saved_tier_filter,
                key="launch_tier_filter",
                on_change=ui_state.save_ui_state,
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

    ds_df = ui_data.load_dataset_df(bench, seed_int).copy()
    if english_only:
        before = len(ds_df)
        ds_df = ds_df[~ds_df["question"].map(fmt.looks_non_english)].reset_index(drop=True)
        hidden = before - len(ds_df)
        if hidden:
            st.caption(f"Hiding {hidden} non-English question(s).")

    # Distribution summary for benchmarks that ship category/type metadata
    # (currently deepsearchqa). Helps the user balance their cherry-pick.
    meta_cols = [c for c in ("problem_category", "answer_type") if c in ds_df.columns]
    for col in meta_cols:
        vc = ds_df[col].value_counts()
        parts = [f"{k} ({v})" for k, v in vc.items()]
        st.caption(f"**{col}**: " + " · ".join(parts))

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
    # Same latest-wins convention as the Tier tab roster (see ui_tiers.tier_run_data).
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
                dur = fmt.fmt_duration_short(s.get("research_duration_seconds"))
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
        if launch_tiers:
            for ti, (_tname, members) in enumerate(launch_tiers.items()):
                if (p, m) in members:
                    return (ti, members.index((p, m)), p, m)
            return (len(launch_tiers), 0, p, m)
        return (0 if p == "tavily" else 1, 0, p, m)

    all_combos = sorted(combo_status.keys(), key=_combo_key)
    if tier_filter != "all":
        members = set(ui_tiers.tier_members(tier_filter, launch_tiers))
        all_combos = [c for c in all_combos if c in members]

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
    # Insert deepsearchqa metadata columns right after q_index/prompt_id so
    # they sit next to the row identifier when picking. Pinned so they stay
    # visible while scrolling through coverage columns on the right.
    next_meta_pos = 2 if "prompt_id" in ds_df.columns else 1
    if "problem_category" in ds_df.columns:
        table_cols.insert(next_meta_pos, "problem_category")
        column_config["problem_category"] = st.column_config.TextColumn(
            "Category", width=180, pinned=True,
            help="problem_category from the source dataset — use the "
                 "distribution caption above the table to balance picks.",
        )
        next_meta_pos += 1
    if "answer_type" in ds_df.columns:
        table_cols.insert(next_meta_pos, "answer_type")
        column_config["answer_type"] = st.column_config.TextColumn(
            "Ans type", width=120, pinned=True,
            help="answer_type from the source dataset.",
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
            st.code(fmt.ranges(sorted(selected_q_indices)), language=None)
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
                initial = ui_state.persisted(
                    ck_key, p == "tavily" and m == default_model
                )
                if st.checkbox(
                    m, value=initial, key=ck_key,
                    on_change=ui_state.save_ui_state,
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
                    already_covered.append((p, m, fmt.ranges(overlap)))
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
    est_cost, missing_costs = ui_costs.estimate_cost(
        provider_models, len(selected_q_indices),
    )
    if ui_costs.load_costs():
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
        est_cost_arg, _missing_costs_arg = ui_costs.estimate_cost(
            provider_models_arg, len(selected_q_indices_arg),
        )
        if ui_costs.load_costs():
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
                    already_covered_now.append((p, m, fmt.ranges(overlap)))
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
                "elapsed": fmt.elapsed_human(r["started_at"]),
                "started": (r["started_at"] or "")[:16].replace("T", " "),
                "note": r["note"] or "",
            }
            for r in in_flight
        ])
        st.dataframe(flight_df, width="stretch", hide_index=True)

    _render_in_flight()
