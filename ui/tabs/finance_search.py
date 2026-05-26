"""Finance search tab.

Focused launcher + head-to-head view for evaluating Tavily *search* (not
research) against EXA and Parallel on the two finance benchmarks. Kept
separate from the generic Launch tab so the framing stays clean: this is
a search-tier competitive eval, not a free-form runner.

Mechanics borrow from `ui/tabs/launch.py` (question picker, cost preview,
launch dialog, in-flight panel) and `ui/tabs/compare.py` (per-question
matrix). Differences:

* Benchmark choice is restricted to `financebench` and `financeqa`.
* Provider list is restricted to the three search-tier providers
  (`tavily_search`, `exa_search`, `parallel_search`).
* Pre-flight key check additionally requires `ANTHROPIC_API_KEY` because
  EXA and Parallel synthesize their final answer via Claude Haiku.
* The latest comparison_set scoped to these providers + benchmark is
  rendered inline at the bottom — no need to flip to the Compare tab.
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

FINANCE_BENCHMARKS = ("financebench", "financeqa")
SEARCH_PROVIDERS = ("tavily_search", "exa_search", "parallel_search")


def _combo_sort_key(combo: tuple[str, str]) -> tuple[int, str, str]:
    """Sort Tavily first, then alphabetical — matches Compare tab."""
    p, m = combo
    return (0 if p == "tavily_search" else 1, p, m)


def render() -> None:
    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_l:
        st.subheader("Tavily search vs. EXA vs. Parallel")
        st.caption(
            "Search-tier evaluation on financial QA benchmarks. Each "
            "provider's `/search` endpoint returns URLs + snippets only "
            "(no provider-side answer synthesis). The same Claude Haiku "
            "model then synthesizes the final answer from those snippets, "
            "so what we're grading is the quality of each provider's "
            "retrieval, not their LLMs."
        )
    with hdr_r:
        # `ui_cache.question_status` is cached at 10s TTL and the in-flight
        # panel runs in a fragment, so the main page does not re-render
        # while runs are landing. This forces a fresh read.
        if st.button(
            "Refresh",
            key="finsearch_refresh",
            width="stretch",
            help="Clear cached query results and rerun the page so newly "
                 "landed run results show up.",
        ):
            ui_cache.question_status.clear()
            st.rerun()

    bench = st.segmented_control(
        "Benchmark",
        FINANCE_BENCHMARKS,
        default=ui_state.persisted("finsearch_bench", FINANCE_BENCHMARKS[0]),
        key="finsearch_bench",
        on_change=ui_state.save_ui_state,
    )
    if not bench:
        bench = FINANCE_BENCHMARKS[0]
    total_q = ui_data.dataset_size(bench)
    st.caption(f"{bench}: {total_q} questions total")

    coverage = storage.get_coverage(bench)

    with st.expander("Run options", expanded=False):
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            seed_str = st.text_input(
                "Seed (blank = original order)",
                value=ui_state.persisted("finsearch_seed", ""),
                key="finsearch_seed",
                on_change=ui_state.save_ui_state,
            )
        with oc2:
            count = st.number_input(
                "Max per batch (cap)",
                min_value=1, max_value=50,
                value=ui_state.persisted("finsearch_count", 10), step=1,
                key="finsearch_count",
                on_change=ui_state.save_ui_state,
            )
        with oc3:
            workers = st.number_input(
                "Workers",
                min_value=1, max_value=16,
                value=ui_state.persisted("finsearch_workers", 4), step=1,
                key="finsearch_workers",
                on_change=ui_state.save_ui_state,
            )
        note = st.text_input(
            "Note (saved with every run)", value="", key="finsearch_note",
        )

    try:
        seed_int = int(seed_str) if seed_str.strip() else None
    except ValueError:
        st.error("Seed must be an integer or empty.")
        seed_int = None

    # ----- Question picker -----
    st.subheader("Questions to run")
    st.caption(
        "Tick rows to pick questions. The right-hand columns show every "
        "search-tier provider:model that has answered this benchmark at "
        "the current seed (✅ correct, ❌ wrong, ⚠ error, blank = not run)."
    )

    ds_df = ui_data.load_dataset_df(bench, seed_int).copy()
    # Surface the benchmark's own metadata columns. They're already
    # cast to str inside `load_dataset_df` if present in the parquet.
    extras_in_df = [
        c for c in ("company", "question_type", "doc_period", "doc_name", "file_name")
        if c in ds_df.columns
    ]
    # Add them as ds_df columns if they're not already there — they aren't,
    # because `load_dataset_df` keeps only the prompt_id + reasoning extras.
    # We need to read the parquet a second time to pull these in.
    if not extras_in_df:
        spec = datasets.REGISTRY[bench]
        raw_df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
        if seed_int is not None:
            raw_df = raw_df.sample(frac=1, random_state=seed_int).reset_index(drop=True)
        raw_df = raw_df.reset_index(drop=False).rename(columns={"index": "q_index"})
        for extra in ("company", "question_type", "doc_period"):
            if extra in raw_df.columns:
                ds_df[extra] = raw_df[extra].astype(str)
                extras_in_df.append(extra)

    # Distribution caption — helpful for picking a balanced sample.
    if "question_type" in ds_df.columns:
        vc = ds_df["question_type"].value_counts()
        parts = [f"{k} ({v})" for k, v in vc.items()]
        st.caption("**question_type**: " + " · ".join(parts))

    ds_df["question"] = ds_df["question"].str.slice(0, 140)
    ds_df["expected_answer"] = ds_df["expected_answer"].str.slice(0, 80)

    # Build coverage cells scoped to search providers only. Latest run wins
    # per (provider, model, q_index) so retries collapse to one cell.
    visible_q_indices = set(ds_df["q_index"].astype(int).tolist())
    status_rows = ui_cache.question_status(bench)
    combo_latest: dict[tuple[str, str], dict[int, dict]] = {}
    for s in status_rows:
        if s["seed"] != seed_int:
            continue
        if s["provider"] not in SEARCH_PROVIDERS:
            continue
        if int(s["q_index"]) not in visible_q_indices:
            continue
        key = (s["provider"], s["model"])
        combo_latest.setdefault(key, {})[int(s["q_index"])] = s

    combo_status: dict[tuple[str, str], dict[int, str]] = {}
    combo_score: dict[tuple[str, str], dict[str, int]] = {}
    for key, qi_to_row in combo_latest.items():
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
            ans = (s.get("extracted_answer") or "").strip()
            if len(ans) > 40:
                ans = ans[:40] + "…"
            head = "⚠ error" if s["error"] else (
                f"{symbol} {ans}" if ans else symbol
            )
            dur = fmt.fmt_duration_short(s.get("research_duration_seconds"))
            status_map[qi] = f"{head} · {dur}" if dur else head
            score["total"] += 1
            if s["is_correct"] is not None:
                score["graded"] += 1
            if s["is_correct"] == 1:
                score["correct"] += 1

    all_combos = sorted(combo_status.keys(), key=_combo_sort_key)
    coverage_cols: list[str] = []
    for p, m in all_combos:
        col = f"{p}:{m}"
        coverage_cols.append(col)
        status_map = combo_status[(p, m)]
        ds_df[col] = ds_df["q_index"].map(
            lambda qi, _s=status_map: _s.get(int(qi), "")
        )

    table_cols = ["q_index"] + extras_in_df + ["question", "expected_answer"] + coverage_cols
    column_config = {
        "q_index": st.column_config.NumberColumn("#", width=60, pinned=True),
        "question": st.column_config.TextColumn("Question", width="large"),
        "expected_answer": st.column_config.TextColumn("Expected", width=140),
    }
    for extra in extras_in_df:
        column_config[extra] = st.column_config.TextColumn(
            extra, width=120, pinned=True,
        )
    for p, m in all_combos:
        col = f"{p}:{m}"
        sc = combo_score.get((p, m), {"correct": 0, "graded": 0, "total": 0})
        graded = sc.get("graded", 0)
        pct = (sc["correct"] / graded * 100) if graded else 0
        pct_label = f"{pct:.0f}%" if graded else "—"
        label = f"{col} ({sc['correct']}/{graded}, {pct_label})"
        column_config[col] = st.column_config.TextColumn(label, width=220)

    table_height = min(720, max(300, 90 + 38 * len(ds_df)))
    sel_version = st.session_state.get("finsearch_table_version", 0)
    sel = st.dataframe(
        ds_df[table_cols],
        on_select="rerun",
        selection_mode="multi-row",
        column_config=column_config,
        width="stretch",
        hide_index=True,
        height=table_height,
        key=f"finsearch_table_{bench}_{seed_int}_v{sel_version}",
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

    if selected_q_indices:
        sel_c1, sel_c2, sel_c3 = st.columns([1, 3, 1])
        sel_c1.metric("Selected", f"{len(selected_q_indices)}")
        with sel_c2:
            st.caption("q_index ranges")
            st.code(fmt.ranges(sorted(selected_q_indices)), language=None)
        with sel_c3:
            if st.button(
                "Deselect all",
                key="finsearch_deselect_all",
                width="stretch",
            ):
                st.session_state["finsearch_table_version"] = sel_version + 1
                st.rerun()
    else:
        st.info("Pick rows in the table to choose which questions to run.")

    # ----- Provider / model matrix (search-tier only) -----
    st.subheader("Providers and models")
    st.caption("Search-tier endpoints only. Tick a model to include it as one run.")
    prov_cols = st.columns(len(SEARCH_PROVIDERS))
    provider_models: dict[str, list[str]] = {}
    selected_providers: list[str] = []
    for col, p in zip(prov_cols, SEARCH_PROVIDERS):
        with col:
            st.markdown(f"**{p}**")
            default_model = providers.PROVIDERS[p].default_model
            chosen: list[str] = []
            for m in providers.PROVIDERS[p].available_models:
                ck_key = f"finsearch_chk_{p}_{m}"
                initial = ui_state.persisted(
                    ck_key, p == "tavily_search" and m == default_model
                )
                if st.checkbox(
                    m, value=initial, key=ck_key,
                    on_change=ui_state.save_ui_state,
                ):
                    chosen.append(m)
            if chosen:
                provider_models[p] = chosen
                selected_providers.append(p)

    # ----- Overlap warning -----
    if selected_q_indices and any(provider_models.values()):
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
                "Some of these questions have already been run at this "
                "seed. Launching again will re-bill them:\n\n" + lines
            )

    # ----- Cost preview -----
    runs_planned = sum(len(ms) for ms in provider_models.values())
    calls_planned = runs_planned * len(selected_q_indices)
    est_cost, missing_costs = ui_costs.estimate_cost(
        provider_models, len(selected_q_indices),
    )
    if ui_costs.load_costs():
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Runs", runs_planned)
        mc2.metric("Search calls", calls_planned)
        mc3.metric("Judge calls", calls_planned)
        cost_label = f"≈ ${est_cost:.2f}" if est_cost else "—"
        help_text = (
            "Rough estimate from model_costs.json. Judge calls included; "
            "synthesis calls (EXA + Parallel) are folded into the judge bucket."
        )
        if missing_costs:
            help_text += (
                "  Missing prices for: "
                + ", ".join(f"{p}:{m}" for p, m in missing_costs)
                + ". Those rows counted as $0."
            )
        mc4.metric("Est. cost", cost_label, help=help_text)
    else:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Runs", runs_planned)
        mc2.metric("Search calls", calls_planned)
        mc3.metric("Judge calls", calls_planned)

    # ----- Pre-flight key check -----
    env = load_env()
    missing: list[str] = []
    for p in selected_providers:
        if not provider_models.get(p):
            continue
        var = providers.PROVIDERS[p].env_var
        if not env.get(var):
            missing.append(f"{p} ({var})")
    if not env.get("ANTHROPIC_API_KEY"):
        # ANTHROPIC_API_KEY drives both the judge and the synthesizer
        # used by exa_search and parallel_search. One key, two callers.
        missing.append("judge + synthesizer (ANTHROPIC_API_KEY)")

    if missing:
        st.error("Missing API keys in `.env`: " + ", ".join(missing))

    @st.dialog("Confirm launch")
    def _confirm_launch_dialog(
        bench_arg, provider_models_arg, selected_q_indices_arg,
        seed_int_arg, workers_arg, note_arg,
        runs_planned_arg, calls_planned_arg, missing_arg, already_covered_arg,
    ):
        est_cost_arg, _ = ui_costs.estimate_cost(
            provider_models_arg, len(selected_q_indices_arg),
        )
        if ui_costs.load_costs():
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Runs", runs_planned_arg)
            dc2.metric("Search calls", calls_planned_arg)
            dc3.metric("Judge calls", calls_planned_arg)
            dc4.metric(
                "Est. cost",
                f"≈ ${est_cost_arg:.2f}" if est_cost_arg else "—",
            )
        else:
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Runs", runs_planned_arg)
            dc2.metric("Search calls", calls_planned_arg)
            dc3.metric("Judge calls", calls_planned_arg)

        if missing_arg:
            st.error("❌ Missing API keys: " + ", ".join(missing_arg))
        if already_covered_arg:
            lines = "\n".join(
                f"- `{p}:{m}` already covers q_index {r}"
                for p, m, r in already_covered_arg
            )
            st.warning(
                "⚠ Some questions already covered at this seed. "
                "Launching again will re-bill them:\n\n" + lines
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
            # Stamp a fresh comparison_set whenever >1 run goes out together
            # so the head-to-head view below can join them later.
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
            st.session_state["_finsearch_last_launch"] = {
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
    if st.button(
        "Launch",
        type="primary",
        disabled=launch_disabled,
        key="finsearch_launch_btn",
    ):
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

    last = st.session_state.get("_finsearch_last_launch")
    if last:
        launched = last["launched"]
        comparison_set = last["comparison_set"]
        st.success(f"✅ Launched {len(launched)} run(s).")
        for p, m, pid, log_name in launched:
            st.caption(f"  `{p}:{m}`  pid `{pid}`  log `{Path(log_name).name}`")
        if comparison_set:
            st.caption(f"comparison_set `{comparison_set[:8]}…`")

    # ----- In-flight runs (auto-refreshing) -----
    st.subheader("In-flight runs")

    @st.fragment(run_every=3)
    def _render_in_flight():
        in_flight = [
            r for r in storage.list_in_progress_runs()
            if r["provider"] in SEARCH_PROVIDERS
        ]
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

    # ----- Head-to-head: latest comparison_set scoped to finance + search -----
    st.subheader("Head-to-head")
    _render_head_to_head()


def _render_head_to_head() -> None:
    """Inline comparison view, scoped to finance benchmarks + search providers.

    Pulls every comparison_set where all runs are search-tier providers and
    the benchmark is one of the two finance datasets. Default selection is
    the most recent set so the freshly-launched run lands in view.
    """
    all_sets = storage.list_comparison_sets()
    scoped: list[dict] = []
    for s in all_sets:
        if s.get("benchmark") not in FINANCE_BENCHMARKS:
            continue
        runs = storage.get_runs_in_set(s["comparison_set"])
        if not runs:
            continue
        if not all(r["provider"] in SEARCH_PROVIDERS for r in runs):
            continue
        scoped.append(s)

    if not scoped:
        st.info(
            "No finance search comparisons yet. Launch one above to "
            "populate this view."
        )
        return

    def _fmt_set(cs: str) -> str:
        row = next(s for s in scoped if s["comparison_set"] == cs)
        short = (cs or "")[:8]
        when = (row.get("started_at") or "")[:16].replace("T", " ")
        return (
            f"{short}…  ·  {row['benchmark']}  ·  "
            f"{row.get('providers') or '?'}  ·  {when}"
        )

    selected_set = st.selectbox(
        "Comparison set",
        [s["comparison_set"] for s in scoped],
        format_func=_fmt_set,
        key="finsearch_set_picker",
    )

    set_runs = storage.get_runs_in_set(selected_set)
    if not set_runs:
        st.warning("This comparison set has no runs.")
        return

    meta = set_runs[0]
    st.markdown(
        f"**Benchmark:** {meta['benchmark']}  ·  "
        f"**Seed:** {meta.get('seed')}  ·  "
        f"**Limit:** {meta.get('limit_n')}  ·  "
        f"**{len(set_runs)} runs**"
    )

    # ----- Per-(provider, model) summary -----
    # accuracy = correct / graded (CLAUDE.md rule 5).
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
        avg_duration=summary_df["avg_seconds_raw"].map(fmt.fmt_duration),
    )[["provider", "model", "total", "graded", "correct", "errors",
       "accuracy", "avg_duration", "run_id"]].reset_index(drop=True)
    st.dataframe(display_summary, width="stretch", hide_index=True)

    # ----- Per-question matrix -----
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
        return
    data = pd.concat(all_results, ignore_index=True)
    combos_in_set = sorted(
        {(p, m) for p, m in zip(data["provider"], data["model_used"])},
        key=_combo_sort_key,
    )
    combo_labels = [f"{p}:{m}" for p, m in combos_in_set]

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

    st.markdown("**Per-question matrix**")
    fc1, fc2 = st.columns([2, 2])
    with fc1:
        finsearch_filter = st.segmented_control(
            "Filter",
            ["all", "any disagreement", "all correct", "all wrong"],
            default="all",
            key="finsearch_filter",
        )
        if not finsearch_filter:
            finsearch_filter = "all"
    with fc2:
        has_tavily = any(p == "tavily_search" for p, _ in combos_in_set)
        tav_pivot = False
        tav_side = None
        if has_tavily:
            tav_pivot = st.toggle(
                "Tavily pivot",
                value=False,
                key="finsearch_tavily_pivot",
                help="Only questions where Tavily is uniquely right or "
                     "uniquely wrong compared to the other providers.",
            )
            if tav_pivot:
                tav_side = st.segmented_control(
                    "Tavily side",
                    ["wins (unique)", "loses (unique)"],
                    default="wins (unique)",
                    key="finsearch_tavily_side",
                )
                if not tav_side:
                    tav_side = "wins (unique)"

    all_qis = sorted(qi_to_correctness.keys())

    def _passes(qi: int) -> bool:
        c = qi_to_correctness.get(qi, {})
        vals = [v for v in c.values() if v is not None]
        if tav_pivot and has_tavily:
            tav = [
                v for combo, v in c.items()
                if combo[0] == "tavily_search" and v is not None
            ]
            oth = [
                v for combo, v in c.items()
                if combo[0] != "tavily_search" and v is not None
            ]
            if not tav or not oth:
                return False
            if tav_side == "wins (unique)":
                return all(v == 1 for v in tav) and all(v == 0 for v in oth)
            if tav_side == "loses (unique)":
                return all(v == 0 for v in tav) and all(v == 1 for v in oth)
            return True
        if not vals:
            return finsearch_filter == "all"
        if finsearch_filter == "all":
            return True
        if finsearch_filter == "any disagreement":
            return len(set(vals)) > 1
        if finsearch_filter == "all correct":
            return all(v == 1 for v in vals)
        if finsearch_filter == "all wrong":
            return all(v == 0 for v in vals)
        return True

    filtered_qis = [qi for qi in all_qis if _passes(qi)]
    st.caption(f"{len(filtered_qis)} of {len(all_qis)} questions shown.")
    if not filtered_qis:
        return

    matrix_rows = []
    for qi in filtered_qis:
        row = {
            "q_index": qi,
            "question": qi_to_question.get(qi) or "",
            "expected": qi_to_expected.get(qi) or "",
        }
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
        file_name=f"finsearch_{selected_set[:8]}_matrix.csv",
        mime="text/csv",
        key=f"finsearch_dl_mat_{selected_set}",
    )

    # ----- Drill into one question -----
    st.markdown("**Drill into one question**")

    def _fmt_q(qi: int) -> str:
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
        format_func=_fmt_q,
        key=f"finsearch_drill_{selected_set}",
    )
    st.markdown(f"**Question:** {qi_to_question.get(drill_qi, '')}")
    st.markdown(f"**Expected answer:** `{qi_to_expected.get(drill_qi, '')}`")

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
            st.caption(fmt.fmt_duration(r["research_duration_seconds"]))
            st.markdown("**Extracted**")
            st.markdown(fmt.quote(r["extracted_answer"]))
            if pd.notna(r["confidence"]):
                st.caption(f"confidence {r['confidence']:.2f}")
            if r["reasoning"]:
                st.caption(r["reasoning"])
            if r["error"]:
                st.error(r["error"])
            with st.expander("Answer"):
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
