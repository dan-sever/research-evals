"""Single run inspector tab.

Picks one run from the list, shows headline metrics + per-question
results table + drill view (full report + sources). Accepts a cross-tab
handoff from the Provider comparison tab via `st.session_state["inspect_run_id"]`.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from benchmarks import storage
from ui import data as ui_data
from ui import format as fmt


def render() -> None:
    runs = storage.list_runs()

    if not runs:
        st.info(
            "No runs yet. Launch one with "
            "`python run.py --provider tavily --benchmark sealqa_seal0 --limit 5`."
        )
        st.stop()

    runs_df = pd.DataFrame(runs)
    # Accuracy uses `graded` (rows the judge could decide on) as the
    # denominator, so errors / ungraded responses do not drag the rate down.
    runs_df["accuracy"] = runs_df.apply(
        lambda r: (
            (r["correct"] or 0) / r["graded"]
            if r.get("graded") else None
        ),
        axis=1,
    )

    st.subheader("All runs")
    st.caption(
        "Click a row to inspect that run. "
        "Accuracy = correct / graded (errored and ungraded rows are excluded)."
    )
    overview = runs_df.assign(
        accuracy_pct=runs_df["accuracy"].map(
            lambda x: f"{x:.1%}" if x is not None else "—"
        ),
        started=runs_df["started_at"].str.slice(0, 16).str.replace("T", " "),
    )[
        [
            "id", "provider", "benchmark", "model", "limit_n", "workers",
            "judge_model", "total", "graded", "correct", "errors",
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
        return

    df = pd.DataFrame(results)
    df["is_correct_bool"] = df["is_correct"].map({1: True, 0: False})
    pid_map = ui_data.prompt_ids(run["benchmark"], run["seed"])
    if pid_map:
        df["prompt_id"] = df["q_index"].map(
            lambda qi: pid_map.get(int(qi), "")
        )

    total = len(df)
    correct = int((df["is_correct"] == 1).sum())
    graded = int(df["is_correct"].notna().sum())
    errors = int(df["error"].notna().sum())
    accuracy = correct / graded if graded else None

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Questions", total, help="Every row recorded for this run.")
    col2.metric(
        "Graded", graded,
        help="Rows the judge could decide on (is_correct in {0, 1}). "
             "This is the accuracy denominator.",
    )
    col3.metric("Correct", correct)
    col4.metric("Errors", errors, help="Provider or judge failures.")
    col5.metric(
        "Accuracy",
        f"{accuracy:.1%}" if accuracy is not None else "—",
        help="correct / graded (errored and ungraded rows excluded)",
    )

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
        return

    view = view.assign(
        duration=view["research_duration_seconds"].map(fmt.fmt_duration)
    )
    display_cols = [
        "q_index", "prompt_id", "question", "expected_answer",
        "extracted_answer", "is_correct_bool", "confidence",
        "duration", "research_status", "error",
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
        return
    row = view.iloc[drill_positions[0]]

    st.subheader("Drill into one question")

    if "prompt_id" in view.columns and row.get("prompt_id"):
        st.caption(f"prompt_id: `{row['prompt_id']}`")

    if row["is_correct"] == 1:
        st.success(f"✅ Correct  ·  confidence {row['confidence']:.2f}")
    elif row["is_correct"] == 0:
        st.error(f"❌ Incorrect  ·  confidence {row['confidence']:.2f}")
    else:
        st.warning("⚠ No grade recorded")
    if row["reasoning"]:
        st.caption(f"Judge: {row['reasoning']}")

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**Expected answer**")
        st.markdown(fmt.quote(row["expected_answer"]))
    with ec2:
        st.markdown("**Extracted answer**")
        st.markdown(fmt.quote(row["extracted_answer"]))

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
        f"research {fmt.fmt_duration(row['research_duration_seconds'])}  ·  "
        f"status {row['research_status']}"
    )
