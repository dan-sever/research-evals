"""Provider comparison tab.

One comparison_set at a time. At the top: per-(provider, model) accuracy
+ average duration table, sorted by accuracy. Below: a per-question
matrix with one column per provider:model and ✅/❌ badges. Drill view
shows every entry's full report and sources side by side.

Tavily pivot tightens the matrix to questions where every Tavily entry
in the set is right (or wrong) and every non-Tavily entry disagrees.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from benchmarks import storage
from ui import data as ui_data
from ui import format as fmt


def render() -> None:
    sets = storage.list_comparison_sets()
    if not sets:
        st.info(
            "No comparison sets yet. Launch one with "
            "`python compare.py --benchmark sealqa_seal0 --limit 10 "
            "--providers tavily:mini,perplexity:sonar-reasoning-pro`."
        )
        return

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
        return

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
        avg_duration=summary_df["avg_seconds_raw"].map(fmt.fmt_duration),
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
        return

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
        return

    cmp_pid_map = ui_data.prompt_ids(meta["benchmark"], meta.get("seed"))
    matrix_rows = []
    for qi in filtered_qis:
        row = {
            "q_index": qi,
            "question": qi_to_question.get(qi) or "",
            "expected": qi_to_expected.get(qi) or "",
        }
        if cmp_pid_map:
            row["prompt_id"] = cmp_pid_map.get(int(qi), "")
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
    if cmp_pid_map:
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

    if cmp_pid_map and cmp_pid_map.get(int(drill_qi)):
        st.caption(f"prompt_id: `{cmp_pid_map[int(drill_qi)]}`")
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
            st.caption(fmt.fmt_duration(r["research_duration_seconds"]))
            st.markdown("**Extracted**")
            st.markdown(fmt.quote(r["extracted_answer"]))
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
