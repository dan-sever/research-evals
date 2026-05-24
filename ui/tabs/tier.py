"""Tier analysis tab.

Groups (provider, model) pairs by model_tiers.json, then shows a roster
+ per-question matrix + drill view restricted to that tier's members.

Mirrors the Compare tab's tuple-keyed matrix convention. The Tavily
pivot here works at the tier level: every Tavily entry in the tier vs
every non-Tavily entry.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from benchmarks import datasets
from ui import cache as ui_cache
from ui import data as ui_data
from ui import format as fmt
from ui import state as ui_state
from ui import tiers as ui_tiers


def render() -> None:
    tiers_def = ui_tiers.load_tiers()
    if not tiers_def:
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
        return

    # ----- Picker row: benchmark, tier (seed only when ambiguous) -----
    pc1, pc2 = st.columns(2)
    with pc1:
        benchmarks = datasets.list_benchmarks()
        saved_bench = ui_state.persisted("tier_bench", benchmarks[0])
        if saved_bench not in benchmarks:
            saved_bench = benchmarks[0]
        tier_bench = st.segmented_control(
            "Benchmark",
            benchmarks,
            default=saved_bench,
            key="tier_bench",
            on_change=ui_state.save_ui_state,
        )
        if not tier_bench:
            tier_bench = benchmarks[0]
    with pc2:
        tier_keys = ["all"] + list(tiers_def.keys())
        saved_tier = ui_state.persisted("tier_name", tier_keys[0])
        if saved_tier not in tier_keys:
            saved_tier = tier_keys[0]
        tier_name = st.segmented_control(
            "Tier",
            tier_keys,
            default=saved_tier,
            key="tier_name",
            on_change=ui_state.save_ui_state,
        )
        if not tier_name:
            tier_name = tier_keys[0]

    if tier_name == "all":
        # Union of every tier's members, preserving first-seen order.
        seen: set[tuple[str, str]] = set()
        tier_member_list: list[tuple[str, str]] = []
        for _tname, members in tiers_def.items():
            for pair in members:
                key = tuple(pair)
                if key not in seen:
                    seen.add(key)
                    tier_member_list.append(key)
    else:
        tier_member_list = ui_tiers.tier_members(tier_name, tiers_def)
    member_set = set(tier_member_list)

    # Distinct seeds with data for this tier on this benchmark.
    status_rows = ui_cache.question_status(tier_bench)
    seeds = sorted(
        {
            s["seed"] for s in status_rows
            if (s["provider"], s["model"]) in member_set
        },
        key=lambda x: (x is None, x),
    )

    # Only surface a seed picker when there's actual ambiguity. When all
    # data is at a single seed (the common case for UI-launched runs),
    # we silently pick it and keep the picker out of sight.
    if not seeds:
        tier_seed = None
    elif len(seeds) == 1:
        tier_seed = seeds[0]
    else:
        saved_seed = ui_state.persisted("tier_seed", seeds[0])
        if saved_seed not in seeds:
            saved_seed = seeds[0]
        tier_seed = st.selectbox(
            "Seed (multiple detected — pick which slice to compare)",
            seeds,
            index=seeds.index(saved_seed),
            format_func=lambda s: "(no seed)" if s is None else str(s),
            key="tier_seed",
            on_change=ui_state.save_ui_state,
            help="Different seeds shuffle the dataset differently, so "
                 "q_index 5 means a different question across seeds. "
                 "Pick one slice to keep comparisons honest.",
        )

    if not seeds:
        st.info(
            f"No runs yet for any member of the `{tier_name}` tier on "
            f"`{tier_bench}`. Once a tier member has run this benchmark, "
            "results will appear here."
        )
        return

    data, run_ids = ui_tiers.tier_run_data(tier_bench, tier_member_list, tier_seed)

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
        return

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
        filter_opts = ["all", "any disagreement", "all correct", "all wrong"]
        saved_filter = ui_state.persisted("tier_filter", "all")
        if saved_filter not in filter_opts:
            saved_filter = "all"
        tier_filter_val = st.segmented_control(
            "Filter",
            filter_opts,
            default=saved_filter,
            key="tier_filter",
            on_change=ui_state.save_ui_state,
        )
        if not tier_filter_val:
            tier_filter_val = "all"
        tier_search = st.text_input(
            "Search question text",
            "",
            key="tier_search",
        )
    with fc2:
        has_tavily = any(p == "tavily" for (p, _m) in providers_in_tier)
        if has_tavily:
            tier_pivot = st.toggle(
                "Tavily pivot",
                value=ui_state.persisted("tier_pivot", False),
                key="tier_pivot",
                on_change=ui_state.save_ui_state,
                help="Show only questions where tavily members are "
                     "uniquely right or wrong vs every other tier "
                     "member. Overrides the filter on the left.",
            )
            if tier_pivot:
                side_opts = ["wins (unique)", "loses (unique)"]
                saved_side = ui_state.persisted("tier_pivot_side", "wins (unique)")
                if saved_side not in side_opts:
                    saved_side = "wins (unique)"
                tier_pivot_side = st.segmented_control(
                    "Tavily side",
                    side_opts,
                    default=saved_side,
                    key="tier_pivot_side",
                    on_change=ui_state.save_ui_state,
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
        if tier_pivot and has_tavily:
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
        return

    # ----- Matrix -----
    pid_map = ui_data.prompt_ids(tier_bench, tier_seed)
    matrix_rows = []
    for qi in filtered_qis:
        row = {
            "q_index": qi,
            "question": qi_to_question.get(qi) or "",
            "expected": qi_to_expected.get(qi) or "",
        }
        if pid_map:
            row["prompt_id"] = pid_map.get(int(qi), "")
        for (p, m) in providers_in_tier:
            c = qi_to_correctness.get(qi, {}).get((p, m))
            ext = qi_to_extracted.get(qi, {}).get((p, m)) or ""
            badge = "✅" if c == 1 else ("❌" if c == 0 else "—")
            row[f"{p}:{m}"] = f"{badge} {ext}".strip()
        matrix_rows.append(row)
    matrix_df = pd.DataFrame(matrix_rows)

    mcc = {
        "q_index": st.column_config.NumberColumn("#", width=60, pinned=True),
        "question": st.column_config.TextColumn(
            "Question", width="large", pinned=True
        ),
        "expected": st.column_config.TextColumn("Expected", width="medium"),
    }
    if pid_map:
        mcc["prompt_id"] = st.column_config.TextColumn(
            "Prompt ID", width=220, pinned=True,
        )
    for (p, m) in providers_in_tier:
        col_name = f"{p}:{m}"
        mcc[col_name] = st.column_config.TextColumn(col_name, width="medium")

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
        return
    drill_qi = int(matrix_df.iloc[drill_positions[0]]["q_index"])
    st.subheader("Drill into one question")
    if pid_map and pid_map.get(drill_qi):
        st.caption(f"prompt_id: `{pid_map[drill_qi]}`")
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
            st.caption(fmt.fmt_duration(r["research_duration_seconds"]))
            st.markdown("**Extracted**")
            st.markdown(fmt.quote(r["extracted_answer"]))
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
