"""Streamlit dashboard for browsing eval runs and comparing providers.

    streamlit run app.py
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from benchmarks import storage

st.set_page_config(page_title="Research Benchmarks", layout="wide")
st.title("Research benchmark runs")

storage.init_db()


def _fmt_duration(seconds) -> str:
    """`45.3s` under a minute, `128.0s (2.1 min)` above."""
    if seconds is None or pd.isna(seconds) or seconds == 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds:.1f}s ({seconds / 60:.1f} min)"


tab_inspect, tab_compare = st.tabs(["Single run inspector", "Provider comparison"])


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
    st.dataframe(overview, width="stretch", hide_index=True)

    def _fmt_run(rid: int) -> str:
        r = next(x for x in runs if x["id"] == rid)
        acc = (r["correct"] or 0) / r["total"] if r["total"] else 0
        when = (r["started_at"] or "")[:16].replace("T", " ")
        provider = r.get("provider") or "?"
        return f"#{rid} · {provider} · {r['benchmark']} · {r['model']} · {acc:.0%} · {when}"

    selected_run = st.selectbox(
        "Inspect run",
        [r["id"] for r in runs],
        format_func=_fmt_run,
    )

    run = storage.get_run(selected_run)
    results = storage.get_results(selected_run)
    if not results:
        st.warning("This run has no results recorded.")
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

        st.subheader("Per-question results")
        fcol, scol = st.columns([1, 2])
        with fcol:
            filter_choice = st.radio(
                "Filter",
                ["all", "correct", "incorrect", "errors"],
                horizontal=True,
                key="inspect_filter",
            )
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
            st.dataframe(view[display_cols], width="stretch", hide_index=True)

            st.subheader("Drill into one question")

            q_text_by_index = dict(zip(view["q_index"], view["question"]))
            correct_by_index = dict(zip(view["q_index"], view["is_correct"]))

            def _fmt_question(qi: int) -> str:
                text = q_text_by_index[qi]
                truncated = text if len(text) <= 90 else text[:90] + "…"
                c = correct_by_index.get(qi)
                badge = "✅" if c == 1 else ("❌" if c == 0 else "⚠")
                return f"{badge}  Q{qi}  ·  {truncated}"

            selected_qi = st.selectbox(
                "Pick a question (type to search)",
                view["q_index"].tolist(),
                format_func=_fmt_question,
                key=f"drill_run_{selected_run}",
            )
            row = view[view["q_index"] == selected_qi].iloc[0]

            if row["is_correct"] == 1:
                st.success(f"✅ Correct  ·  confidence {row['confidence']:.2f}")
            elif row["is_correct"] == 0:
                st.error(f"❌ Incorrect  ·  confidence {row['confidence']:.2f}")
            else:
                st.warning("No grade recorded")
            if row["reasoning"]:
                st.caption(f"Judge: {row['reasoning']}")

            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown("**Expected answer**")
                st.code(row["expected_answer"] or "—", language=None)
            with ec2:
                st.markdown("**Extracted answer**")
                st.code(row["extracted_answer"] or "—", language=None)

            st.markdown("**Question**")
            st.write(row["question"])

            if row["error"]:
                st.error(row["error"])

            with st.expander("Research report"):
                st.markdown(row["research_content"] or "_(empty)_")

            if row["research_sources_json"]:
                sources = json.loads(row["research_sources_json"])
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
            "--providers tavily:mini,perplexity:sonar-pro`."
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
        )[["provider", "model", "total", "correct", "errors", "accuracy", "avg_duration", "run_id"]]
        st.dataframe(display_summary, width="stretch", hide_index=True)

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

        # Filters
        fc1, fc2 = st.columns([2, 2])
        filter_options = ["all"]
        if "tavily" in providers_in_set:
            filter_options += ["tavily wins (unique)", "tavily loses (unique)"]
        filter_options += ["any disagreement", "all correct", "all wrong"]
        with fc1:
            cmp_filter = st.radio(
                "Filter",
                filter_options,
                horizontal=True,
                key="cmp_filter",
            )
        with fc2:
            cmp_search = st.text_input(
                "Search question text",
                "",
                key="cmp_search",
            )

        def _passes(qi: int) -> bool:
            c = qi_to_correctness.get(qi, {})
            vals = [v for v in c.values() if v is not None]
            if not vals:
                return cmp_filter == "all"
            if cmp_filter == "all":
                return True
            if cmp_filter == "tavily wins (unique)":
                tav = c.get("tavily")
                others = [v for k, v in c.items() if k != "tavily" and v is not None]
                return tav == 1 and others and all(v == 0 for v in others)
            if cmp_filter == "tavily loses (unique)":
                tav = c.get("tavily")
                others = [v for k, v in c.items() if k != "tavily" and v is not None]
                return tav == 0 and others and all(v == 1 for v in others)
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
                q = qi_to_question.get(qi) or ""
                row = {
                    "q_index": qi,
                    "question": (q[:80] + "…") if len(q) > 80 else q,
                    "expected": (qi_to_expected.get(qi) or "")[:60],
                }
                for p in providers_in_set:
                    c = qi_to_correctness.get(qi, {}).get(p)
                    ext = qi_to_extracted.get(qi, {}).get(p) or ""
                    badge = "✅" if c == 1 else ("❌" if c == 0 else "—")
                    cell = f"{badge} {ext}"
                    row[p] = cell[:60] + ("…" if len(cell) > 60 else "")
                matrix_rows.append(row)
            st.dataframe(
                pd.DataFrame(matrix_rows),
                width="stretch",
                hide_index=True,
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
                    st.code(r["extracted_answer"] or "—", language=None)
                    if pd.notna(r["confidence"]):
                        st.caption(f"confidence {r['confidence']:.2f}")
                    if r["reasoning"]:
                        st.caption(r["reasoning"])
                    if r["error"]:
                        st.error(r["error"])
                    with st.expander("Report"):
                        st.markdown(r["research_content"] or "_(empty)_")
                    if r["research_sources_json"]:
                        srcs = json.loads(r["research_sources_json"])
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
