"""Insights tab — LLM-generated analysis of where Tavily wins and loses.

Runs Haiku over the latest-per-question matrix for a (benchmark, seed),
returns a structured headline + ranked bullets (claim/evidence/examples/
action). Shows the latest cached insight by default; "Regenerate" calls
Haiku again. History is kept in `results.db`.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import streamlit as st

from benchmarks import insights as insights_mod
from benchmarks import storage
from ui import cache as ui_cache


KIND_BADGE = {
    "gap": ("🔴", "gap"),
    "win": ("🟢", "win"),
    "infra": ("🟡", "infra"),
}


def _format_local(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso


def _render_content(content: dict) -> None:
    headline = content.get("headline") or "(no headline)"
    st.markdown(f"#### {headline}")
    for i, ins in enumerate(content.get("insights", []), 1):
        emoji, _label = KIND_BADGE.get(ins.get("kind", "gap"), ("•", ""))
        claim = ins.get("claim", "").strip()
        st.markdown(f"##### {emoji} {i}. {claim}")
        evidence = ins.get("evidence", "").strip()
        if evidence:
            st.markdown(f"<small><b>Evidence.</b> {evidence}</small>",
                        unsafe_allow_html=True)
        examples = ins.get("examples") or []
        if examples:
            bullets = "<br>".join(f"&nbsp;&nbsp;• {ex}" for ex in examples)
            st.markdown(f"<small><b>Examples.</b><br>{bullets}</small>",
                        unsafe_allow_html=True)
        action = ins.get("action", "").strip()
        if action:
            st.markdown(f"<small><b>Action.</b> {action}</small>",
                        unsafe_allow_html=True)


def _render_history_diff(rows: list[Optional[dict]]) -> None:
    """Render one or two stored insights side by side. Bad rows are
    skipped with a visible error so a corrupt entry doesn't hide the rest."""
    rows = [r for r in rows if r]
    if not rows:
        return
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        with col:
            st.markdown(
                f"**{_format_local(row['generated_at'])}**  \n"
                f"`{row['model']}`  ·  prompt `{row['prompt_version']}`"
            )
            try:
                content = json.loads(row["content_json"])
            except Exception:
                st.error("Stored insight is not valid JSON.")
                continue
            _render_content(content)


def _seed_picker(benchmark: str) -> Optional[int]:
    status_rows = ui_cache.question_status(benchmark)
    seeds = sorted({s["seed"] for s in status_rows},
                   key=lambda x: (x is None, x))
    if not seeds:
        return None
    if len(seeds) == 1:
        seed = seeds[0]
        if seed is not None:
            st.caption(f"seed: {seed}")
        return seed
    saved = st.session_state.get("insights_seed_select", seeds[0])
    if saved not in seeds:
        saved = seeds[0]
    return st.selectbox(
        "Seed",
        seeds,
        index=seeds.index(saved),
        format_func=lambda s: "(no seed)" if s is None else str(s),
        key="insights_seed_select",
    )


def render() -> None:
    st.subheader("Insights")
    st.caption(
        "Tavily-centric pattern analysis. Haiku reads the dimensional "
        "accuracy tables + ~30 wrong-answer examples and surfaces "
        "dimension combinations where Tavily systematically wins or loses."
    )

    benchmarks = [
        "finsearchcomp", "sealqa_seal0", "sealqa_seal_hard", "sealqa_longseal",
    ]
    bench = st.segmented_control(
        "Benchmark", benchmarks, default=benchmarks[0], key="insights_bench",
    )
    if not bench:
        bench = benchmarks[0]
    seed = _seed_picker(bench)

    latest = storage.get_latest_insight(bench, seed)
    history = storage.list_insights(bench, seed)

    cols = st.columns([3, 1])
    with cols[0]:
        if latest:
            st.caption(
                f"Last generated {_format_local(latest['generated_at'])}  ·  "
                f"model: `{latest['model']}`  ·  "
                f"prompt: `{latest['prompt_version']}`"
            )
        else:
            st.caption("No insights generated yet for this benchmark + seed.")
    with cols[1]:
        regenerate = st.button(
            "Regenerate", type="primary", key="insights_regen", width="stretch",
        )

    if regenerate:
        with st.spinner("Calling Haiku..."):
            try:
                insights_mod.generate(bench, seed)
                st.success("Done.")
                latest = storage.get_latest_insight(bench, seed)
                history = storage.list_insights(bench, seed)
            except insights_mod.InsightsError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                return

    if not latest:
        st.info(
            "Click **Regenerate** to run Haiku on the latest data. "
            "Requires `ANTHROPIC_API_KEY`."
        )
        return

    try:
        content = json.loads(latest["content_json"])
    except Exception:
        st.error("Stored insight is not valid JSON.")
        return

    _render_content(content)

    if len(history) > 1:
        with st.expander(f"History + diff ({len(history)} generations)"):
            st.caption(
                "Pick one or two prior generations. Pick two to render them "
                "side by side and compare headlines + insight ordering."
            )

            def _label(h: dict) -> str:
                return (
                    f"{_format_local(h['generated_at'])}  ·  "
                    f"{h['model']}  ·  {h['prompt_version']}"
                )

            label_to_id = {_label(h): h["id"] for h in history}
            picked = st.multiselect(
                "Generations to inspect",
                list(label_to_id.keys()),
                max_selections=2,
                key=f"insights_hist_picker_{bench}_{seed}",
            )

            if picked:
                _render_history_diff(
                    [storage.get_insight(label_to_id[lbl]) for lbl in picked]
                )

    meta = content.get("_meta") or {}
    if meta:
        with st.expander("Payload + token usage"):
            st.json(meta)
