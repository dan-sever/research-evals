"""Export tab — analysis-friendly CSV downloads for Excel / Omni.

Long format: one row per (run, q_index). Omits `research_content` and
`research_sources_json` (use the Single run inspector for those).

Joins in:
- SealQA-Seal0: reasoning/retrieval labels from docs/tags/sealqa_seal0.csv
  and topic/freshness/question_types from the parquet itself. Taxonomy tags
  are anchored to seed=None and only joined when the selected seed is blank.
- FinSearchComp: prompt_id, tier (T1/T2/T3) and region, derived at the
  selected seed from the parquet's `label` column.
- Other benchmarks: just the base columns.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from benchmarks import datasets, storage


# ---------------------------------------------------------------------------
# Cached dataset-side joins (mirrored from the Dashboard tab so neither
# imports the other).
# ---------------------------------------------------------------------------

_SEALQA_SEAL0_TAGS_PATH = Path("docs/tags/sealqa_seal0.csv")


@st.cache_data
def _sealqa_seal0_tags() -> pd.DataFrame:
    if not _SEALQA_SEAL0_TAGS_PATH.exists():
        return pd.DataFrame(columns=["q_index", "reasoning", "retrieval"])
    df = pd.read_csv(_SEALQA_SEAL0_TAGS_PATH)
    keep = ["q_index", "reasoning", "retrieval"]
    return df[[c for c in keep if c in df.columns]].copy()


def _split_finsearchcomp_label(label: str) -> tuple[str, str]:
    if not isinstance(label, str):
        return ("?", "?")
    if "(" in label and label.endswith(")"):
        head, _, tail = label.rpartition("(")
        region = tail[:-1]
    else:
        head, region = label, "?"
    tier_map = {
        "Time-Sensitive_Data_Fetching": "T1",
        "Simple_Historical_Lookup": "T2",
        "Complex_Historical_Investigation": "T3",
    }
    return (tier_map.get(head, head), region)


@st.cache_data
def _parquet_extras(benchmark: str, seed) -> pd.DataFrame:
    """Pull benchmark-specific extra columns from the parquet at the given
    seed. Returns a frame indexed by q_index plus whatever extras are
    relevant for this benchmark."""
    spec = datasets.REGISTRY[benchmark]
    df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.reset_index(drop=False).rename(columns={"index": "q_index"})
    keep = ["q_index"]
    if benchmark.startswith("sealqa"):
        for c in ("topic", "freshness", "question_types"):
            if c in df.columns:
                keep.append(c)
    if benchmark == "finsearchcomp":
        if "label" in df.columns:
            tiers_regions = df["label"].astype(str).apply(_split_finsearchcomp_label)
            df["fin_tier"] = [tr[0] for tr in tiers_regions]
            df["fin_region"] = [tr[1] for tr in tiers_regions]
            keep += ["fin_tier", "fin_region"]
        if "ground_truth" in df.columns:
            df["ground_truth"] = df["ground_truth"].astype(str)
            keep.append("ground_truth")
    out = df[keep].copy()
    if "question_types" in out.columns:
        # Multi-label arrays become semicolon-joined strings so Excel
        # doesn't render them as `[array(['x'], dtype=object)]`.
        out["question_types"] = out["question_types"].apply(
            lambda v: ";".join(map(str, v)) if isinstance(v, (list, tuple)) or hasattr(v, "tolist")
            else ("" if v is None else str(v))
        )
    return out


# ---------------------------------------------------------------------------
# Core: assemble the analysis dataframe for the selected scope
# ---------------------------------------------------------------------------

def _output_columns(benchmark: str, seed) -> list[str]:
    """Benchmark-specific column order for the export.

    Slicer dimensions (the things you'd pivot on in Omni) come right after
    `benchmark` so the leftmost columns of the CSV are exactly the
    categorical axes for analysis. The system label and the question
    content follow. Latency and a couple of low-signal metadata fields
    bring up the rear.
    """
    dims: list[str] = []
    if benchmark.startswith("sealqa"):
        if benchmark == "sealqa_seal0" and seed is None:
            dims += ["reasoning", "retrieval"]
        dims += ["topic", "freshness", "question_types"]
    if benchmark == "finsearchcomp":
        dims += ["fin_tier", "fin_region"]

    cols = (
        ["benchmark"]
        + dims
        + [
            "system",
            "q_index", "question", "expected_answer",
            "extracted_answer", "is_correct",
            "research_duration_seconds",
        ]
    )
    if benchmark == "finsearchcomp":
        cols += ["ground_truth"]
    cols += ["research_status", "started_at"]
    return cols


def _runs_for_benchmark(benchmark: str, seed) -> list[dict]:
    """All runs matching (benchmark, seed). `seed=None` matches runs that
    used no seed."""
    out = []
    for r in storage.list_runs():
        if r["benchmark"] != benchmark:
            continue
        if r["seed"] != seed:
            continue
        out.append(r)
    return out


def _build_analysis_df(
    benchmark: str,
    seed,
    selected_run_ids: list[int],
    latest_only: bool,
) -> pd.DataFrame:
    """Assemble the long-format analysis frame.

    `latest_only=True` dedupes by (provider, model, q_index) keeping the
    row from the most recent run_id, mirroring the Tier-tab convention.
    Off keeps every (run, q_index) row so retries / multiple runs against
    the same questions are visible.
    """
    if not selected_run_ids:
        return pd.DataFrame()

    runs_by_id = {r["id"]: r for r in storage.list_runs()}
    frames: list[pd.DataFrame] = []
    for rid in selected_run_ids:
        run = runs_by_id.get(rid)
        if not run:
            continue
        rows = storage.get_results(rid)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["run_id"] = rid
        df["provider"] = run["provider"]
        df["model"] = run["model"]
        df["benchmark"] = run["benchmark"]
        df["seed"] = run["seed"]
        df["comparison_set"] = run.get("comparison_set")
        df["started_at"] = run.get("started_at")
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    if latest_only:
        # `list_runs()` returns runs DESC by id, but get_results() preserves
        # insertion order within a run. Sort by run_id ASC so keep="last"
        # picks the most recent attempt per (provider, model, q_index).
        df = df.sort_values("run_id", kind="stable").drop_duplicates(
            subset=["provider", "model", "q_index"], keep="last"
        ).reset_index(drop=True)

    # Drop the judge's narrative (`results.reasoning`) so it doesn't collide
    # with the SealQA tag CSV's `reasoning` column on the upcoming merge,
    # and because we don't expose it in the analysis CSV anyway.
    if "reasoning" in df.columns:
        df = df.drop(columns=["reasoning"])

    # Combine provider + model into a single `system` label. Hyphens become
    # spaces so values like `exa deep reasoning` and `perplexity sonar
    # reasoning pro` read cleanly as one categorical dimension in Excel/Omni.
    df["system"] = (
        df["provider"].astype(str) + " "
        + df["model"].astype(str).str.replace("-", " ", regex=False)
    )

    # Collapse is_correct to a true boolean (NaN for errored / ungraded).
    df["is_correct"] = df["is_correct"].map({1: True, 0: False})
    # Round latency to 2 decimals to keep Excel/Omni readable.
    if "research_duration_seconds" in df.columns:
        df["research_duration_seconds"] = pd.to_numeric(
            df["research_duration_seconds"], errors="coerce"
        ).round(2)

    # Benchmark-specific joins
    extras = _parquet_extras(benchmark, seed)
    if not extras.empty:
        df = df.merge(extras, on="q_index", how="left")

    if benchmark == "sealqa_seal0" and seed is None:
        tags = _sealqa_seal0_tags()
        if not tags.empty:
            df = df.merge(tags, on="q_index", how="left")

    # Drop provider-side error rows entirely (and the now-redundant column).
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
        df = df.drop(columns=["error"])

    # Reorder per benchmark, dropping any columns that don't apply.
    present = [c for c in _output_columns(benchmark, seed) if c in df.columns]
    return df[present].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render() -> None:
    st.subheader("Export run data for Excel / Omni")
    st.caption(
        "Long-format CSV. One row per (run, question). Error rows are "
        "excluded. SealQA exports include the reasoning and retrieval "
        "taxonomy labels when the selected seed is blank (the tags CSV is "
        "anchored to the parquet's natural order). FinSearchComp exports "
        "include the tier label (T1/T2/T3) and region."
    )

    bench_list = datasets.list_benchmarks()
    bench = st.segmented_control(
        "Benchmark", bench_list, default=bench_list[0], key="export_bench"
    )
    if not bench:
        bench = bench_list[0]

    # Seeds with data for this benchmark
    all_runs = [r for r in storage.list_runs() if r["benchmark"] == bench]
    if not all_runs:
        st.info(f"No runs yet for `{bench}`.")
        return

    seeds = sorted({r["seed"] for r in all_runs}, key=lambda s: (s is None, s))
    if len(seeds) == 1:
        seed = seeds[0]
        st.caption(
            f"Seed: `{'(no seed)' if seed is None else seed}` (only one in data)"
        )
    else:
        seed = st.selectbox(
            "Seed",
            seeds,
            format_func=lambda s: "(no seed)" if s is None else str(s),
            key="export_seed",
            help="Different seeds shuffle the dataset differently, so q_index "
                 "means different questions across seeds. Pick the seed whose "
                 "runs you want to analyze.",
        )

    matching_runs = _runs_for_benchmark(bench, seed)
    if not matching_runs:
        st.info("No runs at this seed.")
        return

    # Provider / model filter
    combos = sorted({(r["provider"], r["model"]) for r in matching_runs})
    combo_labels = [f"{p}:{m}" for p, m in combos]
    chosen_labels = st.multiselect(
        "Providers / models (blank = all)",
        combo_labels,
        default=[],
        key="export_combos",
        help="Restrict the export to specific provider:model pairs.",
    )
    if chosen_labels:
        chosen_set = set(chosen_labels)
        runs_in_scope = [
            r for r in matching_runs
            if f"{r['provider']}:{r['model']}" in chosen_set
        ]
    else:
        runs_in_scope = matching_runs

    # Comparison set filter (optional, only when any of the in-scope runs
    # has one, otherwise hide the control)
    cmp_sets = sorted({
        r["comparison_set"] for r in runs_in_scope if r.get("comparison_set")
    })
    if cmp_sets:
        cmp_choice = st.selectbox(
            "Comparison set (optional)",
            ["(any)"] + cmp_sets,
            key="export_cmp_set",
            format_func=lambda c: c if c == "(any)" else f"{c[:8]}…",
            help="Restrict to runs that were launched as one comparison batch.",
        )
        if cmp_choice != "(any)":
            runs_in_scope = [
                r for r in runs_in_scope if r.get("comparison_set") == cmp_choice
            ]

    latest_only = st.toggle(
        "Latest run only per (provider, model, question)",
        value=True,
        key="export_latest_only",
        help="On = one row per (provider, model, q_index) using the most "
             "recent run. Off = keep every (run, q_index) row, including "
             "retries.",
    )

    st.caption(
        f"{len(runs_in_scope)} run(s) match this scope: "
        + ", ".join(sorted({f"{r['provider']}:{r['model']}" for r in runs_in_scope}))
        if runs_in_scope else "No runs match this scope."
    )

    if not runs_in_scope:
        return

    df = _build_analysis_df(
        bench, seed, [r["id"] for r in runs_in_scope], latest_only
    )

    if df.empty:
        st.warning("No result rows for the selected runs.")
        return

    # SealQA-specific anchoring warning
    if bench == "sealqa_seal0" and seed is not None:
        st.info(
            "Selected seed is not blank, so the `docs/tags/sealqa_seal0.csv` "
            "taxonomy columns (reasoning / retrieval) are not joined "
            "into this export. Switch to seed `(no seed)` to include them."
        )

    # Headline counts
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Rows", len(df))
    mc2.metric("Questions", df["q_index"].nunique() if "q_index" in df.columns else 0)
    mc3.metric(
        "Systems",
        df["system"].nunique() if "system" in df.columns else 0,
    )
    mc4.metric("Columns", df.shape[1])

    seed_tag = "noseed" if seed is None else f"seed{seed}"
    file_name = f"{bench}_{seed_tag}_analysis.csv"

    st.download_button(
        "Download analysis CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        type="primary",
        key=f"export_dl_{bench}_{seed_tag}",
    )

    with st.expander("Preview (first 20 rows)", expanded=False):
        st.dataframe(df.head(20), width="stretch", hide_index=True)

    with st.expander("Column reference", expanded=False):
        st.markdown(_column_reference(bench, seed))


def _column_reference(benchmark: str, seed) -> str:
    """Human-readable description of every column in the export, in the
    same order as the CSV."""
    meanings = {
        "benchmark": "Benchmark name.",
        "reasoning": "SealQA Scheme A (reasoning hops): "
                     "single-hop / multi-hop / comparative / unanswerable.",
        "retrieval": "SealQA Scheme B (retrieval difficulty): "
                     "common / specialized / fresh / tricky-phrasing.",
        "topic": "Dataset-native topic column.",
        "freshness": "Dataset-native freshness column.",
        "question_types": "Dataset-native multi-label tag, joined with `;`.",
        "fin_tier": "FinSearchComp: T1 (time-sensitive), T2 (simple historical), "
                    "T3 (complex investigation).",
        "fin_region": "FinSearchComp: region in parentheses on the dataset's "
                      "`label` column (e.g. `Greater China`, `Global`).",
        "system": "Combined provider + model label, e.g. `tavily mini`, "
                  "`exa deep reasoning`, `perplexity sonar reasoning pro`. "
                  "Hyphens in the original model name are normalized to spaces.",
        "q_index": "Row index after shuffling at the run's seed. Same q_index "
                   "across runs at the same seed means the same question.",
        "question": "Question text.",
        "expected_answer": "Canonical answer string from the dataset.",
        "extracted_answer": "The model's final answer as extracted by the judge tool.",
        "is_correct": "Boolean. True = judge marked correct, False = incorrect, "
                      "blank = ungraded. Errored rows are excluded from the export.",
        "research_duration_seconds": "How long the provider took to return the "
                                     "report, rounded to 2 decimals.",
        "ground_truth": "FinSearchComp: mostly null. Kept for completeness; "
                        "`expected_answer` is the canonical grading target.",
        "research_status": "Provider-side status. After error filtering this is "
                           "almost always `completed`.",
        "started_at": "When the run started (ISO timestamp).",
    }
    out = ["| column | meaning |", "| --- | --- |"]
    for name in _output_columns(benchmark, seed):
        if name in meanings:
            out.append(f"| `{name}` | {meanings[name]} |")
    return "\n".join(out)
