"""Dashboard tab — per-benchmark analytics.

Read-only. Tavily-centric. Latest run per (provider, model, q_index) wins,
matching the Tier Analysis convention.

Scope v2:
- finsearchcomp: headline ranked bar; tier (T1/T2/T3) and region grouped
  bars; tier × region cross-tab; tavily mini-vs-pro; latency views;
  error-vs-wrong; easy questions Tavily missed; coverage heatmap by tier.
- sealqa_seal0: same headline + latency + mini-vs-pro + error-vs-wrong +
  easy-missed scaffolding; taxonomy slices from docs/tags/sealqa_seal0.csv
  (only when seed is blank, since the CSV is anchored to natural order);
  native-column slices (topic, question_types).

Adding more benchmarks: write a panel function and dispatch in render().
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from benchmarks import datasets, storage
from ui import dashboard_charts as dc

# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------

@st.cache_data
def _benchmark_results_matrix(benchmark: str, seed) -> pd.DataFrame:
    """One row per (provider, model, q_index) for `benchmark` at `seed`,
    keeping the most recent run per question. Columns include provider,
    model, q_index, question, expected_answer, is_correct, extracted_answer,
    research_duration_seconds, error, run_id."""
    rows = storage.get_question_status(benchmark)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["seed"].apply(lambda s: s == seed)].copy()
    if df.empty:
        return df
    # get_question_status() sorts by run_id ASC, so keep="last" picks the
    # most recent answer per (provider, model, q_index).
    df = df.drop_duplicates(
        subset=["provider", "model", "q_index"], keep="last"
    ).reset_index(drop=True)

    # Pull question/expected_answer from the dataset itself so they're
    # available even when a particular run's results lacked them.
    spec = datasets.REGISTRY[benchmark]
    dataset_df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if seed is not None:
        dataset_df = dataset_df.sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)
    dataset_df = dataset_df.reset_index(drop=False).rename(
        columns={"index": "q_index"}
    )
    dataset_df["question"] = dataset_df[spec.question_col].astype(str)
    dataset_df["expected_answer"] = dataset_df[spec.answer_col].astype(str)
    df = df.merge(
        dataset_df[["q_index", "question", "expected_answer"]],
        on="q_index", how="left", suffixes=("", "_ds"),
    )
    return df


def _split_finsearchcomp_label(label: str) -> tuple[str, str]:
    """`Time-Sensitive_Data_Fetching(Greater China)` -> ('T1', 'Greater China').
    Tier numbering follows the dataset's own prompt_id prefixes."""
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
def _finsearchcomp_dims(seed) -> pd.DataFrame:
    """q_index -> (tier, region, label, prompt_id) at the given seed."""
    spec = datasets.REGISTRY["finsearchcomp"]
    df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.reset_index(drop=False).rename(columns={"index": "q_index"})
    tiers_regions = df["label"].astype(str).apply(_split_finsearchcomp_label)
    df["tier"] = [tr[0] for tr in tiers_regions]
    df["region"] = [tr[1] for tr in tiers_regions]
    df["prompt_id"] = df.get("prompt_id", pd.Series(dtype=str)).astype(str)
    return df[["q_index", "tier", "region", "label", "prompt_id"]].copy()


_TAGS_DIR = Path("docs/tags")


@st.cache_data
def _sealqa_tags(benchmark: str) -> pd.DataFrame:
    """Taxonomy CSV at docs/tags/{benchmark}.csv. Anchored to the parquet's
    natural order. Currently only sealqa_seal0 ships one; the other seal
    variants return empty so the panel gracefully omits taxonomy slices."""
    path = _TAGS_DIR / f"{benchmark}.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["q_index", "reasoning", "retrieval", "notes"]
        )
    df = pd.read_csv(path)
    keep = ["q_index", "reasoning", "retrieval", "notes"]
    return df[[c for c in keep if c in df.columns]].copy()


@st.cache_data
def _sealqa_native_dims(benchmark: str, seed) -> pd.DataFrame:
    """q_index -> (topic, freshness, question_types) at the given seed,
    pulled from the parquet itself. `question_types` is kept as a list so
    callers can explode it for per-tag slicing. Missing columns are
    silently skipped — longseal lacks `question_types`, for example."""
    spec = datasets.REGISTRY[benchmark]
    df = pd.read_parquet(datasets.DATA_DIR / spec.parquet)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.reset_index(drop=False).rename(columns={"index": "q_index"})
    keep = ["q_index"]
    for c in ("topic", "freshness", "question_types"):
        if c in df.columns:
            keep.append(c)
    out = df[keep].copy()
    if "question_types" in out.columns:
        out["question_types"] = out["question_types"].apply(
            lambda v: list(v) if v is not None and not isinstance(v, float) else []
        )
    return out


@st.cache_data
def _dataset_size(name: str) -> int:
    spec = datasets.REGISTRY[name]
    import pyarrow.parquet as pq
    return pq.ParquetFile(datasets.DATA_DIR / spec.parquet).metadata.num_rows


# ---------------------------------------------------------------------------
# Helpers for the panels
# ---------------------------------------------------------------------------

def _altair(chart, *, key: str | None = None) -> None:
    """Render an altair chart with sensible defaults, or nothing if None."""
    if chart is None:
        return
    st.altair_chart(chart, width="stretch", key=key)


def _missing_caption(
    slice_df: pd.DataFrame, dim_col: str, expected: list[str] | None
) -> None:
    """Show a small caption when expected dim values have no data."""
    if not expected:
        return
    present = set(slice_df[dim_col].astype(str).unique()) if not slice_df.empty else set()
    missing = [v for v in expected if v not in present]
    if missing:
        st.caption(
            f"_no runs yet for: {', '.join(missing)}. "
            "Launch a run covering those questions to populate them._"
        )


def _render_easy_missed(matrix: pd.DataFrame, dims_lookup: pd.DataFrame | None = None,
                        dim_cols: list[str] | None = None) -> None:
    """Table of questions where ≥2 competitor (provider, model) pairs were
    right and no Tavily model was."""
    missed = dc.easy_questions_tavily_missed(matrix, min_competitors_right=2)
    if missed.empty:
        st.caption(
            "_no 'easy questions Tavily missed' at this seed — either "
            "Tavily got everything answerable, or nobody else did._"
        )
        return
    if dims_lookup is not None and dim_cols:
        cols = ["q_index"] + [c for c in dim_cols if c in dims_lookup.columns]
        missed = missed.merge(dims_lookup[cols], on="q_index", how="left")
    st.dataframe(missed, width="stretch", hide_index=True)


def _render_mini_vs_pro(
    matrix: pd.DataFrame, dims_lookup: pd.DataFrame | None = None,
    dim_cols: list[str] | None = None,
) -> None:
    counts, mini_only, pro_only = dc.mini_vs_pro_split(matrix)
    if counts.empty:
        st.caption(
            "_Tavily mini vs pro view needs both `tavily:mini` and "
            "`tavily:pro` to have answered overlapping questions at this "
            "seed. Launch both, then come back._"
        )
        return
    _altair(dc.mini_vs_pro_outcome_bar(counts), key="mini_vs_pro_outcome")
    cdict = {row["outcome"]: int(row["n"]) for _, row in counts.iterrows()}
    summary = (
        f"both right: **{cdict.get('both right', 0)}**  ·  "
        f"pro only right: **{cdict.get('pro only right', 0)}**  ·  "
        f"mini only right: **{cdict.get('mini only right', 0)}**  ·  "
        f"both wrong: **{cdict.get('both wrong', 0)}**"
    )
    st.markdown(summary)

    def _tag(df: pd.DataFrame) -> pd.DataFrame:
        if dims_lookup is not None and dim_cols:
            cols = ["q_index"] + [c for c in dim_cols if c in dims_lookup.columns]
            return df.merge(dims_lookup[cols], on="q_index", how="left")
        return df

    if not mini_only.empty:
        with st.expander(f"Mini right, Pro wrong ({len(mini_only)} questions)"):
            st.dataframe(_tag(mini_only), width="stretch", hide_index=True)
    if not pro_only.empty:
        with st.expander(f"Pro right, Mini wrong ({len(pro_only)} questions)"):
            st.dataframe(_tag(pro_only), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Per-benchmark panels
# ---------------------------------------------------------------------------

def _render_headline(matrix: pd.DataFrame, total_questions: int) -> None:
    if matrix.empty:
        st.info("No runs for this benchmark + seed yet.")
        return
    st.caption(
        f"{total_questions} total questions in the dataset  ·  "
        f"hiding models with fewer than {dc.MIN_N_DISPLAY} graded responses  ·  "
        "accuracy = correct / graded (errored & ungraded rows excluded)"
    )
    _altair(dc.headline_ranked_bar(matrix), key="headline_ranked_bar")


def _render_finsearchcomp_panel(matrix: pd.DataFrame, dims: pd.DataFrame) -> None:
    if matrix.empty:
        st.info("No finsearchcomp runs yet at this seed.")
        return

    st.markdown("### By task tier")
    tier_df = dc.slice_accuracy(matrix, dims, "tier", dim_order=["T1", "T2", "T3"])
    _missing_caption(tier_df, "tier", ["T1", "T2", "T3"])
    _altair(
        dc.grouped_accuracy_bar(
            tier_df, "tier", ["T1", "T2", "T3"],
            "Accuracy by tier (T1 time-sensitive · T2 historical lookup · T3 complex investigation)",
        ),
        key="finsearch_tier_bar",
    )

    st.markdown("### Coverage heatmap")
    _altair(
        dc.coverage_heatmap(matrix, dims, "tier", dim_order=["T1", "T2", "T3"]),
        key="finsearch_coverage_tier",
    )

    st.markdown("### Tavily mini vs pro")
    _render_mini_vs_pro(matrix, dims_lookup=dims, dim_cols=["tier", "region"])

    st.markdown("### Latency")
    _altair(dc.latency_box(matrix), key="finsearch_latency_box")
    _altair(dc.latency_accuracy_scatter(matrix), key="finsearch_latency_scatter")

    st.markdown("### Error vs wrong-answer split")
    _altair(dc.error_vs_wrong_bar(matrix), key="finsearch_error_vs_wrong")

    st.markdown("### Easy questions Tavily missed")
    _render_easy_missed(matrix, dims_lookup=dims, dim_cols=["tier", "region"])


def _drop_low_n_models(matrix: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Drop (provider, model) pairs with fewer than ``min_n`` graded rows.
    Sparse models distort accuracy slices, latency percentiles, and the
    error-vs-wrong split."""
    if matrix.empty:
        return matrix
    counts = (
        matrix[matrix["is_correct"].notna()]
        .groupby(["provider", "model"]).size()
        .reset_index(name="_n_graded")
    )
    keep = counts[counts["_n_graded"] >= min_n][["provider", "model"]]
    if keep.empty:
        return matrix.iloc[0:0]
    return matrix.merge(keep, on=["provider", "model"], how="inner")


def _render_sealqa_panel(
    matrix: pd.DataFrame, tags: pd.DataFrame, native: pd.DataFrame,
    seed_value, benchmark: str,
) -> None:
    if matrix.empty:
        st.info(f"No {benchmark} runs yet at this seed.")
        return
    matrix = _drop_low_n_models(matrix, dc.MIN_N_DISPLAY)
    if matrix.empty:
        st.info(
            f"No {benchmark} models with ≥{dc.MIN_N_DISPLAY} graded runs yet."
        )
        return

    # Taxonomy slices (only when seed is natural order, since the tags CSV
    # is anchored there). Tags-less benchmarks (seal_hard, longseal) skip
    # this block entirely with no warning.
    tags_path = f"docs/tags/{benchmark}.csv"
    has_tags = not tags.empty
    taxonomy_ok = seed_value is None and has_tags
    if has_tags and not taxonomy_ok and seed_value is not None:
        st.info(
            f"Taxonomy tags in `{tags_path}` are anchored to the parquet's "
            "natural order (seed=blank). Switch to seed=blank to see the "
            "taxonomy breakdown."
        )

    if taxonomy_ok:
        st.markdown(f"### By taxonomy ({tags_path})")
        schemes = [
            ("reasoning",
             ["single-hop", "multi-hop", "comparative", "unanswerable"],
             "By reasoning hops"),
            ("retrieval",
             ["common", "specialized", "fresh", "tricky-phrasing"],
             "By retrieval difficulty"),
        ]
        for col, order, title in schemes:
            slice_df = dc.slice_accuracy(matrix, tags, col, dim_order=order)
            present_in_tags = set(tags[col].astype(str).unique())
            expected = [v for v in order if v in present_in_tags]
            _missing_caption(slice_df, col, expected)
            _altair(
                dc.grouped_accuracy_bar(slice_df, col, order, title),
                key=f"sealqa_tax_{col}",
            )

    # Native-column slices (work at any seed since they live in the parquet).
    if not native.empty:
        st.markdown("### By dataset-native columns")
        for col, title in [
            ("topic", "By topic"),
            ("freshness", "By freshness"),
        ]:
            if col not in native.columns:
                continue
            order = sorted(native[col].dropna().astype(str).unique().tolist())
            slice_df = dc.slice_accuracy(matrix, native, col, dim_order=order)
            _altair(
                dc.grouped_accuracy_bar(slice_df, col, order, title),
                key=f"sealqa_native_{col}",
            )
        if "question_types" in native.columns:
            exploded = native.explode("question_types").rename(
                columns={"question_types": "question_type"}
            ).dropna(subset=["question_type"])
            order = sorted(exploded["question_type"].astype(str).unique())
            slice_df = dc.slice_accuracy(matrix, exploded, "question_type",
                                         dim_order=order)
            _altair(
                dc.grouped_accuracy_bar(slice_df, "question_type", order,
                                        "By question_type (exploded multi-label)"),
                key="sealqa_native_question_type",
            )

    st.markdown("### Coverage heatmap")
    # Use reasoning if available, otherwise fall back to topic.
    if taxonomy_ok:
        _altair(
            dc.coverage_heatmap(
                matrix, tags, "reasoning",
                dim_order=["single-hop", "multi-hop", "comparative", "unanswerable"],
            ),
            key="sealqa_coverage_reasoning",
        )
    elif "topic" in native.columns:
        order = sorted(native["topic"].dropna().astype(str).unique().tolist())
        _altair(
            dc.coverage_heatmap(matrix, native, "topic", dim_order=order),
            key="sealqa_coverage_topic",
        )

    st.markdown("### Tavily mini vs pro")
    dims_for_tag = tags if taxonomy_ok else native
    tag_cols = ["reasoning", "retrieval"] if taxonomy_ok else \
        [c for c in ("topic", "freshness") if c in native.columns]
    _render_mini_vs_pro(matrix, dims_lookup=dims_for_tag, dim_cols=tag_cols)

    st.markdown("### Latency")
    _altair(dc.latency_box(matrix), key="sealqa_latency_box")
    _altair(dc.latency_accuracy_scatter(matrix), key="sealqa_latency_scatter")

    st.markdown("### Error vs wrong-answer split")
    _altair(dc.error_vs_wrong_bar(matrix), key="sealqa_error_vs_wrong")

    st.markdown("### Easy questions Tavily missed")
    _render_easy_missed(matrix, dims_lookup=dims_for_tag, dim_cols=tag_cols)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render() -> None:
    st.subheader("Per-benchmark analytics")
    st.caption(
        "Tavily-centric quality view, sliced by the dimensions each "
        "benchmark actually carries. Latest run per (provider, model, "
        "q_index) wins."
    )

    benchmarks = [
        "finsearchcomp", "sealqa_seal0", "sealqa_seal_hard", "sealqa_longseal",
    ]
    bench = st.segmented_control(
        "Benchmark",
        benchmarks,
        default=benchmarks[0],
        key="dash_bench",
    )
    if not bench:
        bench = benchmarks[0]

    # Seed picker — only surface when ambiguous, mirroring Tier Analysis.
    status_rows = storage.get_question_status(bench)
    seeds = sorted({s["seed"] for s in status_rows},
                   key=lambda x: (x is None, x))
    if not seeds:
        seed = None
    elif len(seeds) == 1:
        seed = seeds[0]
        if seed is not None:
            st.caption(f"seed: {seed}")
    else:
        saved = st.session_state.get("dash_seed_select", seeds[0])
        if saved not in seeds:
            saved = seeds[0]
        seed = st.selectbox(
            "Seed",
            seeds,
            index=seeds.index(saved),
            format_func=lambda s: "(no seed)" if s is None else str(s),
            key="dash_seed_select",
            help="Different seeds shuffle the dataset, so q_index "
                 "identity differs across seeds.",
        )

    total_questions = _dataset_size(bench)
    matrix = _benchmark_results_matrix(bench, seed)

    st.divider()
    st.markdown("## Headline")
    _render_headline(matrix, total_questions)
    st.divider()

    if bench == "finsearchcomp":
        _render_finsearchcomp_panel(matrix, _finsearchcomp_dims(seed))
    elif bench.startswith("sealqa"):
        _render_sealqa_panel(
            matrix,
            _sealqa_tags(bench),
            _sealqa_native_dims(bench, seed),
            seed,
            bench,
        )
