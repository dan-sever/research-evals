"""Altair chart builders for the Dashboard tab.

Every chart in here uses one of two color treatments:

- Two-color (``is_tavily`` boolean): Tavily blue vs neutral gray. Used in
  the headline ranked bar and the latency scatter, where the only signal
  the reader needs is "where's Tavily".
- Tavily-first palette: Tavily blue, then a gray gradient for the rest of
  the providers. Used in grouped bar charts where you want to tell
  competitors apart while still keeping Tavily visually dominant.

All builders return ``alt.Chart`` objects (or tuples for the disagreement
helper). Callers use ``st.altair_chart(chart, use_container_width=True)``.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

# --- Color tokens -----------------------------------------------------------

TAVILY_COLOR = "#1f6feb"           # Tavily blue
TAVILY_LIGHT = "#7aa7ff"           # second Tavily shade (mini vs pro split)
NEUTRAL_DARK = "#374151"           # competitors, darkest
NEUTRAL_MID = "#6b7280"
NEUTRAL_LIGHT = "#9ca3af"
NEUTRAL_PALE = "#cbd5e1"
NEUTRAL_PALETTE = [NEUTRAL_DARK, NEUTRAL_MID, NEUTRAL_LIGHT, NEUTRAL_PALE]

# Outcome colors for the mini-vs-pro split.
OK_GREEN = "#16a34a"
WARN_AMBER = "#f59e0b"
BAD_GRAY = "#9ca3af"

MIN_N_DISPLAY = 10  # provider:model bars with fewer runs are hidden from headline charts


# --- Small helpers ----------------------------------------------------------

def provider_label(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def tavily_first(labels: list[str]) -> list[str]:
    """Tavily entries first, then everything else alphabetical."""
    tavily = sorted(l for l in labels if l.startswith("tavily:"))
    others = sorted(l for l in labels if not l.startswith("tavily:"))
    return tavily + others


def color_scale_for(labels: list[str]) -> alt.Scale:
    """Categorical color scale: Tavily entries get blues, competitors get
    a gray gradient. Order matches ``tavily_first(labels)``."""
    order = tavily_first(labels)
    blues = [TAVILY_COLOR, TAVILY_LIGHT, "#a5c2ff", "#cfdcff"]
    grays = NEUTRAL_PALETTE
    palette: list[str] = []
    bi = gi = 0
    for label in order:
        if label.startswith("tavily:"):
            palette.append(blues[bi % len(blues)])
            bi += 1
        else:
            palette.append(grays[gi % len(grays)])
            gi += 1
    return alt.Scale(domain=order, range=palette)


def provider_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per (provider, model): n, correct, accuracy. Used by the headline bar
    and the latency/accuracy scatter."""
    if matrix.empty:
        return pd.DataFrame(
            columns=["provider", "model", "provider_label", "n", "correct",
                     "accuracy", "is_tavily"]
        )
    df = matrix.assign(
        correct=matrix["is_correct"].fillna(0).astype(int),
        provider_label=[
            provider_label(p, m)
            for p, m in zip(matrix["provider"], matrix["model"])
        ],
    )
    out = (
        df.groupby(["provider", "model", "provider_label"])
        .agg(n=("q_index", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    out["accuracy"] = out["correct"] / out["n"]
    out["is_tavily"] = out["provider"] == "tavily"
    return out


# --- A. Headline ranked bar -------------------------------------------------

def headline_ranked_bar(matrix: pd.DataFrame) -> alt.Chart | None:
    """Horizontal bars per provider:model, sorted descending by accuracy.
    Per-provider color palette (Tavily blues, neutrals for the rest),
    matching the grouped tier chart. Models with fewer than
    ``MIN_N_DISPLAY`` runs are hidden so noisy bars don't crowd the view."""
    summary = provider_summary(matrix)
    if summary.empty:
        return None
    summary = summary[summary["n"] >= MIN_N_DISPLAY].copy()
    if summary.empty:
        return None
    summary = summary.sort_values("accuracy", ascending=False).reset_index(drop=True)
    summary["accuracy_pct"] = (summary["accuracy"] * 100).round(1)
    summary["bar_label"] = [
        f"{a:.1f}%  ·  n={n}"
        for a, n in zip(summary["accuracy_pct"], summary["n"])
    ]

    label_order = summary["provider_label"].tolist()
    palette_scale = color_scale_for(label_order)

    base = alt.Chart(summary).encode(
        y=alt.Y(
            "provider_label:N",
            sort=label_order,  # descending accuracy → best at top
            title=None,
            axis=alt.Axis(labelLimit=400, labelOverlap=False, labelPadding=4),
        ),
    )
    bars = base.mark_bar(size=24).encode(
        x=alt.X("accuracy_pct:Q", title="accuracy (%)",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "provider_label:N", scale=palette_scale,
            legend=alt.Legend(title="provider:model"),
        ),
        tooltip=[
            alt.Tooltip("provider_label:N", title="model"),
            alt.Tooltip("accuracy_pct:Q", title="accuracy %"),
            alt.Tooltip("n:Q", title="n"),
            alt.Tooltip("correct:Q", title="correct"),
        ],
    )
    text = base.mark_text(
        align="left", baseline="middle", dx=4, color="#111",
    ).encode(
        x=alt.X("accuracy_pct:Q"),
        text="bar_label:N",
    )
    chart = (bars + text).properties(
        height=max(60 + 32 * len(summary), 160),
        title=alt.TitleParams(
            f"Accuracy by provider:model (best on top, hiding n < {MIN_N_DISPLAY})",
            anchor="start",
        ),
    )
    return chart


# --- B. Grouped accuracy bar ------------------------------------------------

def grouped_accuracy_bar(
    slice_df: pd.DataFrame, dim_col: str, dim_order: list[str], title: str
) -> alt.Chart | None:
    """Slice_df has columns: {dim_col}, provider_label, accuracy, n.
    Returns a grouped bar chart with Tavily highlighted."""
    if slice_df.empty:
        return None
    df = slice_df.copy()
    df["accuracy_pct"] = (df["accuracy"] * 100).round(1)
    df["is_tavily"] = df["provider_label"].str.startswith("tavily:")
    labels = df["provider_label"].unique().tolist()
    palette_scale = color_scale_for(labels)

    # Facet by dim_col so each slice (e.g. T1/T2/T3) gets its own panel
    # with bars sorted ascending by accuracy within that panel. xOffset
    # with a global sort can't do per-group ordering — facets can.
    base = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            "provider_label:N",
            sort=alt.SortField(field="accuracy_pct", order="ascending"),
            title=None,
            axis=alt.Axis(labels=False, ticks=False),
        ),
        y=alt.Y("accuracy_pct:Q", title="accuracy (%)",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("provider_label:N", scale=palette_scale,
                        legend=alt.Legend(title="provider:model")),
        tooltip=[
            alt.Tooltip("provider_label:N", title="model"),
            alt.Tooltip(f"{dim_col}:N", title=dim_col),
            alt.Tooltip("accuracy_pct:Q", title="accuracy %"),
            alt.Tooltip("n:Q", title="n"),
        ],
    ).properties(height=320, width=alt.Step(28))

    chart = base.facet(
        column=alt.Column(
            f"{dim_col}:N", sort=dim_order,
            header=alt.Header(title=None, labelFontSize=13, labelFontWeight="bold"),
        ),
        spacing=24,
    ).resolve_scale(x="independent").properties(
        title=alt.TitleParams(title, anchor="start"),
    )
    return chart


# --- C. Tavily mini vs pro --------------------------------------------------

def mini_vs_pro_split(
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (counts_df, mini_only_right_df, pro_only_right_df) over the
    set of questions answered by *both* tavily:mini and tavily:pro at this
    seed.

    counts_df has columns: outcome, n, color (4 rows).
    Each *_only_right_df has q_index, question, expected_answer, mini answer,
    pro answer.
    """
    empty_counts = pd.DataFrame(
        columns=["outcome", "n", "color", "order"]
    )
    empty_rows = pd.DataFrame(
        columns=["q_index", "question", "expected_answer",
                 "tavily_mini answer", "tavily_pro answer"]
    )
    if matrix.empty:
        return empty_counts, empty_rows, empty_rows
    tav = matrix[matrix["provider"] == "tavily"].copy()
    if tav.empty:
        return empty_counts, empty_rows, empty_rows
    # Pull mini and pro into separate frames, then inner-merge so we keep
    # only the q_indices both models answered.
    mini = tav[tav["model"] == "mini"][
        ["q_index", "is_correct", "extracted_answer"]
    ].rename(
        columns={"is_correct": "mini_c", "extracted_answer": "mini_ans"}
    )
    pro = tav[tav["model"] == "pro"][
        ["q_index", "is_correct", "extracted_answer", "question",
         "expected_answer"]
    ].rename(
        columns={"is_correct": "pro_c", "extracted_answer": "pro_ans"}
    )
    if mini.empty or pro.empty:
        return empty_counts, empty_rows, empty_rows
    joined = mini.merge(pro, on="q_index", how="inner")
    if joined.empty:
        return empty_counts, empty_rows, empty_rows

    def _bin(row):
        mc = 1 if row["mini_c"] == 1 else 0
        pc = 1 if row["pro_c"] == 1 else 0
        if mc and pc: return "both right"
        if pc and not mc: return "pro only right"
        if mc and not pc: return "mini only right"
        return "both wrong"

    joined["outcome"] = joined.apply(_bin, axis=1)
    counts = (
        joined.groupby("outcome").size()
        .reset_index(name="n")
    )
    ordering = ["both right", "pro only right", "mini only right", "both wrong"]
    palette = {
        "both right": OK_GREEN,
        "pro only right": TAVILY_COLOR,
        "mini only right": WARN_AMBER,
        "both wrong": BAD_GRAY,
    }
    counts["order"] = counts["outcome"].apply(lambda x: ordering.index(x))
    counts["color"] = counts["outcome"].map(palette)
    counts = counts.sort_values("order").reset_index(drop=True)

    cols = ["q_index", "question", "expected_answer", "mini_ans", "pro_ans"]
    mini_only = (
        joined[joined["outcome"] == "mini only right"][cols]
        .rename(columns={"mini_ans": "tavily_mini answer",
                         "pro_ans": "tavily_pro answer"})
        .reset_index(drop=True)
    )
    pro_only = (
        joined[joined["outcome"] == "pro only right"][cols]
        .rename(columns={"mini_ans": "tavily_mini answer",
                         "pro_ans": "tavily_pro answer"})
        .reset_index(drop=True)
    )
    return counts, mini_only, pro_only


def mini_vs_pro_outcome_bar(counts: pd.DataFrame) -> alt.Chart | None:
    """A single horizontal stacked bar showing the 4 outcome buckets."""
    if counts.empty:
        return None
    chart = alt.Chart(counts).mark_bar(size=30).encode(
        x=alt.X("n:Q", stack="zero", title="questions"),
        color=alt.Color(
            "outcome:N",
            scale=alt.Scale(
                domain=counts["outcome"].tolist(),
                range=counts["color"].tolist(),
            ),
            legend=alt.Legend(title="outcome"),
        ),
        order=alt.Order("order:Q"),
        tooltip=[
            alt.Tooltip("outcome:N"),
            alt.Tooltip("n:Q", title="questions"),
        ],
    ).properties(
        height=80,
        title=alt.TitleParams(
            "Tavily mini vs pro (questions both have answered)",
            anchor="start",
        ),
    )
    return chart


# --- D. Latency views -------------------------------------------------------

def latency_box(matrix: pd.DataFrame) -> alt.Chart | None:
    if matrix.empty:
        return None
    df = matrix.dropna(subset=["research_duration_seconds"]).copy()
    df = df[df["research_duration_seconds"] > 0]
    if df.empty:
        return None
    df["provider_label"] = [
        provider_label(p, m) for p, m in zip(df["provider"], df["model"])
    ]
    df["is_tavily"] = df["provider"] == "tavily"

    medians = (
        df.groupby("provider_label")["research_duration_seconds"]
        .median().sort_values().index.tolist()
    )
    chart = alt.Chart(df).mark_boxplot(extent=1.5, size=18).encode(
        x=alt.X("provider_label:N", sort=medians, title=None,
                axis=alt.Axis(labelAngle=-30, labelLimit=200)),
        y=alt.Y(
            "research_duration_seconds:Q",
            scale=alt.Scale(type="log"),
            title="latency (seconds, log)",
        ),
        color=alt.Color(
            "is_tavily:N",
            scale=alt.Scale(domain=[True, False],
                            range=[TAVILY_COLOR, NEUTRAL_MID]),
            legend=None,
        ),
    ).properties(
        height=320,
        title=alt.TitleParams(
            "Latency distribution (sorted by median)", anchor="start",
        ),
    )
    return chart


def latency_accuracy_scatter(matrix: pd.DataFrame) -> alt.Chart | None:
    """One point per provider:model. X = median latency, Y = accuracy %.
    Size = n. Tavily highlighted. Reference cross at overall medians."""
    summary = provider_summary(matrix)
    if summary.empty:
        return None
    lat = matrix.dropna(subset=["research_duration_seconds"]).copy()
    lat = lat[lat["research_duration_seconds"] > 0]
    if lat.empty:
        return None
    lat["provider_label"] = [
        provider_label(p, m) for p, m in zip(lat["provider"], lat["model"])
    ]
    med = (
        lat.groupby("provider_label")["research_duration_seconds"]
        .median().reset_index(name="median_latency")
    )
    df = summary.merge(med, on="provider_label", how="inner")
    if df.empty:
        return None
    df["accuracy_pct"] = (df["accuracy"] * 100).round(1)

    median_lat_all = df["median_latency"].median()
    median_acc_all = df["accuracy_pct"].median()

    points = alt.Chart(df).mark_circle().encode(
        x=alt.X("median_latency:Q",
                scale=alt.Scale(type="log"),
                title="median latency (seconds, log)"),
        y=alt.Y("accuracy_pct:Q", title="accuracy (%)",
                scale=alt.Scale(domain=[0, 100])),
        size=alt.Size("n:Q", scale=alt.Scale(range=[60, 600]),
                      legend=alt.Legend(title="n")),
        color=alt.Color(
            "is_tavily:N",
            scale=alt.Scale(domain=[True, False],
                            range=[TAVILY_COLOR, NEUTRAL_MID]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("provider_label:N", title="model"),
            alt.Tooltip("median_latency:Q", title="median latency (s)",
                        format=".1f"),
            alt.Tooltip("accuracy_pct:Q", title="accuracy %"),
            alt.Tooltip("n:Q", title="n"),
        ],
    )
    labels = alt.Chart(df).mark_text(
        align="left", dx=8, dy=-6, fontSize=11, color="#111",
    ).encode(
        x="median_latency:Q",
        y="accuracy_pct:Q",
        text="provider_label:N",
    )
    ref_x = alt.Chart(pd.DataFrame({"x": [median_lat_all]})).mark_rule(
        color="#cbd5e1", strokeDash=[4, 4]
    ).encode(x="x:Q")
    ref_y = alt.Chart(pd.DataFrame({"y": [median_acc_all]})).mark_rule(
        color="#cbd5e1", strokeDash=[4, 4]
    ).encode(y="y:Q")
    chart = (ref_x + ref_y + points + labels).properties(
        height=360,
        title=alt.TitleParams(
            "Latency × accuracy (upper-left = fast & accurate)",
            anchor="start",
        ),
    )
    return chart


# --- E1. Error vs wrong split -----------------------------------------------

def error_vs_wrong_bar(matrix: pd.DataFrame) -> alt.Chart | None:
    if matrix.empty:
        return None
    df = matrix.copy()
    df["provider_label"] = [
        provider_label(p, m) for p, m in zip(df["provider"], df["model"])
    ]

    def _bucket(row):
        if row.get("error"):
            return "errored"
        if row["is_correct"] == 1:
            return "correct"
        return "wrong"

    df["bucket"] = df.apply(_bucket, axis=1)
    counts = (
        df.groupby(["provider_label", "bucket"]).size()
        .reset_index(name="n")
    )
    # Sort providers by accuracy desc so the best models are on top.
    order = (
        df.assign(c=df["is_correct"].fillna(0).astype(int))
        .groupby("provider_label")["c"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    bucket_order = ["correct", "wrong", "errored"]
    bucket_colors = [OK_GREEN, NEUTRAL_MID, "#dc2626"]

    chart = alt.Chart(counts).mark_bar().encode(
        y=alt.Y("provider_label:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=240)),
        x=alt.X("n:Q", stack="normalize", title="share of questions",
                axis=alt.Axis(format="%")),
        color=alt.Color(
            "bucket:N",
            scale=alt.Scale(domain=bucket_order, range=bucket_colors),
            legend=alt.Legend(title="outcome"),
        ),
        order=alt.Order(
            "bucket:N",
            sort="ascending",
        ),
        tooltip=[
            alt.Tooltip("provider_label:N", title="model"),
            alt.Tooltip("bucket:N", title="outcome"),
            alt.Tooltip("n:Q", title="n"),
        ],
    ).properties(
        height=max(40 + 26 * len(order), 120),
        title=alt.TitleParams(
            "Correct vs wrong vs errored (share of answered questions)",
            anchor="start",
        ),
    )
    return chart


# --- E2. Easy questions Tavily missed ---------------------------------------

def easy_questions_tavily_missed(
    matrix: pd.DataFrame, min_competitors_right: int = 2
) -> pd.DataFrame:
    """Questions where >=``min_competitors_right`` competitor (provider,
    model) pairs got it right and no Tavily model did. Indicates a real
    Tavily-specific gap rather than a benchmark-impossible question."""
    if matrix.empty:
        return pd.DataFrame()
    by_q = matrix.groupby("q_index")
    rows: list[dict] = []
    for qi, grp in by_q:
        tav_grp = grp[grp["provider"] == "tavily"]
        other_grp = grp[grp["provider"] != "tavily"]
        tav_right = (tav_grp["is_correct"] == 1).any()
        if tav_right:
            continue
        n_other_right = int((other_grp["is_correct"] == 1).sum())
        if n_other_right < min_competitors_right:
            continue
        question = grp["question"].iloc[0] if "question" in grp.columns else ""
        expected = grp["expected_answer"].iloc[0] if "expected_answer" in grp.columns else ""
        winners = sorted(
            f"{p}:{m}"
            for p, m, c in zip(other_grp["provider"], other_grp["model"],
                               other_grp["is_correct"])
            if c == 1
        )
        tav_models = sorted(
            f"{m}" for m in tav_grp["model"].unique()
        )
        rows.append({
            "q_index": int(qi),
            "n_competitors_right": n_other_right,
            "winners": ", ".join(winners),
            "tavily models tried": ", ".join(tav_models),
            "question": question,
            "expected_answer": expected,
        })
    out = pd.DataFrame(rows).sort_values(
        ["n_competitors_right", "q_index"], ascending=[False, True]
    ).reset_index(drop=True)
    return out


# --- E3. Coverage heatmap ---------------------------------------------------

def coverage_heatmap(
    matrix: pd.DataFrame,
    dims: pd.DataFrame,
    dim_col: str,
    dim_order: list[str] | None = None,
) -> alt.Chart | None:
    """Heatmap of n runs per (provider:model × dim_col value). Makes
    low-coverage cells visible at a glance."""
    if matrix.empty or dims.empty:
        return None
    joined = matrix.merge(dims[["q_index", dim_col]], on="q_index", how="inner")
    if joined.empty:
        return None
    joined["provider_label"] = [
        provider_label(p, m) for p, m in zip(joined["provider"], joined["model"])
    ]
    cell = (
        joined.groupby(["provider_label", dim_col]).size()
        .reset_index(name="n")
    )
    label_sort = tavily_first(cell["provider_label"].unique().tolist())
    x_sort = dim_order or sorted(cell[dim_col].unique().tolist())

    rect = alt.Chart(cell).mark_rect().encode(
        x=alt.X(f"{dim_col}:N", sort=x_sort, title=dim_col),
        y=alt.Y("provider_label:N", sort=label_sort, title=None,
                axis=alt.Axis(labelLimit=240)),
        color=alt.Color("n:Q",
                        scale=alt.Scale(scheme="blues"),
                        legend=alt.Legend(title="n runs")),
        tooltip=[
            alt.Tooltip("provider_label:N", title="model"),
            alt.Tooltip(f"{dim_col}:N", title=dim_col),
            alt.Tooltip("n:Q", title="n"),
        ],
    )
    text = alt.Chart(cell).mark_text(fontSize=11).encode(
        x=alt.X(f"{dim_col}:N", sort=x_sort),
        y=alt.Y("provider_label:N", sort=label_sort),
        text="n:Q",
        color=alt.condition(
            "datum.n > 30",
            alt.value("white"),
            alt.value("#111"),
        ),
    )
    chart = (rect + text).properties(
        height=max(40 + 24 * len(label_sort), 120),
        title=alt.TitleParams(
            f"Coverage heatmap by {dim_col} (n runs per cell)",
            anchor="start",
        ),
    )
    return chart


# --- Shared slice helper (kept here so it lives next to the chart fns) ------

def slice_accuracy(
    matrix: pd.DataFrame,
    dims: pd.DataFrame,
    dim_col: str,
    dim_order: list[str] | None = None,
) -> pd.DataFrame:
    """Inner-join matrix with dims on q_index, then return (dim_col,
    provider_label, accuracy, n). Cells with n=0 are dropped naturally."""
    if matrix.empty or dims.empty:
        return pd.DataFrame(
            columns=[dim_col, "provider_label", "accuracy", "n"]
        )
    df = matrix.merge(dims[["q_index", dim_col]], on="q_index", how="inner")
    if df.empty:
        return pd.DataFrame(
            columns=[dim_col, "provider_label", "accuracy", "n"]
        )
    df["provider_label"] = [
        provider_label(p, m) for p, m in zip(df["provider"], df["model"])
    ]
    grp = (
        df.groupby([dim_col, "provider_label"], dropna=False)
        .agg(
            n=("is_correct", "size"),
            correct=("is_correct", lambda s: int(s.fillna(0).sum())),
        )
        .reset_index()
    )
    grp["accuracy"] = grp["correct"] / grp["n"]
    if dim_order:
        grp[dim_col] = pd.Categorical(
            grp[dim_col], categories=dim_order, ordered=True
        )
        grp = grp.sort_values([dim_col, "provider_label"]).reset_index(drop=True)
    return grp[[dim_col, "provider_label", "accuracy", "n"]]
