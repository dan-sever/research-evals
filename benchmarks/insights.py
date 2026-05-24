"""Tavily-centric benchmark analysis with a two-stage LLM pipeline.

Stage 1 (enrichment, Haiku): reads each wrong-answer example and tags it
with a failure_mode, source_type_needed, and one-sentence diagnosis. This
is the bulk "read each row" job — cheap, parallel-friendly, structured.

Stage 2 (synthesis, Sonnet): receives the dimensional accuracy tables +
the enriched wrong examples + failure-mode counts, and produces the final
ranked headline + 5–8 bullets. This is the "actually understand the
patterns" job — needs the stronger model.

Per-benchmark dimensions live in BENCHMARK_PAYLOADS so adding a new
benchmark = one function + one registry line.

Output (persisted via storage.save_insight): {headline, insights[...],
_meta{...}} with each insight = {claim, evidence, examples, action, kind}.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from anthropic import Anthropic

from benchmarks import datasets, dimensions, storage


PROMPT_VERSION = "v2-2026-05"
MIN_N_DISPLAY = 10
WRONG_EXAMPLE_BUDGET = 30
ENRICHMENT_MODEL = "claude-haiku-4-5"
SYNTHESIS_MODEL = "claude-sonnet-4-6"
# Back-compat alias for older callers.
DEFAULT_MODEL = SYNTHESIS_MODEL

# ---------------------------------------------------------------------------
# Matrix construction (delegates to benchmarks.dimensions so insights see
# the same latest-wins data the dashboard does)
# ---------------------------------------------------------------------------

def _matrix(benchmark: str, seed: Optional[int]) -> pd.DataFrame:
    df = dimensions.latest_results_matrix(benchmark, seed)
    if df.empty:
        return df
    df["provider_label"] = df["provider"] + ":" + df["model"]
    return df


def _drop_low_n(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    counts = (
        matrix[matrix["is_correct"].notna()]
        .groupby(["provider", "model"]).size()
        .reset_index(name="_n")
    )
    keep = counts[counts["_n"] >= MIN_N_DISPLAY][["provider", "model"]]
    if keep.empty:
        return matrix.iloc[0:0]
    return matrix.merge(keep, on=["provider", "model"], how="inner")


# ---------------------------------------------------------------------------
# Aggregations shared across benchmarks
# ---------------------------------------------------------------------------

def _overall_table(matrix: pd.DataFrame) -> list[dict]:
    """Per provider:model: n_graded, accuracy, n_errored, p50_latency,
    p95_latency. Sorted by accuracy desc."""
    if matrix.empty:
        return []
    rows = []
    for label, grp in matrix.groupby("provider_label"):
        graded = grp[grp["is_correct"].notna()]
        n = len(graded)
        correct = int((graded["is_correct"] == 1).sum())
        errored = int(grp["error"].notna().sum()) if "error" in grp.columns else 0
        lat = grp["research_duration_seconds"].dropna()
        rows.append({
            "provider_label": label,
            "n_graded": n,
            "correct": correct,
            "accuracy_pct": round(correct / n * 100, 1) if n else None,
            "n_errored": errored,
            "p50_latency_s": round(float(lat.quantile(0.5)), 1) if len(lat) else None,
            "p95_latency_s": round(float(lat.quantile(0.95)), 1) if len(lat) else None,
        })
    rows.sort(key=lambda r: (r["accuracy_pct"] is None, -(r["accuracy_pct"] or 0)))
    return rows


def _slice_table(
    matrix: pd.DataFrame, dims: pd.DataFrame, dim_col: str,
) -> list[dict]:
    """Per (provider_label, dim_value): n_graded, accuracy_pct."""
    if matrix.empty or dims.empty or dim_col not in dims.columns:
        return []
    joined = matrix.merge(dims[["q_index", dim_col]], on="q_index", how="inner")
    joined = joined[joined["is_correct"].notna()]
    if joined.empty:
        return []
    grp = (
        joined.groupby([dim_col, "provider_label"])
        .agg(n=("is_correct", "size"),
             correct=("is_correct", lambda s: int((s == 1).sum())))
        .reset_index()
    )
    grp = grp[grp["n"] > 0]
    grp["accuracy_pct"] = (grp["correct"] / grp["n"] * 100).round(1)
    return [
        {
            "dim": dim_col,
            "value": str(r[dim_col]),
            "provider_label": r["provider_label"],
            "n_graded": int(r["n"]),
            "correct": int(r["correct"]),
            "accuracy_pct": float(r["accuracy_pct"]),
        }
        for _, r in grp.iterrows()
    ]


def _cross_table(
    matrix: pd.DataFrame, dims: pd.DataFrame, dim_a: str, dim_b: str,
) -> list[dict]:
    """Per (provider_label, dim_a × dim_b): same shape as _slice_table."""
    if matrix.empty or dims.empty:
        return []
    if dim_a not in dims.columns or dim_b not in dims.columns:
        return []
    joined = matrix.merge(
        dims[["q_index", dim_a, dim_b]], on="q_index", how="inner",
    )
    joined = joined[joined["is_correct"].notna()]
    if joined.empty:
        return []
    grp = (
        joined.groupby([dim_a, dim_b, "provider_label"])
        .agg(n=("is_correct", "size"),
             correct=("is_correct", lambda s: int((s == 1).sum())))
        .reset_index()
    )
    grp = grp[grp["n"] > 0]
    grp["accuracy_pct"] = (grp["correct"] / grp["n"] * 100).round(1)
    return [
        {
            "dim_a": dim_a, "value_a": str(r[dim_a]),
            "dim_b": dim_b, "value_b": str(r[dim_b]),
            "provider_label": r["provider_label"],
            "n_graded": int(r["n"]),
            "correct": int(r["correct"]),
            "accuracy_pct": float(r["accuracy_pct"]),
        }
        for _, r in grp.iterrows()
    ]


def _wrong_examples(
    matrix: pd.DataFrame, dims: pd.DataFrame, dim_cols: list[str], budget: int,
) -> list[dict]:
    """Sample of wrong-answer questions, weighted toward 'easy questions
    Tavily missed' (≥2 competitors right, no Tavily model right). Falls
    back to general Tavily wrongs to fill the budget. Each example carries
    the dimension values + which competitor models got it right."""
    if matrix.empty:
        return []
    have_dims = dims is not None and not dims.empty
    dim_cols = [c for c in dim_cols if have_dims and c in dims.columns]
    if have_dims and dim_cols:
        tagged = matrix.merge(
            dims[["q_index"] + dim_cols], on="q_index", how="left",
        )
    else:
        tagged = matrix.copy()
        for c in dim_cols:
            tagged[c] = None

    examples: list[dict] = []
    seen: set[int] = set()

    # Pass 1: easy questions Tavily missed.
    for qi, grp in tagged.groupby("q_index"):
        tav = grp[grp["provider"] == "tavily"]
        other = grp[grp["provider"] != "tavily"]
        if tav.empty:
            continue
        if (tav["is_correct"] == 1).any():
            continue
        n_other_right = int((other["is_correct"] == 1).sum())
        if n_other_right < 2:
            continue
        examples.append(_example_row(qi, grp, dim_cols, kind="easy_missed",
                                     n_other_right=n_other_right))
        seen.add(int(qi))

    examples.sort(key=lambda e: -e["n_competitors_right"])
    examples = examples[: max(budget // 2, 1)]
    seen = {e["q_index"] for e in examples}

    # Pass 2: any Tavily wrong, stratified across dim values so we don't
    # over-sample one slice.
    pool = []
    for qi, grp in tagged.groupby("q_index"):
        if int(qi) in seen:
            continue
        tav = grp[grp["provider"] == "tavily"]
        if tav.empty:
            continue
        if (tav["is_correct"] == 1).any():
            continue
        if not (tav["is_correct"] == 0).any():
            continue  # only ungraded/errored — skip
        pool.append(_example_row(qi, grp, dim_cols, kind="tavily_wrong"))

    # Stratify by first available dim_col, taking round-robin.
    if dim_cols and pool:
        by_val: dict[str, list[dict]] = {}
        for ex in pool:
            key = str(ex.get(dim_cols[0], "?"))
            by_val.setdefault(key, []).append(ex)
        for v in by_val:
            by_val[v].sort(key=lambda e: e["q_index"])
        remaining = budget - len(examples)
        while remaining > 0 and any(by_val.values()):
            for v in list(by_val.keys()):
                if not by_val[v]:
                    continue
                examples.append(by_val[v].pop(0))
                remaining -= 1
                if remaining <= 0:
                    break
    else:
        examples += pool[: budget - len(examples)]
    return examples[:budget]


def _example_row(
    q_index: int, grp: pd.DataFrame, dim_cols: list[str],
    kind: str, n_other_right: int = 0,
) -> dict:
    tav = grp[grp["provider"] == "tavily"]
    other = grp[grp["provider"] != "tavily"]
    question = grp["question"].iloc[0] if "question" in grp.columns else ""
    expected = grp["expected_answer"].iloc[0] if "expected_answer" in grp.columns else ""
    tavily_answers = [
        f"{m}: {(a or '')[:120]}"
        for m, a in zip(tav["model"], tav["extracted_answer"].fillna(""))
    ]
    winners = sorted(
        f"{p}:{m}"
        for p, m, c in zip(other["provider"], other["model"], other["is_correct"])
        if c == 1
    )
    row: dict = {
        "q_index": int(q_index),
        "kind": kind,
        "question": (question or "")[:240],
        "expected_answer": (expected or "")[:240],
        "tavily_attempts": tavily_answers,
        "competitor_winners": winners,
        "n_competitors_right": int(n_other_right) if n_other_right
                              else int((other["is_correct"] == 1).sum()),
    }
    for c in dim_cols:
        if c in grp.columns:
            v = grp[c].dropna()
            row[c] = str(v.iloc[0]) if len(v) else None
    return row


# ---------------------------------------------------------------------------
# Per-benchmark payload builders
# ---------------------------------------------------------------------------

def _build_finsearchcomp_payload(
    matrix: pd.DataFrame, seed: Optional[int],
) -> dict:
    dims = dimensions.finsearchcomp_dims(seed)[["q_index", "tier", "region"]]
    return {
        "benchmark": "finsearchcomp",
        "dimension_descriptions": {
            "tier": "T1 = time-sensitive data fetching (live prices, latest filings); "
                    "T2 = simple historical lookup; "
                    "T3 = complex historical investigation requiring synthesis.",
            "region": "Global vs Greater China. Greater China questions often need "
                      "Chinese-language sources or local market context.",
        },
        "overall": _overall_table(matrix),
        "by_tier": _slice_table(matrix, dims, "tier"),
        "by_region": _slice_table(matrix, dims, "region"),
        "tier_x_region": _cross_table(matrix, dims, "tier", "region"),
        "wrong_examples": _wrong_examples(
            matrix, dims, ["tier", "region"], WRONG_EXAMPLE_BUDGET,
        ),
    }


def _build_sealqa_payload(
    matrix: pd.DataFrame, seed: Optional[int], benchmark: str = "sealqa_seal0",
) -> dict:
    """Generalized SealQA payload. Taxonomy tags only join when the
    benchmark has a tags CSV AND the seed is blank (the CSV is anchored
    to natural order). Other SealQA variants reuse this with no tags."""
    native = dimensions.sealqa_native_dims(benchmark, seed)
    tags = dimensions.sealqa_tags(benchmark) if seed is None else pd.DataFrame()
    # Merge native + tags into a single dims frame so we can crosstab.
    dims = native.copy()
    if not tags.empty:
        dims = dims.merge(tags, on="q_index", how="left")

    payload: dict = {
        "benchmark": benchmark,
        "dimension_descriptions": {
            "topic": "Native dataset topic label.",
            "freshness": "Whether the answer requires fresh / recent information.",
            "question_types": "Multi-label tag list per question (exploded for analysis).",
        },
        "overall": _overall_table(matrix),
        "by_topic": _slice_table(matrix, dims, "topic"),
        "by_freshness": _slice_table(matrix, dims, "freshness"),
        "topic_x_freshness": _cross_table(matrix, dims, "topic", "freshness"),
    }
    if "question_types" in dims.columns:
        exploded = dims.explode("question_types").rename(
            columns={"question_types": "question_type"}
        ).dropna(subset=["question_type"])
        payload["by_question_type"] = _slice_table(matrix, exploded, "question_type")
    if not tags.empty:
        payload["dimension_descriptions"]["reasoning"] = (
            "Reasoning hops: single-hop, multi-hop, comparative, unanswerable."
        )
        payload["dimension_descriptions"]["retrieval"] = (
            "Retrieval difficulty: common, specialized, fresh, tricky-phrasing."
        )
        payload["by_reasoning"] = _slice_table(matrix, dims, "reasoning")
        payload["by_retrieval"] = _slice_table(matrix, dims, "retrieval")
        payload["reasoning_x_retrieval"] = _cross_table(
            matrix, dims, "reasoning", "retrieval",
        )
    example_dims = [c for c in ("reasoning", "retrieval", "topic", "freshness")
                    if c in dims.columns]
    payload["wrong_examples"] = _wrong_examples(
        matrix, dims, example_dims, WRONG_EXAMPLE_BUDGET,
    )
    return payload


def _sealqa_payload_for(name: str) -> Callable[[pd.DataFrame, Optional[int]], dict]:
    """Bind `_build_sealqa_payload` to a specific benchmark name so the
    registry signature stays `(matrix, seed) -> dict`."""
    def _inner(matrix: pd.DataFrame, seed: Optional[int]) -> dict:
        return _build_sealqa_payload(matrix, seed, benchmark=name)
    _inner.__name__ = f"_build_{name}_payload"
    return _inner


BENCHMARK_PAYLOADS: dict[str, Callable[[pd.DataFrame, Optional[int]], dict]] = {
    "finsearchcomp": _build_finsearchcomp_payload,
    "sealqa_seal0": _sealqa_payload_for("sealqa_seal0"),
    "sealqa_seal_hard": _sealqa_payload_for("sealqa_seal_hard"),
    "sealqa_longseal": _sealqa_payload_for("sealqa_longseal"),
}


# ---------------------------------------------------------------------------
# Stage 1 — Haiku enrichment of wrong examples
# ---------------------------------------------------------------------------

FAILURE_MODES = [
    "missing_fresh_data",       # answer required live/recent data the agent didn't fetch
    "wrong_entity",             # picked a similar-but-wrong company / person / metric
    "hallucinated",             # confidently asserted a fact that isn't true
    "weak_synthesis",           # had the sources, drew the wrong conclusion
    "retrieval_gap",            # right approach, couldn't find the source
    "ambiguous_question",       # the benchmark question itself is unclear
    "language_barrier",         # answer lives in a non-English source the agent missed
    "complex_reasoning_chain",  # multi-step inference that broke down
    "format_mismatch",          # right value, wrong format (judge should have caught — flag)
    "other",
]


ENRICHMENT_SYSTEM = """You are characterizing why Tavily failed on each question in a research benchmark.

For each item you'll see: the question, the expected answer, Tavily's attempts (mini and/or pro), which competitor models got it right, and the dimensional tags.

For each item, return:
- `failure_mode`: pick the single best match from the enum.
- `source_type_needed`: a concrete description of the source that would have answered correctly. Be specific (e.g., "live equity price feed", "Chinese-language CSRC filing", "step-by-step quarterly report cross-reference across 2022–2024"). Avoid generic terms like "web search".
- `diagnosis`: one sentence on what Tavily missed that the winning competitors got. If no competitors won, say what the question fundamentally required.

Be terse and specific. No filler. Return the record_failure_characterizations tool call."""


ENRICHMENT_TOOL = {
    "name": "record_failure_characterizations",
    "description": "Record per-question characterizations of why Tavily failed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "characterizations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "q_index": {"type": "integer"},
                        "failure_mode": {
                            "type": "string", "enum": FAILURE_MODES,
                        },
                        "source_type_needed": {
                            "type": "string",
                            "description": "Specific source description.",
                        },
                        "diagnosis": {
                            "type": "string",
                            "description": "One sentence on what Tavily missed.",
                        },
                    },
                    "required": ["q_index", "failure_mode",
                                 "source_type_needed", "diagnosis"],
                },
            },
        },
        "required": ["characterizations"],
    },
}


def _enrich_wrong_examples(
    examples: list[dict], client: Anthropic,
) -> tuple[list[dict], dict]:
    """Call Haiku once to characterize every wrong example. Returns the
    list of characterizations plus a tokens dict."""
    if not examples:
        return [], {"input": 0, "output": 0}
    # Strip fields the enrichment model doesn't need (provider winners list
    # stays — it's the strongest signal for the diagnosis).
    trimmed = []
    for ex in examples:
        trimmed.append({
            k: v for k, v in ex.items()
            if k in ("q_index", "question", "expected_answer",
                     "tavily_attempts", "competitor_winners",
                     "n_competitors_right", "tier", "region",
                     "topic", "freshness", "reasoning", "retrieval")
        })
    user_msg = (
        "Characterize each of the following Tavily failures.\n\n"
        f"```json\n{json.dumps(trimmed, ensure_ascii=False)}\n```\n\n"
        "Return one characterization per q_index, in the same order. "
        "Use only the enum values for failure_mode."
    )
    msg = client.messages.create(
        model=ENRICHMENT_MODEL,
        max_tokens=4096,
        system=ENRICHMENT_SYSTEM,
        tools=[ENRICHMENT_TOOL],
        tool_choice={"type": "tool", "name": "record_failure_characterizations"},
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use" and \
                block.name == "record_failure_characterizations":
            chars = list(block.input.get("characterizations", []))
            tokens = {
                "input": getattr(msg.usage, "input_tokens", None),
                "output": getattr(msg.usage, "output_tokens", None),
            }
            return chars, tokens
    raise InsightsError(f"Enrichment model did not return tool call. Raw: {msg}")


# ---------------------------------------------------------------------------
# Stage 2 — Sonnet synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are analyzing a research-agent benchmark run from Tavily's perspective.

The user is the PM of Tavily Research. They already see the single-dimension bars on the dashboard. What they need from you is patterns they cannot see at a glance — dimension *combinations* where Tavily systematically wins or loses against Perplexity, EXA, and Parallel, and the *why* behind it.

You receive a JSON payload with:
- `dimension_descriptions`: semantic meaning of each dimension
- `overall`: per-(provider:model) accuracy, error count, p50/p95 latency
- `by_<dim>`: accuracy per (provider:model, dimension value)
- `<dim_a>_x_<dim_b>`: accuracy on dimension crosstabs
- `wrong_examples`: ~30 questions Tavily got wrong, each enriched by a first-pass model with `failure_mode`, `source_type_needed`, and `diagnosis`
- `failure_mode_counts`: how many of the wrong examples fall into each failure_mode

Your job:
1. Produce 5–8 ranked findings, top of the list = biggest lever for Tavily.
2. Prioritize **dimension combinations** (e.g., "multi-hop + fresh", "T3 + Greater China") over single-dimension findings. Single-dim is already visible on the dashboard.
3. Connect the dots: when you see a dimensional gap (e.g., "Tavily lags 18pp on Greater China T3"), look at the enriched wrong examples in that slice — what's the *failure_mode* concentration? what *source_type* do they share? Use that to explain *why*.
4. Call out wins worth marketing, not just gaps.
5. Flag infra issues (errors, latency) only if they materially affect the read.

Constraints:
- `evidence`: must contain concrete numbers from the payload (e.g., "tavily:pro 3/14 = 21% vs perplexity:sonar-reasoning-pro 9/14 = 64%; 5 of 6 wrongs are missing_fresh_data").
- `examples`: 1–3 real q_index references from `wrong_examples`. Format: "Q47 — short paraphrase".
- If a cell has n < 5, do not anchor a finding on it.
- Do not invent dimensions, models, numbers, or q_indices. Use only the payload.
- `action`: one concrete next step (e.g., "add a Chinese-language regulator pass for Greater China queries"), not "consider exploring further".
- Tone: senior PM writing to peers. Direct, no filler, no hedging.

Return the record_insights tool call. Nothing else."""


INSIGHTS_TOOL = {
    "name": "record_insights",
    "description": "Record a ranked list of Tavily-centric findings from this benchmark run.",
    "input_schema": {
        "type": "object",
        "properties": {
            "headline": {
                "type": "string",
                "description": "One sentence stating the single biggest takeaway for Tavily.",
            },
            "insights": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "One-sentence finding focused on Tavily's relative position. Prefer dimension combinations.",
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Concrete numbers + failure_mode aggregates from the payload.",
                        },
                        "examples": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 3,
                            "description": "Up to 3 q_index references with a short paraphrase. Format: 'Q47 — short paraphrase'.",
                        },
                        "action": {
                            "type": "string",
                            "description": "One concrete next step the team can take this week.",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["gap", "win", "infra"],
                            "description": "gap = Tavily underperforms. win = Tavily leads. infra = error/latency issue.",
                        },
                    },
                    "required": ["claim", "evidence", "examples", "action", "kind"],
                },
            },
        },
        "required": ["headline", "insights"],
    },
}


class InsightsError(RuntimeError):
    pass


def generate(
    benchmark: str,
    seed: Optional[int],
    api_key: Optional[str] = None,
    enrichment_model: str = ENRICHMENT_MODEL,
    synthesis_model: str = SYNTHESIS_MODEL,
) -> dict:
    """Two-stage pipeline: Haiku enriches each wrong example with a
    failure_mode + diagnosis, Sonnet synthesizes across enriched examples
    + dimensional aggregates. Persists and returns the structured result."""
    if benchmark not in BENCHMARK_PAYLOADS:
        raise InsightsError(f"Unsupported benchmark: {benchmark}")
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise InsightsError("ANTHROPIC_API_KEY missing — set it in .env.")

    matrix = _drop_low_n(_matrix(benchmark, seed))
    if matrix.empty:
        raise InsightsError(
            f"No {benchmark} rows at this seed with ≥{MIN_N_DISPLAY} graded runs per model."
        )
    payload = BENCHMARK_PAYLOADS[benchmark](matrix, seed)

    client = Anthropic(api_key=api_key)

    # Stage 1: enrich.
    enrichments, enrich_tokens = _enrich_wrong_examples(
        payload.get("wrong_examples", []), client,
    )
    enrich_by_q = {int(c["q_index"]): c for c in enrichments}
    for ex in payload["wrong_examples"]:
        e = enrich_by_q.get(int(ex["q_index"]))
        if e:
            ex["failure_mode"] = e.get("failure_mode")
            ex["source_type_needed"] = e.get("source_type_needed")
            ex["diagnosis"] = e.get("diagnosis")
    payload["failure_mode_counts"] = dict(
        Counter(
            ex.get("failure_mode") for ex in payload["wrong_examples"]
            if ex.get("failure_mode")
        )
    )

    # Stage 2: synthesize.
    user_msg = (
        "Here is the enriched benchmark payload as JSON.\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```\n\n"
        "Return the record_insights tool call."
    )
    msg = client.messages.create(
        model=synthesis_model,
        max_tokens=4096,
        system=SYNTHESIS_SYSTEM,
        tools=[INSIGHTS_TOOL],
        tool_choice={"type": "tool", "name": "record_insights"},
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use" and \
                block.name == "record_insights":
            content = dict(block.input)
            content["_meta"] = {
                "payload_summary": {
                    "n_overall_rows": len(payload.get("overall", [])),
                    "n_wrong_examples": len(payload.get("wrong_examples", [])),
                    "n_enriched": len(enrichments),
                    "failure_mode_counts": payload.get("failure_mode_counts", {}),
                    "dimensions_used": [
                        k.replace("by_", "") for k in payload
                        if k.startswith("by_")
                    ],
                },
                "models": {
                    "enrichment": enrichment_model,
                    "synthesis": synthesis_model,
                },
                "tokens": {
                    "enrichment": enrich_tokens,
                    "synthesis": {
                        "input": getattr(msg.usage, "input_tokens", None),
                        "output": getattr(msg.usage, "output_tokens", None),
                    },
                },
            }
            storage.save_insight(
                benchmark=benchmark, seed=seed,
                model=f"{enrichment_model} → {synthesis_model}",
                prompt_version=PROMPT_VERSION, content=content,
            )
            return content
    raise InsightsError(f"Synthesis model did not return tool call. Raw: {msg}")
