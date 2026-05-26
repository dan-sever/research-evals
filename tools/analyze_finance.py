"""Two-stage LLM analysis of Tavily on financebench + financeqa.

Stage 1 (Haiku): enrich each example with a structured tag.
  - For wrong rows: failure_mode + source_type_needed + diagnosis
  - For Tavily-wins (Tavily right, ≥1 competitor wrong): success_factor + edge
Stage 2 (Sonnet 4.6): synthesize a per-benchmark report covering both
  areas of competence and areas where Tavily falls short, plus a combined
  cross-benchmark summary.

Writes markdown to `finance_analysis/`. One file per benchmark plus a
combined summary. Independent of the in-app insights pipeline so it can
evolve without touching the Streamlit Insights tab.

Run:
    .venv/bin/python tools/analyze_finance.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

# Make the project importable when run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import dimensions, storage  # noqa: E402

load_dotenv(ROOT / ".env")

ENRICHMENT_MODEL = "claude-haiku-4-5"
SYNTHESIS_MODEL = "claude-sonnet-4-6"

OUT_DIR = ROOT / "finance_analysis"
OUT_DIR.mkdir(exist_ok=True)

WRONG_BUDGET = 30
WIN_BUDGET = 20


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _matrix(benchmark: str) -> pd.DataFrame:
    df = dimensions.latest_results_matrix(benchmark, seed=None)
    if df.empty:
        return df
    df["provider_label"] = df["provider"] + ":" + df["model"]
    return df


def _enrich_with_content(matrix: pd.DataFrame) -> pd.DataFrame:
    """Pull research_content + reasoning + research_sources_json for each
    (run_id, q_index) so the enrichment model has the actual answer text,
    not just the extracted label."""
    if matrix.empty:
        return matrix
    run_ids = matrix["run_id"].unique().tolist()
    placeholders = ",".join("?" * len(run_ids))
    with storage.connect(storage.DB_PATH) as conn:
        rows = conn.execute(
            f"""SELECT run_id, q_index, research_content, reasoning, research_sources_json
                FROM results WHERE run_id IN ({placeholders})""",
            run_ids,
        ).fetchall()
    extra = pd.DataFrame([dict(r) for r in rows])
    return matrix.merge(extra, on=["run_id", "q_index"], how="left")


def _dims(benchmark: str) -> pd.DataFrame:
    """Combine taxonomy CSV + parquet-native dims into one frame keyed by q_index."""
    native = dimensions.finance_native_dims(benchmark, seed=None)
    tags = dimensions.taxonomy_tags(benchmark)
    if tags.empty:
        return native
    return native.merge(tags, on="q_index", how="left", suffixes=("_native", ""))


# ---------------------------------------------------------------------------
# Aggregation tables
# ---------------------------------------------------------------------------

def _overall(matrix: pd.DataFrame) -> list[dict]:
    if matrix.empty:
        return []
    rows = []
    for label, grp in matrix.groupby("provider_label"):
        graded = grp[grp["is_correct"].notna()]
        n = len(graded)
        correct = int((graded["is_correct"] == 1).sum())
        errored = int(grp["error"].notna().sum())
        lat = grp["research_duration_seconds"].dropna()
        rows.append({
            "provider_label": label,
            "n_total": len(grp),
            "n_graded": n,
            "correct": correct,
            "accuracy_pct": round(correct / n * 100, 1) if n else None,
            "n_errored": errored,
            "p50_latency_s": round(float(lat.quantile(0.5)), 1) if len(lat) else None,
            "p95_latency_s": round(float(lat.quantile(0.95)), 1) if len(lat) else None,
        })
    rows.sort(key=lambda r: (r["accuracy_pct"] is None, -(r["accuracy_pct"] or 0)))
    return rows


def _slice(matrix: pd.DataFrame, dims: pd.DataFrame, col: str) -> list[dict]:
    if matrix.empty or dims.empty or col not in dims.columns:
        return []
    j = matrix.merge(dims[["q_index", col]], on="q_index", how="inner")
    j = j[j["is_correct"].notna()]
    if j.empty:
        return []
    g = (
        j.groupby([col, "provider_label"])
        .agg(n=("is_correct", "size"),
             correct=("is_correct", lambda s: int((s == 1).sum())))
        .reset_index()
    )
    g = g[g["n"] > 0]
    g["accuracy_pct"] = (g["correct"] / g["n"] * 100).round(1)
    return [
        {
            "dim": col, "value": str(r[col]),
            "provider_label": r["provider_label"],
            "n": int(r["n"]), "correct": int(r["correct"]),
            "accuracy_pct": float(r["accuracy_pct"]),
        }
        for _, r in g.iterrows()
    ]


# ---------------------------------------------------------------------------
# Example sampling: wrongs Tavily should worry about + wins worth marketing
# ---------------------------------------------------------------------------

def _trim(s: Optional[str], n: int) -> str:
    if not s:
        return ""
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _example_base(grp: pd.DataFrame, dims_row: dict) -> dict:
    qi = int(grp["q_index"].iloc[0])
    question = _trim(grp["question"].iloc[0], 280)
    expected = _trim(grp["expected_answer"].iloc[0], 240)
    rows = []
    for _, r in grp.iterrows():
        rows.append({
            "provider_label": r["provider_label"],
            "is_correct": (None if pd.isna(r["is_correct"]) else int(r["is_correct"])),
            "extracted_answer": _trim(r.get("extracted_answer"), 200),
            "research_content_excerpt": _trim(r.get("research_content"), 600),
            "judge_reasoning": _trim(r.get("reasoning"), 220),
        })
    base = {
        "q_index": qi,
        "question": question,
        "expected_answer": expected,
        "attempts": rows,
    }
    for k, v in dims_row.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        base[k] = str(v)
    return base


def _sample_examples(
    matrix: pd.DataFrame, dims: pd.DataFrame,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return three buckets, all from a Tavily-centric lens:
      - tavily_wrong_competitors_right: ≥1 Tavily wrong, ≥1 non-Tavily right
      - tavily_right_competitors_wrong: ≥1 Tavily right, ≥1 non-Tavily wrong
      - all_wrong: every provider that answered got it wrong (joint hard set)
    """
    if matrix.empty:
        return [], [], []
    have_dims = not dims.empty
    dim_cols = [c for c in dims.columns if c != "q_index"] if have_dims else []
    tagged = matrix.copy()
    if have_dims:
        tagged = tagged.merge(dims, on="q_index", how="left")

    losses: list[dict] = []
    wins: list[dict] = []
    joint_hards: list[dict] = []

    for qi, grp in tagged.groupby("q_index"):
        tav = grp[grp["provider"].str.startswith("tavily")]
        other = grp[~grp["provider"].str.startswith("tavily")]
        graded = grp[grp["is_correct"].notna()]
        if graded.empty:
            continue
        dims_row = {c: grp[c].dropna().iloc[0] if grp[c].notna().any() else None
                    for c in dim_cols}
        if (tav["is_correct"] == 0).any() and (other["is_correct"] == 1).any():
            ex = _example_base(grp, dims_row)
            ex["bucket"] = "tavily_wrong_competitors_right"
            ex["n_competitors_right"] = int((other["is_correct"] == 1).sum())
            losses.append(ex)
        elif (tav["is_correct"] == 1).any() and (other["is_correct"] == 0).any():
            ex = _example_base(grp, dims_row)
            ex["bucket"] = "tavily_right_competitors_wrong"
            ex["n_competitors_wrong"] = int((other["is_correct"] == 0).sum())
            wins.append(ex)
        elif (graded["is_correct"] == 0).all() and len(graded) >= 2:
            ex = _example_base(grp, dims_row)
            ex["bucket"] = "all_wrong"
            joint_hards.append(ex)

    losses.sort(key=lambda e: -e["n_competitors_right"])
    wins.sort(key=lambda e: -e["n_competitors_wrong"])
    return (
        losses[:WRONG_BUDGET],
        wins[:WIN_BUDGET],
        joint_hards[:WRONG_BUDGET // 2],
    )


# ---------------------------------------------------------------------------
# Stage 1 — Haiku enrichment
# ---------------------------------------------------------------------------

FAILURE_MODES = [
    "wrong_period",            # right value for wrong fiscal year / quarter
    "wrong_entity",            # picked similar-but-wrong company / subsidiary / metric
    "wrong_metric_definition", # extracted gross profit where revenue was asked, etc.
    "missing_filing_access",   # could not pull the 10-K / 10-Q / 8-K cited as the source
    "calculation_error",       # had the inputs, did the arithmetic wrong
    "stale_data",              # used an older year's figure when current was asked
    "hallucinated",            # confidently stated a fact not supported by any source
    "retrieval_gap",           # right approach, search returned nothing useful
    "format_mismatch",         # right value, judge marked wrong on units / rounding
    "ambiguous_question",      # question itself is unclear or has multiple valid answers
    "other",
]

SUCCESS_FACTORS = [
    "direct_filing_extract",        # pulled the number straight from a 10-K / 10-Q table
    "correct_period_disambiguation",# right fiscal year/quarter when others picked the wrong one
    "multi_step_calculation",       # did the math correctly (margin = X/Y, etc.)
    "definition_handled",           # interpreted the metric correctly (e.g., EBITDA add-backs)
    "cross_reference",              # synthesized across multiple statements / years correctly
    "ambiguity_resolved",           # disambiguated a fuzzy question well
    "snippet_quality",              # retrieved better excerpts than the competitor
    "other",
]

ENRICHMENT_SYSTEM = """You are characterizing one row from a finance research benchmark. Each row is one question, the expected answer, and how each provider (Tavily basic, Tavily advanced, EXA, Parallel) answered it.

There are two kinds of rows:
- `bucket = tavily_wrong_competitors_right`: Tavily got it wrong while at least one competitor got it right. Use the `failure` schema.
- `bucket = tavily_right_competitors_wrong`: Tavily got it right while at least one competitor got it wrong. Use the `win` schema.
- `bucket = all_wrong`: nobody got it right. Use the `failure` schema and tag the failure mode that best fits Tavily's attempt.

For each item, return either a failure or a win record. Be terse and specific. Quote the expected answer or attempt when it makes the diagnosis sharper. Avoid generic language."""

ENRICHMENT_TOOL = {
    "name": "record_finance_characterizations",
    "description": "One characterization per q_index, in the same order as input.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "q_index": {"type": "integer"},
                        "kind": {"type": "string", "enum": ["failure", "win"]},
                        "failure_mode": {"type": "string", "enum": FAILURE_MODES,
                                         "description": "Required when kind=failure."},
                        "success_factor": {"type": "string", "enum": SUCCESS_FACTORS,
                                           "description": "Required when kind=win."},
                        "diagnosis": {
                            "type": "string",
                            "description": "One sentence: for failure, what Tavily missed that the winners got. For win, what Tavily did better than the losers.",
                        },
                        "source_type_needed": {
                            "type": "string",
                            "description": "Concrete source description (e.g., 'FY2018 10-K consolidated cash flow statement, line: capital expenditures').",
                        },
                    },
                    "required": ["q_index", "kind", "diagnosis", "source_type_needed"],
                },
            },
        },
        "required": ["items"],
    },
}


def _enrich(examples: list[dict], client: Anthropic) -> list[dict]:
    if not examples:
        return []
    # Trim before sending so we don't burn tokens on full research_content
    # for every attempt.
    trimmed = []
    for ex in examples:
        t = {
            "q_index": ex["q_index"],
            "bucket": ex["bucket"],
            "question": ex["question"],
            "expected_answer": ex["expected_answer"],
            "attempts": [
                {
                    "provider_label": a["provider_label"],
                    "is_correct": a["is_correct"],
                    "extracted_answer": a["extracted_answer"],
                    "judge_reasoning": a["judge_reasoning"],
                }
                for a in ex["attempts"]
            ],
        }
        for k in ("query_type", "question_type", "question_reasoning",
                  "gics_sector", "doc_type", "doc_period", "company", "notes"):
            if k in ex:
                t[k] = ex[k]
        trimmed.append(t)

    user_msg = (
        "Characterize each of these finance-benchmark rows. "
        "Return one record per q_index in the same order. "
        "Use only the enum values for failure_mode / success_factor.\n\n"
        f"```json\n{json.dumps(trimmed, ensure_ascii=False)}\n```"
    )
    msg = client.messages.create(
        model=ENRICHMENT_MODEL,
        max_tokens=8192,
        system=ENRICHMENT_SYSTEM,
        tools=[ENRICHMENT_TOOL],
        tool_choice={"type": "tool", "name": "record_finance_characterizations"},
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use":
            return list(block.input.get("items", []))
    raise RuntimeError(f"Enrichment did not return tool call. Raw stop_reason={msg.stop_reason}")


# ---------------------------------------------------------------------------
# Stage 2 — Sonnet synthesis (per benchmark)
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are writing a per-benchmark analysis for the PM of Tavily Research. Audience is internal: PM peers and engineering leadership. They already see the headline accuracy bar; what they need from you are the patterns underneath that they cannot see at a glance.

You receive a JSON payload with:
- `benchmark`, `n_questions`
- `overall`: per (provider:model) accuracy, latency, errors
- `by_<dim>`: accuracy per (provider:model, dimension value) across native and taxonomy dims
- `losses`: questions where Tavily lost to a competitor, each tagged with failure_mode + diagnosis
- `wins`: questions where Tavily beat a competitor, each tagged with success_factor + diagnosis
- `joint_hards`: questions nobody got right
- `failure_mode_counts` and `success_factor_counts`

Write a markdown report with the following sections, in this order:
1. `# {benchmark} — Tavily competence and gaps`
2. `## Headline` — one to two sentences naming the single biggest gap and the single biggest area of competence, both with numbers.
3. `## Where Tavily wins` — 3 to 5 bullets, each leading with a concrete pattern (a slice of the data, a failure mode the competition has that Tavily doesn't, etc.), backed by numbers from `overall` + `wins`, with 1-2 quoted q_index examples in the format "Q{q_index}: short paraphrase".
4. `## Where Tavily falls short` — 3 to 6 bullets, same shape, focused on patterns. Pull the dominant failure_modes from `failure_mode_counts`. Be specific about *why* and *what kind of source* would have closed the gap.
5. `## Cross-cutting observations` — 2 to 4 bullets on what the dimensional slices reveal (e.g., "Tavily advanced gains 18pp over basic on `calculation` questions but loses 4pp on `direct-lookup`"). Include hardest joint slices (where all providers struggle).
6. `## What to do next` — 3 to 5 concrete action bullets the PM can take this week. No "explore further" / "consider X". Be specific (a new index, a prompt change, a routing rule, etc.).

Hard rules:
- Every claim that names a number must come from the payload. No invented stats.
- Reference q_indices that exist in the payload's losses / wins lists.
- Do not invent providers, models, or dimensions outside the payload.
- Do not include tables of the raw numbers, only the patterns.
- Tone: direct, opinionated where the evidence is strong, hedged only when genuinely uncertain.
- Avoid em-dashes. No filler openers ("Great", "Overall"), no filler closers ("Hope this helps"). No emojis.
- Markdown only. No code fences around the whole document."""


def _synthesize_benchmark(payload: dict, client: Anthropic) -> str:
    msg = client.messages.create(
        model=SYNTHESIS_MODEL,
        max_tokens=6000,
        system=SYNTHESIS_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                "Here is the enriched payload for one benchmark. Write the report.\n\n"
                f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
            ),
        }],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()
    raise RuntimeError(f"Synthesis returned no text. stop_reason={msg.stop_reason}")


# ---------------------------------------------------------------------------
# Stage 2b — combined cross-benchmark summary
# ---------------------------------------------------------------------------

COMBINED_SYSTEM = """You are writing a short cross-benchmark summary covering both finance benchmarks (financebench and financeqa) from Tavily's perspective.

You receive a JSON list. Each item is one benchmark with `overall`, `failure_mode_counts`, `success_factor_counts`, and the per-benchmark `report_markdown` you already wrote.

Write a markdown summary with:
1. `# Finance benchmarks — combined view`
2. `## Headline` — one sentence on Tavily's position across both datasets.
3. `## Strengths that show up on both` — 2 to 4 bullets, each tied to numbers from both benchmarks when the pattern exists in both.
4. `## Gaps that show up on both` — 2 to 4 bullets, same shape.
5. `## Benchmark-specific notes` — 2 to 3 short bullets per benchmark on things that only appear there.
6. `## Top priorities` — 3 bullets, ranked.

Hard rules: same as the per-benchmark report. Numbers must come from the payload. No em-dashes. Direct tone."""


def _synthesize_combined(per_bench_payloads: list[dict], client: Anthropic) -> str:
    msg = client.messages.create(
        model=SYNTHESIS_MODEL,
        max_tokens=4096,
        system=COMBINED_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                "Here are the per-benchmark payloads + reports. Write the combined view.\n\n"
                f"```json\n{json.dumps(per_bench_payloads, ensure_ascii=False)}\n```"
            ),
        }],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()
    raise RuntimeError(f"Combined synthesis returned no text. stop_reason={msg.stop_reason}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _build_payload(benchmark: str, client: Anthropic) -> dict:
    matrix = _matrix(benchmark)
    matrix = _enrich_with_content(matrix)
    dims = _dims(benchmark)
    overall = _overall(matrix)
    n_questions = int(matrix["q_index"].nunique()) if not matrix.empty else 0

    dim_cols = [c for c in dims.columns if c != "q_index"]
    slices = {f"by_{c}": _slice(matrix, dims, c) for c in dim_cols}

    losses, wins, joint = _sample_examples(matrix, dims)
    print(f"  [{benchmark}] losses={len(losses)} wins={len(wins)} joint_hards={len(joint)}")

    print(f"  [{benchmark}] enriching {len(losses) + len(wins) + len(joint)} examples with Haiku...")
    enriched = _enrich(losses + wins + joint, client)
    by_q = {int(e["q_index"]): e for e in enriched}
    for ex in losses + wins + joint:
        e = by_q.get(int(ex["q_index"]))
        if not e:
            continue
        for k in ("failure_mode", "success_factor", "diagnosis", "source_type_needed"):
            if k in e:
                ex[k] = e[k]

    failure_counts = dict(Counter(
        ex.get("failure_mode")
        for ex in losses + joint
        if ex.get("failure_mode")
    ))
    success_counts = dict(Counter(
        ex.get("success_factor") for ex in wins if ex.get("success_factor")
    ))

    return {
        "benchmark": benchmark,
        "n_questions": n_questions,
        "overall": overall,
        **slices,
        "losses": losses,
        "wins": wins,
        "joint_hards": joint,
        "failure_mode_counts": failure_counts,
        "success_factor_counts": success_counts,
    }


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY missing. Set it in .env.", file=sys.stderr)
        sys.exit(1)
    client = Anthropic(api_key=api_key)

    per_bench = []
    for bench in ("financebench", "financeqa"):
        print(f"[{bench}] building payload...")
        payload = _build_payload(bench, client)
        print(f"[{bench}] synthesizing report with Sonnet 4.6...")
        report = _synthesize_benchmark(payload, client)
        out = OUT_DIR / f"{bench}.md"
        out.write_text(report + "\n", encoding="utf-8")
        print(f"[{bench}] wrote {out.relative_to(ROOT)}")
        # Stash the JSON payload alongside the markdown for traceability.
        # Useful when the report cites a number and you want to verify it.
        (OUT_DIR / f"{bench}.payload.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        per_bench.append({
            "benchmark": bench,
            "overall": payload["overall"],
            "failure_mode_counts": payload["failure_mode_counts"],
            "success_factor_counts": payload["success_factor_counts"],
            "report_markdown": report,
        })

    print("[combined] synthesizing cross-benchmark summary...")
    combined = _synthesize_combined(per_bench, client)
    (OUT_DIR / "combined.md").write_text(combined + "\n", encoding="utf-8")
    print(f"[combined] wrote {(OUT_DIR / 'combined.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
