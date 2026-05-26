"""Classify finance benchmark questions into the financial query-type
taxonomy (Scheme C in docs/question_taxonomy.md).

Reads the full parquet for the target benchmark (not just answered rows
— the finance benchmarks are small enough to label in one shot), sends
every question to Claude Haiku in one batched tool call, and writes
`docs/tags/{benchmark}.csv` with columns `q_index, query_type, notes,
question`. Idempotent — overwrites the file.

Usage:
    python tools/classify_finance_questions.py financebench
    python tools/classify_finance_questions.py financeqa
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks import datasets  # noqa: E402

TAGS_DIR = REPO / "docs" / "tags"

# Mirrors Scheme C in docs/question_taxonomy.md. Keep this list in sync if
# the doc changes — and bump every downstream tab that filters on these
# values as well.
QUERY_TYPES = ["direct-lookup", "calculation", "comparative", "conceptual"]

SUPPORTED = ("financebench", "financeqa")

SYSTEM = """You classify atomic financial QA questions using Scheme C from docs/question_taxonomy.md.

Pick exactly one `query_type` per question. The four categories are mutually exclusive — pick the dominant cognitive move:

- direct-lookup: a single value or fact stated verbatim in a financial document. No math, no judgment, no comparison.
  examples: "What was 3M's FY2018 capex?" / "What is Gross Profit for the year ending 2024?"

- calculation: requires arithmetic over 2+ retrieved values — ratios, margins, growth rates, deltas, sums. The agent must fetch inputs and compute the output.
  examples: "What was Costco's gross margin in FY2024?" / "What is unadjusted EBITDA?" / "What is YoY revenue growth?"

- comparative: relative judgment across 2+ periods, segments, line items, or companies. Retrieve both sides and weigh them.
  examples: "Did 3M's operating margin improve from FY2021 to FY2022?" / "Which segment grew fastest?"

- conceptual: requires finance/accounting interpretation, not just retrieval + math. The answer is a judgment, attribution, or classification.
  examples: "Is 3M capital-intensive?" / "What drove the margin change?" / "How would you classify this covenant?"

Also produce `notes`: a 4-12 word audit hint a human can scan, e.g. "FY2018 capex direct extract" or "operating margin YoY delta + driver attribution". Not a full explanation.

When a question sits on a boundary — for example a question that combines a lookup with a small calculation — pick the harder move (calculation > direct-lookup, comparative > calculation, conceptual > everything else) and mention the alternative in notes.

Be terse and specific. Use only the enum values. Return record_finance_classifications."""

TOOL = {
    "name": "record_finance_classifications",
    "description": "Record financial query-type classifications.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "q_index": {"type": "integer"},
                        "query_type": {"type": "string", "enum": QUERY_TYPES},
                        "notes": {
                            "type": "string",
                            "description": "4-12 word audit hint.",
                        },
                    },
                    "required": ["q_index", "query_type", "notes"],
                },
            },
        },
        "required": ["classifications"],
    },
}


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _classify_batch(
    client: Anthropic, model: str, questions: list[dict],
) -> list[dict]:
    user_msg = (
        "Classify each question below. Return one record per q_index, "
        "in the same order.\n\n"
        f"```json\n{json.dumps(questions, ensure_ascii=False)}\n```"
    )
    msg = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "record_finance_classifications"},
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in msg.content:
        if (getattr(block, "type", None) == "tool_use"
                and block.name == "record_finance_classifications"):
            return list(block.input.get("classifications", []))
    raise SystemExit("Model did not return tool call.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        choices=SUPPORTED,
        help="finance benchmark to classify",
    )
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument(
        "--batch-size", type=int, default=80,
        help="Questions per Haiku call. Lower if hitting max_tokens.",
    )
    args = parser.parse_args()

    _load_env(REPO / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing in .env")

    spec = datasets.REGISTRY[args.benchmark]
    parquet = pd.read_parquet(datasets.DATA_DIR / spec.parquet)

    questions = []
    for qi in range(len(parquet)):
        q_text = str(parquet[spec.question_col].iloc[qi])
        questions.append({"q_index": int(qi), "question": q_text})
    print(f"Classifying {len(questions)} questions for {args.benchmark}")

    client = Anthropic()
    classifications: list[dict] = []
    for start in range(0, len(questions), args.batch_size):
        chunk = questions[start:start + args.batch_size]
        print(f"  batch {start}–{start + len(chunk) - 1}")
        classifications.extend(_classify_batch(client, args.model, chunk))

    qmap = {q["q_index"]: q["question"] for q in questions}
    by_q = {int(c["q_index"]): c for c in classifications}

    missing = sorted(set(qmap) - set(by_q))
    if missing:
        print(f"  warning: {len(missing)} q_indices were not classified: "
              f"{missing[:10]}{'…' if len(missing) > 10 else ''}")

    TAGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TAGS_DIR / f"{args.benchmark}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q_index", "query_type", "notes", "question"])
        for qi in sorted(by_q.keys()):
            c = by_q[qi]
            writer.writerow([
                qi, c["query_type"], c["notes"], qmap.get(qi, ""),
            ])

    print(f"\nWrote {len(by_q)} classifications to {out_path}")
    df = pd.DataFrame(list(by_q.values()))
    print("\nQuery-type distribution:")
    print(df["query_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
