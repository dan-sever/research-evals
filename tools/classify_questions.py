"""Classify answered benchmark questions into the reasoning/retrieval
taxonomy from docs/question_taxonomy.md.

Reads `results.db` for the unique q_indices answered at seed=None for the
target benchmark, sends them to Haiku in one batched tool call, and
writes `docs/tags/{benchmark}.csv`. Idempotent — overwrites the file.

Usage:
    python tools/classify_questions.py deepsearchqa
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks import datasets  # noqa: E402

TAGS_DIR = REPO / "docs" / "tags"
DB_PATH = REPO / "results.db"

REASONING = ["single-hop", "multi-hop", "comparative", "unanswerable"]
RETRIEVAL = ["common", "specialized", "fresh", "tricky-phrasing"]

SYSTEM = """You classify research-benchmark questions using the taxonomy from docs/question_taxonomy.md.

For each question, assign two labels:

- reasoning ∈ {single-hop, multi-hop, comparative, unanswerable}
  * single-hop: one fact, one lookup. Includes superlatives where the list IS the answer (largest, first, oldest).
  * multi-hop: chain 2+ facts, or filter a list then count/sum/intersect. Covers list questions ("list every X that …") that need filtering + enumeration.
  * comparative: explicitly weigh two or more named things against each other.
  * unanswerable: premise is wrong, contradictory, or no such entity exists.

- retrieval ∈ {common, specialized, fresh, tricky-phrasing}
  * common: top Google / Wikipedia / well-known records.
  * specialized: lives in a specific authoritative source — regulator filings, niche databases, non-English sources, technical PDFs.
  * fresh: rankings/counts/leaderboards that change continuously.
  * tricky-phrasing: qualifier or trap easy to skim past ("exclusively", "since 2023", "only", "before Y").

Plus `notes`: a 4–12 word audit hint, e.g. "NBA 60+ point games since 2023" or "Forbes top-10 athletes filtered by golfer + US-resident". Not a full explanation.

When a question sits on a boundary, pick the dominant move and mention the alternative in notes. Be terse and specific. Use only the enum values. Return record_classifications."""

TOOL = {
    "name": "record_classifications",
    "description": "Record reasoning/retrieval classifications for benchmark questions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "q_index": {"type": "integer"},
                        "reasoning": {"type": "string", "enum": REASONING},
                        "retrieval": {"type": "string", "enum": RETRIEVAL},
                        "notes": {
                            "type": "string",
                            "description": "4-12 word audit hint.",
                        },
                    },
                    "required": ["q_index", "reasoning", "retrieval", "notes"],
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


def _answered_q_indices(benchmark: str) -> list[int]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT DISTINCT res.q_index FROM runs r
           JOIN results res ON res.run_id = r.id
           WHERE r.benchmark = ? AND r.seed IS NULL
           ORDER BY res.q_index""",
        (benchmark,),
    ).fetchall()
    return [int(r["q_index"]) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", help="benchmark name from REGISTRY")
    parser.add_argument("--model", default="claude-haiku-4-5")
    args = parser.parse_args()

    _load_env(REPO / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing in .env")

    spec = datasets.REGISTRY[args.benchmark]
    parquet = pd.read_parquet(datasets.DATA_DIR / spec.parquet)

    qis = _answered_q_indices(args.benchmark)
    print(f"Classifying {len(qis)} answered q_indices for {args.benchmark}")

    questions = []
    for qi in qis:
        if qi < 0 or qi >= len(parquet):
            continue
        q_text = str(parquet[spec.question_col].iloc[qi])
        questions.append({"q_index": int(qi), "question": q_text})

    client = Anthropic()
    user_msg = (
        "Classify each question below. Return one record per q_index, "
        "in the same order.\n\n"
        f"```json\n{json.dumps(questions, ensure_ascii=False)}\n```"
    )
    msg = client.messages.create(
        model=args.model,
        max_tokens=8192,
        system=SYSTEM,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "record_classifications"},
        messages=[{"role": "user", "content": user_msg}],
    )

    classifications = None
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use" \
                and block.name == "record_classifications":
            classifications = list(block.input.get("classifications", []))
            break
    if classifications is None:
        raise SystemExit("Model did not return tool call.")

    qmap = {q["q_index"]: q["question"] for q in questions}
    by_q = {int(c["q_index"]): c for c in classifications}

    TAGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TAGS_DIR / f"{args.benchmark}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q_index", "reasoning", "retrieval", "notes", "question"])
        for qi in sorted(by_q.keys()):
            c = by_q[qi]
            writer.writerow([
                qi, c["reasoning"], c["retrieval"], c["notes"], qmap.get(qi, ""),
            ])

    print(f"Wrote {len(by_q)} classifications to {out_path}")
    df = pd.DataFrame(list(by_q.values()))
    print("\nReasoning distribution:")
    print(df["reasoning"].value_counts().to_string())
    print("\nRetrieval distribution:")
    print(df["retrieval"].value_counts().to_string())


if __name__ == "__main__":
    main()
