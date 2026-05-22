"""Run the same benchmark questions across multiple providers, in one batch.

All runs share a single comparison_set id and (importantly) the same seed +
limit, so every provider sees the identical N questions. The dashboard's
"Provider comparison" tab uses comparison_set to group them.

Examples:
    python compare.py --benchmark sealqa_seal0 --limit 20 --seed 42 \\
        --providers tavily:mini,perplexity:sonar-pro,exa:exa,parallel:core

    python compare.py --benchmark finsearchcomp --limit 10 \\
        --providers tavily:pro,perplexity:sonar-reasoning-pro
"""

from __future__ import annotations

import argparse
import sys
import uuid

from benchmarks import datasets, providers
from benchmarks.config import RunConfig, load_env
from benchmarks.runner import run as runner_run


def _parse_providers(spec: str) -> list[tuple[str, str | None]]:
    """`tavily:mini,exa:exa` -> [('tavily','mini'), ('exa','exa')]."""
    out: list[tuple[str, str | None]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, model = item.split(":", 1)
            out.append((name.strip(), model.strip() or None))
        else:
            out.append((item, None))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a benchmark across multiple providers in one batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark", required=True, choices=datasets.list_benchmarks())
    parser.add_argument(
        "--providers",
        required=True,
        help="Comma-separated provider[:model] list, e.g. "
             "tavily:mini,perplexity:sonar-pro,exa:exa,parallel:core",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N questions before applying --limit. "
             "All providers in the batch share the same offset+limit, so the "
             "next batch lines up exactly with the previous one.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Default 42 so every provider sees the same N questions. "
             "Set to a different int to sample a different subset.",
    )
    parser.add_argument("--judge-model", default="claude-haiku-4-5")
    parser.add_argument(
        "--note",
        default="",
        help="Note saved on every run in the batch.",
    )

    args = parser.parse_args(argv)
    specs = _parse_providers(args.providers)
    if not specs:
        parser.error("--providers cannot be empty")

    unknown = [n for n, _ in specs if n not in providers.PROVIDERS]
    if unknown:
        parser.error(
            f"Unknown providers: {unknown}. Known: {list(providers.PROVIDERS)}"
        )

    # Pre-flight: confirm every required key is present before billing anyone.
    env = load_env()
    missing: list[str] = []
    for name, _ in specs:
        var = providers.env_var(name)
        if not env.get(var):
            missing.append(f"{name} ({var})")
    if not env.get("ANTHROPIC_API_KEY"):
        missing.append("judge (ANTHROPIC_API_KEY)")
    if missing:
        parser.error("Missing API keys for: " + ", ".join(missing))

    comparison_set = str(uuid.uuid4())
    print(f"Comparison set: {comparison_set}")
    print(
        f"Benchmark={args.benchmark}  seed={args.seed}  "
        f"limit={args.limit if args.limit is not None else 'all'}"
    )
    print("Providers:")
    for name, model in specs:
        print(f"  - {name}:{model or providers.default_model(name)}")
    print()

    summary: list[tuple[str, str, int]] = []
    for name, model in specs:
        cfg = RunConfig(
            benchmark=args.benchmark,
            provider=name,
            model=model or providers.default_model(name),
            limit=args.limit,
            offset=args.offset,
            workers=args.workers,
            sample_seed=args.seed,
            judge_model=args.judge_model,
            note=args.note,
            comparison_set=comparison_set,
        )
        rid = runner_run(cfg)
        summary.append((name, cfg.model, rid))

    print("\nComparison set complete.")
    print(f"  id        : {comparison_set}")
    print(f"  short id  : {comparison_set[:8]}")
    for name, model, rid in summary:
        print(f"  run #{rid:<3}  {name}:{model}")
    print("\nOpen the dashboard:  streamlit run app.py  (Provider comparison tab)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
