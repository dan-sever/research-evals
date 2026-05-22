"""CLI to launch a single-provider benchmark run.

Examples:
    python run.py --provider tavily --benchmark sealqa_seal0 --model mini --limit 10
    python run.py --provider perplexity --benchmark sealqa_seal_hard --model sonar-pro --limit 25
    python run.py --provider exa --benchmark finsearchcomp --limit 5 --seed 42
"""

from __future__ import annotations

import argparse
import sys

from benchmarks import datasets, providers
from benchmarks.config import RunConfig
from benchmarks.runner import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a research benchmark for one provider.")
    parser.add_argument(
        "--provider",
        default="tavily",
        choices=list(providers.PROVIDERS),
        help="Which research provider to use.",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=datasets.list_benchmarks(),
        help="Which dataset to evaluate.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider-specific model id (e.g. mini/pro/auto for Tavily, "
             "sonar-pro for Perplexity). Defaults to the provider's default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of questions to run. Omit for the full set.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent research+judge workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, shuffles the dataset before applying --limit. "
             "Use the same seed across providers for a fair comparison.",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-haiku-4-5",
        help="Anthropic model for the extract+grade step.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Free-form note saved with the run.",
    )

    args = parser.parse_args(argv)
    model = args.model or providers.default_model(args.provider)

    config = RunConfig(
        benchmark=args.benchmark,
        provider=args.provider,
        model=model,
        limit=args.limit,
        workers=args.workers,
        sample_seed=args.seed,
        judge_model=args.judge_model,
        note=args.note,
    )
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
