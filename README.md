# Research benchmarks

Run Hugging Face QA benchmarks through one or more research providers (Tavily, Perplexity, EXA, Parallel), grade the answers with Claude, and inspect the results side by side.

## One-time setup

1. Install dependencies into the venv:
   ```bash
   .venv/bin/pip install -r requirements.txt
   ```

2. Add the keys you need to `.env`. Only `TAVILY_API_KEY` and `ANTHROPIC_API_KEY` are needed for a Tavily-only run. Add others as you bring them online.
   ```
   TAVILY_API_KEY=...
   ANTHROPIC_API_KEY=...
   HUGGINGFACE_TOKEN=...
   PERPLEXITY_API_KEY=...
   EXA_API_KEY=...
   PARALLEL_API_KEY=...
   ```

3. Download the benchmark datasets to local parquet (run once):
   ```bash
   python load-datasets.py
   ```
   Files land in `data/`. The pipeline reads from there and never modifies them.

## Running an eval

### Single provider

```bash
python run.py --provider tavily --benchmark sealqa_seal0 --model mini --limit 10
python run.py --provider perplexity --benchmark sealqa_seal_hard --model sonar-pro --limit 25
python run.py --provider exa --benchmark finsearchcomp --limit 5 --seed 42
```

Flags:

| flag | meaning |
| --- | --- |
| `--provider` | `tavily`, `perplexity`, `exa`, `parallel` |
| `--benchmark` | `sealqa_seal0`, `sealqa_seal_hard`, `sealqa_longseal`, `finsearchcomp` |
| `--model` | provider-specific model id; defaults to the provider's default |
| `--limit` | how many questions to run |
| `--workers` | parallel workers (default 4) |
| `--seed` | shuffle before applying `--limit`. Use the same seed across providers for fair comparison |
| `--judge-model` | Anthropic model for extract+grade (default `claude-haiku-4-5`) |
| `--note` | label saved with the run |

### Compare multiple providers on the same questions

```bash
python compare.py --benchmark sealqa_seal0 --limit 20 --seed 42 \
    --providers tavily:mini,perplexity:sonar-pro,exa:exa,parallel:core
```

This launches one run per provider sequentially, all sharing the same seed + limit + `comparison_set` id, so every provider answers the identical N questions. The dashboard groups them automatically.

## Dashboard (launching + inspecting)

```bash
streamlit run app.py
```

Three tabs:

1. **Launch run.** Pick a benchmark, see which `q_index` ranges have already been covered (by which provider/model/seed), pick the next offset+limit, choose one or more providers + models, click Launch. Runs spawn as detached subprocesses and survive Streamlit reloads. The in-flight panel shows live progress (rows landing in the DB).
2. **Single run inspector.** Pick any run, see headline accuracy, filter rows by correct/incorrect/errors, search question text, drill into one question for the full research report and sources.
3. **Provider comparison.** Pick a comparison set, see the per-provider accuracy matrix at the top, then a per-question grid with one column per provider (✅/❌ plus extracted answer). Filters include "Tavily wins (unique)", "Tavily loses (unique)", "any disagreement", "all correct", "all wrong". Click into one question to see every provider's full report side by side.

The CLIs (`run.py`, `compare.py`) still work; they're equivalent to launching from the UI. Logs from UI-launched runs land in `logs/`.

If you prefer raw SQL, `results.db` is plain SQLite. Tables: `runs`, `results`.

## How grading works

For each question:

1. The provider runs research (submit + poll for Tavily/Parallel, single sync call for Perplexity/EXA) and returns content plus sources.
2. Claude Haiku gets the question, the expected answer, and the full report. It extracts a short final answer and grades it against the expected one, with format-tolerant matching (`"12 players"` matches `"12"`, `"$1.2B"` matches `"1.2 billion"`).
3. Question, expected, full report, sources, extracted answer, correct/incorrect, confidence, and reasoning all land in `results.db`.

## File map

| file | role |
| --- | --- |
| `load-datasets.py` | Bootstrap. Downloads HF datasets to `data/*.parquet`. |
| `run.py` | CLI for a single-provider run. |
| `compare.py` | CLI that fans the same questions across multiple providers, sharing one `comparison_set`. |
| `app.py` | Streamlit dashboard. Single-run + provider-comparison tabs. Read-only. |
| `benchmarks/` | Library. |
| `benchmarks/providers/` | One file per provider, plus a shared `ResearchProvider` ABC and registry. |
| `data/` | Source parquet files. Read-only. |
| `results.db` | SQLite output. Created on first run. |

## Adding a new provider

1. Create `benchmarks/providers/myprovider.py` with a class that subclasses `ResearchProvider` and implements `run(question, model) -> ProviderResult`.
2. Register it in `benchmarks/providers/__init__.py`.
3. Add its API key var to `benchmarks/config.load_env`.

That's it. `run.py --provider myprovider ...` and `compare.py --providers ...,myprovider:somemodel` work without further changes.

## Adding a new benchmark

1. Add a download function to `load-datasets.py`.
2. Add a `BenchmarkSpec` entry to `REGISTRY` in `benchmarks/datasets.py` naming the question and answer columns in the parquet.
