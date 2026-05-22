# Project context

Internal eval harness owned by the PM of Tavily Research. Used to measure Tavily's research quality on public QA benchmarks and to compare Tavily head-to-head against Perplexity, EXA, and Parallel. Both quality measurement and competitive positioning live here.

Read `README.md` for setup and user-facing flows. This file captures invariants, gotchas, and conventions that aren't obvious from the code.

## Architecture invariants

1. **`data/*.parquet` is read-only.** The pipeline reads it. No code path writes back. All eval state (configs, results, coverage) lives in `results.db`, never in the dataset.

2. **`q_index` identity is scoped by seed.** `q_index 5` only means the same question across two runs when both used the same `seed` (or both used no seed). Coverage queries, comparison-set joins, and the launch UI's coverage table all filter by seed for this reason. Don't compare q_indices across different seeds.

3. **`comparison_set` is a UUID stamped on every run launched together.** `compare.py` and the Streamlit launcher set it when multiple runs go out as one batch. A `comparison_set` plus a shared `seed` is what makes side-by-side comparison honest. Single `run.py` runs leave it `NULL`.

4. **Storage migrations live in `_migrate(conn)`** in `benchmarks/storage.py`. Use `ALTER TABLE ... ADD COLUMN` with a sensible default for back-compat. Indexes on new columns must be created inside `_migrate`, not in the top-level `SCHEMA`, so older DBs migrate cleanly. Existing rows must continue to load.

5. **Provider abstraction.** Every vendor subclasses `ResearchProvider` (`benchmarks/providers/base.py`), implements `run(question, model, **kwargs) -> ProviderResult`, declares `name`, `env_var`, `default_model`, `available_models`, and registers itself in `PROVIDERS` in `benchmarks/providers/__init__.py`. The runner does not know provider specifics. Adding a vendor = one file + one registry line + one entry in `load_env`.

6. **Pre-flight key check before any billable call.** `compare.py` and the Streamlit launch tab both validate every required env var (provider + judge) before launching anything. New code paths that spend API credits must do the same. Never start a run that's going to error out per-row on missing auth.

## Provider notes

- **Tavily:** async submit + poll via `get_research(request_id)`. Verified.
- **Perplexity:** sync chat-completions endpoint. `citations` at top level.
- **EXA:** sync `/answer` endpoint. `citations` field.
- **Parallel:** task-runner at `https://api.parallel.ai/v1/tasks/runs`. Endpoint shapes match public docs at time of writing but have not been verified against a live key. Expect to tweak `benchmarks/providers/parallel.py` on first real call.

## Judge

Anthropic Claude Haiku 4.5 via tool use (`record_grade` tool). One call does both extraction and grading. Format-tolerant matching is the whole point of the judge (`"12 players"` matches `"12"`, `"$1.2B"` matches `"1.2 billion"`). Don't replace it with regex; you'll lose the tolerance and silently regress accuracy.

## Dataset gotcha

`finsearchcomp` has a mostly-null `ground_truth` column. The canonical expected answer is `response_reference` — this is what the dataset's own judge prompt uses, and it's what `REGISTRY` in `benchmarks/datasets.py` points to. If you ever wire in another finance dataset that looks similar, check which column actually has the answer before picking.

## Selection model

`datasets.load(...)` accepts either `offset + limit` (range mode) or `q_indices=[...]` (cherry-pick mode). `q_indices` overrides offset+limit. `run.py --q-indices "7,20,15"` and `RunConfig.q_indices` propagate it through. The Streamlit launch tab uses cherry-pick exclusively now; the auto-fill "first uncovered" default was removed deliberately because the user wants explicit per-question control.

## Subprocess detachment

UI-launched runs use `subprocess.Popen(..., start_new_session=True, close_fds=True)` in `benchmarks/launcher.py`. They survive Streamlit hot reloads and browser tab closes. Logs land in `logs/{timestamp}_{provider}_{model}.log`. Don't switch to threading; you'll lose detachment.

## Don't

- Mutate `data/`.
- Drop or rename `runs`/`results` columns; only add via migration.
- Add destructive UI (delete run, drop data) without a confirm flow.
- Add a "first uncovered" auto-default back to the launch tab.
- Replace the LLM judge with string equality or regex.
- Skip the pre-flight key check on any new billable code path.

## Where to find what

| concern | file |
| --- | --- |
| Provider implementations + protocol | `benchmarks/providers/` |
| Storage schema, migrations, coverage helpers | `benchmarks/storage.py` |
| Judge prompt and tool schema | `benchmarks/judge.py` |
| Dataset column mapping per benchmark | `benchmarks/datasets.py` (`REGISTRY`) |
| Subprocess launcher for UI runs | `benchmarks/launcher.py` |
| Run config dataclass | `benchmarks/config.py` |
| Streamlit dashboard (3 tabs, single file) | `app.py` |
| CLI: single run | `run.py` |
| CLI: multi-provider fan-out | `compare.py` |
| CLI: dataset bootstrap (run once) | `load-datasets.py` |
| Question taxonomy doc (3 schemes) | `docs/question_taxonomy.md` |
| Per-benchmark question tags | `docs/tags/{benchmark}.csv` |
