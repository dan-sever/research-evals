# Research benchmarks

Run public QA benchmarks through one or more research providers (Tavily, Perplexity, EXA, Parallel), grade the answers with Claude Haiku, and inspect or compare the results in a Streamlit dashboard.

Built for quality measurement and head-to-head comparison of search/research APIs on the same set of questions.

---

## Quick start

```bash
.venv/bin/pip install -r requirements.txt

# Add at minimum TAVILY_API_KEY and ANTHROPIC_API_KEY to .env
echo 'TAVILY_API_KEY=...'    >> .env
echo 'ANTHROPIC_API_KEY=...' >> .env

# One-time: download benchmark datasets to local parquet
python load-datasets.py

# Either launch via UI ...
streamlit run app.py

# ... or run from the CLI
python run.py --provider tavily --benchmark sealqa_seal0 --limit 5
```

After a run, the dashboard's "Single run inspector" tab shows every question, the research report, the extracted answer, and the correct/incorrect grade.

---

## Concepts

**Benchmark** — A QA dataset downloaded from Hugging Face. Four are wired in:

| name | source | size | what it tests |
| --- | --- | --- | --- |
| `sealqa_seal0` | vtllms/sealqa, `seal_0` | 111 | conflicting web sources, freshness |
| `sealqa_seal_hard` | vtllms/sealqa, `seal_hard` | 254 | the hardest SealQA subset |
| `sealqa_longseal` | vtllms/sealqa, `longseal` | 254 | long-context variant |
| `finsearchcomp` | ByteSeedXpert/FinSearchComp | 635 | financial lookups (Chinese + English) |

**Provider** — A research API. Four are wired in: `tavily`, `perplexity`, `exa`, `parallel`. Each declares its own default model and the list of models it accepts.

**Model** — Provider-specific string. `mini`/`pro`/`auto` for Tavily, `sonar-reasoning-pro`/`sonar-deep-research` for Perplexity, `deep-lite`/`deep`/`deep-reasoning` for EXA, `lite`/`base`/`core`/`core-fast`/`pro`/`ultra` for Parallel.

**Run** — One execution of one `(provider, model)` over one batch of questions. Every run gets a numeric `id` and a row in the `runs` table.

**Comparison set** — A UUID stamped on every run launched together. The dashboard groups them in the "Provider comparison" tab so you can see all providers' answers to the same question side by side.

**`q_index`** — A question's position within the (possibly shuffled) dataset. `q_index 5` only means the same question across two runs **when both runs used the same `seed`**. Always reuse one seed when you want apples-to-apples comparison; that's what `compare.py` does automatically.

**Judge** — Claude Haiku 4.5 via Anthropic tool use. It receives the question, the expected answer, and the provider's full research report. It then extracts the short final answer from the report and grades it against the expected answer with format-tolerant matching (`"12 players"` matches `"12"`, `"$1.2B"` matches `"1.2 billion"`, `"Šerban Ghenea"` matches `"Serban Ghenea"`).

---

## Setup in detail

1. Activate the venv, then install:
   ```bash
   source .venv/bin/activate          # macOS / Linux
   # .venv\Scripts\activate           # Windows PowerShell
   pip install -r requirements.txt
   ```
   When you're done working in the project, `deactivate` returns the shell to your system Python. The Quick start commands above use `.venv/bin/pip` / `.venv/bin/python` directly so they work without activation; once activated, you can drop the `.venv/bin/` prefix and just type `pip`, `python`, `streamlit`.

2. Add the keys you have to `.env`. The minimum is `TAVILY_API_KEY` + `ANTHROPIC_API_KEY` for a Tavily-only setup. Add others as you bring providers online.
   ```
   TAVILY_API_KEY=...
   ANTHROPIC_API_KEY=...
   HUGGINGFACE_TOKEN=...
   PERPLEXITY_API_KEY=...
   EXA_API_KEY=...
   PARALLEL_API_KEY=...
   ```
   Missing keys are caught before any billable call.

3. Download datasets (writes to `data/`, idempotent, only need to run once):
   ```bash
   python load-datasets.py
   ```

---

## Dashboard

```bash
streamlit run app.py
```

Three tabs:

### Launch run

Cherry-pick mode for new evals.

1. Pick a benchmark (segmented control at the top).
2. Pick providers (multi-select) and their models (multi-select per provider).
3. Set seed (blank = original dataset order), Max per batch (hard cap, default 5), and worker count.
4. The dataset table below shows every row: `#`, `Question`, `Expected`, then one column per `provider:model` that has ever run this benchmark for the current seed.
   - ✅ correct
   - ❌ incorrect
   - ⚠ error (research or judge failed)
   - blank = not run
5. Click row checkboxes to pick exactly which questions to run. The selection is capped at "Max per batch".
6. Cost preview and overlap warning update live (overlap = questions some of your selected providers have already covered).
7. Tick the confirm checkbox, click **Launch**.

Each `(provider, model)` becomes its own detached subprocess that survives Streamlit hot reloads and browser tab closes. Logs land in `logs/{timestamp}_{provider}_{model}.log`. The "In-flight runs" panel at the bottom shows live row counts.

### Single run inspector

Pick any run from the dropdown, see headline accuracy and the per-question table. Filter by correct/incorrect/errors, search question text, drill into one question to read the full research report and the cited sources.

### Provider comparison

Pick a comparison set (made via `compare.py` or by selecting multiple providers in the Launch tab). At the top: provider accuracy + average duration matrix, sorted by accuracy. Below: a per-question grid with one column per provider showing ✅/❌ and the extracted answer.

Filters:
- **Tavily wins (unique)** — Tavily correct, every other provider in the set incorrect.
- **Tavily loses (unique)** — Tavily incorrect, every other provider correct.
- **Any disagreement** — providers disagreed on this question.
- **All correct** / **All wrong** — full agreement either way.

Click into one question to see every provider's full report and sources side by side.

---

## CLI

The CLIs do everything the UI does and are useful for scripting.

### `python run.py`

One run, one provider, one model.

| flag | meaning |
| --- | --- |
| `--provider` | `tavily`, `perplexity`, `exa`, `parallel` |
| `--benchmark` | one of the dataset names above |
| `--model` | provider-specific model string; defaults to the provider's default |
| `--limit` | number of questions to run |
| `--offset` | skip the first N questions (pairs with `--limit`) |
| `--q-indices` | comma-separated list of exact `q_index` values to run, e.g. `"7,20,15,2,9"`. Overrides `--offset`/`--limit` |
| `--seed` | shuffle the dataset before applying the range. Use the same seed across runs for fair comparison |
| `--workers` | parallel research+judge workers (default 4) |
| `--judge-model` | Anthropic model id (default `claude-haiku-4-5`) |
| `--note` | free-form label saved with the run |
| `--comparison-set` | UUID to group with other runs. Set by `compare.py` and the UI launcher |

Examples:
```bash
python run.py --provider tavily --benchmark sealqa_seal0 --model mini --limit 10
python run.py --provider perplexity --benchmark sealqa_seal_hard --model sonar-reasoning-pro --limit 25
python run.py --provider tavily --benchmark sealqa_seal0 --q-indices "7,20,15,2,9"
```

### `python compare.py`

Fan the same questions across multiple providers in one batch. All runs share a seed, a limit, and a `comparison_set` UUID so the dashboard groups them.

```bash
python compare.py --benchmark sealqa_seal0 --limit 20 --seed 42 \
    --providers tavily:mini,perplexity:sonar-reasoning-pro,exa:deep-lite,parallel:core
```

Pre-flight checks every required env var before running anything.

### `python load-datasets.py`

Downloads the four datasets to `data/*.parquet`. Run once at setup.

---

## Storage

`results.db` is plain SQLite at the repo root. Two tables:

**`runs`** — one row per launch.

| column | type | notes |
| --- | --- | --- |
| `id` | int | primary key |
| `provider` | text | e.g. `tavily` |
| `benchmark` | text | e.g. `sealqa_seal0` |
| `model` | text | provider-specific |
| `limit_n` | int | as passed; null = full set |
| `workers` | int | concurrency |
| `seed` | int | null = original order |
| `judge_model` | text | e.g. `claude-haiku-4-5` |
| `note` | text | free-form |
| `comparison_set` | text | UUID, null = standalone |
| `started_at` | text | ISO 8601, UTC |
| `finished_at` | text | null while in flight |
| `config_json` | text | full `RunConfig` as JSON, includes `q_indices` |

**`results`** — one row per question per run.

| column | type | notes |
| --- | --- | --- |
| `id` | int | primary key |
| `run_id` | int | FK to `runs.id` |
| `q_index` | int | position in (possibly shuffled) dataset |
| `question` | text | full text |
| `expected_answer` | text | from the benchmark |
| `research_status` | text | provider's terminal status |
| `research_content` | text | full report |
| `research_sources_json` | text | normalized `[{title, url}]` list |
| `research_request_id` | text | provider's request id |
| `research_duration_seconds` | real | wall time |
| `extracted_answer` | text | judge's short extraction |
| `is_correct` | int | 1 / 0 / null |
| `confidence` | real | judge's 0..1 confidence |
| `reasoning` | text | judge's one-line rationale |
| `error` | text | non-null if research or judge failed |
| `created_at` | text | ISO 8601, UTC |

Constraint: `UNIQUE(run_id, q_index)`. Re-running the same `q_index` within the same `run_id` overwrites the row; across different `run_id`s, history is preserved.

---

## File map

| file | role |
| --- | --- |
| `load-datasets.py` | One-time HF download bootstrap. |
| `run.py` | CLI for a single-provider run. |
| `compare.py` | CLI that fans the same questions across providers. |
| `app.py` | Streamlit dashboard (Launch + Inspect + Compare tabs). |
| `benchmarks/config.py` | `RunConfig` dataclass + `load_env`. |
| `benchmarks/datasets.py` | Parquet loader + per-benchmark `REGISTRY`. |
| `benchmarks/judge.py` | Anthropic judge with structured tool use. |
| `benchmarks/storage.py` | SQLite schema, migrations, query helpers. |
| `benchmarks/runner.py` | Orchestrates load → research → judge → store. |
| `benchmarks/launcher.py` | Subprocess wrapper for UI-launched runs. |
| `benchmarks/providers/` | One file per provider plus the `ResearchProvider` ABC. |
| `data/` | Source parquet files. Read-only. |
| `logs/` | Per-launch stdout/stderr from UI subprocesses. |
| `results.db` | SQLite output. Created on first run. |

---

## Extending

### Add a new provider

1. Create `benchmarks/providers/myprovider.py` subclassing `ResearchProvider`:
   ```python
   class MyProvider(ResearchProvider):
       name = "myprovider"
       default_model = "default"
       available_models = ("default", "premium")
       env_var = "MY_API_KEY"

       def run(self, question, model=None, **kwargs):
           # Submit, poll if async, return ProviderResult.
           ...
   ```
2. Register it in `benchmarks/providers/__init__.py` under `PROVIDERS`.
3. Add `MY_API_KEY` to the dict in `benchmarks/config.load_env`.

Done. The CLI, the comparison tool, and the UI all pick it up automatically.

### Add a new benchmark

1. Add a download function to `load-datasets.py`.
2. Add a `BenchmarkSpec` entry to `REGISTRY` in `benchmarks/datasets.py`, naming the question and answer columns in the parquet:
   ```python
   "my_dataset": BenchmarkSpec(
       name="my_dataset",
       parquet="my_dataset.parquet",
       question_col="prompt",
       answer_col="expected",
       extra_cols=("topic",),
   ),
   ```

---

## Troubleshooting

**`q_index` confusion across runs.** `q_index` is position in the dataset *after seed shuffle*. Same seed = same question at the same index. Different seeds = different questions at the same index. The dashboard's coverage table filters by seed for exactly this reason.

**`finsearchcomp` shows null answers.** The dataset's `ground_truth` column is mostly null. The pipeline uses `response_reference` (the column the dataset's own judge prompt uses). If you swap to another finance dataset, double-check which column actually holds the answer.

**"Missing API keys" before a launch.** Add the listed keys to `.env`. The Streamlit Launch tab disables the Launch button when keys are missing; `compare.py` errors out before any billable call.

**Subprocess runs disappear after Streamlit reload.** They shouldn't — `subprocess.Popen(..., start_new_session=True)` detaches them. Check `logs/` for the run's stdout; if the process is gone, the log will show why.

**"ALTER TABLE no such column" on startup.** The schema migration runs in `_migrate` inside `benchmarks/storage.py`. If you added a column to `SCHEMA` without also adding it to `_migrate`, fresh DBs will work but existing DBs will break. Move the `ALTER TABLE` (and any index on the new column) into `_migrate`.
