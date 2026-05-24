# Research benchmarks

Run public QA benchmarks through one or more research providers (Tavily, Perplexity, EXA, Parallel), grade the answers with Claude Haiku, and inspect, compare, slice, and explain the results in a Streamlit dashboard.

Built for quality measurement and head-to-head positioning of search/research APIs on the same set of questions. The dashboard is Tavily-centric on purpose — colors, pivots, and insight prompts all read the world from Tavily's seat.

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

After a run, the "Single run inspector" tab shows every question, the research report, the extracted answer, and the correct/incorrect grade. The "Dashboard" tab visualizes the same data sliced by taxonomy / tier / region. The "Insights" tab writes a structured Tavily-centric narrative on top.

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

**Tier** — A named group of `[provider, model]` pairs declared in `model_tiers.json` (e.g. `fast`, `heavy`). Pure UI overlay used by the Tier analysis tab and by the Launch tab's column ordering. Tiers never affect which runs are launched, only how they're grouped.

**`q_index`** — A question's position within the (possibly shuffled) dataset. `q_index 5` only means the same question across two runs **when both runs used the same `seed`**. Always reuse one seed when you want apples-to-apples comparison; that's what `compare.py` does automatically.

**Latest run wins.** When the same `(provider, model, q_index)` is run multiple times at the same seed (e.g. a retry after a rate-limit), the most recent `run_id` is the canonical row everywhere except the Single run inspector (which scopes to one chosen run). The Launch coverage table, Tier roster, Dashboard analytics, Insights matrix, and the Export tab's "latest only" mode all dedupe this way.

**`graded`** — Rows the judge could decide on (`is_correct IN (0, 1)`). Accuracy is always `correct / graded`, never `correct / total`. Errored and ungraded rows are surfaced separately so the infra picture stays visible, but they never drag the accuracy rate down.

**Judge** — Claude Haiku 4.5 via Anthropic tool use. It receives the question, the expected answer, and the provider's full research report. It then extracts the short final answer from the report and grades it against the expected answer with format-tolerant matching (`"12 players"` matches `"12"`, `"$1.2B"` matches `"1.2 billion"`, `"Šerban Ghenea"` matches `"Serban Ghenea"`).

**Insights** — A separate LLM pipeline that reads the dimensional accuracy tables + a sample of wrong-answer examples and produces a ranked list of Tavily-centric findings. Two stages: Haiku enriches each wrong example with a structured `failure_mode + source_type_needed + diagnosis`, Sonnet synthesizes the final `headline + insights[]`. Persisted in `results.db`.

**Taxonomy** — Two question-level labeling schemes (reasoning hops, retrieval difficulty) documented in `docs/question_taxonomy.md` and stored per-benchmark in `docs/tags/{benchmark}.csv`. Currently populated for `sealqa_seal0` only. Used by the Dashboard, Insights, and Export tabs as slice dimensions.

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
   `ANTHROPIC_API_KEY` is used both by the judge and by the Insights pipeline. Missing keys are caught before any billable call.

3. Download datasets (writes to `data/`, idempotent, only need to run once):
   ```bash
   python load-datasets.py
   ```

4. (Optional) Define tier groupings in `model_tiers.json` if you want the Tier analysis tab and the Launch tab's tier-column filter. Shape:
   ```json
   {
     "fast":  [["tavily", "mini"], ["perplexity", "sonar-reasoning-pro"], ["exa", "deep"], ["parallel", "core"]],
     "heavy": [["tavily", "pro"],  ["perplexity", "sonar-deep-research"], ["exa", "deep-reasoning"], ["parallel", "pro"]]
   }
   ```
   The file is committed to git. Missing or malformed file hides every tier-aware control without breaking the rest of the app.

---

## Dashboard

```bash
streamlit run app.py
```

Seven tabs.

### 1. Launch run

Cherry-pick mode for new evals.

1. Pick a benchmark (segmented control at the top).
2. (Optional) seed, max-per-batch cap, worker count, free-form note.
3. The dataset table below shows every row: `#`, optional `prompt_id` (finsearchcomp), `question`, `expected`, then one column per `provider:model` that has ever run this benchmark for the current seed.
   - ✅ correct
   - ❌ incorrect
   - ⚠ error (research or judge failed)
   - blank = not run
4. Toggles on top of the table:
   - **Show answer and duration in coverage cells** — adds the extracted answer + run duration to each cell. Off keeps just the symbol so more provider columns fit.
   - **English only** — hides questions containing CJK characters. Useful for finsearchcomp which mixes English and Chinese prompts. `q_index` numbering is preserved.
   - **Tier columns** (only when `model_tiers.json` exists) — restrict the visible provider columns to one tier's members.
5. Click row checkboxes to pick exactly which questions to run. The selection is capped at "Max per batch".
6. Tick provider models in the matrix below the table. Each ticked model becomes one run.
7. Cost preview and overlap warning update live (overlap = questions some of your selected providers have already covered).
8. Click **Launch** to open a confirm dialog with the final cost preview and any overlap warning. Confirm to spend credits.

Each `(provider, model)` becomes its own detached subprocess that survives Streamlit hot reloads and browser tab closes. Logs land in `logs/{timestamp}_{provider}_{model}.log`. The "In-flight runs" panel at the bottom auto-refreshes every 3 seconds and shows live row counts, correct counts, error counts, and elapsed time.

UI state (seed, last-used providers, toggles, tier filter) persists across reloads via a gitignored `.ui_state.json`. Safe to delete; the app will recreate it.

### 2. Single run inspector

Pick any run from the table, see headline accuracy (`correct / graded`) and the per-question table. Filter by correct/incorrect/errors, search question text, drill into one question to read the full research report and the cited sources. The Compare tab can preselect a run here via the "Open in inspector" button.

### 3. Provider comparison

Pick a comparison set (made via `compare.py` or by selecting multiple providers in the Launch tab). At the top: provider accuracy + average duration matrix, sorted by accuracy. Below: a per-question grid with one column per provider showing ✅/❌ and the extracted answer.

Filters:
- **Filter** — `all`, `any disagreement`, `all correct`, `all wrong`.
- **Tavily pivot** (when Tavily is in the set) — show only questions where Tavily is **uniquely right** or **uniquely wrong** vs every other provider. Overrides the left-hand filter.
- **Search** — substring match on the question text.

Click into one question to see every provider's full report, extracted answer, judge reasoning, and sources side by side.

### 4. Tier analysis

Requires `model_tiers.json`. Group `(provider, model)` pairs by tier and compare just those members.

1. Pick a benchmark and a tier (`all` shows every tier member as a union).
2. Seed picker appears only when there's ambiguity (multiple seeds with data).
3. **Roster** — one metric per tier member with accuracy + a help tooltip listing the latest run_id, graded/correct counts, and errored count.
4. **Matrix** — one row per question, one column per `provider:model`. Same Tavily-pivot + filter + search controls as the Compare tab, but pivoting against *all* Tavily models in the tier vs *all* non-Tavily models.
5. **Drill view** — click a matrix row, see side-by-side reports for every tier member.
6. **CSV downloads** — roster + matrix.

The Tier tab is the place to ask "across providers at the same compute tier, who wins on this benchmark?" without inventing arbitrary pairings.

### 5. Dashboard

Per-benchmark analytics. Tavily-centric color scheme: Tavily blue for Tavily models, gray gradient for competitors.

Currently wired for `finsearchcomp` and all three SealQA variants (`sealqa_seal0`, `sealqa_seal_hard`, `sealqa_longseal`). Each panel uses the latest-run-wins matrix at the selected seed, then drops any `(provider, model)` with fewer than 10 graded responses.

Common to both benchmarks:
- **Headline ranked bar** — accuracy per `provider:model`, sorted, with `n` graded.
- **Tavily mini vs pro** — outcome split (both right / pro only / mini only / both wrong) over the questions both have answered, plus expandable tables of the mini-only-right and pro-only-right questions.
- **Latency** — box-and-whisker (p5–p95 whiskers, p25–p75 box, p50 tick), plus a latency-vs-accuracy scatter with median reference lines (upper-left = fast and accurate).
- **Error vs wrong-answer split** — stacked share of correct / wrong / errored per `provider:model`, sorted by accuracy.
- **Easy questions Tavily missed** — questions where ≥2 competitor pairs got it right and no Tavily model did.
- **Coverage heatmap** — success rate (% correct) per `(provider:model × dim_value)` cell.

`finsearchcomp` adds:
- **By task tier** — T1 (time-sensitive) / T2 (simple historical lookup) / T3 (complex investigation), derived from the parquet's `label` column.

`sealqa_seal0` adds:
- **By taxonomy** (only when seed is blank, since `docs/tags/sealqa_seal0.csv` is anchored to natural order):
  - Reasoning: single-hop / multi-hop / comparative / unanswerable.
  - Retrieval: common / specialized / fresh / tricky-phrasing.
- **By dataset-native columns** — topic, freshness, exploded `question_types`.

### 6. Insights

Two-stage LLM analysis layered on top of the Dashboard data.

- **Stage 1 (Haiku):** reads each of ~30 wrong-answer examples and tags it with `failure_mode` (closed enum), `source_type_needed` (specific source description), and a one-sentence `diagnosis`.
- **Stage 2 (Sonnet):** receives the dimensional accuracy tables + enriched wrong examples + failure-mode counts. Produces `headline` + 5–8 ranked `insights` where each item is `{claim, evidence, examples, action, kind}` and `kind ∈ {gap, win, infra}`.

The latest cached insight loads on tab open. **Regenerate** calls Haiku + Sonnet again and persists the new generation in the `insights` table. History is browsable for the active `(benchmark, seed)`. Each insight stores the model pair, prompt version, and token usage so older generations stay traceable when the prompt evolves.

Requires `ANTHROPIC_API_KEY`. Benchmarks currently supported: `finsearchcomp`, `sealqa_seal0`, `sealqa_seal_hard`, `sealqa_longseal`. The three SealQA variants share a generalized payload builder; only `sealqa_seal0` has a taxonomy CSV (`docs/tags/sealqa_seal0.csv`), so the reasoning/retrieval slices only appear there.

### 7. Export data

Long-format CSV download for Excel / Omni / any downstream analysis tool. One row per `(run, q_index)` by default; toggle "Latest run only per (provider, model, question)" to dedupe to one row per `(provider, model, q_index)`.

The exporter excludes provider-side error rows entirely (the `error` column is dropped) and omits `research_content` and `research_sources_json` (use the Single run inspector for those).

Joins applied per benchmark:
- **sealqa_seal0** — taxonomy labels (`reasoning`, `retrieval`) from `docs/tags/sealqa_seal0.csv` when the selected seed is blank, plus parquet-native `topic`, `freshness`, `question_types` (semicolon-joined string).
- **sealqa_seal_hard** / **sealqa_longseal** — parquet-native columns where present.
- **finsearchcomp** — parsed `fin_tier` (T1/T2/T3) and `fin_region`, plus `prompt_id` and the raw `ground_truth` column.

Filters: provider/model multi-select, comparison-set filter (only shown when matching runs have one). The export shows row/question/system/column counts before download and a "Column reference" expander explaining every column.

---

## CLI

The CLIs do everything the run-launching UI does and are useful for scripting.

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

`results.db` is plain SQLite at the repo root. Three tables.

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
| `reasoning` | text | judge's one-line rationale (not the SealQA taxonomy column) |
| `error` | text | non-null if research or judge failed |
| `created_at` | text | ISO 8601, UTC |

Constraint: `UNIQUE(run_id, q_index)`. Re-running the same `q_index` within the same `run_id` overwrites the row; across different `run_id`s, history is preserved (and latest-wins dedupe in views uses `run_id` ordering).

**`insights`** — one row per LLM-generated insight.

| column | type | notes |
| --- | --- | --- |
| `id` | int | primary key |
| `benchmark` | text | which dataset the insight was generated for |
| `seed` | int | which seed slice; null matches null |
| `generated_at` | text | ISO 8601, UTC |
| `model` | text | `"{enrichment} → {synthesis}"`, e.g. `claude-haiku-4-5 → claude-sonnet-4-6` |
| `prompt_version` | text | bumped whenever system prompt or payload shape changes materially |
| `content_json` | text | full `{headline, insights[], _meta{...}}` blob |

---

## File map

| file | role |
| --- | --- |
| `load-datasets.py` | One-time HF download bootstrap. |
| `run.py` | CLI for a single-provider run. |
| `compare.py` | CLI that fans the same questions across providers. |
| `app.py` | Streamlit entrypoint; owns Launch / Inspect / Compare / Tier tabs inline. |
| `ui/tabs/dashboard.py` | Dashboard tab. Per-benchmark analytics. |
| `ui/tabs/insights.py` | Insights tab. Renders LLM-generated findings + history. |
| `ui/tabs/export.py` | Export tab. Long-format CSV with parquet + taxonomy joins. |
| `ui/dashboard_charts.py` | Altair chart builders (Tavily blue vs neutral gray palette). |
| `benchmarks/config.py` | `RunConfig` dataclass + `load_env`. |
| `benchmarks/datasets.py` | Parquet loader + per-benchmark `REGISTRY`. |
| `benchmarks/judge.py` | Anthropic judge with structured tool use. |
| `benchmarks/storage.py` | SQLite schema, migrations, query helpers, insights persistence. |
| `benchmarks/runner.py` | Orchestrates load → research → judge → store. |
| `benchmarks/dimensions.py` | Shared per-benchmark dimension helpers + latest-wins matrix builder. Imported by Dashboard, Insights, and Export. |
| `benchmarks/launcher.py` | Subprocess wrapper for UI-launched runs. |
| `benchmarks/insights.py` | Two-stage LLM analysis pipeline (Haiku + Sonnet). |
| `benchmarks/providers/` | One file per provider plus the `ResearchProvider` ABC. |
| `model_tiers.json` | Tier groupings of `[provider, model]` pairs (UI overlay, committed). |
| `.ui_state.json` | Streamlit widget state (gitignored, auto-created). |
| `docs/question_taxonomy.md` | Two labeling schemes for benchmark questions. |
| `docs/tags/{benchmark}.csv` | Per-question taxonomy labels. |
| `data/` | Source parquet files. Read-only. |
| `logs/` | Per-launch stdout/stderr from UI subprocesses. |
| `backups/` | Manual snapshots of `results.db` taken before risky migrations (gitignored). |
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

The CLI, the comparison tool, and every UI tab pick it up automatically. If you want it inside a tier, add `["myprovider", "default"]` to the relevant tier list in `model_tiers.json`.

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
3. (Optional) To unlock the Dashboard analytics + Insights tab, write a payload builder in `benchmarks/insights.py` and register it in `BENCHMARK_PAYLOADS`, then add a panel function + dispatch line in `ui/tabs/dashboard.py`.
4. (Optional) Run the classification prompt at the bottom of `docs/question_taxonomy.md` to generate `docs/tags/my_dataset.csv` for taxonomy slices.

### Add a new tier

Edit `model_tiers.json`. The file is a flat object mapping tier name to a list of `[provider, model]` pairs. Restart Streamlit (not just rerun) to pick up the change mid-session.

### Add a new persistent widget

If you want a Streamlit widget to remember its value across reloads, give its key one of the prefixes in `_UI_PERSIST_PREFIXES` in `app.py`, and call `on_change=_save_ui_state` on the widget. State lands in `.ui_state.json`. Don't put eval data there.

---

## Troubleshooting

**`q_index` confusion across runs.** `q_index` is position in the dataset *after seed shuffle*. Same seed = same question at the same index. Different seeds = different questions at the same index. The Launch coverage, Tier matrix, Dashboard analytics, Insights, and Export all filter by seed for this reason. The taxonomy CSV (`docs/tags/sealqa_seal0.csv`) is anchored to natural order, so the Dashboard and Export tabs only join it when seed is blank.

**`finsearchcomp` shows null answers.** The dataset's `ground_truth` column is mostly null. The pipeline uses `response_reference` (the column the dataset's own judge prompt uses). If you swap to another finance dataset, double-check which column actually holds the answer.

**"Missing API keys" before a launch.** Add the listed keys to `.env`. The Streamlit Launch tab disables the Launch button when keys are missing; `compare.py` errors out before any billable call. The Insights tab raises a visible error before calling either Haiku or Sonnet if `ANTHROPIC_API_KEY` is missing.

**Subprocess runs disappear after Streamlit reload.** They shouldn't — `subprocess.Popen(..., start_new_session=True)` detaches them. Check `logs/` for the run's stdout; if the process is gone, the log will show why.

**"ALTER TABLE no such column" on startup.** The schema migration runs in `_migrate` inside `benchmarks/storage.py`. If you added a column to `SCHEMA` without also adding it to `_migrate`, fresh DBs will work but existing DBs will break. Move the `ALTER TABLE` (and any index on the new column) into `_migrate`.

**Accuracy looks wrong in one view.** Confirm both numerator and denominator: it is always `correct / graded`, never `correct / total`. Errored and ungraded rows are excluded. If you're seeing a denominator that includes errors, that's a bug.

**Cell counts in the Launch coverage header don't match what I see.** The header counts each `(provider, model)` *after* latest-wins dedupe, so a flaky provider that retried the same q_indices many times shows the deduped denominator. This is intentional.

**Dashboard shows fewer models than I expected.** Anything with fewer than 10 graded responses is hidden from headline + slice charts (`MIN_N_DISPLAY = 10` in `ui/dashboard_charts.py`). Run more questions on that model or lower the threshold deliberately.

**Tier tab is empty.** Either no tier members have run yet for this `(benchmark, seed)`, or `model_tiers.json` is missing/malformed. The tab renders an inline example of the expected JSON shape when the file isn't loadable.

**Insights regenerate fails with `InsightsError: No <benchmark> rows at this seed with ≥10 graded runs per model`.** You need at least one `(provider, model)` with 10+ graded responses on this benchmark and seed. Run more, or pick a different seed where you have coverage.

**Widget keeps resetting between reloads.** Check that its key is covered by one of the `_UI_PERSIST_PREFIXES` in `app.py`, and that the widget call passes `on_change=_save_ui_state`.
