# Project context

Internal eval harness owned by the PM of Tavily Research. Used to measure Tavily's research quality on public QA benchmarks and to compare Tavily head-to-head against Perplexity, EXA, and Parallel. Both quality measurement and competitive positioning live here.

Read `README.md` for setup and user-facing flows. This file captures invariants, gotchas, and conventions that aren't obvious from the code. **Read before editing.** The app is in active PM use, so don't refactor for refactoring's sake. If a change looks invasive, ask first.

---

## Architecture invariants

These are load-bearing. Breaking one breaks the app's correctness without obvious test failure.

1. **`data/*.parquet` is read-only.** The pipeline reads it. No code path writes back. All eval state (configs, results, coverage, insights) lives in `results.db`, never in the dataset.

2. **`q_index` identity is scoped by seed.** `q_index 5` only means the same question across two runs when both used the same `seed` (or both used no seed). Coverage queries, comparison-set joins, the Launch coverage table, the Tier matrix, the Dashboard analytics, the Insights payload, and the Export CSV all filter by seed for this reason. Don't compare q_indices across different seeds.

3. **`comparison_set` is a UUID stamped on every run launched together.** `compare.py` and the Streamlit launcher set it when multiple runs go out as one batch. A `comparison_set` plus a shared `seed` is what makes side-by-side comparison honest. Single `run.py` runs leave it `NULL`.

4. **Latest run wins per `(provider, model, q_index)`.** Whenever a question gets answered more than once at the same seed, the most recent `run_id` is the canonical row. The Launch coverage table, the Tier tab roster + matrix, the Dashboard analytics, the Insights matrix, and the Export tab's "latest only" mode all dedupe this way (sort by `run_id` ASC, then `drop_duplicates(keep="last")`). If you add a new view that aggregates across runs, follow this convention or you will double-count retries.

5. **Accuracy denominator = `graded`, not `total`.** `graded` counts rows where `is_correct IN (0, 1)`. Errored and ungraded rows are excluded from both numerator and denominator. Surface `n_total`, `n_errored`, `n_ungraded` separately so the infra picture stays visible, but never put errors into the accuracy fraction. Header labels, headline bars, slice bars, the Tier roster, and the Insights `overall` table all follow this. If you compute accuracy as `correct / total`, you have a bug.

6. **`MIN_N_DISPLAY = 10` is the floor for showing a `(provider, model)` row in analytics.** Defined in `ui/dashboard_charts.py` and mirrored in `benchmarks/insights.py`. Models with fewer than 10 graded responses are filtered out of the Dashboard headline, the coverage heatmap, the SealQA panel, and the Insights synthesis. Don't lower this without a deliberate reason; sparse rows make latency percentiles and slice accuracies meaningless.

7. **Storage migrations live in `_migrate(conn)`** in `benchmarks/storage.py`. Use `ALTER TABLE ... ADD COLUMN` with a sensible default for back-compat. Indexes on new columns must be created inside `_migrate`, not in the top-level `SCHEMA`, so older DBs migrate cleanly. Existing rows must continue to load.

8. **Provider abstraction.** Every vendor subclasses `ResearchProvider` (`benchmarks/providers/base.py`), implements `run(question, model, **kwargs) -> ProviderResult`, declares `name`, `env_var`, `default_model`, `available_models`, and registers itself in `PROVIDERS` in `benchmarks/providers/__init__.py`. The runner does not know provider specifics. Adding a vendor = one file + one registry line + one entry in `load_env`.

9. **Pre-flight key check before any billable call.** `compare.py` and the Streamlit Launch tab both validate every required env var (provider + judge) before launching anything. The Insights tab also requires `ANTHROPIC_API_KEY` and raises `InsightsError` before any API call. New code paths that spend API credits must do the same. Never start a run that's going to error out per-row on missing auth.

10. **`.ui_state.json` is widget state, not eval state.** Lives at repo root, gitignored, persists `launch_*` and `tier_*` widget keys across Streamlit reloads. Safe to delete at any time — the app re-creates it on the next interaction. Never put data that needs to survive a wipe in this file; it goes in `results.db`.

11. **Backups are gitignored snapshots.** `backups/results.db.bak.*` are manual snapshots taken before risky migrations. They are intentionally not in git (`.gitignore` matches `results.db.bak.*`). Take one with `python tools/backup_db.py` (optionally `--label preinsights` for context in the filename). Don't add a code path that writes here automatically; this is a human-driven safety net.

---

## Tab map (what code owns what UI surface)

`app.py` is a thin entrypoint — config, DB init, log prune, tab dispatch. Every tab lives in `ui/tabs/<name>.py` and exposes a `render()` function. New tabs go in the same place.

| tab | code | reads | writes |
| --- | --- | --- | --- |
| Launch run | `ui/tabs/launch.py` | `results.db`, parquet, `.ui_state.json`, `model_tiers.json`, `model_costs.json` | spawns subprocesses via `benchmarks/launcher.py` |
| Single run inspector | `ui/tabs/inspect.py` | `results.db`, parquet | nothing |
| Provider comparison | `ui/tabs/compare.py` | `results.db` (comparison_set rows) | nothing |
| Tier analysis | `ui/tabs/tier.py` | `results.db`, `model_tiers.json`, `.ui_state.json` | nothing |
| Dashboard | `ui/tabs/dashboard.py` + `ui/dashboard_charts.py` | `results.db`, parquet, `docs/tags/sealqa_seal0.csv` | nothing |
| Insights | `ui/tabs/insights.py` + `benchmarks/insights.py` | `results.db`, parquet, `docs/tags/sealqa_seal0.csv` | `results.db` `insights` table (LLM-generated) |
| Export data | `ui/tabs/export.py` | `results.db`, parquet, `docs/tags/sealqa_seal0.csv` | downloads CSV (nothing persistent) |

Shared helpers under `ui/` (Streamlit-aware): `state.py` (widget persistence), `format.py` (formatters), `tiers.py` (model_tiers.json loader + tier_run_data), `data.py` (parquet readers), `costs.py` (model_costs.json loader + estimate), `cache.py` (10s TTL wrappers around hot DB reads), `dashboard_charts.py` (Altair builders).

Pure (Streamlit-free) helpers under `benchmarks/`: `dimensions.py` (label parsing + latest-wins matrix builder).

### Tab intent in one sentence each

- **Launch** — cherry-pick mode for new evals. Tick rows in the dataset table, tick providers/models, see live cost + overlap warnings, confirm dialog, detached subprocesses with a refreshing in-flight panel.
- **Single run inspector** — drill into one run's per-question results, judge reasoning, full research report, sources.
- **Provider comparison** — side-by-side matrix across providers for one comparison_set, with a Tavily pivot for unique wins/losses.
- **Tier analysis** — group provider:model pairs by tier (fast/heavy) defined in `model_tiers.json`, then show a roster + matrix + drill view restricted to those members.
- **Dashboard** — per-benchmark analytics: headline ranked bar, taxonomy slices, latency views, error-vs-wrong, easy questions Tavily missed, coverage heatmap. Tavily-centric color scheme.
- **Insights** — two-stage LLM analysis (Haiku enriches each wrong example, Sonnet synthesizes ranked findings). Persisted history, regenerate button.
- **Export data** — long-format CSV download for Excel / Omni. Joins parquet native columns + taxonomy CSV when applicable.

---

## Provider notes

- **Tavily:** async submit + poll via `get_research(request_id)`. Verified.
- **Perplexity:** sync chat-completions endpoint. `citations` at top level.
- **EXA:** sync `/answer` endpoint. `citations` field.
- **Parallel:** task-runner at `https://api.parallel.ai/v1/tasks/runs`. Endpoint shapes match public docs at time of writing but have not been verified against a live key. Expect to tweak `benchmarks/providers/parallel.py` on first real call.

Models per provider are declared in `available_models` on each provider class. They surface automatically in the Launch tab and in `model_tiers.json` membership.

---

## Judge

Anthropic Claude Haiku 4.5 via tool use (`record_grade` tool). One call does both extraction and grading. Format-tolerant matching is the whole point of the judge (`"12 players"` matches `"12"`, `"$1.2B"` matches `"1.2 billion"`). Don't replace it with regex; you'll lose the tolerance and silently regress accuracy.

The judge's `reasoning` text is one line of rationale. It's stored on the `results` row and surfaced in the Inspector drill view. Don't conflate it with the SealQA taxonomy's `reasoning` column — the Export tab specifically drops the judge column before joining the taxonomy CSV to avoid a name collision.

---

## Insights pipeline

`benchmarks/insights.py` is a two-stage LLM workflow that produces structured Tavily-centric findings. Persisted to the `insights` table; rendered by `ui/tabs/insights.py`.

- **Stage 1 (enrichment, Haiku):** reads each wrong-answer example and tags it with `failure_mode` (closed enum in `FAILURE_MODES`), `source_type_needed` (free-form, must be specific), and a one-sentence `diagnosis`. Cheap, parallel-friendly, structured tool call.
- **Stage 2 (synthesis, Sonnet):** receives the dimensional accuracy tables + enriched wrong examples + failure-mode counts. Produces `headline` + ranked `insights[]` where each item is `{claim, evidence, examples, action, kind}` and `kind ∈ {gap, win, infra}`.

Per-benchmark payload builders live in `BENCHMARK_PAYLOADS` (currently `finsearchcomp`, `sealqa_seal0`, `sealqa_seal_hard`, `sealqa_longseal`, `deepsearchqa`). The three SealQA variants share one generalized builder (`_build_sealqa_payload`) bound to each name via `_sealqa_payload_for`. Adding a new benchmark to insights = one payload function + one registry line. The payload must include `overall`, at least one `by_<dim>` slice, ideally a `<dim_a>_x_<dim_b>` crosstab, and `wrong_examples`. If the dims aren't already in the keep-list of `_enrich_wrong_examples`, add them there so Haiku sees the tags in the wrong-example trim.

`PROMPT_VERSION` is stamped onto every saved insight so when the prompt changes, old runs stay traceable. Bump it whenever you materially change either system prompt or the payload shape.

---

## Model tiers

`model_tiers.json` at repo root maps a tier name to a list of `[provider, model]` pairs. Pure UI overlay — never persisted into `results.db`, never affects how runs are launched, only how they're grouped in the Tier and Launch tabs.

- Edit freely. The file is committed to git.
- Members are `[provider, model]` pairs. A model can appear in multiple tiers, but the first tier (in JSON-key order) wins for column ordering in the Launch table.
- Missing or malformed file = tier-aware UI controls silently hide. The rest of the app keeps working.
- Loaded once per session via `_load_tiers()` in `app.py`; cached in `st.session_state["_tiers_cache"]`. Restart Streamlit (not just rerun) to pick up an edit mid-session.

---

## Question taxonomy

Two schemes documented in `docs/question_taxonomy.md`:

- **Scheme A (reasoning):** `single-hop`, `multi-hop`, `comparative`, `unanswerable`.
- **Scheme B (retrieval):** `common`, `specialized`, `fresh`, `tricky-phrasing`.

Per-benchmark labels live in `docs/tags/{benchmark}.csv` with columns `q_index, reasoning, retrieval, notes, question`. Currently populated for `sealqa_seal0` (full coverage of all 110 questions) and `deepsearchqa` (partial coverage: only the q_indices that have been answered, generated via `python tools/classify_questions.py deepsearchqa`). Re-run that script after answering more questions to extend the CSV.

**Anchoring rule:** the tags CSV is anchored to the parquet's *natural order* (no seed). Every join that uses it must condition on `seed is None`. The Dashboard, Insights, and Export tabs all enforce this; if you wire taxonomy joins into a new view, do the same. With a non-None seed, `q_index` no longer matches the CSV's row indices.

`docs/question_taxonomy.md` ends with a "Classification prompt" block intended to be pasted into a fresh Claude session to label a new benchmark. If you add a benchmark and want taxonomy slices in the Dashboard / Insights / Export tabs, run that prompt to produce the CSV.

---

## Dataset gotcha

`finsearchcomp` has a mostly-null `ground_truth` column. The canonical expected answer is `response_reference` — this is what the dataset's own judge prompt uses, and it's what `REGISTRY` in `benchmarks/datasets.py` points to. The Export tab keeps `ground_truth` as a separate column for completeness, but never grade against it.

`finsearchcomp` also carries a `label` column shaped like `Time-Sensitive_Data_Fetching(Greater China)`. The Dashboard, Insights, and Export tabs all parse this into `(tier, region)` via `_split_finsearchcomp_label`. Tiers are `T1` (time-sensitive), `T2` (simple historical lookup), `T3` (complex investigation). If you add a finance dataset that looks similar, check both the answer column and the label parsing.

The `prompt_id` column on `finsearchcomp` (e.g. `(T2)Simple_Historical_Lookup_001`) is surfaced through the app as a pinned column in the Launch, Inspector, and Compare drill views — easier to talk about a specific question with a stable id than a seed-dependent q_index.

`sealqa_*` benchmarks carry native `topic`, `freshness`, and `question_types` columns (the last is multi-label). The Dashboard and Export tabs join these at the active seed. `question_types` is exploded for per-tag slicing in the Dashboard and joined as a semicolon-separated string in the Export CSV (so Excel doesn't render the array literal).

---

## Selection model

`datasets.load(...)` accepts either `offset + limit` (range mode) or `q_indices=[...]` (cherry-pick mode). `q_indices` overrides offset+limit. `run.py --q-indices "7,20,15"` and `RunConfig.q_indices` propagate it through. The Streamlit Launch tab uses cherry-pick exclusively now; the auto-fill "first uncovered" default was removed deliberately because the user wants explicit per-question control. Don't add it back.

---

## Subprocess detachment

UI-launched runs use `subprocess.Popen(..., start_new_session=True, close_fds=True)` in `benchmarks/launcher.py`. They survive Streamlit hot reloads and browser tab closes. Logs land in `logs/{timestamp}_{provider}_{model}.log`. Don't switch to threading; you'll lose detachment. The Launch tab's "In-flight runs" panel auto-refreshes every 3 seconds via `@st.fragment(run_every=3)` to show progress without forcing a full rerun.

---

## Don't

- Mutate `data/`.
- Drop or rename `runs`/`results`/`insights` columns; only add via migration.
- Add destructive UI (delete run, drop data) without an explicit confirm flow.
- Add a "first uncovered" auto-default back to the Launch tab.
- Replace the LLM judge with string equality or regex.
- Skip the pre-flight key check on any new billable code path.
- Compute accuracy as `correct / total` anywhere — always `correct / graded`.
- Aggregate across multiple runs at the same `(provider, model, q_index)` without latest-wins dedupe.
- Join `docs/tags/sealqa_seal0.csv` against a non-None seed; the CSV is anchored to natural order.
- Lower `MIN_N_DISPLAY` without a written reason in the diff.
- Auto-write to `backups/`; that's a human-driven safety net.
- Put eval state in `.ui_state.json`; that file is widget state only and is gitignored.
- Spawn subprocesses without `start_new_session=True` — they will die on Streamlit reload.

---

## Where to find what

| concern | file |
| --- | --- |
| Provider implementations + protocol | `benchmarks/providers/` |
| Storage schema, migrations, query helpers (runs, results, insights) | `benchmarks/storage.py` |
| Judge prompt and tool schema | `benchmarks/judge.py` |
| Dataset column mapping per benchmark | `benchmarks/datasets.py` (`REGISTRY`) |
| Subprocess launcher for UI runs | `benchmarks/launcher.py` |
| Run config dataclass | `benchmarks/config.py` |
| Run orchestration (load → research → judge → store) | `benchmarks/runner.py` |
| Shared dimension + matrix helpers (label parsing, latest-wins matrix) | `benchmarks/dimensions.py` |
| Streamlit-side caches (`get_question_status` etc.) | `ui/cache.py` |
| Two-stage LLM insights pipeline | `benchmarks/insights.py` |
| Streamlit dashboard entrypoint (thin dispatcher) | `app.py` |
| Launch / Inspect / Compare / Tier tabs | `ui/tabs/launch.py` / `inspect.py` / `compare.py` / `tier.py` |
| Dashboard analytics tab | `ui/tabs/dashboard.py` |
| Altair chart builders (Tavily-centric colors) | `ui/dashboard_charts.py` |
| Insights tab UI | `ui/tabs/insights.py` |
| Export CSV tab | `ui/tabs/export.py` |
| Tier definitions (UI overlay) | `model_tiers.json` |
| Approx per-call pricing for Launch cost preview | `model_costs.json` |
| Widget state persistence (gitignored) | `.ui_state.json` |
| CLI: single run | `run.py` |
| CLI: multi-provider fan-out | `compare.py` |
| CLI: dataset bootstrap (run once) | `load-datasets.py` |
| Question taxonomy doc (2 schemes) | `docs/question_taxonomy.md` |
| Per-benchmark question tags | `docs/tags/{benchmark}.csv` |
| Manual DB snapshots (gitignored) | `backups/` |
| Snapshot helper | `tools/backup_db.py` |

---

## Editing checklist

Before merging any change to this repo:

1. If you touched `runs`, `results`, or `insights` schema — added a column? added the migration in `_migrate`? old DBs still load?
2. If you added a new view that aggregates across runs — latest-wins dedupe? accuracy denominator = graded?
3. If you added a billable code path — pre-flight env var check?
4. If you added a new benchmark — `REGISTRY` entry? loader in `load-datasets.py`? checked which column actually holds the answer?
5. If you added a new provider — `available_models` declared? registered in `PROVIDERS`? key in `load_env`?
6. If you added new UI state you want to persist — added the prefix to `_UI_PERSIST_PREFIXES` in `app.py`?
7. If you changed the insights system prompt or payload shape — bumped `PROMPT_VERSION`?
8. If you touched anything analytical — `MIN_N_DISPLAY` still respected for headline charts?
