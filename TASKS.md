# Improvement tasks

Ordered for safety. Smallest, most-isolated changes first. Each task gets its own commit. Each commit verified to parse and (where applicable) the app still loads before moving on.

The big modularization (#11) is intentionally last so all the small refactors land first and reduce its blast radius.

## Status

- [X] **1. Backup helper script** — `tools/backup_db.py` snapshots `results.db` to `backups/results.db.bak.{timestamp}`. New file; touches nothing existing. **DONE**
- [X] **2. Log pruning at startup** — keep last N logs in `logs/`, prune the rest. Called once on app boot. **DONE**
- [X] **3. Inspector `Graded` metric** — surface graded count in the metric grid, not just the tooltip. One-line UI tweak. **DONE**
- [X] **4. Wire `sealqa_seal_hard` and `sealqa_longseal` into Dashboard + Insights** — generalize the `_sealqa_seal0_*` helpers to accept a benchmark parameter, register all three variants. Taxonomy CSV still only applies to seal0. **DONE**
- [X] **5. Deduplicate dimension/matrix helpers** — pulled into `benchmarks/dimensions.py`. Dashboard, Insights, and Export all import the shared module instead of carrying their own copies. Behavior unchanged. **DONE**
- [X] **6. Cache `get_question_status`** — `ui/cache.py` exposes `question_status(benchmark)` with a 10s TTL. Launch / Tier / Dashboard / Insights tabs all route through it. **DONE**
- [X] **7. Cost preview in $** — `model_costs.json` (optional, committed) drives an `Est. cost` metric in the Launch tab and confirm dialog. Missing providers/models flagged in the tooltip. **DONE**
- [x] **8. Compare tab bug: multiple Tavily models collapse** — `qi_to_correctness` keys by `provider` not `(provider, model)`. Fix so `tavily:mini` + `tavily:pro` in one comparison set both show. Tavily pivot logic also updated.
- [x] **9. Insights diff view** — side-by-side comparison of two stored insight generations.
- [x] **10. Surface `graded` in Provider comparison summary** — minor cleanup uncovered while reviewing #8.
- [ ] **11. Modularize `app.py`** — move Launch / Inspect / Compare / Tier tabs into `ui/tabs/`. Mirror the existing Dashboard/Insights/Export pattern. Done one tab per commit so each step is reversible.

## Conventions for this batch

- Verify with `python -c "import ast; ast.parse(open(X).read())"` after every edit.
- Smoke test by importing key modules where possible.
- One task = one commit. Commit message starts with the task number.
- Never rename or drop a column. Never touch `data/`.
- Never spawn a billable code path without a pre-flight env var check.
- After each commit, re-read CLAUDE.md invariants and confirm nothing was violated.
