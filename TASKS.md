# Improvement tasks

Ordered for safety. Smallest, most-isolated changes first. Each task gets its own commit. Each commit verified to parse and (where applicable) the app still loads before moving on.

The big modularization (#11) is intentionally last so all the small refactors land first and reduce its blast radius.

## Status

- [X] **1. Backup helper script** — `tools/backup_db.py` snapshots `results.db` to `backups/results.db.bak.{timestamp}`. New file; touches nothing existing. **DONE**
- [x] **2. Log pruning at startup** — keep last N logs in `logs/`, prune the rest. Called once on app boot.
- [x] **3. Inspector `Graded` metric** — surface graded count in the metric grid, not just the tooltip. One-line UI tweak.
- [x] **4. Wire `sealqa_seal_hard` and `sealqa_longseal` into Dashboard + Insights** — generalize the `_sealqa_seal0_*` helpers to accept a benchmark parameter, register all three variants. Taxonomy CSV still only applies to seal0.
- [x] **5. Deduplicate dimension/matrix helpers** — pull repeated helpers (`_split_finsearchcomp_label`, `_sealqa_seal0_tags`, `_sealqa_seal0_native_dims`, `_finsearchcomp_dims`, latest-wins matrix builder) into `benchmarks/dimensions.py`. Three files lose copy-paste; behavior unchanged.
- [x] **6. Cache `get_question_status`** — small `@st.cache_data` wrapper (10s TTL) so Launch + Tier tabs don't re-query SQLite on every widget interaction.
- [x] **7. Cost preview in $** — add rough per-call cost lookup, surface estimated $ alongside the runs/research/judge counts in the Launch tab.
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
