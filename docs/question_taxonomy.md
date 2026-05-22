# Question taxonomy

Three orthogonal schemes for labeling benchmark questions. Each scheme surfaces a different failure mode, so a single question gets one label per scheme.

The goal is diagnostic: when a provider gets a question wrong, you can slice the failures by these labels and see whether the problem is reasoning, retrieval, or instruction-following.

The dataset's own `topic`, `freshness`, and `question_types` columns are kept untouched. These taxonomies sit alongside them, in `docs/tags/{benchmark}.csv`.

---

## Scheme A — Reasoning structure

"Where does the model struggle to *think*?" When the right facts are findable, did the model reason over them correctly?

| label | definition | example |
| --- | --- | --- |
| **lookup** | one fact from a canonical source; includes "what is X" superlatives where the list itself is the answer | q41 (longest border by length), q48 (valency in HgCl), q73 (largest card processor) |
| **aggregate** | count, sum, compute, or pick max/min from a filtered list | q15 (videos >29M likes), q22 (lowest-grossing $150M+ film), q58 (mortality decline 2010–2021) |
| **multi-set** | combine two or more lists (intersect, subtract, both-of) | q12 (top-20 by non-football accounts also in overall top-20), q38 (countries both Trump and Biden visited) |
| **refuse** | the question's frame is wrong; correct answer is to push back | q35, q43, q44, q68, q79 |

---

## Scheme B — Knowledge access

"Where do the *search results* fail us?" When the model is wrong, was the right info even retrievable from open web sources?

| label | definition | example |
| --- | --- | --- |
| **mainstream** | well-indexed on the open web, top-of-Google content | NBA scoring records, Grammys, Hollywood box office, IATA membership |
| **authoritative** | answer lives in a specific named source (UN, WHO, EPA, FAO, Forbes, government agency report) | q54 (FAO rice ranking), q57–59 (UN parliamentary seats), q86 (EDGAR emissions) |
| **niche** | thin web coverage of the entity itself; long-tail | q31, q33 (Sahitya Akademi), q49 (Polish smoke grenades), q60 (DC 911 specific date), q110 (Akhil Bharatiya Marathi Sahitya Mahamandal) |
| **live-state** | rankings or leaderboards that change frequently | q11–14 (Instagram/YouTube leaderboards), q70 (most-followed agency on X), q93–94 (record-holding posts) |
| **non-english** | answer requires reading non-English source material | q63 (Vietnamese fuel pricing), q65 (Vietnamese YouTube video), q95 (KOSPI 1981) |

---

## Scheme C — Trap density

"The model *should* have gotten this. What made it slip?" Surfaces instruction-following and adversarial robustness.

| label | definition | example |
| --- | --- | --- |
| **clean** | no qualifiers that change the answer | q41 (longest border), q48 (valency), q73 (largest card processor) |
| **time-bound** | temporal qualifier carries weight: "since X", "before Y", "during last year" | q5 (books since 2020), q34 (offices since Jan 2022), q50 (Brad Pitt before COVID) |
| **qualifier-trap** | stacked filters OR a single adversarial word easy to miss | q49 ("*exclusively* from the Soviet Union"), q70 ("*non-American*"), q12 (multi-condition filter) |
| **false-premise** | question's frame is wrong; the correct answer is refusal | q35, q36, q43, q44, q68, q69, q79 |

---

## Applying these to a new dataset

When you add a benchmark (`BenchmarkSpec` in `benchmarks/datasets.py`):

1. Read a sample of questions (50+) to verify the schemes still apply. Public QA datasets often skew toward one mode (FinSearchComp is mostly `lookup`+`mainstream`, SealQA is mostly `aggregate`+`live-state`).
2. If a scheme doesn't carry signal (e.g., everything is `clean` because the dataset isn't adversarial), drop it for that dataset rather than forcing labels.
3. Tag the questions into `docs/tags/{benchmark}.csv` with columns `q_index, reasoning, knowledge, traps, notes, question`. Notes column is for short labels you'll want to recognize later, not full explanations.
4. Run a baseline (e.g., Tavily mini at 20 questions per group) to confirm the tags carry signal — i.e., accuracy varies meaningfully between labels. If a label has uniform accuracy, it's not separating anything useful.

---

## Caveats

- Some questions sit on a boundary. For example, q3 ("current age of the last American experimental Nobel physicist") could be `multi-set` (filter + lookup + compute) or `aggregate`. Pick the dominant move and note the alternative in the `notes` column.
- The schemes are orthogonal but not independent. False-premise questions almost always also test `refuse` under Scheme A. That's fine.
- The judge layer (Claude Haiku) sees the same answer regardless of label, so labels don't influence grading. They only inform the dashboard's failure analysis.

---

## Where labels live

- This file: scheme definitions, examples, doc.
- `docs/tags/{benchmark}.csv`: per-question labels.
- Not in `data/` (kept immutable) and not in `results.db` (yet). If labels prove useful, wire a `question_tags` table into storage so the dashboard can slice accuracy by tag.

---

## Classification prompt

Drop the block below into a Claude session in this repo. Replace `{BENCHMARK}` with the benchmark name from `benchmarks.datasets.REGISTRY` (e.g. `sealqa_seal_hard`, `finsearchcomp`). Claude will read the parquet, classify every question, and write `docs/tags/{BENCHMARK}.csv`. It must not modify the parquet or `results.db`.

```
You are classifying questions in the benchmark `{BENCHMARK}` using the taxonomy defined in docs/question_taxonomy.md. Read that file first if you have not already.

Source of truth: the parquet at data/{BENCHMARK}.parquet. The column mapping for question/answer is in benchmarks/datasets.py REGISTRY[{BENCHMARK}].

Rules:
- Do not modify data/ or results.db. Read-only.
- Output a single CSV at docs/tags/{BENCHMARK}.csv with the columns:
    q_index, reasoning, knowledge, traps, notes, question
  Match q_index to the row position after applying NO seed (original dataset order). Do not shuffle.
- One row per question. Every q_index in the dataset must appear exactly once.

Label values:
- reasoning ∈ {lookup, aggregate, multi-set, refuse}
- knowledge ∈ {mainstream, authoritative, niche, live-state, non-english}
- traps    ∈ {clean, time-bound, qualifier-trap, false-premise}

Apply the definitions from docs/question_taxonomy.md exactly. When a question sits on a boundary between two labels for a scheme, pick the dominant move and mention the alternative in the notes column.

The notes column is a 4-12 word hint that helps a human auditor recognize the question's classification at a glance. Not a full explanation. Examples:
- "NBA 60+ point games since 2023"
- "Forbes top-10 athletes filtered by golfer + US-resident"
- "Fermat: cube cannot be sum of two cubes"

If a scheme does not carry signal for this dataset (for example, every row is `clean` because the dataset is not adversarial), still emit the label, but note this at the end of your reply so the human can decide whether to drop the scheme for this dataset.

After writing the CSV, print:
1. The path you wrote to.
2. A per-scheme distribution table (label, count, % of total).
3. A short note flagging any 3-5 boundary calls you found hardest, so the human can audit those rows first.
```

The output CSV is human-reviewable in any spreadsheet. Edit cells in place if you disagree with a label; the file is the source of truth from that point on.
