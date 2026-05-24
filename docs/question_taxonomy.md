# Question taxonomy

Two schemes for labeling benchmark questions. Each question gets one label per scheme.

The goal is diagnostic. When your agent gets a question wrong, slice the failures by these labels and see whether the problem is reasoning or retrieval. The schemes are independent on purpose: a single hard question can be `multi-hop` + `specialized`, and that combination tells you more than either label alone.

The dataset's own `topic`, `freshness`, and `question_types` columns stay untouched. These taxonomies sit alongside them, in `docs/tags/{benchmark}.csv`.

---

## Scheme A — Reasoning hops

**Question it answers:** "If retrieval works, can the agent still get to the answer?"

This is about the shape of the reasoning move after retrieval succeeds. An agent can have perfect search and still fail here because it never decomposed the question, never aggregated the list it found, or never noticed the premise was wrong.

| label | definition | example queries |
| --- | --- | --- |
| **single-hop** | One fact, one lookup. Includes superlatives where the list itself is the answer (`largest`, `first`, `oldest`). | "Who holds the all-time record at the Grammys for the most wins in album of the year?" / "What is the largest wildfire in Northern California last year?" |
| **multi-hop** | Chain two or more facts, or filter a list then count, sum, or intersect. Covers both counting-after-filtering and combining independent lists. | "How many NBA players have scored 60+ points in a regular season game since 2023?" / "How many countries did both Trump (first term) and Biden visit during their presidencies?" |
| **comparative** | Explicitly weigh two or more named things against each other. Kept separate because agents fail on these in a specific way — they answer about one entity and forget the others. | "Which of Apple, Microsoft, and Google had the highest R&D spend in 2024?" / "Between the EU AI Act and California's SB 1047, which imposes stricter disclosure requirements?" |
| **unanswerable** | The question's frame is wrong. No such entity exists, the premise contains a contradiction, or it presupposes something false. The correct response is to push back. | "What is the smallest cube number expressible as a sum of two different positive cubes in two different ways?" (Fermat: no such number) / "Who is the only female artist to top Spotify's most-streamed list in three consecutive years 2020–2022?" (no artist did) |

---

## Scheme B — Retrieval difficulty

**Question it answers:** "How hard is it to find the right source?"

This is about the retrieval surface, not the reasoning. Even a perfect reasoner cannot answer a question whose facts are buried in a paywalled filing or a leaderboard that changed yesterday. Different labels stress different parts of the search stack.

| label | definition | example queries |
| --- | --- | --- |
| **common** | Top Google results, Wikipedia, major news, well-known records. The agent often has it in weights already. | "Who holds the all-time record at the Grammys for the most wins in album of the year?" / "How many airlines are members of IATA?" |
| **specialized** | Lives in a specific authoritative source you have to actually fetch — regulator filings, conference proceedings, niche databases, government reports, non-English sources. Anything else is hearsay. | "What was the acceptance rate for the most recent EMNLP including Findings?" / "According to Vietcombank's exchange rate table on May 1, 2025, what was the cash buying rate for USD?" |
| **fresh** | Rankings, leaderboards, or counts that change continuously. A correct answer last month is wrong this month. Stresses index freshness and recency awareness. | "What is the most recent film to join the top 10 highest-grossing films of all time?" / "How many YouTube music videos have surpassed 7 billion views?" |
| **tricky-phrasing** | Retrieval is fine, but a qualifier or trap is easy to skim past (`exclusively`, `non-American`, `only`, `since 2023`, `before Y`). The agent fires off a search that ignores the qualifier and confidently returns the wrong answer. | "How many of the top 10 highest-paid athletes in the latest Forbes list are professional golfers living in the U.S.?" / "How many books has Jamie Oliver published since 2020?" |

---

## Applying these to a new dataset

When you add a benchmark (`BenchmarkSpec` in `benchmarks/datasets.py`):

1. Read a sample of at least 50 questions and check the schemes still apply. Public QA datasets skew toward one mode. FinSearchComp is mostly `single-hop` + `common`. SealQA-Seal0 is mostly `multi-hop` + `fresh`.
2. If a scheme doesn't carry signal (everything labels the same way), drop it for that dataset rather than forcing labels.
3. Tag every question into `docs/tags/{benchmark}.csv` with columns `q_index, reasoning, retrieval, notes, question`. The notes column is a 4-to-12-word hint a human auditor can scan, not a full explanation.
4. Run a baseline (Tavily mini at ~20 questions per group is enough) and confirm accuracy varies meaningfully across labels. If a label has uniform accuracy, it isn't separating anything useful and you should drop it.

---

## Caveats

- Some questions sit on a boundary. "Current age of the last American experimental Nobel physicist" is `multi-hop` (filter + lookup + compute age). When in doubt, pick the dominant move and note the alternative in the `notes` column.
- The schemes are orthogonal but not statistically independent. `unanswerable` questions are often also `tricky-phrasing` because the trap is the false premise itself. That's expected.
- The judge (Claude Haiku) sees the same expected answer regardless of label, so labels don't influence grading. They only inform the failure analysis in the dashboard.
- If your first run shows `tricky-phrasing` failures look genuinely different from `specialized` and `fresh` failures, pull traps out into a third scheme. Start lean and add complexity only when the data demands it.
- These schemes are tuned for short-answer benchmarks (SealQA, FinSearchComp). Synthesis-style questions ("compare the AI strategies of Microsoft, Google, and Meta from 2023 to 2025") don't slot cleanly into Scheme A — extend Scheme A rather than overloading `multi-hop`.

---

## Where labels live

- This file: scheme definitions, examples, doc.
- `docs/tags/{benchmark}.csv`: per-question labels.
- Not in `data/` (kept immutable) and not in `results.db` (yet). If labels prove useful for slicing dashboard accuracy, wire a `question_tags` table into storage.

---

## Classification prompt

Drop the block below into a Claude session in this repo. Replace `{BENCHMARK}` with the benchmark name from `benchmarks.datasets.REGISTRY` (e.g. `sealqa_seal_hard`, `finsearchcomp`). Claude will read the parquet, classify every question, and write `docs/tags/{BENCHMARK}.csv`. It must not modify the parquet or `results.db`.

```
You are classifying questions in the benchmark `{BENCHMARK}` using the taxonomy defined in docs/question_taxonomy.md. Read that file first if you have not already.

Source of truth: the parquet at data/{BENCHMARK}.parquet. The column mapping for question/answer is in benchmarks/datasets.py REGISTRY[{BENCHMARK}].

Rules:
- Do not modify data/ or results.db. Read-only.
- Output a single CSV at docs/tags/{BENCHMARK}.csv with the columns:
    q_index, reasoning, retrieval, notes, question
  Match q_index to the row position after applying NO seed (original dataset order). Do not shuffle.
- One row per question. Every q_index in the dataset must appear exactly once.

Label values:
- reasoning ∈ {single-hop, multi-hop, comparative, unanswerable}
- retrieval ∈ {common, specialized, fresh, tricky-phrasing}

Apply the definitions from docs/question_taxonomy.md exactly. When a question sits on a boundary between two labels for a scheme, pick the dominant move and mention the alternative in the notes column.

The notes column is a 4-12 word hint that helps a human auditor recognize the question's classification at a glance. Not a full explanation. Examples:
- "NBA 60+ point games since 2023"
- "Forbes top-10 athletes filtered by golfer + US-resident"
- "Fermat: cube cannot be sum of two cubes"

If a scheme does not carry signal for this dataset (for example, every row is `common` because the dataset is not adversarial), still emit the label, but note this at the end of your reply so the human can decide whether to drop the scheme for this dataset.

After writing the CSV, print:
1. The path you wrote to.
2. A per-scheme distribution table (label, count, % of total).
3. A short note flagging any 3-5 boundary calls you found hardest, so the human can audit those rows first.
```

The output CSV is human-reviewable in any spreadsheet. Edit cells in place if you disagree with a label; the file is the source of truth from that point on.