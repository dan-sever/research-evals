# financeqa — Tavily competence and gaps

## Headline

The single biggest gap is multi-step financial calculation: 11 of 24 graded losses are tagged `calculation_error`, and on the dominant `calculation` query type (49 questions), tavily_advanced hits only 16.3% accuracy versus exa_search:auto's 18.4%. The single biggest area of competence is direct-lookup of well-disclosed line items from Costco's filings, where tavily_basic correctly pulled ROE (30.3%) and asset turnover (3.7x) while exa_search:auto failed both.

## Where Tavily wins

- **Direct lookup of clearly stated financial ratios beats exa_search:auto.** On Q65 (ROE 2024), tavily_basic extracted 30.3%, matching the expected 30.27%, while exa_search:auto returned 31.19% and tavily_advanced returned 28.45% using TTM data. On Q80 (Asset Turnover 2024), tavily_basic returned 3.7x (expected 3.67x) while both exa_search:auto and tavily_advanced returned 3.57x, a material miss. All three wins carry `direct_filing_extract` as the success factor.

- **Tavily_advanced wins sole-provider victories on footnote-sourced items.** Q23 (marginal tax rate 2024) is the only question where one Tavily tier wins while exa_search:auto fails: tavily_advanced extracted 24.40% against an expected 24.00%, while exa_search:auto and tavily_basic both returned no answer. The source was the statutory tax rate from the tax footnote, a detail exa_search:auto could not locate.

- **Conceptual finance questions are a shared strength, not a Tavily weakness.** On the `conceptual` query type (7 questions), tavily_advanced matches exa_search:auto exactly at 57.1%. On the `question_type` = conceptual slice (14 questions), both hit 64.3%. Neither provider has a systematic edge here, meaning Tavily is not leaving points on the table relative to competition on knowledge-type questions.

## Where Tavily falls short

- **Calculation errors are the dominant failure mode (11 losses).** Tavily_advanced computed 5.49% EBITDA margin for Q5 (expected 4.53%) by adding EBIT of $11,543M and D&A of $2,426M rather than using the $11,522M unadjusted EBITDA directly. The source that would close this gap is the FY2024 non-GAAP reconciliation table, which states the EBITDA line explicitly rather than requiring a build-up from income statement components.

- **Tavily_basic has a systematic retrieval failure on structured financial data (5 `missing_filing_access` losses).** On Q9 (Operating Profit 2024), Q7 (Operating Profit Margin), Q137 (asset write-down three-statement impact), and Q137, tavily_basic repeatedly returned "cannot find" or blank, while exa_search:auto and tavily_advanced answered correctly. These are not hard questions by subject matter — Q137 is purely conceptual and requires no filing access — but tavily_basic failed to surface any usable content. The source type needed is the FY2024 consolidated income statement, which competitors retrieved without issue.

- **Wrong metric definition errors lose winnable questions (3 losses).** On Q66 (Customer Retention Rate), tavily_basic returned 92.3% (U.S./Canada membership renewal rate) instead of 90.5% (global rate). The agent identified the terminology ambiguity but still selected the wrong scope. On Q19 and Q20 (market debt-to-equity and debt-to-value), all three providers extracted book-value or capital-structure ratios rather than market-value-weighted ones, producing results of 0.18–0.31 against expected values of 2.73% and 2.65%. A prompt-level definition of "market value" weighting in financial ratio questions would directly address this.

- **All providers fail at adjusted/non-GAAP metrics that require a reconciliation table (joint hard cluster).** Q1 (unadjusted EBITDA $11,522M), Q2 (adjusted EBITDA $11,969M), Q4 (adjusted EBIT $9,396M), and Q13 (EV/EBITDA 35.44x) are all-wrong. Tavily_advanced calculated $12.81B for unadjusted EBITDA by mixing EBIT and D&A from different sources; exa_search:auto computed $9.74B using a D&A figure an order of magnitude too small. No provider retrieved the reconciliation table. The source gap is a structured non-GAAP reconciliation disclosure, not an income statement.

- **Valuation multiples requiring a specific point-in-time stock price fail consistently.** Q10 and Q11 (P/E using Basic and Diluted EPS) produced 53.5x–55.2x across all providers against expected 56.66x and 56.86x respectively. The diagnosis is a wrong stock price input: providers retrieved a period-average or historical price rather than the fiscal year-end close. Q14 (Enterprise Value) showed spreads of $398B to $466B against expected $424B. This is a retrieval routing problem: the fiscal year-end closing price needs to be explicitly anchored in the query or fetched from a structured market data source.

- **Tavily_basic underperforms tavily_advanced by 10pp overall (14.3% vs 24.3%) and the gap is especially visible on `basic` question type (12.9% vs 25.8%).** The basic tier appears to surface summarized or rounded figures rather than the source-level data needed for financial precision. On Q5, basic returned 4.5% while advanced returned 5.49% (both wrong, but basic is rounding an already-wrong cached summary).

## Cross-cutting observations

- **The `assumption` question type is a complete zero for all providers (0/25, 0.0%).** Every assumption question failed every provider. These appear to require multi-step derivations that chain adjusted EBITDA, market-value capital structures, and non-GAAP definitions — the exact cluster where all providers fail. This is a benchmark-structural problem as much as a retrieval problem, but Tavily needs a dedicated multi-hop calculation path to compete here.

- **The `direct-lookup` query type is Tavily_advanced's best relative slice.** Tavily_advanced hits 33.3% on direct-lookup (12 questions) versus exa_search:auto's 25.0% — an 8pp advantage. This is the one slice where advanced leads. Basic collapses to 8.3% on the same slice, suggesting the retrieval depth difference between basic and advanced is most consequential when an exact figure needs to be pulled from a specific document location.

- **Wrong-period retrieval is an underreported failure.** Q12 (Diluted EPS) shows exa_search:auto returning Q1 2026 data ($4.58) and tavily_advanced returning FY2025 data ($18.21) when FY2024 ($16.53) was required. Q69 (same-store sales growth) has tavily_basic anchoring to Q2 2026 (7%) rather than the FY2024 period that matches the expected answer of 5%. Without explicit fiscal period anchoring in queries, providers consistently drift to the most recently indexed data.

- **Costco-specific structured data (balance sheet line items, tax footnotes, lease schedules) is a universal gap, not a Tavily-specific one.** Questions on ROU lease assets (Q35, Q36), operating deferred tax assets (Q27, Q28), OWC delta (Q26), and invested capital with goodwill are all-wrong across every provider. These require parsing footnote tables in Costco's 10-K, not web summaries. No provider retrieves this level of document structure.

## What to do next

- **Build a structured FY2024 Costco 10-K index with section-level chunking.** Parse the income statement, balance sheet, cash flow statement, tax footnote, and non-GAAP reconciliation tables as separate retrievable documents. Tag each chunk with fiscal year, section name, and line item. This directly addresses the `missing_filing_access` failures on Q9, Q7, Q27, Q28 and the all-wrong non-GAAP cluster (Q1, Q2, Q4).

- **Add a prompt-level fiscal period constraint to all financeqa queries.** Every query referencing a year (e.g., "2024") should inject "fiscal year ended September 1, 2024" as a filter in the search call. This eliminates the wrong-period failures in Q12 (Diluted EPS) and Q69 (same-store sales) where providers pulled FY2025 or Q1 2026 data instead.

- **Add a routing rule that fetches the fiscal year-end closing stock price from a structured market data endpoint (not web search) whenever a query requires enterprise value, P/E, or market-cap calculation.** The P/E errors in Q10 and Q11 (all providers off by 3pp) and EV errors in Q14 (spread of $68B) trace directly to using period-average or non-fiscal-year-end prices. Route market price lookups to a point-in-time equity data source keyed to the filing date.

- **Add a metric definition disambiguation step to the tavily_basic prompt for ratio questions.** Before computing debt-to-equity, retention rate, or EBITDA margin, the prompt should explicitly resolve whether "market value" or "book value" weighting is required, and whether "global" or "regional" scope applies. This targets the `wrong_metric_definition` failures in Q19, Q20, and Q66 without requiring new index work.

- **Instrument a calculation verification step for tavily_advanced on multi-input ratios.** On Q5 (EBITDA margin), tavily_advanced built its own EBITDA rather than reading the stated figure, and got 5.49% instead of 4.53%. Add a post-retrieval check: if a non-GAAP metric is asked, prefer the explicitly stated reconciliation value over a manual build-up from raw components. This can be implemented as a retrieval preference rule ("if a reconciliation table is found, use its stated output; do not recompute from constituent lines").
