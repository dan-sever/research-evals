# Finance benchmarks — combined view

## Headline

Across both financebench (57.1%) and financeqa (24.3%), Tavily advanced matches or leads Exa on direct-lookup and period-specific extraction tasks but loses ground consistently on multi-step calculation accuracy and retrieval coverage of structured filing documents that are not in its index.

## Strengths that show up on both

- **Direct extraction from primary filings beats Exa when the right document is indexed.** On financebench, Tavily advanced was the sole correct provider on Amazon FY2017 DPO and PepsiCo revolving credit aggregation. On financeqa, Tavily basic alone returned correct ROE (30.3%) and asset turnover (3.7x) from Costco's FY2024 filing while Exa failed both. The `direct_filing_extract` success factor appears in the wins set on both benchmarks.

- **Tavily advanced holds parity with Exa on conceptual and qualitative question types.** On financebench, Tavily advanced correctly identified 3M's Consumer segment as the FY2022 organic growth drag and J&J's gross margin drivers where Exa returned wrong or empty answers. On financeqa, Tavily advanced ties Exa exactly at 57.1% on conceptual questions. Neither benchmark shows Tavily losing ground on questions that do not require a precise numeric extraction.

- **Footnote-level and non-standard line item retrieval is a relative Tavily advantage.** On financeqa Q23, Tavily advanced was the only provider to surface the statutory marginal tax rate from the tax footnote (24.40% vs. expected 24.00%). On financebench Q38, Tavily advanced was the only provider to correctly conclude American Express had zero registered exchange-traded debt securities, a footnote-level negative finding that Exa answered incorrectly with a specific security.

## Gaps that show up on both

- **Retrieval gaps on specific historical or structured filing documents are the top failure mode across both benchmarks.** Financebench logs 21 losses tagged `retrieval_gap`, concentrated on pre-2021 SEC filings from companies such as Adobe, Nike, and Pfizer (0 of 5 Pfizer questions correct). Financeqa logs 5 `missing_filing_access` losses on Costco's FY2024 income statement and footnote tables. In both cases the failure is not a reasoning problem: the document is either not indexed or ranked too low to surface.

- **Calculation errors are the second-largest failure mode on both benchmarks (10 on financebench, 11 on financeqa).** On financebench, Tavily advanced returned $1,603M for 3M capex instead of $1,577M and 3.59 for Nike inventory turnover instead of 3.46, both times accepting a secondary aggregator figure over the primary filing. On financeqa, Tavily advanced computed 5.49% EBITDA margin by building from raw EBIT and D&A instead of reading the reconciliation table's stated 4.53% figure. The shared root cause is preferring a computed or aggregated value over an explicitly stated primary source value.

- **Wrong metric definition errors appear on both benchmarks and share the same prompt-level cause.** Financebench logs 9 `wrong_metric_definition` losses, including Tavily advanced dismissing PayPal working capital as irrelevant and answering PepsiCo geography at the wrong scope. Financeqa logs 3, including Tavily basic selecting the U.S./Canada membership renewal rate (92.3%) instead of the global rate (90.5%) and all providers returning book-value debt ratios when market-value weighting was required. In each case the model substituted its prior about the appropriate metric for what the question literally asked.

- **Wrong-period retrieval without explicit fiscal period anchoring causes losses on both benchmarks.** Financebench logs 4 `wrong_period` losses including Tavily basic citing Q2 FY2023 quick ratio data from December 2023 instead. Financeqa shows Tavily advanced returning FY2025 diluted EPS ($18.21) when FY2024 ($16.53) was required, and Tavily basic anchoring same-store sales growth to Q2 2026 instead of FY2024. Both benchmarks reveal that without a fiscal period constraint injected into the query, providers drift to the most recently indexed data.

## Benchmark-specific notes

**financebench**
- Tavily basic is effectively non-functional on this benchmark at 12.9% accuracy, a 44.2pp gap from advanced (57.1%). Every question type where basic fails requires locating a specific line item from a dated SEC filing. Basic should not be used for any financial statement analysis task without advanced-tier retrieval.
- 10-Q coverage is materially weaker than 10-K coverage for Tavily advanced: 28.6% on quarterly filing questions versus 59.2% on annual filing questions, a 30.6pp gap. Exa holds 57.1% on 10-Qs. The quarterly period-specific balance sheets are where Tavily advanced most frequently cites the wrong period or returns empty.
- Pfizer is a complete single-company indexing failure: 0 of 5 questions correct, all retrieval failures. The FY2021 10-K referencing the Trillium, Array, and Therachon acquisitions and the Upjohn spinoff is evidently not indexed.

**financeqa**
- The `assumption` question type (25 questions) returned 0% accuracy across every provider including Tavily. These questions chain adjusted EBITDA, market-value capital structures, and non-GAAP definitions in a sequence no provider handled. This is the single largest accuracy ceiling on the benchmark.
- Valuation multiples requiring a point-in-time fiscal year-end stock price fail consistently across providers: P/E errors of 3pp on Q10 and Q11 and an enterprise value spread of $68B on Q14. Providers retrieved period-average or recent prices rather than the fiscal year-end close, a routing problem that prompt engineering alone cannot fix.
- Non-GAAP reconciliation tables are a universal retrieval gap: Q1 (unadjusted EBITDA), Q2 (adjusted EBITDA), Q4 (adjusted EBIT), and Q13 (EV/EBITDA) are all-wrong across every provider because no provider retrieved the reconciliation table itself, only income statement components from which they built incorrect approximations.

## Top priorities

1. **Index the primary filing documents that drive the largest share of losses across both benchmarks.** For financebench, this means a dedicated SEC EDGAR corpus covering all 28 benchmark companies for 10-K, 10-Q, 8-K, and earnings releases from 2015 to 2024, with Pfizer and 3M 10-Qs as the first priority. For financeqa, this means section-level chunking of Costco's FY2024 10-K including the income statement, balance sheet, non-GAAP reconciliation tables, tax footnote, and lease schedules. The 21 `retrieval_gap` losses on financebench and 5 `missing_filing_access` losses on financeqa cannot be addressed by prompt changes.

2. **Add a source-type re-ranking rule that downgrades financial aggregator pages in favor of SEC EDGAR URLs, and add a reconciliation-table preference rule that instructs the model to use explicitly stated non-GAAP figures over manual build-ups from raw components.** This single intervention targets the 10 calculation errors on financebench (wrong aggregator figures for Nike, 3M, Costco) and the 11 calculation errors on financeqa (EBITDA margin built from EBIT and D&A instead of read from the reconciliation table) simultaneously.

3. **Inject explicit fiscal period constraints and a metric definition disambiguation step into all financial queries.** Every query referencing a specific year or quarter should bind to the corresponding filing date range rather than the most recently indexed document. Every ratio query should resolve market-value vs. book-value weighting and geographic scope before retrieval. These two prompt-level rules directly address the `wrong_period` losses on both benchmarks (4 on financebench, 2 on financeqa) and the `wrong_metric_definition` losses on both (9 on financebench, 3 on financeqa).
