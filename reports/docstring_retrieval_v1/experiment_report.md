# LocalBench: Docstring-Derived Query Retrieval Evaluation v1

**Date:** 2026-11-07  
**Status:** BM25 completed; dense retrieval pending local Ollama access  
**Cohort:** 30 stratified docstring-derived queries

## Objective

Evaluate retrieval for 30 new natural-language queries written from CodeUnit docstrings, while excluding docstrings from the indexed source representation. This prevents direct retrieval of the same docstring text used to author a query.

Implementation source was not inspected during target selection or query writing. The docstrings themselves originate from the code corpus, so this is not an independent external-query benchmark.

## Cohort and sampling

- Test corpus: 2,493 Python CodeUnits across two repositories.
- Eligible pool: non-empty docstrings of at least 100 characters.
- Previous nine source-informed target IDs excluded.
- Sample: 30 targets, stratified across repository and symbol type.
- Sampling seed: 20261107.
- Queries paraphrase docstrings only.
- Each query is linked to the CodeUnit whose docstring supplied its description.

## Leakage control

An initial audit found docstring text embedded in the `source_code` field for 29 of 30 selected targets. A preliminary BM25 run against unmodified source code produced Hit@1 0.7333 and Hit@10 0.9000, but those figures are **invalid for this evaluation** because the indexed documents contained the same documentation used to author the queries. They are excluded from the reported result.

To control this, all 2,493 source records were parsed successfully with Python's AST. First string-literal expression docstrings were removed from module, class, function, and async-function bodies before indexing. A total of 1,199 docstring blocks were removed, including docstrings from all 30 selected targets. The uploaded dataset was not modified.

## BM25 results

The LocalBench BM25 algorithm was applied to the docstring-stripped source representation using its default parameters (`k1=1.5`, `b=0.75`), code-aware tokenizer, and top-10 cutoff.

| Metric | Score |
|---|---:|
| Hit@1 | 0.2667 |
| Hit@5 | 0.3667 |
| Hit@10 | 0.4333 |
| MRR | 0.3151 |

Target ranks (blank means not present in top 10):

| Query | Target rank |
|---|---:|
| D001 | 1 |
| D002 | 2 |
| D003 | 1 |
| D004 | 1 |
| D005 | — |
| D006 | 2 |
| D007 | 1 |
| D008 | 2 |
| D009 | 1 |
| D010 | 1 |
| D011 | 1 |
| D012 | 2 |
| D013 | 7 |
| D014 | — |
| D015 | 1 |
| D016 | — |
| D017 | 1 |
| D018 | 1 |
| D019 | — |
| D020 | — |
| D021 | — |
| D022 | — |
| D023 | — |
| D024 | — |
| D025 | 1 |
| D026 | — |
| D027 | — |
| D028 | — |
| D029 | 5 |
| D030 | — |

## Relevance-label limitations

The intended target is the CodeUnit whose docstring was used to write each query. This is a **silver positive association**, not independent human adjudication. Other potentially relevant CodeUnits were not judged. Hit@K and MRR therefore use a single-positive assumption, not exhaustive relevance labels.

## Dense retrieval status

Dense retrieval has not been run. The previous experiment used `nomic-embed-text:latest` through the user's local Ollama service. This execution environment cannot access that local service, and no remote or substitute embedding model was used.

## Provenance

- Original corpus SHA-256: `37b27223c1701126a8f24559e33212f66a5e70ce85fcbcefd01aa116dbe6ff0e`
- Docstring-stripped indexed representation SHA-256: `87d6ea149454bb2c9bcb9bca1349b4b717a2073a8616ad0546e3b04e36dbc95f`
- Query JSONL SHA-256: `a03ab022725441392fb00502a6e9074cb5db9a07437d7f9bba71e1c3e1d0b72a`

The transformed representation is defined by the deterministic AST docstring-removal procedure recorded in `manifest.json`. The original dataset was not changed.

## Human relevance review preparation

A blinded top-10 review workflow is now documented in `HUMAN_REVIEW_PROTOCOL.md`. The committed helper `scripts/prepare_docstring_retrieval_review.py` regenerates a 300-row worksheet from the test split and frozen query set. It reproduces the LocalBench BM25 tokenizer and scoring parameters, strips AST docstrings before indexing, and leaves all relevance judgment fields blank. The helper's rankings were independently checked against the frozen BM25 rankings: all 30 query rankings matched exactly (300 candidate rows total).

The generated worksheet is published at `human_relevance_review.csv` and contains 300 query–candidate rows (30 queries × 10 candidates), covering 208 unique CodeUnits. All judgment and reviewer-note fields are blank. Its SHA-256 is `779ab5475010f882704878368529dc9b487fe8a19b0dbc7a6cc48075584ef081`.

Human judgments have **not** yet been collected. The silver target labels remain separate from the blinded worksheet and must not be treated as human relevance judgments.


## Human relevance evaluation tooling

The review workflow now includes `scripts/evaluate_docstring_relevance.py`. Once human labels are entered, it validates the CSV and produces a machine-readable summary plus per-query metrics. It supports graded nDCG@10 (relevant=2, partially relevant=1, not relevant=0), strict Precision@10, Hit@10, MRR, and judged-pool coverage. Blank and uncertain rows are excluded from judgment denominators rather than treated as negatives. Because only the BM25 top-10 pool is labeled, corpus-wide recall remains unmeasurable from this worksheet.

The evaluator was checked against a synthetic fixture covering positive, partial, negative, and uncertain judgments. No real human labels or retrieval scores were added by this implementation.


## Dense retrieval runner implementation

A reproducible local-only dense retrieval runner is now available at `scripts/run_docstring_dense_retrieval.py`. It uses the pinned `nomic-embed-text:latest` model through Ollama's local `/api/embed` endpoint, removes AST docstrings from indexed source, embeds queries and CodeUnits, ranks with cosine similarity, and records Hit@1/5/10 and MRR against the existing single-positive silver associations. SQLite caching allows interrupted embedding runs to resume. The runner writes raw top-k rankings and a metrics artifact; it does not silently fall back to a cloud model.

**Execution remains pending:** this environment cannot reach the user's local Ollama service, so no dense scores are claimed. Run instructions are in `HUMAN_REVIEW_PROTOCOL.md`.


## Hybrid retrieval implementation

A BM25 + dense fusion runner is available at `scripts/run_docstring_hybrid_retrieval.py`. It recomputes the frozen docstring-stripped BM25 rankings, reads the local dense top-100 rankings, and applies Reciprocal Rank Fusion with a default constant of 60. It writes fused top-k rankings and silver-label Hit@1/5/10 and MRR metrics. The RRF constant is fixed by default rather than tuned against the 30-query evaluation cohort.

Hybrid execution is pending the local dense run. The exact commands and artifact paths are documented in `HUMAN_REVIEW_PROTOCOL.md`.
