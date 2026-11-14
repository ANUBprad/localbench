# Human relevance review: docstring retrieval v1

## Goal
Independently judge the top-10 BM25 candidates for each of the 30 docstring-derived queries.

## Generate the worksheet
Run the committed helper from the repository root:

```powershell
python scripts/prepare_docstring_retrieval_review.py --units dataset/splits/test.jsonl --queries reports/docstring_retrieval_v1/queries.jsonl --output reports/docstring_retrieval_v1/human_relevance_review.csv
```

The helper reproduces the LocalBench BM25 tokenizer and scoring parameters (k1=1.5, b=0.75), strips AST docstrings from indexed source, and leaves the judgment columns blank. It does not write to the dataset or query files.

## Labels
- `relevant`: directly implements the requested behavior or is a clearly valid answer.
- `partially_relevant`: related behavior, but incomplete, indirect, or only one part of the request.
- `not_relevant`: does not materially answer the query.
- `uncertain`: insufficient context to decide.

## Review rules
1. Judge each candidate against the query; do not use candidate rank as evidence.
2. Review all candidates for a query, preferably without looking up the original silver target.
3. Use the source included in the worksheet; docstrings are removed to avoid query-text leakage.
4. Add a concise reason for relevant, partially relevant, or uncertain judgments.
5. If possible, have a second reviewer independently label a subset and resolve disagreements.

## Evaluate collected judgments
After saving labels in the worksheet, run:

```powershell
python scripts/evaluate_docstring_relevance.py --input reports/docstring_retrieval_v1/human_relevance_review.csv --output-dir reports/docstring_retrieval_v1/human_evaluation
```

The evaluator writes `summary.json` and `per_query_metrics.csv`. It reports macro-averaged strict Precision@10, Hit@10, MRR, graded nDCG@10, and judgment coverage. `relevant=2`, `partially_relevant=1`, and `not_relevant=0` are used as graded gains for nDCG. Blank and `uncertain` labels remain unjudged; they are not silently counted as negatives. Strict Hit@10 and MRR count only `relevant` labels.

## Limitations
- Queries were authored from docstrings in the same corpus, not from external user requests.
- The original intended target is a silver association, not necessarily the only relevant CodeUnit.
- The worksheet covers only BM25 top-10 candidates; it cannot estimate corpus-wide recall.
- No human relevance judgments have been made yet.
