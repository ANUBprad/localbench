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

## Local dense retrieval baseline
The dense baseline is implemented as a resumable local-only runner. With the test split and frozen query file available in the repository, run it on a machine with Ollama and the pinned embedding model:

```powershell
ollama pull nomic-embed-text:latest
python scripts/run_docstring_dense_retrieval.py --units dataset/splits/test.jsonl --queries reports/docstring_retrieval_v1/queries.jsonl --output-dir reports/docstring_retrieval_v1/dense --model nomic-embed-text:latest --top-k 100
```

The runner removes AST docstrings from indexed CodeUnit source, embeds documents and natural-language queries through Ollama's local `/api/embed` endpoint, and ranks by cosine similarity. It caches vectors in SQLite so interrupted runs can resume. Use `--top-k 100` to retain a wider candidate pool for hybrid fusion; Hit@1/5/10 are still computed from the full ranking. It writes `dense_results.json` and `dense_metrics.json`. It has no cloud fallback. Run it on the machine hosting Ollama; this chat environment cannot reach that local service.

## Hybrid retrieval (BM25 + dense)
After the dense run completes, fuse the two top-100 rankings with Reciprocal Rank Fusion:

```powershell
python scripts/run_docstring_hybrid_retrieval.py --units dataset/splits/test.jsonl --queries reports/docstring_retrieval_v1/queries.jsonl --dense-results reports/docstring_retrieval_v1/dense/dense_results.json --output-dir reports/docstring_retrieval_v1/hybrid --rrf-k 60 --top-k 10
```

The default RRF constant is 60. The script recomputes the frozen BM25 rankings, fuses ranks without tuning against the silver labels, and writes `hybrid_results.json` and `hybrid_metrics.json`.

## Pooled human review and cross-retriever evaluation
The first 300-row worksheet judges BM25 candidates only. To compare BM25, dense, and hybrid fairly, first build the union of their top-10 candidate pools after both local runs finish:

```powershell
python scripts/prepare_pooled_relevance_review.py --units dataset/splits/test.jsonl --queries reports/docstring_retrieval_v1/queries.jsonl --dense-results reports/docstring_retrieval_v1/dense/dense_results.json --hybrid-results reports/docstring_retrieval_v1/hybrid/hybrid_results.json --existing-review reports/docstring_retrieval_v1/human_relevance_review.csv --output reports/docstring_retrieval_v1/pooled_relevance_review.csv --top-k 10
```

The generator preserves existing labels and notes, and adds blank judgments for candidates newly surfaced by dense or hybrid retrieval. Review all newly added rows before comparing systems. Then run the evaluator once per rank column:

```powershell
python scripts/evaluate_docstring_relevance.py --input reports/docstring_retrieval_v1/pooled_relevance_review.csv --rank-column candidate_rank_bm25 --output-dir reports/docstring_retrieval_v1/evaluation/bm25
python scripts/evaluate_docstring_relevance.py --input reports/docstring_retrieval_v1/pooled_relevance_review.csv --rank-column candidate_rank_dense --output-dir reports/docstring_retrieval_v1/evaluation/dense
python scripts/evaluate_docstring_relevance.py --input reports/docstring_retrieval_v1/pooled_relevance_review.csv --rank-column candidate_rank_hybrid --output-dir reports/docstring_retrieval_v1/evaluation/hybrid
```

The evaluator supports method-specific rank columns. Blank ranks mean a candidate was not retrieved by that method. It reports judged coverage separately; unjudged labels are excluded, and nDCG uses a condensed judged ranking so unjudged candidates are not silently treated as negative labels.

## Limitations
- Queries were authored from docstrings in the same corpus, not from external user requests.
- The original intended target is a silver association, not necessarily the only relevant CodeUnit.
- The worksheet covers only BM25 top-10 candidates; it cannot estimate corpus-wide recall.
- No human relevance judgments have been made yet.
