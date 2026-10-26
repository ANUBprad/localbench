# LocalBench

Offline-first local LLM tooling, with an active **code-retrieval dataset and query-generation research workflow**.

> **Project status (October 2026):** Work in progress. Repository ingestion, CodeUnit extraction, query generation, selection, blind review tooling, and a Semantic IR A/B experiment are implemented. A complete retrieval index, ranking evaluation runner, and reproducible multi-model comparison are **not yet shipped**. Do not interpret experimental generation success as retrieval accuracy.

## Install

Python 3.10+ is required. Ollama is optional for unit tests but required for local inference.

```bash
git clone https://github.com/ANUBprad/localbench.git
cd localbench
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev,semantic-ir]"
python -m pytest -q
```

## Available CLI commands

```bash
localbench --help
localbench --version
localbench system
localbench models            # requires a running Ollama service
localbench ask "Explain a hash table"  # requires Ollama and a suitable model
localbench review --help
localbench review status     # requires local review artifacts
```

The `review` CLI concerns the selected 45-query benchmark. It is **separate** from the development-only Semantic IR A/B blind review.

## Semantic IR A/B review (development only)

The original experiment contains 60 blinded pairs (120 review items). Its frozen inputs and results live under `artifacts/experiments/query_representation_ab/v1/` and are local/ignored; a fresh clone will not contain them. Never fabricate missing artifacts or human judgments.

Check the existing full review:

```bash
python scripts/analyze_query_representation_ab_review.py --validate-only
```

If the full review is incomplete, a deterministic 20-pair (40-item) subset can be reviewed without overwriting the original:

```bash
python scripts/run_blind_query_ab_review.py --review-pairs 20
python scripts/analyze_query_representation_ab_review.py \
  --input artifacts/experiments/query_representation_ab/v1/blind_review_input_20pairs.json \
  --review-file artifacts/experiments/query_representation_ab/v1/blind_review_results_20pairs.json \
  --analysis-dir artifacts/experiments/query_representation_ab/v1/analysis_20pairs
```

Complete the human review before analyzing. A 20-pair result is directional development evidence, not a definitive benchmark-quality claim. Preserve the frozen 45-query evaluation set and its ground truth.

## Development

```bash
python -m pytest -q
python -m pytest --cov=localbench --cov-report=term-missing
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
```

CI runs offline unit tests without a local Ollama server or ignored experiment artifacts. The standalone experiment scripts may require their own local data and dependencies.

## Roadmap

1. Close the Semantic IR A/B decision using real blind judgments and a documented stopping rule.
2. Verify and version the frozen dataset, query-to-CodeUnit relevance labels, and repository-disjoint splits.
3. Implement BM25 and fixed local-embedding retrieval baselines with Hit@K and MRR.
4. Benchmark local models with comparable configurations, full-denominator failure accounting, timing, and hardware metadata.
5. Publish reproducible runs, documentation, and a v1 release.

## Engineering principles

- **Offline first:** no cloud inference fallback.
- **Benchmark integrity:** no development tuning against frozen test labels.
- **Evidence over claims:** preserve raw artifacts and report failed cases.
- **Reproducibility:** record dataset hashes, seeds, model IDs, configuration, and hardware.
- **Typed boundaries:** validate structured outputs and fail closed on malformed records.

Detailed design documents in the repository describe planned architecture and should not be mistaken for proof that every module is implemented.

## License

MIT; see [LICENSE](LICENSE).
