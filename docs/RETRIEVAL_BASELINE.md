# Retrieval baseline (Phase 2)

The first retrieval implementation is a deterministic, offline BM25 source-code baseline.
It is **not** a full model comparison or a completed benchmark release.

## API

```python
from localbench.workloads.code_retrieval.retrieval import BM25Index, code_unit_documents
from localbench.workloads.code_retrieval.metrics import evaluate_rankings

index = BM25Index({"unit-a": "def retry_payment(): ...", "unit-b": "def parse_csv(): ..."})
rankings = {"query-1": [hit.code_unit_id for hit in index.search("retry payment")]}
metrics = evaluate_rankings(rankings, {"query-1": ["unit-a"]})
print(metrics)
```

`code_unit_documents(units)` accepts validated canonical CodeUnit instances and indexes their
source code only. BM25 uses normalized term frequencies and IDF, deterministic sorting,
and stable CodeUnit IDs. Empty query terms return no hits. No LLM, network, or dataset labels
are required for indexing.

`evaluate_rankings` computes Hit@1, Hit@5, Hit@10, and MRR across **all labeled
queries**; missing rankings are misses, not silently dropped. This avoids inflating
end-to-end scores when generation or retrieval fails.

## Remaining Phase 2 work

- Fixed local embedding baseline and optional hybrid fusion
- Dataset artifact loader, hash validation, and leakage checks
- Reproducible CLI runner, per-query raw results, and comparison report
- End-to-end integration tests against the actual frozen benchmark

No results should be published until frozen labels are verified and the entire pipeline
has been run with recorded configurations.
