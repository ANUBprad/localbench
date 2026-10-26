# Local retrieval comparison runner

This is an **opt-in evaluation runner** for explicitly labeled query cases.
It does not infer labels from generated candidates or bypass human review.

Inputs:

- `--units`: JSONL, each row has `id` and `source_code`.
- `--queries`: JSONL, each row has `id`, `query`, and
  `relevant_code_units` (list of adjudicated corpus IDs).

Run from repository root with the package installed (for example,
`python -m pip install -e .`):

```powershell
# BM25 only:
python -m scripts.run_retrieval_comparison --units dataset/splits/test.jsonl --queries path/to/adjudicated_queries.jsonl --output reports/bm25.json

# BM25 vs Ollama dense (requires local Ollama and a model):
ollama pull nomic-embed-text
python -m scripts.run_retrieval_comparison --units dataset/splits/test.jsonl --queries path/to/adjudicated_queries.jsonl --model nomic-embed-text --output reports/comparison.json
```

The path `path/to/adjudicated_queries.jsonl` is a placeholder: supply a
genuine reviewed/labeled file. The existing frozen 45-candidate selection
is **not** automatically equivalent to 45 accepted ground-truth queries.

Output includes per-query rankings, Hit@1/5/10, MRR, SHA-256 hashes of both
input files, source-code representation, model name and top-k. The model name
is recorded, but a complete scientific run should also record the installed
model digest and Ollama version separately. Results are not benchmark claims
until labels and split provenance have been independently validated.

The adapter uses `/api/embed` with `truncate=false`. A large corpus may
exceed available RAM or Ollama's batch/context limits; persistent caching,
batch sizing, and strict frozen-artifact validation are follow-up work.
