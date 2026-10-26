# Export reviewed ground truth for LocalBench

The frozen 45-query selection is **not automatically a labeled benchmark**.
The blind review artifact stores decisions, not relevance labels. This
exporter requires accepted review decisions and separately adjudicated labels.

1. Inspect your existing `dataset/queries/review_artifact.json`. Do not
   regenerate it if human decisions are already present: rebuilding resets
   review states to pending. If absent, run
   `python scripts/build_review_artifact.py`.
2. Complete human review and record `review.state = "accepted"` or
   `"rejected"` for each reviewed item. Rejections require notes. Keep
   review decisions in the existing artifact; do not insert relevance fields.
3. Create a separate local JSONL label file, e.g.
   `dataset/queries/adjudicated_labels.jsonl`, with one line per **accepted**
   candidate, containing independently verified relevant corpus IDs:

```json
{"candidate_id":"<actual selected candidate ID>","relevant_code_units":["<verified CodeUnit ID>"]}
```

These strings are placeholders, not verified labels. Include all accepted
candidates and none that are pending/rejected. The export fails closed on
missing labels, duplicates, unknown candidates, or corpus ID mismatches.

```powershell
python -m scripts.export_adjudicated_queries --labels dataset/queries/adjudicated_labels.jsonl --output reports/adjudicated_queries.jsonl
python -m scripts.run_retrieval_comparison --units dataset/splits/test.jsonl --queries reports/adjudicated_queries.jsonl --model nomic-embed-text --output reports/comparison.json
```

Do not commit private or unreviewed datasets. Only publish metrics after
verifying split provenance and adjudication. The export does not invent
additional relevance labels or assume the selected target is the sole
relevant CodeUnit. Model digest and large-corpus batching remain future work.
