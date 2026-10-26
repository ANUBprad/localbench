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
3. Generate an independent source audit (only once; it refuses to overwrite):
   `python -m scripts.create_source_audit`. Inspect
   `dataset/queries/source_code_audit.json`, compare each query to its
   source implementation, and set `status` to `passed` or `failed`
   with a nonempty explanation. Leave uncertain items `pending`.
   The template never auto-approves. Source hashes and query text are
   checked against the current review artifact. The audit must contain
   every human-accepted candidate; only audit-passed candidates are
   eligible for export.
4. Create a separate local JSONL label file, e.g.
   `dataset/queries/adjudicated_labels.jsonl`, with one line per **accepted**
   candidate, containing independently verified relevant corpus IDs:

```json
{"candidate_id":"<actual selected candidate ID>","relevant_code_units":["<verified CodeUnit ID>"]}
```

These strings are placeholders, not verified labels. Include only audit-passed, human-accepted\ncandidates. Do not label pending/failed audits or human-rejected candidates. The export fails closed on
missing labels, duplicates, unknown candidates, failed audits, or corpus ID mismatches.

```powershell
python -m scripts.export_adjudicated_queries --labels dataset/queries/adjudicated_labels.jsonl --audit dataset/queries/source_code_audit.json --output reports/adjudicated_queries.jsonl
python -m scripts.run_retrieval_comparison --units dataset/splits/test.jsonl --queries reports/adjudicated_queries.jsonl --model nomic-embed-text --output reports/comparison.json
```

Do not commit private or unreviewed datasets. Only publish metrics after
verifying split provenance and adjudication. The export does not invent
additional relevance labels or assume the selected target is the sole
relevant CodeUnit. Model digest and large-corpus batching remain future work.
