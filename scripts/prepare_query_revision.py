"""Prepare a non-destructive worksheet for revising source-audit failures.

Example:
    python scripts/prepare_query_revision.py \
      --audit reports/query_generation_v2/batch30/source_audit_worksheet_completed.csv \
      --output reports/query_generation_v2/batch30/query_revision_worksheet.csv

Only failed source-audit rows are included. Revised query text and rationale
remain blank for human/model-assisted drafting and subsequent review.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localbench.workloads.code_retrieval.source_audit import (  # noqa: E402
    prepare_query_revision_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")

    with args.audit.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Source-audit CSV is missing its header")
        rows = list(reader)
    if not rows:
        raise ValueError("Source-audit CSV is empty")

    revisions = prepare_query_revision_rows(rows)
    fields = [
        "review_id",
        "candidate_id",
        "code_unit_id",
        "repository",
        "file_path",
        "symbol",
        "original_query",
        "source_sha256",
        "source_code",
        "source_evidence",
        "audit_notes",
        "revised_query",
        "revision_rationale",
        "revision_status",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(revisions)

    print(f"Queries requiring revision: {len(revisions)} / {len(rows)}")
    print(f"Worksheet: {args.output}")
    print("Revised queries remain blank; no query was auto-approved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
