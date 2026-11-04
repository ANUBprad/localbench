"""Validate completed query revisions and emit a review-only report.

Example:
    python scripts/validate_query_revision.py \
      --input reports/query_generation_v2/batch30/query_revision_worksheet.csv \
      --output reports/query_generation_v2/batch30/query_revision_assessment.json

This command does not approve revisions or claim retrieval success.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localbench.workloads.code_retrieval.query_revision import (  # noqa: E402
    assess_query_revision_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Query-revision CSV is missing its header")
        rows = list(reader)

    report = {
        "artifact": "query_revision_assessment",
        "version": 1,
        "approval": "not_approved",
        "items": assess_query_revision_rows(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    flagged = sum(bool(item["quality_flags"]) for item in report["items"])
    print(f"Revisions assessed: {len(report['items'])}")
    print(f"Rows with advisory quality flags: {flagged}")
    print(f"Review report: {args.output}")
    print("All revisions still require manual source and retrieval review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
