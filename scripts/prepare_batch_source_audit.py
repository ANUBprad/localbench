"""Prepare a fail-closed source-audit worksheet from a completed blind review.

The script maps accepted review rows to their CodeUnit source. It never
marks a query as source-verified; that decision and its evidence remain blank.

Example:
    python scripts/prepare_batch_source_audit.py \
      --review reports/query_generation_v2/batch30/blind_review_ai_assisted.csv \
      --key reports/query_generation_v2/batch30/blind_review_key.json \
      --candidates reports/query_generation_v2/batch30/candidates.jsonl \
      --split dataset/splits/test.jsonl \
      --output reports/query_generation_v2/batch30/source_audit_worksheet.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localbench.workloads.code_retrieval.source_audit import (  # noqa: E402
    prepare_batch_audit_rows,
)


def _read_jsonl(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records


def _unique_index(records: list[dict], field: str, label: str) -> dict:
    result = {}
    for record in records:
        identity = record.get(field)
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{label} contains a missing {field}")
        if identity in result:
            raise ValueError(f"{label} contains duplicate {field}: {identity}")
        result[identity] = record
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")

    with args.review.open(encoding="utf-8-sig", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    if not review_rows:
        raise ValueError("Review CSV is empty")
    key = json.loads(args.key.read_text(encoding="utf-8"))
    if not isinstance(key, dict):
        raise ValueError("Blind-review key must be a JSON object")

    candidates = _unique_index(
        _read_jsonl(args.candidates), "candidate_id", "Candidate artifact"
    )
    units = _unique_index(_read_jsonl(args.split), "id", "Test split")
    rows = prepare_batch_audit_rows(review_rows, key, candidates, units)

    fields = [
        "review_id",
        "candidate_id",
        "code_unit_id",
        "query",
        "repository",
        "file_path",
        "symbol",
        "source_sha256",
        "source_code",
        "source_verified",
        "source_evidence",
        "audit_notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Accepted for source audit: {len(rows)} / {len(review_rows)}")
    print(f"Worksheet: {args.output}")
    print("Source decisions remain blank; verify each query against its source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
