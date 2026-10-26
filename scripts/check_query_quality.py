"""Generate advisory quality flags for review-artifact queries.

Usage: python -m scripts.check_query_quality --review FILE --output FILE
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from localbench.workloads.code_retrieval.query_quality import assess_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.review.resolve() == args.output.resolve():
        parser.error("output must differ from review artifact")
    data = json.loads(args.review.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("review artifact must be a JSON object")
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("review artifact must contain an items list")
    rows = assess_batch(items)
    counts = Counter(flag for row in rows for flag in row["flags"])
    report = {
        "artifact": "query_quality_advisory",
        "version": 1,
        "candidate_count": len(rows),
        "flag_counts": dict(sorted(counts.items())),
        "items": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {len(rows)} queries; flagged {sum(bool(r['flags']) for r in rows)}")
    print(f"Saved advisory report to {args.output}")


if __name__ == "__main__":
    main()
