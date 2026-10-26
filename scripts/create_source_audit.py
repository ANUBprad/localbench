"""Create a pending source-code audit template without overwriting decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localbench.workloads.code_retrieval.review import validate_review_artifact
from localbench.workloads.code_retrieval.source_audit import write_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review", type=Path,
        default=Path("dataset/queries/review_artifact.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("dataset/queries/source_code_audit.json"),
    )
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    errors = validate_review_artifact(review)
    if errors:
        raise ValueError("Invalid review artifact: " + "; ".join(errors))
    write_template(review, args.output)
    print(f"Pending audit template created at {args.output}")


if __name__ == "__main__":
    main()
