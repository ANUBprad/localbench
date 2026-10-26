"""Export human-approved, explicitly labeled queries for retrieval evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from localbench.workloads.code_retrieval.ground_truth_export import export_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("dataset/queries/review_artifact.json"),
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--units", type=Path, default=Path("dataset/splits/test.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/adjudicated_queries.jsonl")
    )
    args = parser.parse_args()
    count = export_from_files(
        args.review, args.labels, args.units, args.output, args.audit
    )
    print(f"Exported {count} accepted, labeled queries to {args.output}")


if __name__ == "__main__":
    main()
