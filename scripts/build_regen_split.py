"""Build the strictly filtered regeneration input for conflicted units.

Reads the conflict report produced by
``recover_query_candidates.py --isolate-conflicts`` and copies the
matching rows verbatim out of the canonical test split into a subset
JSONL that feeds ``generate_query_candidates.py --test-split``.

Validation (aborts before writing anything):
- every conflicted CodeUnit ID must be found in the canonical test
  split,
- no CodeUnit ID may match more than one row,
- the conflict report must not contain duplicate IDs.

The canonical split itself is only read, never modified.

Usage:
    python scripts/build_regen_split.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localbench.workloads.code_retrieval.extraction import (
    _build_code_unit_id,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
CONFLICT_REPORT_PATH = DATASET_ROOT / "queries" / "conflict_report.json"
TEST_SPLIT_PATH = DATASET_ROOT / "splits" / "test.jsonl"
OUTPUT_PATH = DATASET_ROOT / "queries" / "regen_conflicted_input.jsonl"

EXIT_OK = 0
EXIT_ERROR = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conflict-report",
        type=Path,
        default=CONFLICT_REPORT_PATH,
        help="Conflict report written by recover_query_candidates.py.",
    )
    parser.add_argument(
        "--test-split",
        type=Path,
        default=TEST_SPLIT_PATH,
        help="Canonical test split (read-only source of input rows).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Filtered subset JSONL to generate.",
    )
    return parser.parse_args(argv)


def _row_code_unit_id(row: dict) -> str:
    return _build_code_unit_id(
        row["repository"], row["file_path"], row["symbol"], row["content_hash"]
    )


def build_subset(
    conflicted_ids: list[str],
    test_split_path: Path,
    output_path: Path,
) -> int:
    expected = set(conflicted_ids)
    if len(expected) != len(conflicted_ids):
        logger.error(
            "Conflict report contains duplicate CodeUnit IDs; refusing "
            "to proceed."
        )
        return EXIT_ERROR

    matched_lines: list[str] = []
    matched_ids: set[str] = set()
    with open(test_split_path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            row_id = _row_code_unit_id(json.loads(raw_line))
            if row_id not in expected:
                continue
            if row_id in matched_ids:
                logger.error(
                    "%s:%d contains a second row for conflicted CodeUnit "
                    "'%s'; refusing to proceed.",
                    test_split_path,
                    line_number,
                    row_id,
                )
                return EXIT_ERROR
            matched_ids.add(row_id)
            matched_lines.append(raw_line.rstrip("\n"))

    missing = sorted(expected - matched_ids)
    if missing:
        logger.error(
            "%d conflicted CodeUnit(s) missing from %s: %s",
            len(missing),
            test_split_path,
            missing,
        )
        return EXIT_ERROR

    temp_path = output_path.with_name(output_path.name + ".tmp")
    with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(line + "\n" for line in matched_lines))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, output_path)

    logger.info(
        "Wrote %d input rows (%d conflicted units requested) to %s",
        len(matched_lines),
        len(conflicted_ids),
        output_path,
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = json.loads(args.conflict_report.read_text(encoding="utf-8"))
    conflicted_ids = report["conflicted_unit_ids"]
    logger.info(
        "Conflict report lists %d conflicted CodeUnit(s)", len(conflicted_ids)
    )
    try:
        return build_subset(conflicted_ids, args.test_split, args.output)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
