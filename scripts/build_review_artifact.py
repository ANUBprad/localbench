"""Build the benchmark-blind human-review artifact for the selected 45.

Loads the frozen selection record, the candidate pool, and the test
CodeUnits, then delegates to ``review.build_review_artifact``.  Writes
the result to ``dataset/queries/review_artifact.json``.

Usage:
    python scripts/build_review_artifact.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localbench.workloads.code_retrieval.review import (  # noqa: E402
    ReviewArtifactError,
    build_review_artifact,
    validate_review_artifact,
)

SELECTION_PATH = REPO_ROOT / "dataset" / "queries" / "final_45_selection.json"
CANDIDATES_PATH = REPO_ROOT / "dataset" / "queries" / "candidates.jsonl"
TEST_SPLIT_PATH = REPO_ROOT / "dataset" / "splits" / "test.jsonl"
TRAIN_SPLIT_PATH = REPO_ROOT / "dataset" / "splits" / "train.jsonl"
VALIDATION_SPLIT_PATH = REPO_ROOT / "dataset" / "splits" / "validation.jsonl"
OUTPUT_PATH = REPO_ROOT / "dataset" / "queries" / "review_artifact.json"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    print("Loading selection record …")
    with open(SELECTION_PATH, encoding="utf-8") as f:
        selection_record = json.load(f)

    print("Loading candidates …")
    all_candidates = _load_jsonl(CANDIDATES_PATH)
    candidates_by_id = {c["candidate_id"]: c for c in all_candidates}

    print("Loading test CodeUnits …")
    test_units = _load_jsonl(TEST_SPLIT_PATH)
    units_by_id = {u["id"]: u for u in test_units}
    test_code_unit_ids = {u["id"] for u in test_units}

    print("Loading train/val CodeUnits for leakage check …")
    train_units = _load_jsonl(TRAIN_SPLIT_PATH) if TRAIN_SPLIT_PATH.exists() else []
    validation_units = (
        _load_jsonl(VALIDATION_SPLIT_PATH) if VALIDATION_SPLIT_PATH.exists() else []
    )
    train_ids = {u["id"] for u in train_units}
    validation_ids = {u["id"] for u in validation_units}

    print("Building review artifact …")
    try:
        artifact = build_review_artifact(
            selection_record=selection_record,
            candidates_by_id=candidates_by_id,
            units_by_id=units_by_id,
            test_code_unit_ids=test_code_unit_ids,
            train_code_unit_ids=train_ids,
            validation_code_unit_ids=validation_ids,
        )
    except ReviewArtifactError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Validating review artifact …")
    errors = validate_review_artifact(artifact)
    if errors:
        for err in errors:
            print(f"  VIOLATION: {err}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print(f"Review artifact written to {OUTPUT_PATH}")
    print(f"  Items: {len(artifact['items'])}")
    all_pending = all(
        item["review"]["state"] == "pending" for item in artifact["items"]
    )
    print(f"  All pending: {all_pending}")


if __name__ == "__main__":
    main()
