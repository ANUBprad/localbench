"""Phase 4F-I-C2: deterministic final-45 query selection (§4.4.5).

Executes the frozen methodology:

    generation -> automated eligibility -> deterministic selection of 45
    -> benchmark-blind human review of the selected 45 -> freeze -> ground truth

Pipeline:
  1. Load candidate artifacts and canonical splits
  2. Recompute the eligible pool (§4.4.5 conditions, duplicates abort)
  3. Verify expected pool size
  4. Hash the canonical pool representation
  5. Select 45 via random.Random(42).sample
  6. Independently reproduce and compare
  7. Post-selection audit
  8. Write the selection record artifact

Usage:
    python scripts/select_final_queries.py \
        --generation-source-commit ca257dd \
        [--expected-pool-size 2077] \
        [--selection-version 1.0.0] \
        [--output dataset/queries/final_45_selection.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localbench.workloads.code_retrieval.selection import (
    FINAL_QUERY_COUNT,
    SelectionError,
    build_eligible_pool,
    build_selection_record,
    distribution,
    pool_hash,
    select_final_queries,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = REPO_ROOT / "dataset" / "queries" / "candidates.jsonl"
SPLITS_DIR = REPO_ROOT / "dataset" / "splits"
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "queries" / "final_45_selection.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("select_final_queries")


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _split_ids(name: str) -> set[str]:
    return {record["code_unit_id"] for record in _load_jsonl(SPLITS_DIR / name)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--expected-pool-size", type=int, default=2077)
    parser.add_argument("--selection-version", default="1.0.0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    candidates = _load_jsonl(CANDIDATES_PATH)
    logger.info("loaded %d candidate records", len(candidates))
    train_ids = _split_ids("train.jsonl")
    validation_ids = _split_ids("validation.jsonl")
    test_records = {
        r["code_unit_id"]: r for r in _load_jsonl(SPLITS_DIR / "test.jsonl")
    }
    logger.info(
        "splits: train=%d validation=%d test=%d",
        len(train_ids),
        len(validation_ids),
        len(test_records),
    )

    pool = build_eligible_pool(
        candidates,
        test_code_unit_ids=set(test_records),
        train_code_unit_ids=train_ids,
        validation_code_unit_ids=validation_ids,
    )
    if len(pool) != args.expected_pool_size:
        logger.error(
            "eligible pool is %d candidates; expected exactly %d - STOP",
            len(pool),
            args.expected_pool_size,
        )
        return 2
    logger.info("eligible pool: %d candidates (matches expectation)", len(pool))

    digest = pool_hash(pool)
    logger.info("eligible_pool_sha256: %s", digest)

    selected = select_final_queries(pool)

    verify_pool_hash = pool_hash(pool)
    verify_selected = select_final_queries(pool)
    if verify_pool_hash != digest or [c["code_unit_id"] for c in verify_selected] != [
        c["code_unit_id"] for c in selected
    ]:
        logger.error("reproducibility verification FAILED - STOP")
        return 3
    logger.info("reproducibility verification passed (independent rerun identical)")

    units = [c["code_unit_id"] for c in selected]
    cand_ids = [c["candidate_id"] for c in selected]
    assert len(selected) == FINAL_QUERY_COUNT
    assert len(set(units)) == FINAL_QUERY_COUNT
    assert len(set(cand_ids)) == FINAL_QUERY_COUNT
    assert all(u in test_records for u in units)
    assert not (set(units) & train_ids)
    assert not (set(units) & validation_ids)
    assert all(c["leakage_passed"] and c["validation_passed"] for c in selected)
    pool_by_unit = {c["code_unit_id"]: c for c in pool}
    assert all(u in pool_by_unit for u in units)
    logger.info("post-selection audit passed")

    repositories = [test_records[u]["repository"] for u in units]
    styles = [c["query_style"] for c in selected]
    repo_by_unit = {u: rec["repository"] for u, rec in test_records.items()}
    record = build_selection_record(
        ordered_pool=pool,
        selected=selected,
        repository_by_code_unit_id=repo_by_unit,
        generation_source_commit=args.generation_source_commit,
        selection_version=args.selection_version,
    )
    assert record["selected_repository_distribution"] == distribution(repositories)
    assert record["selected_query_style_distribution"] == distribution(styles)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("selection record written: %s", args.output)

    print(f"eligible_pool_sha256: {digest}")
    print(f"repository_distribution: {record['selected_repository_distribution']}")
    print(f"query_style_distribution: {record['selected_query_style_distribution']}")
    print("selected_code_unit_ids:")
    for unit in units:
        print(f"  {unit}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SelectionError as error:
        logger.error("%s", error)
        sys.exit(2)
