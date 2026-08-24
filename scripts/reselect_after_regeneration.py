"""Phase 4F-I-C3: deterministic re-selection after regeneration.

Rebuilds the eligible candidate pool from original + v2 candidates,
re-executes the frozen deterministic selection algorithm (seed=42),
and builds a fresh benchmark-blind review artifact.

The selection algorithm is IDENTICAL to the frozen section 4.4.5:
- Filter eligible candidates (schema + leakage + non-trivial + test split + unique)
- Sort by code_unit_id ascending (Python string ordering)
- random.Random(42).sample(pool, 45)
- No quotas, no manual picks

Usage:
    python scripts/reselect_after_regeneration.py \
        --generation-source-commit <commit> \
        [--expected-pool-size 2077]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localbench.workloads.code_retrieval.review import build_review_artifact
from localbench.workloads.code_retrieval.selection import (
    FINAL_QUERY_COUNT,
    SelectionError,
    build_eligible_pool,
    build_selection_record,
    pool_hash,
    select_final_queries,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = REPO_ROOT / "dataset" / "queries" / "candidates.jsonl"
V2_CANDIDATES_PATH = REPO_ROOT / "dataset" / "queries" / "candidates_v2.jsonl"
SPLITS_DIR = REPO_ROOT / "dataset" / "splits"
SELECTION_OUTPUT = REPO_ROOT / "dataset" / "queries" / "final_45_selection.json"
REVIEW_OUTPUT = REPO_ROOT / "dataset" / "queries" / "review_artifact.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("reselect_after_regeneration")


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _split_ids(name: str) -> set[str]:
    return {record["id"] for record in _load_jsonl(SPLITS_DIR / name)}


def _deduplicate_candidates(
    original: list[dict], v2: list[dict]
) -> list[dict]:
    """Merge original + v2 candidates, with v2 taking precedence.

    For each code_unit_id, the v2 record replaces the original.
    Original records are preserved in the audit trail but not in the pool.
    """
    by_unit: dict[str, dict] = {}
    for rec in original:
        by_unit[rec["code_unit_id"]] = rec
    for rec in v2:
        by_unit[rec["code_unit_id"]] = rec
    return list(by_unit.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--expected-pool-size", type=int, default=2077)
    parser.add_argument("--selection-version", default="1.0.0")
    args = parser.parse_args()

    original = _load_jsonl(CANDIDATES_PATH)
    v2 = _load_jsonl(V2_CANDIDATES_PATH)
    logger.info(
        "loaded %d original + %d v2 candidate records",
        len(original),
        len(v2),
    )

    candidates = _deduplicate_candidates(original, v2)
    logger.info("after deduplication: %d unique candidates", len(candidates))

    train_ids = _split_ids("train.jsonl")
    validation_ids = _split_ids("validation.jsonl")
    test_records = {r["id"]: r for r in _load_jsonl(SPLITS_DIR / "test.jsonl")}
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

    v2_in_pool = sum(
        1 for c in pool if c.get("candidate_id", "").startswith("candidate_v2_")
    )
    logger.info(
        "eligible pool: %d candidates (%d v2, %d original)",
        len(pool),
        v2_in_pool,
        len(pool) - v2_in_pool,
    )

    if len(pool) < FINAL_QUERY_COUNT:
        logger.error(
            "eligible pool has %d candidates; need at least %d - STOP",
            len(pool),
            FINAL_QUERY_COUNT,
        )
        return 2

    digest = pool_hash(pool)
    logger.info("eligible_pool_sha256: %s", digest)

    selected = select_final_queries(pool)

    verify_pool_hash = pool_hash(pool)
    verify_selected = select_final_queries(pool)
    if verify_pool_hash != digest or [
        c["code_unit_id"] for c in verify_selected
    ] != [c["code_unit_id"] for c in selected]:
        logger.error("reproducibility verification FAILED - STOP")
        return 3
    logger.info("reproducibility verification passed")

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

    repo_by_unit = {u: rec["repository"] for u, rec in test_records.items()}
    record = build_selection_record(
        ordered_pool=pool,
        selected=selected,
        repository_by_code_unit_id=repo_by_unit,
        generation_source_commit=args.generation_source_commit,
        selection_version=args.selection_version,
    )

    SELECTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SELECTION_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("selection record written: %s", SELECTION_OUTPUT)

    units_by_id = {uid: test_records[uid] for uid in units}
    candidates_by_id = {c["candidate_id"]: c for c in selected}
    review_artifact = build_review_artifact(
        selection_record=record,
        candidates_by_id=candidates_by_id,
        units_by_id=units_by_id,
        test_code_unit_ids=set(test_records),
        train_code_unit_ids=train_ids,
        validation_code_unit_ids=validation_ids,
    )

    with REVIEW_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(review_artifact, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("review artifact written: %s", REVIEW_OUTPUT)

    print(f"eligible_pool_sha256: {digest}")
    print(
        f"repository_distribution: {record['selected_repository_distribution']}"
    )
    print(
        f"query_style_distribution: {record['selected_query_style_distribution']}"
    )
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
