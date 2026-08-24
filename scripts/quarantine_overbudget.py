"""Phase 4F-I-C5B: quarantine over-budget v2 candidates and rebuild clean pool.

Dynamically detects CodeUnits whose total generation attempts exceed the
frozen 3-attempt maximum (DATASET_SPECIFICATION.md section 4.4.3), quarantines
the offending v2 candidate records, and rebuilds a clean eligible pool,
fresh selection, and review artifact.

No new queries are generated. No human review is performed.
No benchmarks are run. No ground truth is assigned.

Usage:
    python scripts/quarantine_overbudget.py \
        --generation-source-commit <commit> \
        [--selection-version 1.0.0]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
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
DATASET_ROOT = REPO_ROOT / "dataset"
CANDIDATES_PATH = DATASET_ROOT / "queries" / "candidates.jsonl"
FAILURES_PATH = DATASET_ROOT / "queries" / "candidate_failures.jsonl"
V2_CANDIDATES_PATH = DATASET_ROOT / "queries" / "candidates_v2.jsonl"
V2_FAILURES_PATH = DATASET_ROOT / "queries" / "candidate_failures_v2.jsonl"
SPLITS_DIR = DATASET_ROOT / "splits"
QUARANTINE_OUTPUT = DATASET_ROOT / "queries" / "quarantine_overbudget.json"
SELECTION_OUTPUT = DATASET_ROOT / "queries" / "final_45_selection.json"
REVIEW_OUTPUT = DATASET_ROOT / "queries" / "review_artifact.json"

MAX_ATTEMPTS = 3

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("quarantine_overbudget")


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


def _count_total_attempts(
    code_unit_id: str,
    orig_cands: dict[str, list[dict]],
    orig_fails: dict[str, list[dict]],
    v2_cands: dict[str, list[dict]],
    v2_fails: dict[str, list[dict]],
) -> int:
    """Sum attempt_count across all original + v2 records for a CodeUnit."""
    count = 0
    for rec in orig_cands.get(code_unit_id, []):
        count += rec.get("attempt_count", 0)
    for rec in orig_fails.get(code_unit_id, []):
        count += rec.get("attempt_count", 0)
    for rec in v2_cands.get(code_unit_id, []):
        count += rec.get("attempt_count", 0)
    for rec in v2_fails.get(code_unit_id, []):
        count += rec.get("attempt_count", 0)
    return count


def detect_overbudget(
    original_candidates: list[dict],
    original_failures: list[dict],
    v2_candidates: list[dict],
    v2_failures: list[dict],
) -> list[dict]:
    """Dynamically detect over-budget CodeUnits from artifact histories.

    Returns a list of quarantine entries for v2 candidates whose total
    generation attempts (original + v2) exceed MAX_ATTEMPTS.
    """
    # Index records by code_unit_id
    orig_cands_by_id: dict[str, list[dict]] = {}
    for rec in original_candidates:
        orig_cands_by_id.setdefault(rec["code_unit_id"], []).append(rec)

    orig_fails_by_id: dict[str, list[dict]] = {}
    for rec in original_failures:
        orig_fails_by_id.setdefault(rec["code_unit_id"], []).append(rec)

    v2_cands_by_id: dict[str, list[dict]] = {}
    for rec in v2_candidates:
        v2_cands_by_id.setdefault(rec["code_unit_id"], []).append(rec)

    v2_fails_by_id: dict[str, list[dict]] = {}
    for rec in v2_failures:
        v2_fails_by_id.setdefault(rec["code_unit_id"], []).append(rec)

    # All CodeUnits with any v2 record
    v2_code_unit_ids = set(v2_cands_by_id.keys()) | set(v2_fails_by_id.keys())

    quarantine_entries = []
    for uid in sorted(v2_code_unit_ids):
        total = _count_total_attempts(
            uid,
            orig_cands_by_id,
            orig_fails_by_id,
            v2_cands_by_id,
            v2_fails_by_id,
        )
        if total > MAX_ATTEMPTS:
            # Collect all v2 records for this CodeUnit
            v2_recs = v2_cands_by_id.get(uid, []) + v2_fails_by_id.get(uid, [])
            orig_rec_count = len(
                orig_cands_by_id.get(uid, []) + orig_fails_by_id.get(uid, [])
            )
            for v2_rec in v2_recs:
                quarantine_entries.append({
                    "code_unit_id": uid,
                    "candidate_id": v2_rec.get("candidate_id", "?"),
                    "generation_version": "v2",
                    "attempt_history": v2_rec.get("attempts", []),
                    "total_attempts": total,
                    "max_allowed_attempts": MAX_ATTEMPTS,
                    "quarantine_reason": (
                        f"total generation attempts ({total}) exceeds "
                        f"maximum ({MAX_ATTEMPTS})"
                    ),
                    "quarantine_timestamp": datetime.now(timezone.utc).isoformat(),
                    "original_record_count": orig_rec_count,
                    "v2_record_count": len(v2_recs),
                })

    return quarantine_entries


def build_clean_pool(
    original_candidates: list[dict],
    v2_candidates: list[dict],
    quarantined_ids: set[str],
    test_code_unit_ids: set[str],
    train_code_unit_ids: set[str] | None = None,
    validation_code_unit_ids: set[str] | None = None,
) -> list[dict]:
    """Build eligible pool excluding quarantined v2 candidates.

    For each CodeUnit:
    - If the v2 candidate is quarantined, use only the original (if eligible)
    - If the v2 candidate is not quarantined and successful, use v2
    - Otherwise use original (if eligible)
    """
    # Index original candidates by code_unit_id
    orig_by_id: dict[str, dict] = {}
    for rec in original_candidates:
        orig_by_id[rec["code_unit_id"]] = rec

    # Index v2 candidates by code_unit_id (only non-quarantined)
    v2_by_id: dict[str, dict] = {}
    for rec in v2_candidates:
        if rec["candidate_id"] not in quarantined_ids:
            v2_by_id[rec["code_unit_id"]] = rec

    # Build merged candidate list: v2 takes precedence where not quarantined
    merged = []
    seen_ids: set[str] = set()
    for rec in original_candidates:
        uid = rec["code_unit_id"]
        if uid in seen_ids:
            continue
        if uid in v2_by_id:
            merged.append(v2_by_id[uid])
        else:
            merged.append(rec)
        seen_ids.add(uid)

    return build_eligible_pool(
        merged,
        test_code_unit_ids=test_code_unit_ids,
        train_code_unit_ids=train_code_unit_ids,
        validation_code_unit_ids=validation_code_unit_ids,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--selection-version", default="1.0.0")
    args = parser.parse_args()

    # Load all artifacts
    original_candidates = _load_jsonl(CANDIDATES_PATH)
    original_failures = _load_jsonl(FAILURES_PATH)
    v2_candidates = _load_jsonl(V2_CANDIDATES_PATH)
    v2_failures = _load_jsonl(V2_FAILURES_PATH)
    logger.info(
        "loaded: %d orig cands, %d orig fails, %d v2 cands, %d v2 fails",
        len(original_candidates),
        len(original_failures),
        len(v2_candidates),
        len(v2_failures),
    )

    # Detect over-budget
    quarantine_entries = detect_overbudget(
        original_candidates, original_failures, v2_candidates, v2_failures
    )
    quarantined_v2_ids = {e["candidate_id"] for e in quarantine_entries}
    quarantined_unit_ids = {e["code_unit_id"] for e in quarantine_entries}
    logger.info(
        "detected %d over-budget CodeUnits, %d quarantined v2 records",
        len(quarantined_unit_ids),
        len(quarantined_v2_ids),
    )
    for entry in quarantine_entries:
        logger.info(
            "  QUARANTINE: %s (candidate=%s, total=%d)",
            entry["code_unit_id"],
            entry["candidate_id"],
            entry["total_attempts"],
        )

    # Write quarantine artifact
    quarantine_artifact = {
        "quarantine_version": "1.0.0",
        "quarantine_reason": (
            "v2 candidate generation exceeded "
            "frozen 3-attempt maximum"
        ),
        "frozen_maximum_attempts": MAX_ATTEMPTS,
        "total_overbudget_code_units": len(quarantined_unit_ids),
        "total_quarantined_v2_records": len(quarantined_v2_ids),
        "quarantine_timestamp": datetime.now(timezone.utc).isoformat(),
        "entries": quarantine_entries,
    }
    QUARANTINE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with QUARANTINE_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(quarantine_artifact, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("quarantine artifact written: %s", QUARANTINE_OUTPUT)

    # Load splits
    train_ids = _split_ids("train.jsonl")
    validation_ids = _split_ids("validation.jsonl")
    test_records = {r["id"]: r for r in _load_jsonl(SPLITS_DIR / "test.jsonl")}
    logger.info(
        "splits: train=%d validation=%d test=%d",
        len(train_ids),
        len(validation_ids),
        len(test_records),
    )

    # Build clean eligible pool
    pool = build_clean_pool(
        original_candidates,
        v2_candidates,
        quarantined_v2_ids,
        test_code_unit_ids=set(test_records),
        train_code_unit_ids=train_ids,
        validation_code_unit_ids=validation_ids,
    )

    v2_in_pool = sum(
        1 for c in pool if c.get("candidate_id", "").startswith("candidate_v2_")
    )
    orig_in_pool = len(pool) - v2_in_pool
    logger.info(
        "clean eligible pool: %d candidates (%d v2, %d original)",
        len(pool),
        v2_in_pool,
        orig_in_pool,
    )

    if len(pool) < FINAL_QUERY_COUNT:
        logger.error(
            "clean pool has %d candidates; need at least %d - STOP",
            len(pool),
            FINAL_QUERY_COUNT,
        )
        return 2

    # Compute pool hash
    digest = pool_hash(pool)
    logger.info("eligible_pool_sha256: %s", digest)

    # Deterministic selection
    selected = select_final_queries(pool)

    # Verify reproducibility
    verify_pool_hash = pool_hash(pool)
    verify_selected = select_final_queries(pool)
    if verify_pool_hash != digest or [
        c["code_unit_id"] for c in verify_selected
    ] != [c["code_unit_id"] for c in selected]:
        logger.error("reproducibility verification FAILED - STOP")
        return 3
    logger.info("reproducibility verification passed")

    # Post-selection audit
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
    # Verify no quarantined v2 candidates in selection
    assert not (set(cand_ids) & quarantined_v2_ids)
    logger.info("post-selection audit passed")

    # Build selection record
    repo_by_unit = {u: rec["repository"] for u, rec in test_records.items()}
    record = build_selection_record(
        ordered_pool=pool,
        selected=selected,
        repository_by_code_unit_id=repo_by_unit,
        generation_source_commit=args.generation_source_commit,
        selection_version=args.selection_version,
    )

    # Write selection record
    SELECTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SELECTION_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("selection record written: %s", SELECTION_OUTPUT)

    # Build review artifact
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

    # Summary
    print(f"quarantined_code_units: {len(quarantined_unit_ids)}")
    print(f"quarantined_v2_records: {len(quarantined_v2_ids)}")
    print(f"clean_pool_size: {len(pool)}")
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
