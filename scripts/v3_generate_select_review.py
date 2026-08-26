"""V3.0 generation + pool rebuild + selection + review artifact.

End-to-end orchestration for the first v3.0-methodology run:
  1. Generate candidates for all test CodeUnits using v3.0 pipeline
  2. Build eligible pool (§4.4.5)
  3. Select exactly 45 (seed=42)
  4. Build fresh benchmark-blind review artifact

Historical candidate files are NOT modified.  The existing candidates
are backed up before generation overwrites them.

Usage:
    python scripts/v3_generate_select_review.py
    python scripts/v3_generate_select_review.py --limit 5   # smoke test
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localbench.workloads.code_retrieval.review import (
    ReviewArtifactError,
    build_review_artifact,
    validate_review_artifact,
)
from localbench.workloads.code_retrieval.selection import (
    SelectionError,
    build_eligible_pool,
    build_selection_record,
    distribution,
    pool_hash,
    select_final_queries,
)

SPLITS_DIR = REPO_ROOT / "dataset" / "splits"
QUERIES_DIR = REPO_ROOT / "dataset" / "queries"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _split_ids(name: str) -> set[str]:
    return {record["id"] for record in _load_jsonl(SPLITS_DIR / name)}


def step1_generate(limit: int | None = None) -> int:
    """Run the v3.0 generation pipeline via the existing script."""
    from scripts.generate_query_candidates import main as gen_main

    argv = ["--fresh"]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    return gen_main(argv)


def step2_select() -> int:
    """Build eligible pool, select 45, write selection record."""
    candidates_path = QUERIES_DIR / "candidates.jsonl"
    candidates = _load_jsonl(candidates_path)
    logger.info("loaded %d candidate records", len(candidates))

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
    logger.info("eligible pool: %d candidates", len(pool))

    if len(pool) < 45:
        logger.error(
            "eligible pool has %d candidates; need at least 45", len(pool)
        )
        return 2

    digest = pool_hash(pool)
    logger.info("eligible_pool_sha256: %s", digest)

    selected = select_final_queries(pool)
    assert len(selected) == 45

    verify_digest = pool_hash(pool)
    verify_selected = select_final_queries(pool)
    assert verify_digest == digest
    assert [c["code_unit_id"] for c in verify_selected] == [
        c["code_unit_id"] for c in selected
    ]
    logger.info("reproducibility verification passed")

    units = [c["code_unit_id"] for c in selected]
    cand_ids = [c["candidate_id"] for c in selected]
    assert len(set(units)) == 45
    assert len(set(cand_ids)) == 45
    assert all(u in test_records for u in units)
    assert not (set(units) & train_ids)
    assert not (set(units) & validation_ids)
    assert all(c["leakage_passed"] and c["validation_passed"] for c in selected)
    pool_by_unit = {c["code_unit_id"]: c for c in pool}
    assert all(u in pool_by_unit for u in units)
    logger.info("post-selection audit passed")

    repo_by_unit = {u: rec["repository"] for u, rec in test_records.items()}
    import platform
    record = build_selection_record(
        ordered_pool=pool,
        selected=selected,
        repository_by_code_unit_id=repo_by_unit,
        generation_source_commit="HEAD",
        selection_version="3.0.0",
    )

    output = QUERIES_DIR / "final_45_selection.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("selection record written: %s", output)

    print(f"eligible_pool_sha256: {digest}")
    print(f"repository_distribution: {record['selected_repository_distribution']}")
    print(f"query_style_distribution: {record['selected_query_style_distribution']}")
    print("selected_code_unit_ids:")
    for unit in units:
        print(f"  {unit}")
    return 0


def step3_review() -> int:
    """Build the benchmark-blind review artifact."""
    selection_path = QUERIES_DIR / "final_45_selection.json"
    candidates_path = QUERIES_DIR / "candidates.jsonl"
    test_split_path = SPLITS_DIR / "test.jsonl"
    train_split_path = SPLITS_DIR / "train.jsonl"
    validation_split_path = SPLITS_DIR / "validation.jsonl"
    output_path = QUERIES_DIR / "review_artifact.json"

    with open(selection_path, encoding="utf-8") as f:
        selection_record = json.load(f)

    all_candidates = _load_jsonl(candidates_path)
    candidates_by_id = {c["candidate_id"]: c for c in all_candidates}

    test_units = _load_jsonl(test_split_path)
    units_by_id = {u["id"]: u for u in test_units}
    test_code_unit_ids = {u["id"] for u in test_units}

    train_units = _load_jsonl(train_split_path) if train_split_path.exists() else []
    validation_units = (
        _load_jsonl(validation_split_path) if validation_split_path.exists() else []
    )
    train_ids = {u["id"] for u in train_units}
    validation_ids = {u["id"] for u in validation_units}

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
        logger.error("Review artifact build failed: %s", exc)
        return 1

    errors = validate_review_artifact(artifact)
    if errors:
        for err in errors:
            logger.error("VIOLATION: %s", err)
        return 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    all_pending = all(item["review"]["state"] == "pending" for item in artifact["items"])
    logger.info(
        "Review artifact written: %s (%d items, all_pending=%s)",
        output_path,
        len(artifact["items"]),
        all_pending,
    )
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate for only the first N test units (smoke testing).",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation step (use existing candidates).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()

    if not args.skip_generation:
        logger.info("=" * 60)
        logger.info("STEP 1: V3.0 CANDIDATE GENERATION")
        logger.info("=" * 60)
        rc = step1_generate(limit=args.limit)
        if rc != 0:
            logger.error("Generation failed with exit code %d", rc)
            return rc
    else:
        logger.info("Skipping generation (--skip-generation)")

    logger.info("=" * 60)
    logger.info("STEP 2: POOL REBUILD + SELECTION")
    logger.info("=" * 60)
    rc = step2_select()
    if rc != 0:
        return rc

    logger.info("=" * 60)
    logger.info("STEP 3: REVIEW ARTIFACT")
    logger.info("=" * 60)
    rc = step3_review()
    if rc != 0:
        return rc

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("V3.0 PIPELINE COMPLETE (%.1f s)", elapsed)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
