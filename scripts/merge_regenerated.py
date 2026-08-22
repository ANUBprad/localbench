"""Merge regenerated conflicted-unit results into the recovered checkpoint.

Combines the isolated recovery checkpoint (``candidates.jsonl`` /
``candidate_failures.jsonl``) with the staged regeneration output
produced by ``generate_query_candidates.py`` so that every completed
CodeUnit ends up with exactly one active record:

    recovered non-conflicted candidates
    + regenerated successful candidates
    -> candidates.jsonl

    recovered failure audit records
    + regenerated failure audit records
    -> candidate_failures.jsonl

Safety properties:
- Read-only audits first: any ID overlap between the two sources, any
  duplicate ID inside a source, or any unresolved recovery conflict in
  the merged sets aborts with nothing written.
- Current artifacts are copied to ``<name>.pre-merge.bak`` first; an
  existing backup aborts instead of being overwritten.
- Merged content is written to temporary files, validated by loading
  them through CandidateStore, and moved into place with ``os.replace``.
- Re-running after a complete merge is a no-op.

Usage:
    python scripts/merge_regenerated.py --dry-run
    python scripts/merge_regenerated.py
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localbench.workloads.code_retrieval.artifact_recovery import (
    ArtifactRecoveryError,
    _semantic_signature,
    plan_recovery,
    read_jsonl,
    serialize_records,
)
from localbench.workloads.code_retrieval.candidate_store import CandidateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
CANDIDATES_FILENAME = "candidates.jsonl"
FAILURES_FILENAME = "candidate_failures.jsonl"
BACKUP_SUFFIX = ".pre-merge.bak"
TEMP_SUFFIX = ".merge-tmp"

EXIT_OK = 0
EXIT_ERROR = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries-dir",
        type=Path,
        default=DATASET_ROOT / "queries",
        help="Directory holding the recovered checkpoint.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DATASET_ROOT / "queries" / "regen_staging",
        help="Directory holding the staged regeneration results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and report only; never write.",
    )
    return parser.parse_args(argv)


def _write_temp(target: Path, content: str) -> Path:
    temp_path = target.with_name(target.name + TEMP_SUFFIX)
    with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return temp_path


def _validate_pair(candidates_path: Path, failures_path: Path) -> tuple[int, int]:
    store = CandidateStore(candidates_path, failures_path)
    try:
        successful, failed = store.load()
    finally:
        store.close()
    return len(successful), len(failed)


def _audit_disjoint(
    label_a: str, ids_a: set[str], label_b: str, ids_b: set[str]
) -> None:
    overlap = ids_a & ids_b
    if overlap:
        raise ArtifactRecoveryError(
            f"CodeUnit IDs present in both {label_a} and {label_b}: "
            f"{sorted(overlap)[:5]}"
        )


def _audit_unique(label: str, records: list[dict]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        unit_id = record["code_unit_id"]
        if unit_id in seen:
            duplicates.append(unit_id)
        seen.add(unit_id)
    if duplicates:
        raise ArtifactRecoveryError(
            f"Duplicate CodeUnit IDs in {label}: {sorted(duplicates)[:5]}"
        )


def merge(queries_dir: Path, staging_dir: Path, dry_run: bool) -> int:
    candidates_path = queries_dir / CANDIDATES_FILENAME
    failures_path = queries_dir / FAILURES_FILENAME
    staging_candidates_path = staging_dir / CANDIDATES_FILENAME
    staging_failures_path = staging_dir / FAILURES_FILENAME

    recovered_candidates = read_jsonl(candidates_path)
    recovered_failures = read_jsonl(failures_path)
    regen_candidates = read_jsonl(staging_candidates_path)
    regen_failures = read_jsonl(staging_failures_path)

    logger.info(
        "Recovered checkpoint: %d candidates / %d failures",
        len(recovered_candidates),
        len(recovered_failures),
    )
    logger.info(
        "Staged regeneration:  %d candidates / %d failures",
        len(regen_candidates),
        len(regen_failures),
    )

    staged_records = regen_candidates + regen_failures
    checkpoint_records = recovered_candidates + recovered_failures
    if staged_records and all(
        record in checkpoint_records for record in staged_records
    ):
        logger.info(
            "Staged results are already merged into the checkpoint "
            "(%d candidates / %d failures); nothing to do.",
            len(recovered_candidates),
            len(recovered_failures),
        )
        return EXIT_OK

    _audit_unique("recovered candidates", recovered_candidates)
    _audit_unique("recovered failures", recovered_failures)
    _audit_unique("staged candidates", regen_candidates)
    _audit_unique("staged failures", regen_failures)

    _audit_disjoint(
        "recovered candidates",
        {r["code_unit_id"] for r in recovered_candidates},
        "staged candidates",
        {r["code_unit_id"] for r in regen_candidates},
    )
    _audit_disjoint(
        "recovered candidates",
        {r["code_unit_id"] for r in recovered_candidates},
        "staged failures",
        {r["code_unit_id"] for r in regen_failures},
    )
    _audit_disjoint(
        "recovered failures",
        {r["code_unit_id"] for r in recovered_failures},
        "staged candidates",
        {r["code_unit_id"] for r in regen_candidates},
    )
    _audit_disjoint(
        "recovered failures",
        {r["code_unit_id"] for r in recovered_failures},
        "staged failures",
        {r["code_unit_id"] for r in regen_failures},
    )

    merged_candidates = sorted(
        recovered_candidates + regen_candidates,
        key=lambda r: r["code_unit_id"],
    )
    merged_failures = sorted(
        recovered_failures + regen_failures,
        key=lambda r: r["code_unit_id"],
    )

    plan = plan_recovery(merged_candidates, merged_failures)
    if not plan.recoverable:
        logger.error(
            "Merged sets still contain %d unresolved conflict(s):",
            len(plan.conflicts),
        )
        for conflict in plan.conflicts[:10]:
            logger.error(
                "  [%s] %s: %s",
                conflict.kind,
                conflict.code_unit_id,
                conflict.detail,
            )
        return EXIT_ERROR
    signatures: dict[str, tuple] = {}
    for record in merged_candidates:
        unit_id = record["code_unit_id"]
        signature = _semantic_signature(record)
        if unit_id in signatures and signatures[unit_id] != signature:
            raise ArtifactRecoveryError(
                f"Merged candidates contain diverging payloads for "
                f"'{unit_id}'."
            )
        signatures[unit_id] = signature

    new_candidates = serialize_records(merged_candidates)
    new_failures = serialize_records(merged_failures)

    current_candidates = (
        candidates_path.read_text(encoding="utf-8")
        if candidates_path.exists()
        else ""
    )
    current_failures = (
        failures_path.read_text(encoding="utf-8")
        if failures_path.exists()
        else ""
    )
    if (
        new_candidates == current_candidates
        and new_failures == current_failures
    ):
        logger.info(
            "Checkpoint already merged (%d candidates / %d failures); "
            "nothing to do.",
            len(merged_candidates),
            len(merged_failures),
        )
        return EXIT_OK

    logger.info(
        "Merge plan: %d candidates / %d failures "
        "(%d + %d and %d + %d)",
        len(merged_candidates),
        len(merged_failures),
        len(recovered_candidates),
        len(regen_candidates),
        len(recovered_failures),
        len(regen_failures),
    )
    if dry_run:
        logger.info("Dry run: no files were modified.")
        return EXIT_OK

    for artifact in (candidates_path, failures_path):
        backup = artifact.with_name(artifact.name + BACKUP_SUFFIX)
        if backup.exists():
            logger.error(
                "Backup %s already exists; refusing to overwrite the "
                "previous snapshot. Remove it deliberately first.",
                backup,
            )
            return EXIT_ERROR
        shutil.copy2(artifact, backup)
        logger.info("Backed up %s -> %s", artifact, backup)

    temp_candidates = _write_temp(candidates_path, new_candidates)
    try:
        temp_failures = _write_temp(failures_path, new_failures)
        try:
            loaded_success, loaded_failed = _validate_pair(
                temp_candidates, temp_failures
            )
            expected = (len(merged_candidates), len(merged_failures))
            if (loaded_success, loaded_failed) != expected:
                raise ArtifactRecoveryError(
                    f"Merged pair failed self-check: loaded "
                    f"{loaded_success}/{loaded_failed}, planned {expected}."
                )
            os.replace(temp_candidates, candidates_path)
            os.replace(temp_failures, failures_path)
        finally:
            temp_failures.unlink(missing_ok=True)
    finally:
        temp_candidates.unlink(missing_ok=True)

    final_success, final_failed = _validate_pair(candidates_path, failures_path)
    logger.info(
        "Merge complete: %d successful candidates, %d failures; "
        "checkpoint loads cleanly.",
        final_success,
        final_failed,
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return merge(args.queries_dir, args.staging_dir, args.dry_run)
    except ArtifactRecoveryError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
