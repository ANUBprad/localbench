"""Phase 4F-I-B: recover duplicated query-generation artifacts.

Repairs ``candidates.jsonl`` / ``candidate_failures.jsonl`` produced by
the overlapping pre-lock generation runs into a checkpoint that
CandidateStore can load, following the deterministic rules implemented
in ``localbench.workloads.code_retrieval.artifact_recovery``.

Safety properties:
- Read-only planning first: semantic candidate conflicts or ambiguous
  cross-file outcomes abort with exit code 2 and touch nothing.
- Originals are copied to ``<name>.pre-recovery.bak`` before any
  change; an existing backup aborts instead of being overwritten.
- Recovered content is written to temporary files in the same
  directory, validated by loading them through CandidateStore, and
  moved into place with ``os.replace``.
- Re-running on already-recovered artifacts is a no-op.

Usage:
    python scripts/recover_query_candidates.py --dry-run
    python scripts/recover_query_candidates.py
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
    cross_file_overview,
    file_stats,
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
BACKUP_SUFFIX = ".pre-recovery.bak"
TEMP_SUFFIX = ".recover-tmp"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFLICTS = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries-dir",
        type=Path,
        default=DATASET_ROOT / "queries",
        help="Directory holding the two JSONL artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and report only; never write.",
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


def recover(queries_dir: Path, dry_run: bool) -> int:
    candidates_path = queries_dir / CANDIDATES_FILENAME
    failures_path = queries_dir / FAILURES_FILENAME

    candidates = read_jsonl(candidates_path)
    failures = read_jsonl(failures_path)

    logger.info("Input candidates: %s", file_stats(candidates))
    logger.info("Input failures:   %s", file_stats(failures))
    overview = cross_file_overview(candidates, failures)
    logger.info(
        "Cross-file: only-candidates=%d only-failures=%d both=%d",
        len(overview["only_candidates"]),
        len(overview["only_failures"]),
        len(overview["both"]),
    )

    plan = plan_recovery(candidates, failures)
    if not plan.recoverable:
        logger.error(
            "Recovery refused: %d unresolved conflict(s). Nothing was "
            "written; decide each case explicitly before retrying:",
            len(plan.conflicts),
        )
        for conflict in sorted(
            plan.conflicts, key=lambda c: (c.kind, c.code_unit_id)
        ):
            logger.error(
                "  [%s] %s: %s",
                conflict.kind,
                conflict.code_unit_id,
                conflict.detail,
            )
        return EXIT_CONFLICTS

    new_candidates = serialize_records(plan.candidates)
    new_failures = serialize_records(plan.failures)

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
            "Artifacts are already recovered (%d candidates / %d "
            "failures); nothing to do.",
            len(plan.candidates),
            len(plan.failures),
        )
        return EXIT_OK

    logger.info(
        "Plan: %d canonical candidates, %d canonical failures, "
        "%d superseded records",
        len(plan.candidates),
        len(plan.failures),
        len(plan.superseded),
    )
    if dry_run:
        logger.info("Dry run: no files were modified.")
        return EXIT_OK

    for artifact in (candidates_path, failures_path):
        backup = artifact.with_name(artifact.name + BACKUP_SUFFIX)
        if backup.exists():
            logger.error(
                "Backup %s already exists; refusing to overwrite the "
                "original snapshot. Remove it deliberately first.",
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
            expected = (len(plan.candidates), len(plan.failures))
            if (loaded_success, loaded_failed) != expected:
                raise ArtifactRecoveryError(
                    f"Recovered pair failed self-check: loaded "
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
        "Recovery complete: %d successful candidates, %d failures; "
        "checkpoint loads cleanly.",
        final_success,
        final_failed,
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return recover(args.queries_dir, args.dry_run)
    except ArtifactRecoveryError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
