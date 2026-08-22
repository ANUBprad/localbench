"""Exclusive single-process lock for a candidate output directory.

Candidate generation appends to shared JSONL checkpoint artifacts inside
one output directory. Two overlapping processes corrupt that checkpoint
(observed as duplicate CodeUnit records during Phase 4F-I-B), so a
process may write there only while holding this lock.

Mechanism: an ``O_CREAT | O_EXCL`` sentinel file inside the directory,
whose creation is atomic on Windows and POSIX. The file records the
owning PID so a sentinel left behind by a crashed run can be recognized.
A stale sentinel is broken only when its recorded PID provably no longer
exists; a process that exists can still call ``release()``, so breaking
its lock could let a third process acquire a freed sentinel. PID reuse
makes detection conservative (a reused PID looks alive), which can only
refuse a start — never allow two concurrent generators.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)

LOCK_FILENAME = ".generate_query_candidates.lock"


class GenerationLockError(Exception):
    """Another live process holds the candidate generation lock."""


@contextmanager
def generation_run_lock(output_dir: Path):
    """Hold exclusive generation access to *output_dir*.

    Raises ``GenerationLockError`` when another live process owns the
    directory. The sentinel is removed on normal exit and on exceptions,
    so an interrupted run never blocks the next one.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILENAME
    _acquire(lock_path)
    try:
        yield lock_path
    finally:
        lock_path.unlink(missing_ok=True)
        logger.info("Released generation lock %s", lock_path)


def _acquire(lock_path: Path) -> None:
    while True:
        try:
            handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            _handle_existing(lock_path)
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pid": os.getpid(),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                },
                f,
            )
            f.flush()
            os.fsync(f.fileno())
        logger.info("Acquired generation lock %s", lock_path)
        return


def _handle_existing(lock_path: Path) -> None:
    """Fail fast on a live owner, otherwise move a stale sentinel aside."""
    owner_pid = _read_owner_pid(lock_path)
    if owner_pid is not None and psutil.pid_exists(owner_pid):
        raise GenerationLockError(
            f"Another generation run is already using '{lock_path.parent}': "
            f"lock {lock_path} is held by PID {owner_pid}. "
            "Stop that process or wait for it to finish."
        )
    if not _break_stale(lock_path):
        raise GenerationLockError(
            f"The generation lock {lock_path} exists but could not be "
            "read or identified. Verify no generator process is running, "
            f"then delete {lock_path} and rerun."
        )


def _read_owner_pid(lock_path: Path) -> int | None:
    """Return the owning PID, or ``None`` if unreadable or malformed."""
    try:
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        return int(owner["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _break_stale(lock_path: Path) -> bool:
    """Rename a stale sentinel aside so a fresh one can be created.

    Returns ``True`` when the path is free to retry (we renamed it, or
    another process renamed it first and the next exclusive create will
    arbitrate). Returns ``False`` only when the sentinel cannot be
    interpreted at all.
    """
    if _read_owner_pid(lock_path) is None:
        return False
    stale_name = f"{lock_path.name}.stale-{uuid.uuid4().hex[:8]}"
    try:
        os.replace(lock_path, lock_path.with_name(stale_name))
    except FileNotFoundError:
        # Another process broke it first; retrying acquisition finds
        # either free space or that winner's fresh lock.
        return True
    logger.warning("Broke stale generation lock %s", lock_path)
    return True
