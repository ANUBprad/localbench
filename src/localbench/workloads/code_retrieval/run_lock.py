"""Exclusive single-process lock for a candidate output directory.

Candidate generation appends to shared JSONL checkpoint artifacts inside
one output directory. Two overlapping processes corrupt that checkpoint
(observed as duplicate CodeUnit records during Phase 4F-I-B), so a
process may write there only while holding this lock.

Mechanism: an ``O_CREAT | O_EXCL`` sentinel file inside the directory,
whose creation is atomic on Windows and POSIX. Acquisition is a single
attempt: if the sentinel exists, another run owns the directory and this
one fails immediately. The sentinel content (owner PID, creation time)
is written for human inspection only and is never read back, so there is
no window in which a contender can misjudge an owner as stale.

Stale locks: a hard crash (power loss, kill -9) can leave a sentinel
behind. It is never removed automatically — deleting another process's
lock cannot be made safe from file contents alone. Recovery is manual
and deliberate: verify no generator process is running, delete the
sentinel, rerun; checkpoint resume then continues normally.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LOCK_FILENAME = ".generate_query_candidates.lock"


class GenerationLockError(Exception):
    """The candidate generation lock could not be acquired."""


@contextmanager
def generation_run_lock(output_dir: Path):
    """Hold exclusive generation access to *output_dir*.

    Raises ``GenerationLockError`` immediately when the directory is
    already locked; never waits or retries. The sentinel is removed on
    normal exit and on exceptions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILENAME
    _acquire(lock_path)
    try:
        yield lock_path
    finally:
        try:
            lock_path.unlink(missing_ok=True)
            logger.info("Released generation lock %s", lock_path)
        except PermissionError:
            # An external handle (editor, indexer, antivirus) is holding
            # the sentinel open. The run itself is finished; leaving the
            # file behind is recoverable via the documented manual
            # deletion, whereas raising here would mask completed work.
            logger.warning(
                "Could not remove generation lock %s; if no generator "
                "process is running, delete it manually.",
                lock_path,
            )


def _acquire(lock_path: Path) -> None:
    """Create the sentinel atomically or fail fast if it exists."""
    try:
        handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise GenerationLockError(
            f"Another generation run is already using "
            f"'{lock_path.parent}': lock {lock_path} exists. Stop that "
            "run or wait for it to finish. If no generator process is "
            "running (for example after a crash), verify that, delete "
            f"{lock_path} manually, and rerun. The file records the "
            "owning PID and creation time for inspection."
        ) from None
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
