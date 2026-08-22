"""Tests for the fail-fast exclusive generation output-directory lock."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from localbench.workloads.code_retrieval.run_lock import (
    LOCK_FILENAME,
    GenerationLockError,
    generation_run_lock,
)

_SRC_ROOT = str(Path(__file__).resolve().parents[3] / "src")

_RACE_CHILD = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    sys.path.insert(0, {src!r})
    from localbench.workloads.code_retrieval.run_lock import (
        GenerationLockError,
        generation_run_lock,
    )
    time.sleep({delay})
    try:
        with generation_run_lock(Path({directory!r})):
            time.sleep({hold})
            print("ACQUIRED", flush=True)
    except GenerationLockError:
        print("REFUSED", flush=True)
    """
)


def _write_sentinel(directory: Path, content: bytes) -> None:
    (directory / LOCK_FILENAME).write_bytes(content)


class TestAcquisitionAndRelease:
    def test_first_acquire_creates_sentinel(self, tmp_path):
        with generation_run_lock(tmp_path) as lock_path:
            assert lock_path == tmp_path / LOCK_FILENAME
            owner = json.loads(lock_path.read_text(encoding="utf-8"))
            assert owner["pid"] == os.getpid()
            assert owner["created_utc"]

    def test_second_concurrent_acquisition_fails(self, tmp_path):
        with generation_run_lock(tmp_path):
            with pytest.raises(GenerationLockError, match="already using"):
                with generation_run_lock(tmp_path):
                    pass
            # The refused attempt left the live owner's sentinel intact.
            assert (tmp_path / LOCK_FILENAME).exists()

    def test_release_on_normal_exit(self, tmp_path):
        with generation_run_lock(tmp_path):
            pass
        assert not (tmp_path / LOCK_FILENAME).exists()

    def test_release_on_exception(self, tmp_path):
        with pytest.raises(RuntimeError):
            with generation_run_lock(tmp_path):
                raise RuntimeError("generator crashed mid-run")
        assert not (tmp_path / LOCK_FILENAME).exists()

    def test_reacquire_after_release(self, tmp_path):
        for _ in range(2):
            with generation_run_lock(tmp_path):
                assert (tmp_path / LOCK_FILENAME).exists()
        assert list(tmp_path.iterdir()) == []


class TestFailureMessage:
    def test_failure_message_is_actionable(self, tmp_path):
        with generation_run_lock(tmp_path):
            with pytest.raises(GenerationLockError) as excinfo:
                with generation_run_lock(tmp_path):
                    pass
        message = str(excinfo.value)
        assert str(tmp_path) in message
        assert "Stop that run" in message
        assert "delete" in message
        assert "rerun" in message


class TestExistingSentinelIsNeverStolen:
    def test_existing_sentinel_refused_without_being_modified(
        self, tmp_path
    ):
        _write_sentinel(tmp_path, b'{"pid": 999999999}')
        with pytest.raises(GenerationLockError, match="already using"):
            with generation_run_lock(tmp_path):
                pass
        assert (
            (tmp_path / LOCK_FILENAME).read_bytes()
            == b'{"pid": 999999999}'
        )

    def test_corrupt_sentinel_content_still_refuses(self, tmp_path):
        _write_sentinel(tmp_path, b"not json at all")
        with pytest.raises(GenerationLockError, match="already using"):
            with generation_run_lock(tmp_path):
                pass
        assert (
            (tmp_path / LOCK_FILENAME).read_bytes() == b"not json at all"
        )

    def test_manual_deletion_restores_availability(self, tmp_path):
        """Documented operator recovery for a crashed run's sentinel."""
        _write_sentinel(tmp_path, b'{"pid": 1}')
        with pytest.raises(GenerationLockError):
            with generation_run_lock(tmp_path):
                pass
        (tmp_path / LOCK_FILENAME).unlink()
        with generation_run_lock(tmp_path):
            pass
        assert not (tmp_path / LOCK_FILENAME).exists()


class TestProcessRace:
    def test_exactly_one_of_two_processes_wins(self, tmp_path):
        winner = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _RACE_CHILD.format(
                    src=_SRC_ROOT,
                    directory=str(tmp_path),
                    delay=0.0,
                    hold=1.0,
                ),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.35)  # winner owns the lock before contender attempts
        contender = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _RACE_CHILD.format(
                    src=_SRC_ROOT,
                    directory=str(tmp_path),
                    delay=0.0,
                    hold=0.1,
                ),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        outcome_winner = winner.communicate(timeout=30)[0].strip()
        outcome_contender = contender.communicate(timeout=30)[0].strip()
        assert outcome_winner == "ACQUIRED"
        assert outcome_contender == "REFUSED"
        # Winner released on exit; contender never created anything.
        assert list(tmp_path.iterdir()) == []
