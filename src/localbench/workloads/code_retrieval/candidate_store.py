"""Incremental, crash-safe persistence for candidate audit records.

Candidate generation runs can last many hours, so every completed
CodeUnit — successful or failed — is appended to its JSONL artifact
immediately and flushed to disk. The two JSONL files double as the
checkpoint: on restart, completed CodeUnit IDs are reconstructed from
them and skipped.

Crash safety:
- Each record is written as one line followed by flush + fsync.
- A torn trailing line (process died mid-write) is detected at load
  time and truncated; it is never treated as a successful candidate.
- A corrupt record in the middle of a file, a record with an invalid
  shape, or a duplicated CodeUnit ID (within one file or across both)
  aborts loading: these indicate real corruption that must not be
  silently repaired.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"code_unit_id": str, "success": bool}


class CandidateArtifactError(Exception):
    """Candidate artifacts are corrupted and cannot be safely used."""


def validate_record(record: object) -> None:
    """Raise ``CandidateArtifactError`` if *record* is not well-formed."""
    if not isinstance(record, dict):
        raise CandidateArtifactError(
            f"Record must be a JSON object, got {type(record).__name__}."
        )
    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in record:
            raise CandidateArtifactError(f"Record missing required key '{key}'.")
        if not isinstance(record[key], expected_type):
            raise CandidateArtifactError(
                f"Record key '{key}' expected {expected_type.__name__}, "
                f"got {type(record[key]).__name__}."
            )
        if expected_type is str and not record[key]:
            raise CandidateArtifactError(f"Record key '{key}' must be non-empty.")


class CandidateStore:
    """Append-only JSONL store for successful and failed candidates."""

    def __init__(self, candidates_path: Path, failures_path: Path) -> None:
        self.candidates_path = candidates_path
        self.failures_path = failures_path
        self._seen_ids: set[str] = set()
        self._handles: dict[Path, object] = {}

    # -- checkpoint reconstruction ------------------------------------

    def load(self) -> tuple[list[dict], list[dict]]:
        """Load existing records from both artifact files.

        Returns ``(successful_records, failed_records)``. Repairs a torn
        trailing line; raises ``CandidateArtifactError`` on corruption,
        invalid records, or duplicate CodeUnit IDs.
        """
        self._repair_torn_tail(self.candidates_path)
        self._repair_torn_tail(self.failures_path)

        successful, success_ids = self._load_file(
            self.candidates_path, expected_success=True
        )
        failed, failure_ids = self._load_file(
            self.failures_path, expected_success=False
        )

        cross_duplicates = success_ids & failure_ids
        if cross_duplicates:
            raise CandidateArtifactError(
                "CodeUnit IDs present in both candidates and failures "
                f"artifacts: {sorted(cross_duplicates)[:5]}"
            )

        self._seen_ids = success_ids | failure_ids
        return successful, failed

    @property
    def completed_ids(self) -> set[str]:
        """CodeUnit IDs already recorded as completed (either outcome)."""
        return set(self._seen_ids)

    # -- incremental persistence ---------------------------------------

    def append_success(self, record: dict) -> None:
        """Persist a successful candidate record immediately."""
        self._append(self.candidates_path, record, expected_success=True)

    def append_failure(self, record: dict) -> None:
        """Persist a failed-candidate audit record immediately."""
        self._append(self.failures_path, record, expected_success=False)

    def close(self) -> None:
        """Close open file handles."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    # -- internals ------------------------------------------------------

    def _append(self, path: Path, record: dict, *, expected_success: bool) -> None:
        validate_record(record)
        unit_id = record["code_unit_id"]
        if unit_id in self._seen_ids:
            raise CandidateArtifactError(
                f"Duplicate candidate record for CodeUnit '{unit_id}'."
            )
        if record["success"] is not expected_success:
            raise CandidateArtifactError(
                f"Record for '{unit_id}' filed under the wrong outcome."
            )

        if path not in self._handles:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handles[path] = open(path, "a", encoding="utf-8")
        handle = self._handles[path]
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._seen_ids.add(unit_id)

    def _repair_torn_tail(self, path: Path) -> None:
        """Truncate an incomplete final line left by an interrupted write."""
        if not path.exists() or path.stat().st_size == 0:
            return
        with open(path, "rb") as f:
            raw = f.read()
        if raw.endswith(b"\n"):
            return

        last_newline = raw.rfind(b"\n")
        keep = last_newline + 1  # may be 0: drop everything
        logger.warning(
            "%s ends with an incomplete record; truncating to byte %d",
            path,
            keep,
        )
        with open(path, "rb+") as f:
            os.truncate(f.fileno(), keep)
            os.fsync(f.fileno())

    def _load_file(
        self, path: Path, *, expected_success: bool
    ) -> tuple[list[dict], set[str]]:
        if not path.exists():
            return [], set()

        records: list[dict] = []
        seen: set[str] = set()
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CandidateArtifactError(
                        f"{path}:{line_number} is not valid JSON: {exc}"
                    ) from exc
                validate_record(record)
                unit_id = record["code_unit_id"]
                if unit_id in seen:
                    raise CandidateArtifactError(
                        f"Duplicate record for CodeUnit '{unit_id}' "
                        f"in {path}."
                    )
                if record["success"] is not expected_success:
                    raise CandidateArtifactError(
                        f"{path}:{line_number} contains a record with "
                        f"success={record['success']}."
                    )
                seen.add(unit_id)
                records.append(record)
        return records, seen
