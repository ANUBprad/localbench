"""Tests for crash-safe candidate record persistence."""

from __future__ import annotations

import json
import os

import pytest

from localbench.workloads.code_retrieval.candidate_store import (
    CandidateArtifactError,
    CandidateStore,
    validate_record,
)


def _record(unit_id: str = "repo003_py_a_py__f_abc123", success: bool = True) -> dict:
    """Minimal well-formed audit record with representative metadata."""
    return {
        "code_unit_id": unit_id,
        "candidate_id": f"candidate_{unit_id}",
        "query": "Find the function that formats numbers." if success else "",
        "query_style": "natural" if success else "",
        "query_intent": "find_implementation" if success else "",
        "model": "qwen2.5-coder:7b",
        "model_version": "7b",
        "prompt_version": "1.0.0",
        "seed": 42,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 128,
        "attempt_count": 1,
        "attempts": [
            {
                "attempt_number": 1,
                "status": "success" if success else "failed",
                "will_retry": False,
                "generation_ms": 1000.0,
                "errors": [],
                "raw_text": "{}",
            }
        ],
        "generation_ms": 1000.0,
        "validation_ms": 1.0,
        "validation_passed": success,
        "leakage_passed": success,
        "leakage_violations": [],
        "success": success,
        "failure_category": None if success else "malformed_json",
        "failure_reason": None if success else "Model output is not valid JSON.",
        "completed_utc": "2026-08-21T00:00:00+00:00",
    }


@pytest.fixture
def paths(tmp_path):
    return (
        tmp_path / "candidates.jsonl",
        tmp_path / "candidate_failures.jsonl",
    )


# ===========================================================================
# Incremental persistence
# ===========================================================================


class TestImmediatePersistence:
    def test_success_record_visible_to_new_store(self, paths):
        store = CandidateStore(*paths)
        record = _record(success=True)
        store.append_success(record)
        store.close()

        reloaded_store = CandidateStore(*paths)
        successful, failed = reloaded_store.load()
        assert len(successful) == 1
        assert failed == []
        assert successful[0]["query"] == record["query"]
        assert successful[0]["seed"] == 42

    def test_failure_record_persisted_immediately(self, paths):
        store = CandidateStore(*paths)
        record = _record(success=False)
        store.append_failure(record)
        store.close()

        reloaded_store = CandidateStore(*paths)
        successful, failed = reloaded_store.load()
        assert successful == []
        assert len(failed) == 1
        assert failed[0]["failure_category"] == "malformed_json"

    def test_record_is_flushed_without_close(self, paths):
        """A crash before close() must not lose appended records."""
        store = CandidateStore(*paths)
        store.append_success(_record("repo006_py_x_py__g_def456"))
        # No close(): simulate reading from another process right away.
        content = paths[0].read_text(encoding="utf-8")
        assert "repo006_py_x_py__g_def456" in content
        store.close()

    def test_metadata_round_trip(self, paths):
        """Attempt history and reproducibility fields survive reload."""
        record = _record()
        store = CandidateStore(*paths)
        store.append_success(record)
        store.close()

        successful, _ = CandidateStore(*paths).load()
        assert successful[0] == record


# ===========================================================================
# Checkpoint reconstruction
# ===========================================================================


class TestCheckpointReconstruction:
    def test_completed_ids_include_both_outcomes(self, paths):
        store = CandidateStore(*paths)
        store.append_success(_record("unit_ok"))
        store.append_failure(_record("unit_bad", success=False))
        completed = store.completed_ids
        store.close()

        reloaded = CandidateStore(*paths)
        reloaded.load()
        assert completed == {"unit_ok", "unit_bad"}
        assert reloaded.completed_ids == {"unit_ok", "unit_bad"}

    def test_load_is_deterministic(self, paths):
        store = CandidateStore(*paths)
        store.append_success(_record("unit_1"))
        store.append_failure(_record("unit_2", success=False))
        store.close()

        first = CandidateStore(*paths).load()
        second = CandidateStore(*paths).load()
        assert first == second


# ===========================================================================
# Idempotency
# ===========================================================================


class TestIdempotency:
    def test_duplicate_append_rejected(self, paths):
        store = CandidateStore(*paths)
        store.append_success(_record("unit_1"))
        with pytest.raises(CandidateArtifactError, match="Duplicate"):
            store.append_success(_record("unit_1"))

    def test_duplicate_across_outcomes_rejected(self, paths):
        store = CandidateStore(*paths)
        store.append_success(_record("unit_1"))
        with pytest.raises(CandidateArtifactError, match="Duplicate"):
            store.append_failure(_record("unit_1", success=False))

    def test_duplicate_inside_file_rejected_on_load(self, paths):
        candidates_path, failures_path = paths
        record = json.dumps(_record("unit_1"))
        candidates_path.write_text(f"{record}\n{record}\n", encoding="utf-8")

        with pytest.raises(CandidateArtifactError, match="Duplicate record"):
            CandidateStore(candidates_path, failures_path).load()

    def test_same_id_in_both_files_rejected_on_load(self, paths):
        candidates_path, failures_path = paths
        candidates_path.write_text(
            json.dumps(_record("unit_1")) + "\n", encoding="utf-8"
        )
        failures_path.write_text(
            json.dumps(_record("unit_1", success=False)) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(CandidateArtifactError, match="both"):
            CandidateStore(candidates_path, failures_path).load()


# ===========================================================================
# Crash safety
# ===========================================================================


class TestCrashSafety:
    def test_torn_trailing_line_is_truncated(self, paths):
        candidates_path, failures_path = paths
        good = json.dumps(_record("unit_1")) + "\n"
        torn = '{"code_unit_id": "unit_2", "success": true, "qu'
        candidates_path.write_text(good + torn, encoding="utf-8")

        store = CandidateStore(candidates_path, failures_path)
        successful, failed = store.load()

        assert [r["code_unit_id"] for r in successful] == ["unit_1"]
        assert failed == []
        # Repaired file ends cleanly so future appends stay parseable.
        raw = candidates_path.read_bytes()
        assert raw.endswith(b"\n")
        assert b'"unit_2"' not in raw

    def test_torn_only_record_leaves_empty_pool(self, paths):
        candidates_path, failures_path = paths
        candidates_path.write_bytes(
            b'{"code_unit_id": "unit_2", "succ'
        )

        store = CandidateStore(candidates_path, failures_path)
        successful, failed = store.load()
        assert successful == []
        assert failed == []
        assert candidates_path.read_bytes() == b""

    def test_midfile_corrupt_line_is_fatal(self, paths):
        candidates_path, failures_path = paths
        good = json.dumps(_record("unit_1")) + "\n"
        bad = "this is not json\n"
        tail = json.dumps(_record("unit_2")) + "\n"
        candidates_path.write_text(good + bad + tail, encoding="utf-8")

        with pytest.raises(CandidateArtifactError, match="not valid JSON"):
            CandidateStore(candidates_path, failures_path).load()

    def test_invalid_record_shape_is_fatal(self, paths):
        candidates_path, failures_path = paths
        candidates_path.write_text(
            json.dumps({"query": "no identity or outcome"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(CandidateArtifactError, match="required key"):
            CandidateStore(candidates_path, failures_path).load()

    def test_non_object_record_is_fatal(self, paths):
        candidates_path, failures_path = paths
        candidates_path.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(CandidateArtifactError, match="JSON object"):
            CandidateStore(candidates_path, failures_path).load()

    def test_success_flag_mismatch_is_fatal(self, paths):
        candidates_path, failures_path = paths
        candidates_path.write_text(
            json.dumps(_record("unit_1", success=False)) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(CandidateArtifactError, match="success=False"):
            CandidateStore(candidates_path, failures_path).load()


# ===========================================================================
# Record validation
# ===========================================================================


class TestValidateRecord:
    def test_valid_record_passes(self):
        validate_record(_record())

    def test_missing_key_rejected(self):
        record = _record()
        del record["code_unit_id"]
        with pytest.raises(CandidateArtifactError, match="code_unit_id"):
            validate_record(record)

    def test_wrong_type_rejected(self):
        record = _record()
        record["success"] = "yes"
        with pytest.raises(CandidateArtifactError, match="bool"):
            validate_record(record)

    def test_empty_identity_rejected(self):
        record = _record()
        record["code_unit_id"] = ""
        with pytest.raises(CandidateArtifactError, match="non-empty"):
            validate_record(record)


class TestDurability:
    def test_fsync_called_per_append(self, paths, monkeypatch):
        calls = []
        monkeypatch.setattr(
            os, "fsync", lambda fd: calls.append(fd), raising=True
        )
        store = CandidateStore(*paths)
        store.append_success(_record("unit_ok"))
        store.append_failure(_record("unit_bad", success=False))
        assert len(calls) == 2
        store.close()
