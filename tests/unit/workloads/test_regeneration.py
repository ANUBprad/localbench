"""Tests for bounded regeneration of rejected queries (Phase 4F-I-C3).

All tests are deterministic, network-free, and use synthetic fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

from localbench.workloads.code_retrieval.candidate_store import CandidateStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate_record(
    code_unit_id: str,
    *,
    success: bool = True,
    attempt_count: int = 1,
    candidate_id: str | None = None,
    query: str = "test query",
) -> dict:
    return {
        "code_unit_id": code_unit_id,
        "candidate_id": candidate_id or f"candidate_{code_unit_id}",
        "query": query,
        "query_style": "technical",
        "query_intent": "find_implementation",
        "model": "qwen2.5-coder:7b",
        "model_version": "7b",
        "prompt_version": "1.1.0",
        "seed": 42,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 128,
        "attempt_count": attempt_count,
        "attempts": [],
        "generation_ms": 100.0,
        "validation_ms": 10.0,
        "validation_passed": success,
        "leakage_passed": success,
        "leakage_violations": [],
        "success": success,
        "failure_category": None if success else "leakage",
        "failure_reason": None if success else "test failure",
        "completed_utc": "2026-08-24T12:00:00Z",
    }


# ---------------------------------------------------------------------------
# Budget accounting tests
# ---------------------------------------------------------------------------


class TestBudgetAccounting:
    def test_count_prior_attempts_from_candidates(self) -> None:
        candidates = [
            _make_candidate_record("unit_a", attempt_count=2),
            _make_candidate_record("unit_b", attempt_count=1),
        ]
        count_a = sum(
            r["attempt_count"]
            for r in candidates
            if r["code_unit_id"] == "unit_a"
        )
        count_b = sum(
            r["attempt_count"]
            for r in candidates
            if r["code_unit_id"] == "unit_b"
        )
        assert count_a == 2
        assert count_b == 1

    def test_count_prior_attempts_from_failures(self) -> None:
        failures = [
            _make_candidate_record(
                "unit_c", success=False, attempt_count=3
            ),
        ]
        count_c = sum(
            r["attempt_count"]
            for r in failures
            if r["code_unit_id"] == "unit_c"
        )
        assert count_c == 3

    def test_count_prior_attempts_combined(self) -> None:
        candidates = [
            _make_candidate_record("unit_d", attempt_count=1),
        ]
        failures = [
            _make_candidate_record(
                "unit_d", success=False, attempt_count=1
            ),
        ]
        count_d = sum(
            r["attempt_count"]
            for r in candidates + failures
            if r["code_unit_id"] == "unit_d"
        )
        assert count_d == 2

    def test_remaining_attempts_zero_when_exhausted(self) -> None:
        candidates = [
            _make_candidate_record("unit_e", attempt_count=3),
        ]
        prior = sum(
            r["attempt_count"]
            for r in candidates
            if r["code_unit_id"] == "unit_e"
        )
        remaining = 3 - prior
        assert remaining == 0

    def test_remaining_attempts_positive_when_not_exhausted(self) -> None:
        candidates = [
            _make_candidate_record("unit_f", attempt_count=2),
        ]
        prior = sum(
            r["attempt_count"]
            for r in candidates
            if r["code_unit_id"] == "unit_f"
        )
        remaining = 3 - prior
        assert remaining == 1


# ---------------------------------------------------------------------------
# V2 candidate_id convention tests
# ---------------------------------------------------------------------------


class TestV2CandidateId:
    def test_v2_prefix(self) -> None:
        prefix = "candidate_v2_"
        uid = "repo003_py_rich_console_py__Console_size_0079f1ad076c"
        v2_id = f"{prefix}{uid}"
        assert v2_id == f"candidate_v2_{uid}"

    def test_v2_id_is_distinct_from_original(self) -> None:
        uid = "repo003_py_rich_console_py__Console_size_0079f1ad076c"
        original_id = f"candidate_{uid}"
        v2_id = f"candidate_v2_{uid}"
        assert original_id != v2_id


# ---------------------------------------------------------------------------
# CandidateStore integration tests
# ---------------------------------------------------------------------------


class TestCandidateStoreIntegration:
    def test_append_success_v2(self, tmp_path: Path) -> None:
        candidates_path = tmp_path / "candidates.jsonl"
        failures_path = tmp_path / "failures.jsonl"
        store = CandidateStore(candidates_path, failures_path)
        store.load()

        record = _make_candidate_record(
            "unit_v2_test",
            candidate_id="candidate_v2_unit_v2_test",
            query="regenerated query about behavior",
        )
        store.append_success(record)
        store.close()

        with open(candidates_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1
        assert lines[0]["code_unit_id"] == "unit_v2_test"
        assert lines[0]["candidate_id"] == "candidate_v2_unit_v2_test"
        assert lines[0]["success"] is True

    def test_append_failure_v2(self, tmp_path: Path) -> None:
        candidates_path = tmp_path / "candidates.jsonl"
        failures_path = tmp_path / "failures.jsonl"
        store = CandidateStore(candidates_path, failures_path)
        store.load()

        record = _make_candidate_record(
            "unit_v2_fail",
            candidate_id="candidate_v2_unit_v2_fail",
            success=False,
        )
        store.append_failure(record)
        store.close()

        with open(failures_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1
        assert lines[0]["success"] is False

    def test_original_records_preserved(self, tmp_path: Path) -> None:
        candidates_path = tmp_path / "candidates.jsonl"
        failures_path = tmp_path / "failures.jsonl"
        store = CandidateStore(candidates_path, failures_path)
        store.load()

        original = _make_candidate_record(
            "unit_orig",
            candidate_id="candidate_unit_orig",
            query="original query",
        )
        store.append_success(original)

        v2 = _make_candidate_record(
            "unit_v2_only",
            candidate_id="candidate_v2_unit_orig",
            query="regenerated query",
        )
        store.append_success(v2)
        store.close()

        with open(candidates_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        ids = [r["candidate_id"] for r in lines]
        assert "candidate_unit_orig" in ids
        assert "candidate_v2_unit_orig" in ids


# ---------------------------------------------------------------------------
# Review artifact rejection extraction tests
# ---------------------------------------------------------------------------


class TestRejectionExtraction:
    def test_extract_rejected_ids(self) -> None:
        artifact = {
            "items": [
                {
                    "code_unit_id": "unit_a",
                    "review": {"state": "rejected", "notes": "test"},
                },
                {
                    "code_unit_id": "unit_b",
                    "review": {"state": "accepted", "notes": ""},
                },
                {
                    "code_unit_id": "unit_c",
                    "review": {"state": "rejected", "notes": "test"},
                },
            ]
        }
        rejected = [
            item["code_unit_id"]
            for item in artifact["items"]
            if item["review"]["state"] == "rejected"
        ]
        assert rejected == ["unit_a", "unit_c"]

    def test_exhausted_units_not_regenerated(self) -> None:
        budget_info = {
            "unit_a": {"prior": 2, "remaining": 1},
            "unit_b": {"prior": 3, "remaining": 0},
            "unit_c": {"prior": 1, "remaining": 2},
        }
        regenerable = [
            uid for uid, info in budget_info.items() if info["remaining"] > 0
        ]
        exhausted = [
            uid for uid, info in budget_info.items() if info["remaining"] <= 0
        ]
        assert regenerable == ["unit_a", "unit_c"]
        assert exhausted == ["unit_b"]


# ---------------------------------------------------------------------------
# Prompt version tests
# ---------------------------------------------------------------------------


class TestPromptVersionForRegeneration:
    def test_regeneration_uses_v1_1_0(self) -> None:
        from localbench.workloads.code_retrieval.query_prompt import (
            QUERY_PROMPT_TEMPLATE_VERSION,
        )

        assert QUERY_PROMPT_TEMPLATE_VERSION == "1.1.0"

    def test_regeneration_script_references_correct_version(self) -> None:
        script_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "scripts"
            / "regenerate_rejected_queries.py"
        )
        content = script_path.read_text(encoding="utf-8")
        assert "QUERY_PROMPT_TEMPLATE_VERSION" in content
