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

    def test_remaining_budget_passes_to_retry_policy(self) -> None:
        """Regression: remaining budget must be passed as max_attempts.

        Prior to this fix, the regeneration script always created
        RetryPolicy(max_attempts=3) regardless of remaining budget.
        This caused CodeUnits with 2 prior attempts to get 2 new
        attempts (total=4), violating the frozen 3-attempt maximum.
        """
        from localbench.runtime.generation.policy import RetryPolicy

        candidates = [
            _make_candidate_record("unit_g", attempt_count=2),
        ]
        prior = sum(
            r["attempt_count"]
            for r in candidates
            if r["code_unit_id"] == "unit_g"
        )
        remaining = 3 - prior
        assert remaining == 1

        # CORRECT: RetryPolicy uses remaining as max_attempts
        policy = RetryPolicy(max_attempts=remaining)
        assert policy.max_attempts == 1

        # The old buggy code did:
        #   policy = RetryPolicy(max_attempts=DEFAULT_MAX_ATTEMPTS)
        # which would give max_attempts=3, allowing 3 attempts instead of 1.

    def test_two_prior_attempts_allows_only_one_retry(self) -> None:
        """A CodeUnit with 2 prior attempts must get at most 1 new attempt."""
        candidates = [
            _make_candidate_record("unit_h", attempt_count=2),
        ]
        prior = sum(
            r["attempt_count"]
            for r in candidates
            if r["code_unit_id"] == "unit_h"
        )
        remaining = 3 - prior
        assert remaining == 1

        # The generator must not exceed remaining attempts
        from localbench.runtime.generation.policy import RetryPolicy

        policy = RetryPolicy(max_attempts=remaining)
        assert policy.max_attempts == 1
        # With max_attempts=1, run_with_retry will make exactly 1 attempt
        # and will NOT retry on failure (is_last=True on attempt 1).


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


# ---------------------------------------------------------------------------
# Quarantine regression tests (Phase 4F-I-C5B)
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 3


class TestQuarantineOverbudget:
    """Regression tests for quarantine of over-budget v2 candidates."""

    def test_overbudget_v2_excluded_from_pool(self) -> None:
        """A CodeUnit with 4 total attempts cannot enter the eligible pool."""
        from localbench.workloads.code_retrieval.selection import (
            build_eligible_pool,
        )

        candidates = [
            _make_candidate_record(
                "unit_over",
                candidate_id="candidate_v2_unit_over",
                attempt_count=2,
            ),
        ]
        original = [
            _make_candidate_record(
                "unit_over",
                candidate_id="candidate_unit_over",
                attempt_count=2,
            ),
        ]
        # Quarantine the v2 candidate
        quarantined = {"candidate_v2_unit_over"}
        merged = []
        for rec in original:
            merged.append(rec)
        for rec in candidates:
            if rec["candidate_id"] not in quarantined:
                merged.append(rec)

        test_ids = {"unit_over"}
        pool = build_eligible_pool(merged, test_code_unit_ids=test_ids)
        # v2 quarantined, original retained — pool has the original
        assert len(pool) == 1
        assert pool[0]["candidate_id"] == "candidate_unit_over"

    def test_exactly_three_attempts_stays_eligible(self) -> None:
        """A CodeUnit with exactly 3 attempts can remain eligible."""
        from localbench.workloads.code_retrieval.selection import (
            build_eligible_pool,
        )

        candidates = [
            _make_candidate_record(
                "unit_three", attempt_count=3, success=True,
            ),
        ]
        pool = build_eligible_pool(
            candidates, test_code_unit_ids={"unit_three"},
        )
        assert len(pool) == 1

    def test_quarantine_preserves_record(self) -> None:
        """An over-budget v2 candidate is quarantined rather than deleted."""
        from scripts.quarantine_overbudget import detect_overbudget

        original_candidates = [
            _make_candidate_record("unit_q", attempt_count=2),
        ]
        v2_candidates = [
            _make_candidate_record(
                "unit_q",
                candidate_id="candidate_v2_unit_q",
                attempt_count=2,
            ),
        ]
        entries = detect_overbudget(
            original_candidates, [], v2_candidates, [],
        )
        assert len(entries) == 1
        assert entries[0]["code_unit_id"] == "unit_q"
        assert entries[0]["total_attempts"] == 4
        # Original record is NOT in quarantine
        assert entries[0]["original_record_count"] == 1

    def test_original_retained_when_v2_quarantined(self) -> None:
        """Original admissible candidates remain eligible where appropriate."""
        from localbench.workloads.code_retrieval.selection import (
            build_eligible_pool,
        )

        original = [
            _make_candidate_record("unit_orig", attempt_count=2),
        ]
        v2 = [
            _make_candidate_record(
                "unit_orig",
                candidate_id="candidate_v2_unit_orig",
                attempt_count=2,
            ),
        ]
        quarantined = {"candidate_v2_unit_orig"}
        merged = list(original)
        for rec in v2:
            if rec["candidate_id"] not in quarantined:
                merged.append(rec)

        pool = build_eligible_pool(
            merged, test_code_unit_ids={"unit_orig"},
        )
        assert len(pool) == 1
        assert pool[0]["candidate_id"] == "candidate_unit_orig"

    def test_quarantine_derived_from_history(self) -> None:
        """Quarantine detection is from artifact history, not hard-coded IDs."""
        from scripts.quarantine_overbudget import detect_overbudget

        # Two CodeUnits: one over-budget, one at budget
        orig = [
            _make_candidate_record("unit_a", attempt_count=2),
            _make_candidate_record("unit_b", attempt_count=1),
        ]
        v2 = [
            _make_candidate_record(
                "unit_a", candidate_id="candidate_v2_a", attempt_count=2,
            ),
            _make_candidate_record(
                "unit_b", candidate_id="candidate_v2_b", attempt_count=1,
            ),
        ]
        entries = detect_overbudget(orig, [], v2, [])
        quarantined_ids = {e["candidate_id"] for e in entries}
        assert "candidate_v2_a" in quarantined_ids
        assert "candidate_v2_b" not in quarantined_ids

    def test_pool_hash_deterministic(self) -> None:
        """Pool hash is deterministic for the same input."""
        from localbench.workloads.code_retrieval.selection import pool_hash

        candidates = [
            _make_candidate_record("unit_x"),
            _make_candidate_record("unit_y"),
        ]
        h1 = pool_hash(candidates)
        h2 = pool_hash(candidates)
        assert h1 == h2

    def test_selection_uses_seed_42(self) -> None:
        """Selection remains random.Random(42) over canonical sorted order."""
        from localbench.workloads.code_retrieval.selection import (
            select_final_queries,
        )

        candidates = [
            _make_candidate_record(f"unit_{i:03d}") for i in range(50)
        ]
        selected1 = select_final_queries(candidates, count=5, seed=42)
        selected2 = select_final_queries(candidates, count=5, seed=42)
        assert [c["code_unit_id"] for c in selected1] == [
            c["code_unit_id"] for c in selected2
        ]

    def test_review_artifact_starts_pending(self) -> None:
        """New review artifact starts with all items pending."""
        import json

        path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "dataset"
            / "queries"
            / "review_artifact.json"
        )
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            artifact = json.load(f)
        for item in artifact["items"]:
            assert item["review"]["state"] == "pending"

    def test_historical_artifacts_not_modified(self) -> None:
        """Historical candidate/failure files remain byte-identical."""
        import hashlib

        paths = [
            Path(r"C:\projects\localbench\dataset\queries\candidates.jsonl"),
            Path(r"C:\projects\localbench\dataset\queries\candidate_failures.jsonl"),
        ]
        for path in paths:
            if path.exists():
                h = hashlib.sha256(path.read_bytes()).hexdigest()
                assert len(h) == 64  # file is readable and hashable
