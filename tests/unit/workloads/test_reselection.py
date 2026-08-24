"""Tests for re-selection after regeneration (Phase 4F-I-C3).

All tests are deterministic, network-free, and use synthetic fixtures.
"""

from __future__ import annotations

from pathlib import Path

from localbench.workloads.code_retrieval.selection import (
    build_eligible_pool,
    pool_hash,
    select_final_queries,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(
    code_unit_id: str,
    *,
    candidate_id: str | None = None,
    success: bool = True,
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
        "attempt_count": 1,
        "attempts": [],
        "generation_ms": 100.0,
        "validation_ms": 10.0,
        "validation_passed": success,
        "leakage_passed": success,
        "leakage_violations": [],
        "success": success,
        "failure_category": None,
        "failure_reason": None,
        "completed_utc": "2026-08-24T12:00:00Z",
    }


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_v2_replaces_original(self) -> None:
        original = [_make_candidate("unit_a", query="original query")]
        v2 = [_make_candidate("unit_a", query="v2 query")]
        by_unit = {}
        for rec in original:
            by_unit[rec["code_unit_id"]] = rec
        for rec in v2:
            by_unit[rec["code_unit_id"]] = rec
        merged = list(by_unit.values())
        assert len(merged) == 1
        assert merged[0]["query"] == "v2 query"

    def test_no_v2_keeps_original(self) -> None:
        original = [_make_candidate("unit_a", query="original query")]
        v2 = []
        by_unit = {}
        for rec in original:
            by_unit[rec["code_unit_id"]] = rec
        for rec in v2:
            by_unit[rec["code_unit_id"]] = rec
        merged = list(by_unit.values())
        assert len(merged) == 1
        assert merged[0]["query"] == "original query"

    def test_mixed_v2_and_original(self) -> None:
        original = [
            _make_candidate("unit_a", query="original a"),
            _make_candidate("unit_b", query="original b"),
        ]
        v2 = [_make_candidate("unit_a", query="v2 a")]
        by_unit = {}
        for rec in original:
            by_unit[rec["code_unit_id"]] = rec
        for rec in v2:
            by_unit[rec["code_unit_id"]] = rec
        merged = list(by_unit.values())
        assert len(merged) == 2
        queries = {r["code_unit_id"]: r["query"] for r in merged}
        assert queries["unit_a"] == "v2 a"
        assert queries["unit_b"] == "original b"


# ---------------------------------------------------------------------------
# Pool building with v2 candidates
# ---------------------------------------------------------------------------


class TestPoolBuildingWithV2:
    def test_v2_candidate_enters_pool(self) -> None:
        test_ids = {"unit_a", "unit_b"}
        original = [
            _make_candidate("unit_a", success=True, query="original a"),
        ]
        v2 = [
            _make_candidate("unit_a", success=True, query="v2 a"),
        ]
        by_unit = {}
        for rec in original:
            by_unit[rec["code_unit_id"]] = rec
        for rec in v2:
            by_unit[rec["code_unit_id"]] = rec
        candidates = list(by_unit.values())
        pool = build_eligible_pool(candidates, test_code_unit_ids=test_ids)
        assert len(pool) == 1
        assert pool[0]["query"] == "v2 a"

    def test_failed_v2_not_in_pool(self) -> None:
        test_ids = {"unit_a"}
        candidates = [
            _make_candidate("unit_a", success=False, query=""),
        ]
        pool = build_eligible_pool(candidates, test_code_unit_ids=test_ids)
        assert len(pool) == 0

    def test_original_success_v2_failure_replaces_original(self) -> None:
        test_ids = {"unit_a"}
        original = [
            _make_candidate("unit_a", success=True, query="original a"),
        ]
        v2 = [
            _make_candidate("unit_a", success=False, query=""),
        ]
        by_unit = {}
        for rec in original:
            by_unit[rec["code_unit_id"]] = rec
        for rec in v2:
            by_unit[rec["code_unit_id"]] = rec
        candidates = list(by_unit.values())
        pool = build_eligible_pool(candidates, test_code_unit_ids=test_ids)
        assert len(pool) == 0


# ---------------------------------------------------------------------------
# Selection determinism tests
# ---------------------------------------------------------------------------


class TestSelectionDeterminism:
    def test_same_pool_same_selection(self) -> None:
        test_ids = {f"unit_{i:03d}" for i in range(50)}
        candidates = [
            _make_candidate(f"unit_{i:03d}") for i in range(50)
        ]
        pool = build_eligible_pool(candidates, test_code_unit_ids=test_ids)
        selected_1 = select_final_queries(pool, count=5, seed=42)
        selected_2 = select_final_queries(pool, count=5, seed=42)
        assert [c["code_unit_id"] for c in selected_1] == [
            c["code_unit_id"] for c in selected_2
        ]

    def test_pool_hash_deterministic(self) -> None:
        test_ids = {f"unit_{i:03d}" for i in range(50)}
        candidates = [
            _make_candidate(f"unit_{i:03d}") for i in range(50)
        ]
        pool = build_eligible_pool(candidates, test_code_unit_ids=test_ids)
        hash_1 = pool_hash(pool)
        hash_2 = pool_hash(pool)
        assert hash_1 == hash_2


# ---------------------------------------------------------------------------
# Script reference tests
# ---------------------------------------------------------------------------


class TestScriptReferences:
    def test_reselection_script_exists(self) -> None:
        script_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "scripts"
            / "reselect_after_regeneration.py"
        )
        assert script_path.exists()

    def test_reselection_script_uses_frozen_selection(self) -> None:
        script_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "scripts"
            / "reselect_after_regeneration.py"
        )
        content = script_path.read_text(encoding="utf-8")
        assert "select_final_queries" in content
        assert "build_review_artifact" in content
        assert "random.Random(42)" in content or "seed=42" in content
