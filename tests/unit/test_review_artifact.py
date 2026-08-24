"""Tests for the benchmark-blind human-review artifact (§4.4.4/§4.4.5).

Proves that the review artifact meets all invariant properties required
before human review can begin.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localbench.workloads.code_retrieval.review import (
    FINAL_QUERY_COUNT,
    PENDING,
    build_review_artifact,
    review_progress,
    validate_review_artifact,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SELECTION_PATH = REPO_ROOT / "dataset" / "queries" / "final_45_selection.json"
CANDIDATES_PATH = REPO_ROOT / "dataset" / "queries" / "candidates.jsonl"
TEST_SPLIT_PATH = REPO_ROOT / "dataset" / "splits" / "test.jsonl"
REVIEW_ARTIFACT_PATH = REPO_ROOT / "dataset" / "queries" / "review_artifact.json"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def selection_record() -> dict:
    with open(SELECTION_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def candidates_by_id() -> dict[str, dict]:
    return {c["candidate_id"]: c for c in _load_jsonl(CANDIDATES_PATH)}


@pytest.fixture(scope="module")
def test_units() -> list[dict]:
    return _load_jsonl(TEST_SPLIT_PATH)


@pytest.fixture(scope="module")
def units_by_id(test_units: list[dict]) -> dict[str, dict]:
    return {u["id"]: u for u in test_units}


@pytest.fixture(scope="module")
def test_code_unit_ids(test_units: list[dict]) -> set[str]:
    return {u["id"] for u in test_units}


@pytest.fixture(scope="module")
def review_artifact(
    selection_record: dict,
    candidates_by_id: dict[str, dict],
    units_by_id: dict[str, dict],
    test_code_unit_ids: set[str],
) -> dict:
    """Build the review artifact from real data (module-scoped for speed)."""
    return build_review_artifact(
        selection_record=selection_record,
        candidates_by_id=candidates_by_id,
        units_by_id=units_by_id,
        test_code_unit_ids=test_code_unit_ids,
    )


# ---------------------------------------------------------------------------
# Invariant: exactly 45 items
# ---------------------------------------------------------------------------


class TestItemCount:
    def test_exactly_45_items(self, review_artifact: dict) -> None:
        assert len(review_artifact["items"]) == FINAL_QUERY_COUNT

    def test_selected_count_matches(self, review_artifact: dict) -> None:
        assert review_artifact["basis"]["selected_count"] == FINAL_QUERY_COUNT


# ---------------------------------------------------------------------------
# Invariant: all selected IDs represented, no duplicates
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_all_candidate_ids_present(self, review_artifact: dict) -> None:
        ids = {item["candidate_id"] for item in review_artifact["items"]}
        assert len(ids) == FINAL_QUERY_COUNT

    def test_all_code_unit_ids_present(self, review_artifact: dict) -> None:
        ids = {item["code_unit_id"] for item in review_artifact["items"]}
        assert len(ids) == FINAL_QUERY_COUNT

    def test_no_duplicate_candidate_ids(self, review_artifact: dict) -> None:
        ids = [item["candidate_id"] for item in review_artifact["items"]]
        assert len(ids) == len(set(ids))

    def test_no_duplicate_code_unit_ids(self, review_artifact: dict) -> None:
        ids = [item["code_unit_id"] for item in review_artifact["items"]]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Invariant: no train/validation leakage
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_no_code_unit_in_train(self, review_artifact: dict) -> None:
        train_path = REPO_ROOT / "dataset" / "splits" / "train.jsonl"
        if not train_path.exists():
            pytest.skip("train.jsonl not found")
        train_ids = {u["id"] for u in _load_jsonl(train_path)}
        review_ids = {item["code_unit_id"] for item in review_artifact["items"]}
        overlap = review_ids & train_ids
        assert not overlap, f"Leakage into train split: {overlap}"

    def test_no_code_unit_in_validation(self, review_artifact: dict) -> None:
        val_path = REPO_ROOT / "dataset" / "splits" / "validation.jsonl"
        if not val_path.exists():
            pytest.skip("validation.jsonl not found")
        val_ids = {u["id"] for u in _load_jsonl(val_path)}
        review_ids = {item["code_unit_id"] for item in review_artifact["items"]}
        overlap = review_ids & val_ids
        assert not overlap, f"Leakage into validation split: {overlap}"


# ---------------------------------------------------------------------------
# Invariant: initial state is pending
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_all_items_pending(self, review_artifact: dict) -> None:
        for item in review_artifact["items"]:
            assert item["review"]["state"] == PENDING

    def test_pending_is_not_accepted(self, review_artifact: dict) -> None:
        for item in review_artifact["items"]:
            assert item["review"]["state"] != "accepted"

    def test_pending_is_not_rejected(self, review_artifact: dict) -> None:
        for item in review_artifact["items"]:
            assert item["review"]["state"] != "rejected"

    def test_no_decided_timestamps(self, review_artifact: dict) -> None:
        for item in review_artifact["items"]:
            assert item["review"]["decided_utc"] is None


# ---------------------------------------------------------------------------
# Invariant: forbidden benchmark information cannot enter
# ---------------------------------------------------------------------------


class TestBenchmarkBlindness:
    FORBIDDEN_MARKERS = (
        "hit",
        "mrr",
        "latency",
        "throughput",
        "ranking",
        "benchmark",
        "relevanc",
        "ground_truth",
        "score",
    )

    def _check_keys_recursive(self, obj: object, path: str = "root") -> list[str]:
        violations = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                lowered = key.lower()
                for marker in self.FORBIDDEN_MARKERS:
                    if marker in lowered:
                        violations.append(f"{path}.{key}")
                violations.extend(self._check_keys_recursive(value, f"{path}.{key}"))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                violations.extend(self._check_keys_recursive(item, f"{path}[{i}]"))
        return violations

    def test_no_benchmark_keys_in_artifact(self, review_artifact: dict) -> None:
        violations = self._check_keys_recursive(review_artifact)
        assert not violations, f"Forbidden benchmark fields found: {violations}"

    def test_no_relevance_scores(self, review_artifact: dict) -> None:
        for item in review_artifact["items"]:
            assert "relevance_score" not in item
            assert "relevance_label" not in item


# ---------------------------------------------------------------------------
# Invariant: deterministic ordering
# ---------------------------------------------------------------------------


class TestDeterministicOrdering:
    def test_positions_are_sequential(self, review_artifact: dict) -> None:
        for i, item in enumerate(review_artifact["items"], start=1):
            assert item["position"] == i

    def test_order_is_deterministic_across_builds(self) -> None:
        if not REVIEW_ARTIFACT_PATH.exists():
            pytest.skip("review_artifact.json not yet built")
        with open(REVIEW_ARTIFACT_PATH, encoding="utf-8") as f:
            first = json.load(f)
        with open(REVIEW_ARTIFACT_PATH, encoding="utf-8") as f:
            second = json.load(f)
        first_ids = [item["code_unit_id"] for item in first["items"]]
        second_ids = [item["code_unit_id"] for item in second["items"]]
        assert first_ids == second_ids


# ---------------------------------------------------------------------------
# Invariant: structure correctness
# ---------------------------------------------------------------------------


class TestStructure:
    def test_artifact_type(self, review_artifact: dict) -> None:
        assert review_artifact["artifact"] == "human_review"

    def test_review_version(self, review_artifact: dict) -> None:
        assert review_artifact["review_version"] == "1.0.0"

    def test_basis_has_required_fields(self, review_artifact: dict) -> None:
        basis = review_artifact["basis"]
        assert "selection_version" in basis
        assert "eligible_pool_sha256" in basis
        assert "generation_source_commit" in basis
        assert "selection_created_utc" in basis
        assert "selected_count" in basis

    def test_each_item_has_target(self, review_artifact: dict) -> None:
        for item in review_artifact["items"]:
            target = item["target"]
            assert "repository" in target
            assert "file_path" in target
            assert "symbol" in target
            assert "source_code" in target

    def test_each_item_has_automated_validation(self, review_artifact: dict) -> None:
        for item in review_artifact["items"]:
            av = item["automated_validation"]
            assert "validation_passed" in av
            assert "leakage_passed" in av
            assert av["validation_passed"] is True
            assert av["leakage_passed"] is True


# ---------------------------------------------------------------------------
# Validation function itself
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_returns_no_errors(self, review_artifact: dict) -> None:
        errors = validate_review_artifact(review_artifact)
        assert errors == []

    def test_validation_catches_wrong_count(self) -> None:
        artifact = {
            "artifact": "human_review",
            "review_version": "1.0.0",
            "basis": {
                "selection_version": "1.0.0",
                "eligible_pool_sha256": "abc",
                "generation_source_commit": "abc",
                "selection_created_utc": "2026-01-01T00:00:00Z",
                "selected_count": 2,
            },
            "items": [],
        }
        errors = validate_review_artifact(artifact)
        assert any("45" in e for e in errors)


# ---------------------------------------------------------------------------
# Review progress helper
# ---------------------------------------------------------------------------


class TestReviewProgress:
    def test_all_pending_initially(self, review_artifact: dict) -> None:
        progress = review_progress(review_artifact)
        assert progress["pending"] == FINAL_QUERY_COUNT
        assert progress["accepted"] == 0
        assert progress["rejected"] == 0
        assert progress["total"] == FINAL_QUERY_COUNT


# ---------------------------------------------------------------------------
# On-disk artifact (if it exists)
# ---------------------------------------------------------------------------


class TestOnDiskArtifact:
    def test_artifact_file_exists(self) -> None:
        if not REVIEW_ARTIFACT_PATH.exists():
            pytest.skip("review_artifact.json not yet built")

    def test_artifact_file_valid(self) -> None:
        if not REVIEW_ARTIFACT_PATH.exists():
            pytest.skip("review_artifact.json not yet built")
        with open(REVIEW_ARTIFACT_PATH, encoding="utf-8") as f:
            artifact = json.load(f)
        errors = validate_review_artifact(artifact)
        assert errors == []
