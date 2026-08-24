"""Tests for the benchmark-blind human-review workflow (§4.4.4/§4.4.5)."""

from __future__ import annotations

import json

import pytest

from localbench.workloads.code_retrieval.review import (
    ACCEPTED,
    FINAL_QUERY_COUNT,
    PENDING,
    REJECTED,
    ReviewArtifactError,
    build_review_artifact,
    build_review_item,
    review_progress,
    validate_review_artifact,
)


def _unit(unit_id: str) -> dict:
    return {
        "id": unit_id,
        "repository": unit_id.split("_")[0],
        "file_path": f"src/{unit_id}.py",
        "symbol": "Symbol",
        "symbol_type": "function",
        "docstring": "Docstring.",
        "source_code": "def Symbol():\n    return 1\n",
        "split": "test",
    }


def _candidate(unit_id: str) -> dict:
    return {
        "candidate_id": f"cand_{unit_id}",
        "code_unit_id": unit_id,
        "query": f"Find logic related to {unit_id} behavior.",
        "query_style": "technical",
        "query_intent": "find_implementation",
        "validation_passed": True,
        "leakage_passed": True,
    }


def _fixture(count: int = FINAL_QUERY_COUNT) -> tuple[dict, dict, dict]:
    units = {f"repo006_u{i:03d}": _unit(f"repo006_u{i:03d}") for i in range(count)}
    candidates = {
        f"cand_{unit_id}": _candidate(unit_id) for unit_id in sorted(units)
    }
    ordered_units = sorted(units)
    selection = {
        "selection_version": "1.0.0",
        "eligible_pool_sha256": "ab" * 32,
        "generation_source_commit": "ca257dd",
        "selection_created_utc": "2026-08-23T00:00:00Z",
        "selected_count": count,
        "selected_candidate_ids": [f"cand_{u}" for u in ordered_units],
        "selected_code_unit_ids": ordered_units,
    }
    return selection, candidates, units


def _build(selection=None, candidates=None, units=None) -> dict:
    selection = selection or _fixture()[0]
    candidates = candidates if candidates is not None else _fixture()[1]
    units = units if units is not None else _fixture()[2]
    return build_review_artifact(
        selection,
        candidates,
        units,
        test_code_unit_ids=set(units),
    )


class TestBuildReviewArtifact:
    def test_exactly_45_review_records(self) -> None:
        artifact = _build()
        assert len(artifact["items"]) == FINAL_QUERY_COUNT
        assert artifact["basis"]["selected_count"] == FINAL_QUERY_COUNT

    def test_all_selected_ids_represented_in_order(self) -> None:
        selection, _, _ = _fixture()
        artifact = _build(selection)
        assert [i["candidate_id"] for i in artifact["items"]] == selection[
            "selected_candidate_ids"
        ]
        assert [i["code_unit_id"] for i in artifact["items"]] == selection[
            "selected_code_unit_ids"
        ]

    def test_no_duplicate_identities_allowed(self) -> None:
        selection, candidates, units = _fixture()
        selection["selected_candidate_ids"][1] = selection["selected_candidate_ids"][0]
        with pytest.raises(ReviewArtifactError, match="duplicate identity"):
            build_review_artifact(
                selection,
                candidates,
                units,
                test_code_unit_ids=set(units),
            )

    def test_train_validation_leakage_refused(self) -> None:
        selection, candidates, units = _fixture()
        all_ids = set(units)
        leaked_id = sorted(all_ids)[0]
        test_ids = all_ids - {leaked_id}
        train_ids = {leaked_id}
        with pytest.raises(ReviewArtifactError, match="outside canonical test split"):
            build_review_artifact(
                selection,
                candidates,
                units,
                test_code_unit_ids=test_ids,
                train_code_unit_ids=train_ids,
            )

    def test_wrong_selection_size_refused(self) -> None:
        selection, candidates, units = _fixture()
        selection["selected_candidate_ids"] = selection["selected_candidate_ids"][:10]
        with pytest.raises(ReviewArtifactError, match="exactly 45"):
            build_review_artifact(
                selection,
                candidates,
                units,
                test_code_unit_ids=set(units),
            )

    def test_initial_state_pending_with_empty_decision_metadata(self) -> None:
        item = _build()["items"][0]
        assert item["review"] == {"state": PENDING, "notes": "", "decided_utc": None}

    def test_deterministic_ordering_identical_bytes(self) -> None:
        first = json.dumps(_build(), ensure_ascii=False, sort_keys=False)
        second = json.dumps(_build(), ensure_ascii=False, sort_keys=False)
        assert first == second

    def test_item_carries_permitted_context_only(self) -> None:
        item = _build()["items"][0]
        assert set(item["target"]) == {
            "repository",
            "file_path",
            "symbol",
            "symbol_type",
            "docstring",
            "source_code",
        }
        assert set(item["automated_validation"]) == {
            "validation_passed",
            "leakage_passed",
        }
        assert "relevant_code_units" not in item


class TestValidateReviewArtifact:
    def test_fresh_artifact_has_zero_violations(self) -> None:
        assert validate_review_artifact(_build()) == []

    def test_pending_differs_from_accepted_states(self) -> None:
        artifact = _build()
        artifact["items"][0]["review"]["state"] = ACCEPTED
        errors = validate_review_artifact(artifact)
        assert errors == []
        assert artifact["items"][0]["review"]["state"] != PENDING

    def test_unknown_state_rejected(self) -> None:
        artifact = _build()
        artifact["items"][2]["review"]["state"] = "auto_accepted"
        errors = validate_review_artifact(artifact)
        assert any("invalid review state" in e for e in errors)

    def test_rejected_requires_auditable_reason(self) -> None:
        artifact = _build()
        artifact["items"][3]["review"].update(state=REJECTED, notes="   ")
        errors = validate_review_artifact(artifact)
        assert any("auditable reason" in e for e in errors)
        artifact["items"][3]["review"]["notes"] = "Query leaks parameter names."
        assert validate_review_artifact(artifact) == []

    def test_benchmark_information_cannot_enter_items(self) -> None:
        artifact = _build()
        artifact["items"][0]["hit_at_1"] = 1.0
        with pytest.raises(ReviewArtifactError, match="forbidden field 'hit_at_1'"):
            validate_review_artifact(artifact)

    def test_ground_truth_cannot_enter_items(self) -> None:
        artifact = _build()
        artifact["items"][0]["relevance_scores"] = [1.0]
        with pytest.raises(ReviewArtifactError, match="forbidden field"):
            validate_review_artifact(artifact)

    def test_mrr_key_forbidden_at_top_level(self) -> None:
        artifact = _build()
        artifact["mrr"] = 0.5
        with pytest.raises(ReviewArtifactError, match="forbidden field 'mrr'"):
            validate_review_artifact(artifact)

    def test_position_ordering_enforced(self) -> None:
        artifact = _build()
        artifact["items"][5]["position"] = 99
        assert any("canonical order" in e for e in validate_review_artifact(artifact))

    def test_pending_item_cannot_carry_decision_timestamp(self) -> None:
        artifact = _build()
        artifact["items"][7]["review"]["decided_utc"] = "2026-08-23T01:00:00Z"
        errors = validate_review_artifact(artifact)
        assert any("decision timestamp" in e for e in errors)


class TestProgress:
    def test_progress_counts_by_state(self) -> None:
        artifact = _build()
        artifact["items"][0]["review"]["state"] = ACCEPTED
        artifact["items"][1]["review"].update(state=REJECTED, notes="too generic")
        progress = review_progress(artifact)
        assert progress == {
            PENDING: 43,
            ACCEPTED: 1,
            REJECTED: 1,
            "total": 45,
        }


def test_build_review_item_shape() -> None:
    unit = _unit("repo006_x")
    candidate = _candidate("repo006_x")
    item = build_review_item(4, candidate, unit)
    assert item["position"] == 4
    assert item["candidate_id"] == "cand_repo006_x"
    assert item["target"]["source_code"].startswith("def Symbol")
