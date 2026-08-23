"""Tests for deterministic final-45 selection (§4.4.5)."""

from __future__ import annotations

import random

import pytest

from localbench.workloads.code_retrieval.selection import (
    FINAL_QUERY_COUNT,
    REQUIRED_RECORD_FIELDS,
    SEED,
    SelectionError,
    build_eligible_pool,
    build_selection_record,
    distribution,
    pool_hash,
    select_final_queries,
)


def _candidate(unit: str, style: str = "technical", **overrides) -> dict:
    record = {
        "candidate_id": f"cand_{unit}",
        "code_unit_id": unit,
        "query": f"Locate the logic in {unit}.",
        "query_style": style,
        "success": True,
        "validation_passed": True,
        "leakage_passed": True,
    }
    record.update(overrides)
    return record


def _pool(n: int = 50, start: int = 0) -> list[dict]:
    return [_candidate(f"repo006_u{index:04d}") for index in range(start, start + n)]


TEST_IDS = {f"repo006_u{index:04d}" for index in range(0, 500)}
TRAIN_IDS = {f"repo001_u{index:04d}" for index in range(0, 100)}
VAL_IDS = {f"repo005_u{index:04d}" for index in range(0, 100)}


class TestEligiblePool:
    def test_canonical_ordering_is_code_unit_id_lexicographic(self) -> None:
        shuffled = _pool(10)
        random.Random(7).shuffle(shuffled)
        ordered = build_eligible_pool(shuffled, TEST_IDS, TRAIN_IDS, VAL_IDS)
        assert [c["code_unit_id"] for c in ordered] == sorted(
            c["code_unit_id"] for c in shuffled
        )

    def test_ineligible_records_excluded(self) -> None:
        candidates = [
            *_pool(3),
            _candidate("repo006_bad1", success=False),
            _candidate("repo006_bad2", validation_passed=False),
            _candidate("repo006_bad3", leakage_passed=False),
            _candidate("repo006_bad4", query="   "),
            _candidate("repo001_train_only"),
            _candidate("repo005_val_only"),
        ]
        ordered = build_eligible_pool(candidates, TEST_IDS, TRAIN_IDS, VAL_IDS)
        assert [c["code_unit_id"] for c in ordered] == [
            "repo006_u0000",
            "repo006_u0001",
            "repo006_u0002",
        ]

    def test_duplicate_id_refusal(self) -> None:
        candidates = [*_pool(3), _candidate("repo006_u0000")]
        with pytest.raises(SelectionError, match="duplicate code_unit_id"):
            build_eligible_pool(candidates, TEST_IDS, TRAIN_IDS, VAL_IDS)


class TestPoolHash:
    def test_hash_deterministic_across_input_orderings(self) -> None:
        pool = _pool(20)
        reordered = list(pool)
        random.Random(9).shuffle(reordered)
        assert pool_hash(pool) == pool_hash(reordered)

    def test_hash_changes_when_record_content_changes(self) -> None:
        pool = _pool(5)
        mutated = [dict(pool[0], query="different"), *pool[1:]]
        assert pool_hash(mutated) != pool_hash(pool)

    def test_hash_matches_frozen_serialization_recipe(self) -> None:
        import hashlib
        import json

        pool = _pool(4)
        payload = "\n".join(
            json.dumps(c, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for c in sorted(pool, key=lambda x: x["code_unit_id"])
        )
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert pool_hash(sorted(pool, key=lambda x: x["code_unit_id"])) == expected


class TestSampling:
    def test_exact_sample_size_and_uniqueness(self) -> None:
        pool = build_eligible_pool(_pool(60), TEST_IDS, TRAIN_IDS, VAL_IDS)
        selected = select_final_queries(pool)
        assert len(selected) == FINAL_QUERY_COUNT
        units = [c["code_unit_id"] for c in selected]
        cand_ids = [c["candidate_id"] for c in selected]
        assert len(set(units)) == FINAL_QUERY_COUNT
        assert len(set(cand_ids)) == FINAL_QUERY_COUNT

    def test_seed_reproducibility_same_ids_same_order(self) -> None:
        pool = build_eligible_pool(_pool(60), TEST_IDS, TRAIN_IDS, VAL_IDS)
        first = select_final_queries(pool)
        second = select_final_queries(pool)
        reference = random.Random(SEED).sample(list(pool), FINAL_QUERY_COUNT)
        assert first == second == reference

    def test_insufficient_pool_refusal(self) -> None:
        pool = build_eligible_pool(_pool(44), TEST_IDS, TRAIN_IDS, VAL_IDS)
        with pytest.raises(SelectionError, match="45 required"):
            select_final_queries(pool)

    def test_timestamp_neutrality(self) -> None:
        pool = build_eligible_pool(_pool(60), TEST_IDS, TRAIN_IDS, VAL_IDS)
        base = build_selection_record(
            ordered_pool=pool,
            selected=select_final_queries(pool),
            repository_by_code_unit_id={u: "repo006" for u in TEST_IDS},
            generation_source_commit="ca257dd",
            selection_version="1.0.0",
            created_utc="2026-08-23T00:00:00Z",
        )
        later = build_selection_record(
            ordered_pool=pool,
            selected=select_final_queries(pool),
            repository_by_code_unit_id={u: "repo006" for u in TEST_IDS},
            generation_source_commit="ca257dd",
            selection_version="1.0.0",
            created_utc="2099-01-01T00:00:00Z",
        )
        assert pool_hash(pool) == base["eligible_pool_sha256"]
        assert base["selected_code_unit_ids"] == later["selected_code_unit_ids"]
        differing = {k for k in base if base[k] != later[k]}
        assert differing == {"selection_created_utc"}


class TestSelectionRecord:
    def test_metadata_completeness_and_no_forbidden_fields(self) -> None:
        pool = build_eligible_pool(_pool(60), TEST_IDS, TRAIN_IDS, VAL_IDS)
        record = build_selection_record(
            ordered_pool=pool,
            selected=select_final_queries(pool),
            repository_by_code_unit_id={u: "repo006" for u in TEST_IDS},
            generation_source_commit="ca257dd",
            selection_version="1.0.0",
            created_utc="2026-08-23T00:00:00Z",
        )
        assert set(record) == set(REQUIRED_RECORD_FIELDS)
        forbidden = ("hit", "mrr", "latency", "benchmark", "ground_truth")
        assert not any(any(word in key for word in forbidden) for key in record)
        assert record["seed"] == SEED
        assert record["prng"] == "python.random.Random"
        assert record["sampling_method"] == "sample_without_replacement"
        assert record["canonical_order"] == "code_unit_id_lexicographic_ascending"
        assert record["eligible_candidate_count"] == len(pool)
        assert record["selected_count"] == FINAL_QUERY_COUNT

    def test_distributions_are_observations_sorted_by_key(self) -> None:
        styles = ["technical", "natural", "natural", "technical", "verbose"]
        assert distribution(styles) == {"natural": 2, "technical": 2, "verbose": 1}
