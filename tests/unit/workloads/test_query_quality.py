"""Tests for advisory query checks."""

from localbench.workloads.code_retrieval.query_quality import (
    assess_batch,
    assess_candidate,
    assess_query,
)


def test_generic_template_flagged():
    assert "generic_template" in assess_query(
        "What is the function's purpose, inputs, and outputs, "
        "and how does it interact with specific domain concepts?"
    )


def test_count_claim_flagged():
    assert "exact_count_claim_review" in assess_query(
        "Why does the function make three serialization calls?"
    )


def test_clear_query_not_flagged():
    assert (
        assess_query(
            "How does Color.parse validate hexadecimal color values "
            "and raise errors for invalid strings?"
        )
        == []
    )


def test_empty_query():
    assert assess_query("  ") == ["empty_query"]


def test_candidate_preserves_identifiers():
    row = assess_candidate(
        {
            "candidate_id": "candidate-a",
            "code_unit_id": "unit-a",
            "query": "What does this function do?",
        }
    )
    assert row["candidate_id"] == "candidate-a"
    assert row["code_unit_id"] == "unit-a"
    assert "generic_template" in row["flags"]


def test_generic_function_question_flagged():
    assert "generic_template" in assess_query("How does this function work?")


def test_implementation_count_question_flagged():
    assert "exact_count_claim_review" in assess_query(
        "Which function performs exactly three method calls?"
    )


def test_specific_behavior_question_not_generic():
    flags = assess_query(
        "How does this function handle invalid hexadecimal "
        "color values when parsing user input?"
    )
    assert "generic_template" not in flags


def test_batch_exact_duplicate_normalizes_case_punctuation_whitespace():
    candidates = [
        {"candidate_id": "a", "query": "How are invalid colors handled?"},
        {"candidate_id": "b", "query": "  HOW are invalid colors handled!!! "},
    ]
    rows = assess_batch(candidates)
    assert len(rows) == 2
    assert "duplicate_query" not in rows[0]["flags"]
    assert "duplicate_query" in rows[1]["flags"]
    assert rows[1]["similar_to_candidate_id"] == "a"
    assert rows[1]["similarity_score"] == 1.0


def test_batch_near_duplicate_and_distinct_query():
    candidates = [
        {
            "candidate_id": "a",
            "query": "How does the parser handle invalid color strings?",
        },
        {
            "candidate_id": "b",
            "query": "How does the parser handle invalid color string?",
        },
        {"candidate_id": "c", "query": "Where are database connections initialized?"},
    ]
    rows = assess_batch(candidates, similarity_threshold=0.85)
    assert "near_duplicate_query" in rows[1]["flags"]
    assert rows[1]["similar_to_candidate_id"] == "a"
    assert not any("duplicate" in flag for flag in rows[2]["flags"])


def test_batch_deterministic_and_non_mutating():
    candidates = [
        {"candidate_id": "a", "code_unit_id": "u1", "query": "Parse color values"},
        {"candidate_id": "b", "code_unit_id": "u2", "query": "Parse color values"},
        {"candidate_id": "c", "code_unit_id": "u3", "query": "Parse color values"},
    ]
    snapshot = [dict(candidate) for candidate in candidates]
    first = assess_batch(candidates)
    assert first == assess_batch(candidates)
    assert candidates == snapshot
    assert [row["candidate_id"] for row in first] == ["a", "b", "c"]
    assert first[2]["similar_to_candidate_id"] == "a"


def test_batch_empty_queries_not_duplicates():
    rows = assess_batch(
        [
            {"candidate_id": "a", "query": ""},
            {"candidate_id": "b", "query": "!!!"},
        ]
    )
    assert all("similar_to_candidate_id" not in row for row in rows)


def test_batch_invalid_threshold():
    import pytest

    with pytest.raises(ValueError):
        assess_batch([], similarity_threshold=0)
    with pytest.raises(ValueError):
        assess_batch([], similarity_threshold=1.1)
