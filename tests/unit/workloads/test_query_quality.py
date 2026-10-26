"""Tests for advisory query checks."""

from localbench.workloads.code_retrieval.query_quality import (
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
