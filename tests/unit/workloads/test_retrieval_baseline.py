"""Offline retrieval baseline regression tests."""

import pytest

from localbench.workloads.code_retrieval.metrics import evaluate_rankings
from localbench.workloads.code_retrieval.retrieval import BM25Index, tokenize


def test_identifier_tokenization():
    assert tokenize("retryPayment_retry_count") == [
        "retry",
        "payment",
        "retry",
        "count",
    ]


def test_bm25_ranks_matching_code_first():
    index = BM25Index(
        {
            "b": "def validate_email(): return email",
            "a": "def retry_payment(): retry payment transaction",
        }
    )
    hits = index.search("retry payment")
    assert hits[0].code_unit_id == "a"
    assert hits[0].score > 0


def test_bm25_ties_are_stable():
    index = BM25Index({"z": "same text", "a": "same text"})
    assert [hit.code_unit_id for hit in index.search("same")] == ["a", "z"]


def test_bm25_empty_query_and_invalid_k():
    index = BM25Index({"a": "hello"})
    assert index.search(" ") == []
    with pytest.raises(ValueError):
        index.search("hello", k=0)


def test_metrics_include_missing_rankings_in_denominator():
    result = evaluate_rankings(
        {"q1": ["wrong", "target"]},
        {"q1": ["target"], "q2": ["another"]},
    )
    assert result.query_count == 2
    assert result.hit_at_1 == 0
    assert result.hit_at_5 == 0.5
    assert result.hit_at_10 == 0.5
    assert result.mrr == 0.25


def test_metrics_validate_duplicates_and_unknown_queries():
    with pytest.raises(ValueError):
        evaluate_rankings({"q": ["a", "a"]}, {"q": ["a"]})
    with pytest.raises(ValueError):
        evaluate_rankings({"other": ["a"]}, {"q": ["a"]})
