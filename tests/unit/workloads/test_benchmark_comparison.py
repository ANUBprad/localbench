"""Comparison runner regression tests without network or model downloads."""

import pytest

from localbench.workloads.code_retrieval.benchmark import run_comparison


def test_bm25_comparison():
    result = run_comparison(
        {"a": "retry payment", "b": "validate email"},
        {"q": "retry payment"},
        {"q": ["a"]},
    )
    assert result["query_count"] == 1
    assert result["methods"]["bm25"]["metrics"]["hit_at_1"] == 1.0


def test_dense_comparison_uses_same_labels():
    vectors = {
        "retry payment": [1.0, 0.0],
        "validate email": [0.0, 1.0],
        "payment retry": [1.0, 0.0],
    }
    result = run_comparison(
        {"a": "retry payment", "b": "validate email"},
        {"q": "payment retry"},
        {"q": ["a"]},
        encoder=lambda texts: [vectors[text] for text in texts],
    )
    assert set(result["methods"]) == {"bm25", "dense"}
    assert result["methods"]["dense"]["metrics"]["hit_at_1"] == 1.0


def test_missing_labels_rejected():
    with pytest.raises(ValueError):
        run_comparison({"a": "text"}, {"q": "text"}, {})


def test_unknown_gold_rejected():
    with pytest.raises(ValueError):
        run_comparison({"a": "text"}, {"q": "text"}, {"q": ["missing"]})


def test_short_k_rejected():
    with pytest.raises(ValueError):
        run_comparison({"a": "text"}, {"q": "text"}, {"q": ["a"]}, k=5)
