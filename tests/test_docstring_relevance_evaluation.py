from __future__ import annotations

import csv

import pytest

from scripts.evaluate_docstring_relevance import evaluate_query, load_reviews


def test_evaluate_query_uses_graded_ndcg_and_strict_mrr() -> None:
    rows = [
        {"candidate_rank": 1, "relevance_label": "not_relevant"},
        {"candidate_rank": 2, "relevance_label": "partially_relevant"},
        {"candidate_rank": 3, "relevance_label": "relevant"},
        {"candidate_rank": 4, "relevance_label": "uncertain"},
    ]

    result = evaluate_query(rows, 4)

    assert result["judged_candidates"] == 3
    assert result["uncertain_or_blank"] == 1
    assert result["strict_relevant_count"] == 1
    assert result["mrr_strict"] == pytest.approx(1 / 3)
    assert result["precision_at_k_judged"] == pytest.approx(1 / 3)
    assert 0 < result["ndcg_at_k_judged_pool"] < 1


def test_load_reviews_keeps_blank_and_uncertain_unjudged(tmp_path) -> None:
    path = tmp_path / "review.csv"
    fields = ["query_id", "candidate_rank", "candidate_code_unit_id", "relevance_label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"query_id": "D001", "candidate_rank": 1, "candidate_code_unit_id": "unit-a", "relevance_label": ""},
            {"query_id": "D001", "candidate_rank": 2, "candidate_code_unit_id": "unit-b", "relevance_label": "uncertain"},
            {"query_id": "D001", "candidate_rank": 3, "candidate_code_unit_id": "unit-c", "relevance_label": "relevant"},
        ])

    rows = load_reviews(path)["D001"]

    assert [row["relevance_label"] for row in rows] == ["", "uncertain", "relevant"]


def test_load_reviews_rejects_duplicate_query_rank(tmp_path) -> None:
    path = tmp_path / "duplicate.csv"
    fields = ["query_id", "candidate_rank", "candidate_code_unit_id", "relevance_label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"query_id": "D001", "candidate_rank": 1, "candidate_code_unit_id": "unit-a", "relevance_label": "relevant"},
            {"query_id": "D001", "candidate_rank": 1, "candidate_code_unit_id": "unit-b", "relevance_label": "not_relevant"},
        ])

    with pytest.raises(ValueError, match="duplicate query/rank"):
        load_reviews(path)


def test_load_reviews_rejects_unknown_label(tmp_path) -> None:
    path = tmp_path / "bad-label.csv"
    fields = ["query_id", "candidate_rank", "candidate_code_unit_id", "relevance_label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "query_id": "D001",
            "candidate_rank": 1,
            "candidate_code_unit_id": "unit-a",
            "relevance_label": "maybe",
        })

    with pytest.raises(ValueError, match="unsupported relevance label"):
        load_reviews(path)



def test_evaluate_query_does_not_count_unranked_candidates_as_retrieved() -> None:
    rows = [
        {"candidate_rank": 1, "relevance_label": "not_relevant"},
        {"candidate_rank": None, "relevance_label": "relevant"},
    ]

    result = evaluate_query(rows, 1)

    assert result["judged_candidates"] == 2
    assert result["hit_at_k_strict"] == 0
    assert result["mrr_strict"] == 0
    assert result["precision_at_k_judged"] == 0
    assert result["ndcg_at_k_judged_pool"] < 1


def test_load_reviews_supports_method_specific_rank_column(tmp_path) -> None:
    path = tmp_path / "pooled.csv"
    fields = ["query_id", "candidate_rank_dense", "candidate_code_unit_id", "relevance_label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"query_id": "D001", "candidate_rank_dense": "1", "candidate_code_unit_id": "unit-a", "relevance_label": "relevant"},
            {"query_id": "D001", "candidate_rank_dense": "", "candidate_code_unit_id": "unit-b", "relevance_label": "not_relevant"},
        ])

    rows = load_reviews(path, "candidate_rank_dense")["D001"]

    assert [row["candidate_rank"] for row in rows] == [1, None]
