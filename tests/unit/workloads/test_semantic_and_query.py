"""Tests for semantic label, query case, and relevance contracts."""

import pytest
from pydantic import ValidationError

from localbench.workloads.code_retrieval.schemas import (
    QueryCase,
    QueryRelevance,
    SemanticLabel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_DESCRIPTION = (
    "Retries a failed payment transaction using exponential backoff "
    "with configurable maximum attempts. Logs each retry attempt and "
    "raises an exception after exhausting retries."
)

_MIN_CONCEPTS = ["retry logic", "exponential backoff"]


def _make_label(**overrides) -> dict:
    base = {
        "code_unit_id": "repo001_py_class_Pay_method_retry",
        "description": _VALID_DESCRIPTION,
        "summary": "Payment retry with exponential backoff",
        "concepts": _MIN_CONCEPTS,
        "input_types": ["int (transaction_id)", "int (max_attempts)"],
        "output_type": "bool",
        "side_effects": ["Logs to payment logger"],
        "created_by": "human",
        "label_version": "1.0.0",
    }
    base.update(overrides)
    return base


def _make_query(**overrides) -> dict:
    base = {
        "id": "query_test_0001",
        "query": "Where are payment transactions retried when they fail?",
        "query_style": "natural",
        "query_intent": "find_error_handling",
        "relevant_code_units": [
            "repo001_py_class_Pay_method_retry",
        ],
        "related_concepts": ["retry logic", "exponential backoff"],
        "split": "test",
        "difficulty": "medium",
        "created_at": "2026-08-19T10:30:00Z",
    }
    base.update(overrides)
    return base


def _make_relevance(**overrides) -> dict:
    base = {
        "query_id": "query_test_0001",
        "code_unit_id": "repo001_py_class_Pay_method_retry",
        "relevance_score": 1.0,
        "relevance_label": "direct_match",
        "explanation": "Directly implements retry logic.",
    }
    base.update(overrides)
    return base


# ===========================================================================
# SemanticLabel
# ===========================================================================


class TestSemanticLabelValid:
    def test_valid_label(self):
        label = SemanticLabel(**_make_label())
        assert label.code_unit_id == "repo001_py_class_Pay_method_retry"
        assert label.created_by == "human"
        assert len(label.concepts) == 2

    def test_model_generated_label(self):
        label = SemanticLabel(**_make_label(created_by="model_generated"))
        assert label.created_by == "model_generated"

    def test_hybrid_label(self):
        label = SemanticLabel(**_make_label(created_by="hybrid"))
        assert label.created_by == "hybrid"

    def test_empty_optional_fields(self):
        label = SemanticLabel(**_make_label(
            input_types=[], output_type="", side_effects=[],
        ))
        assert label.input_types == []

    def test_many_concepts(self):
        concepts = [f"concept_{i}" for i in range(10)]
        label = SemanticLabel(**_make_label(concepts=concepts))
        assert len(label.concepts) == 10


class TestSemanticLabelInvalid:
    def test_description_too_short(self):
        with pytest.raises(ValidationError, match="words"):
            SemanticLabel(**_make_label(description="Short."))

    def test_description_too_long(self):
        long_desc = "word " * 257
        with pytest.raises(ValidationError, match="words"):
            SemanticLabel(**_make_label(description=long_desc))

    def test_too_few_concepts(self):
        with pytest.raises(ValidationError, match="items"):
            SemanticLabel(**_make_label(concepts=["only one"]))

    def test_zero_concepts(self):
        with pytest.raises(ValidationError):
            SemanticLabel(**_make_label(concepts=[]))

    def test_missing_code_unit_id(self):
        with pytest.raises(ValidationError):
            SemanticLabel(**{k: v for k, v in _make_label().items()
                           if k != "code_unit_id"})

    def test_missing_description(self):
        with pytest.raises(ValidationError):
            SemanticLabel(**{k: v for k, v in _make_label().items()
                           if k != "description"})

    def test_missing_created_by(self):
        with pytest.raises(ValidationError):
            SemanticLabel(**{k: v for k, v in _make_label().items()
                           if k != "created_by"})

    def test_invalid_created_by(self):
        with pytest.raises(ValidationError):
            SemanticLabel(**_make_label(created_by="automated"))

    def test_missing_label_version(self):
        with pytest.raises(ValidationError):
            SemanticLabel(**{k: v for k, v in _make_label().items()
                           if k != "label_version"})


# ===========================================================================
# QueryCase
# ===========================================================================


class TestQueryCaseValid:
    def test_valid_query(self):
        q = QueryCase(**_make_query())
        assert q.split == "test"
        assert len(q.relevant_code_units) == 1

    def test_multiple_relevant_units(self):
        q = QueryCase(**_make_query(
            relevant_code_units=["unit_a", "unit_b", "unit_c"],
        ))
        assert len(q.relevant_code_units) == 3

    def test_all_styles(self):
        for style in ("natural", "technical", "verbose", "concise"):
            q = QueryCase(**_make_query(query_style=style))
            assert q.query_style == style

    def test_all_difficulties(self):
        for diff in ("easy", "medium", "hard"):
            q = QueryCase(**_make_query(difficulty=diff))
            assert q.difficulty == diff

    def test_empty_related_concepts(self):
        q = QueryCase(**_make_query(related_concepts=[]))
        assert q.related_concepts == []


class TestQueryCaseInvalid:
    def test_no_relevant_code_units(self):
        with pytest.raises(ValidationError, match="At least one"):
            QueryCase(**_make_query(relevant_code_units=[]))

    def test_invalid_split(self):
        with pytest.raises(ValidationError):
            QueryCase(**_make_query(split="train"))

    def test_invalid_style(self):
        with pytest.raises(ValidationError):
            QueryCase(**_make_query(query_style="informal"))

    def test_invalid_difficulty(self):
        with pytest.raises(ValidationError):
            QueryCase(**_make_query(difficulty="extreme"))

    def test_missing_query(self):
        with pytest.raises(ValidationError):
            QueryCase(**{k: v for k, v in _make_query().items()
                        if k != "query"})

    def test_missing_id(self):
        with pytest.raises(ValidationError):
            QueryCase(**{k: v for k, v in _make_query().items()
                        if k != "id"})


# ===========================================================================
# QueryRelevance
# ===========================================================================


class TestQueryRelevanceValid:
    def test_direct_match(self):
        r = QueryRelevance(**_make_relevance())
        assert r.relevance_score == 1.0
        assert r.relevance_label == "direct_match"

    def test_highly_relevant(self):
        r = QueryRelevance(**_make_relevance(
            relevance_score=0.8,
            relevance_label="highly_relevant",
        ))
        assert r.relevance_score == 0.8

    def test_related(self):
        r = QueryRelevance(**_make_relevance(
            relevance_score=0.5,
            relevance_label="related",
        ))
        assert r.relevance_label == "related"

    def test_not_relevant(self):
        r = QueryRelevance(**_make_relevance(
            relevance_score=0.0,
            relevance_label="not_relevant",
        ))
        assert r.relevance_score == 0.0

    def test_boundary_scores(self):
        for score in (0.0, 0.25, 0.5, 0.75, 1.0):
            r = QueryRelevance(**_make_relevance(relevance_score=score))
            assert r.relevance_score == score


class TestQueryRelevanceInvalid:
    def test_score_below_zero(self):
        with pytest.raises(ValidationError):
            QueryRelevance(**_make_relevance(relevance_score=-0.1))

    def test_score_above_one(self):
        with pytest.raises(ValidationError):
            QueryRelevance(**_make_relevance(relevance_score=1.1))

    def test_invalid_label(self):
        with pytest.raises(ValidationError):
            QueryRelevance(**_make_relevance(relevance_label="maybe"))

    def test_missing_query_id(self):
        with pytest.raises(ValidationError):
            QueryRelevance(**{k: v for k, v in _make_relevance().items()
                            if k != "query_id"})

    def test_missing_code_unit_id(self):
        with pytest.raises(ValidationError):
            QueryRelevance(**{k: v for k, v in _make_relevance().items()
                            if k != "code_unit_id"})

    def test_missing_explanation(self):
        with pytest.raises(ValidationError):
            QueryRelevance(**{k: v for k, v in _make_relevance().items()
                            if k != "explanation"})
