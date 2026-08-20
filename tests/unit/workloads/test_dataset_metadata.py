"""Tests for dataset metadata, query generation metadata, and split types."""

import pytest
from pydantic import ValidationError

from localbench.workloads.code_retrieval.schemas import (
    DatasetMetadata,
    QueryGenerationMetadata,
    SplitType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_query_gen(**overrides) -> dict:
    base = {
        "model_name": "gemma-2b",
        "model_version": "1.0",
        "prompt_template_version": "v1",
        "seed": 42,
    }
    base.update(overrides)
    return base


def _make_metadata(**overrides) -> dict:
    base = {
        "version": "1.0.0",
        "schema_version": "1.0.0",
        "release_date": "2026-09-30",
        "repositories": ["repo001", "repo002", "repo003"],
        "repository_commits": {
            "repo001": "abc1234",
            "repo002": "def5678",
            "repo003": "ghi9012",
        },
        "total_code_units": 450,
        "train_cases": 225,
        "validation_cases": 112,
        "test_cases": 113,
        "total_queries": 45,
        "split_seed": 42,
        "query_generation": QueryGenerationMetadata(**_make_query_gen()),
        "parser": "python_ast",
        "frozen": True,
    }
    base.update(overrides)
    return base


# ===========================================================================
# SplitType literal
# ===========================================================================


class TestSplitType:
    def test_valid_values(self):
        for split in ("train", "validation", "test"):
            assert split in ("train", "validation", "test")

    def test_is_literal_type(self):
        assert SplitType.__args__ == ("train", "validation", "test")


# ===========================================================================
# QueryGenerationMetadata
# ===========================================================================


class TestQueryGenerationMetadata:
    def test_valid(self):
        qg = QueryGenerationMetadata(**_make_query_gen())
        assert qg.model_name == "gemma-2b"
        assert qg.seed == 42

    def test_missing_model_name(self):
        with pytest.raises(ValidationError):
            QueryGenerationMetadata(**{
                k: v for k, v in _make_query_gen().items()
                if k != "model_name"
            })

    def test_missing_seed(self):
        with pytest.raises(ValidationError):
            QueryGenerationMetadata(**{
                k: v for k, v in _make_query_gen().items()
                if k != "seed"
            })

    def test_missing_prompt_template_version(self):
        with pytest.raises(ValidationError):
            QueryGenerationMetadata(**{
                k: v for k, v in _make_query_gen().items()
                if k != "prompt_template_version"
            })


# ===========================================================================
# DatasetMetadata — valid
# ===========================================================================


class TestDatasetMetadataValid:
    def test_valid_metadata(self):
        dm = DatasetMetadata(**_make_metadata())
        assert dm.version == "1.0.0"
        assert dm.total_code_units == 450
        assert dm.frozen is True

    def test_counts_correct(self):
        dm = DatasetMetadata(**_make_metadata())
        assert dm.train_cases + dm.validation_cases + dm.test_cases == 450

    def test_split_seed_default(self):
        dm = DatasetMetadata(version="1.0.0", schema_version="1.0.0")
        assert dm.split_seed == 42

    def test_frozen_false(self):
        dm = DatasetMetadata(**_make_metadata(frozen=False))
        assert dm.frozen is False

    def test_empty_metadata(self):
        dm = DatasetMetadata(version="1.0.0", schema_version="1.0.0")
        assert dm.repositories == []
        assert dm.total_queries == 0
        assert dm.query_generation is None

    def test_custom_parser(self):
        dm = DatasetMetadata(**_make_metadata(parser="tree_sitter"))
        assert dm.parser == "tree_sitter"


# ===========================================================================
# DatasetMetadata — invalid
# ===========================================================================


class TestDatasetMetadataInvalid:
    def test_counts_mismatch(self):
        with pytest.raises(ValidationError, match="total_code_units"):
            DatasetMetadata(**_make_metadata(
                total_code_units=400,
                train_cases=225,
                validation_cases=112,
                test_cases=113,
            ))

    def test_missing_version(self):
        with pytest.raises(ValidationError):
            DatasetMetadata(**{
                k: v for k, v in _make_metadata().items()
                if k != "version"
            })

    def test_missing_schema_version(self):
        with pytest.raises(ValidationError):
            DatasetMetadata(**{
                k: v for k, v in _make_metadata().items()
                if k != "schema_version"
            })


# ===========================================================================
# Cross-model integration
# ===========================================================================


class TestCrossModelIntegration:
    def test_metadata_with_all_schemas(self):
        """DatasetMetadata can hold full provenance."""
        qg = QueryGenerationMetadata(**_make_query_gen())
        dm = DatasetMetadata(**_make_metadata(query_generation=qg))
        assert dm.query_generation.model_name == "gemma-2b"
        assert dm.query_generation.seed == 42
        assert dm.split_seed == 42

    def test_metadata_serialization_roundtrip(self):
        """Metadata survives model_dump → model_validate."""
        dm = DatasetMetadata(**_make_metadata())
        data = dm.model_dump()
        dm2 = DatasetMetadata.model_validate(data)
        assert dm.version == dm2.version
        assert dm.total_code_units == dm2.total_code_units
        assert dm.query_generation.seed == dm2.query_generation.seed
