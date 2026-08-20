"""Tests for the structured validation pipeline."""

import json

from pydantic import BaseModel, Field

from localbench.runtime.generation.failures import (
    ConstraintViolationError,
    MalformedJSONError,
    MissingFieldError,
)
from localbench.runtime.generation.validate import validate_structured


class SampleArtifact(BaseModel):
    """Minimal schema for testing validation."""

    description: str
    concepts: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class StrictArtifact(BaseModel):
    """Schema with required fields and constrained types."""

    name: str
    count: int = Field(ge=0)
    tags: list[str]


class TestValidOutput:
    def test_valid_json_passes(self):
        """Valid JSON matching schema returns valid result."""
        raw = json.dumps({
            "description": "A retry mechanism.",
            "concepts": ["retry", "backoff"],
            "confidence": 0.9,
        })
        result = validate_structured(raw, SampleArtifact)
        assert result.valid is True
        assert result.data.description == "A retry mechanism."
        assert result.data.concepts == ["retry", "backoff"]
        assert result.errors == []

    def test_valid_fenced_json_passes(self):
        """Fenced JSON is extracted and validated."""
        raw = (
            '```json\n'
            '{"description": "Test.", "concepts": ["a"], '
            '"confidence": 0.5}\n'
            '```'
        )
        result = validate_structured(raw, SampleArtifact)
        assert result.valid is True
        assert result.data.description == "Test."

    def test_valid_with_prefix_text(self):
        """JSON with conversational prefix is handled."""
        raw = 'Here you go:\n{"description": "X", "concepts": ["y"], "confidence": 0.1}'
        result = validate_structured(raw, SampleArtifact)
        assert result.valid is True

    def test_result_preserves_raw_text(self):
        """Raw text is always preserved in the result."""
        raw = '{"description": "X", "concepts": ["y"], "confidence": 0.5}'
        result = validate_structured(raw, SampleArtifact)
        assert result.raw_text == raw


class TestMalformedJSON:
    def test_empty_input_returns_failure(self):
        """Empty input produces MalformedJSON failure."""
        result = validate_structured("", SampleArtifact)
        assert result.valid is False
        assert len(result.errors) == 1
        assert isinstance(result.errors[0], MalformedJSONError)

    def test_non_json_text_returns_failure(self):
        """Non-JSON text produces MalformedJSON failure."""
        result = validate_structured("Hello world", SampleArtifact)
        assert result.valid is False
        assert isinstance(result.errors[0], MalformedJSONError)

    def test_truncated_json_returns_failure(self):
        """Truncated JSON produces MalformedJSON failure."""
        result = validate_structured('{"desc', SampleArtifact)
        assert result.valid is False
        assert isinstance(result.errors[0], MalformedJSONError)


class TestValidationErrors:
    def test_missing_required_field(self):
        """Missing required field produces MissingFieldError."""
        raw = json.dumps({"concepts": ["a"], "confidence": 0.5})
        result = validate_structured(raw, SampleArtifact)
        assert result.valid is False
        assert any(
            isinstance(e, MissingFieldError) and e.field == "description"
            for e in result.errors
        )

    def test_wrong_type_returns_error(self):
        """Wrong field type produces TypeMismatch or ConstraintViolation."""
        raw = json.dumps({
            "description": "OK",
            "concepts": ["a"],
            "confidence": "not_a_number",
        })
        result = validate_structured(raw, SampleArtifact)
        assert result.valid is False
        assert len(result.errors) >= 1

    def test_constraint_violation_below_min(self):
        """Value below minimum produces ConstraintViolationError."""
        raw = json.dumps({
            "description": "OK",
            "concepts": ["a"],
            "confidence": -0.5,
        })
        result = validate_structured(raw, SampleArtifact)
        assert result.valid is False
        assert any(
            isinstance(e, ConstraintViolationError)
            for e in result.errors
        )

    def test_constraint_violation_above_max(self):
        """Value above maximum produces ConstraintViolationError."""
        raw = json.dumps({
            "description": "OK",
            "concepts": ["a"],
            "confidence": 1.5,
        })
        result = validate_structured(raw, SampleArtifact)
        assert result.valid is False
        assert any(
            isinstance(e, ConstraintViolationError)
            for e in result.errors
        )

    def test_empty_list_violates_min_length(self):
        """Empty list violates min_length constraint."""
        raw = json.dumps({
            "description": "OK",
            "concepts": [],
            "confidence": 0.5,
        })
        result = validate_structured(raw, SampleArtifact)
        assert result.valid is False

    def test_multiple_errors_detected(self):
        """Multiple validation errors are all captured."""
        raw = json.dumps({"name": "test"})
        result = validate_structured(raw, StrictArtifact)
        assert result.valid is False
        assert len(result.errors) >= 1


class TestTimingInfo:
    def test_extraction_ms_populated(self):
        """extraction_ms is set on successful extraction."""
        raw = '{"description": "X", "concepts": ["y"], "confidence": 0.1}'
        result = validate_structured(raw, SampleArtifact)
        assert result.extraction_ms is not None
        assert result.extraction_ms >= 0

    def test_validation_ms_populated_on_success(self):
        """validation_ms is set on successful validation."""
        raw = '{"description": "X", "concepts": ["y"], "confidence": 0.1}'
        result = validate_structured(raw, SampleArtifact)
        assert result.validation_ms is not None
        assert result.validation_ms >= 0

    def test_extraction_ms_on_failure(self):
        """extraction_ms is set even when extraction fails."""
        result = validate_structured("not json", SampleArtifact)
        assert result.extraction_ms is not None
        assert result.validation_ms is None
