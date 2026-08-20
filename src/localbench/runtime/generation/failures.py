"""Structural failure taxonomy for structured output validation.

Covers failures in the parse → validate pipeline. Semantic failures
(incoherent, grinding output) are a separate concern for later phases.
"""

from __future__ import annotations


class StructuredError(Exception):
    """Base for all structured output failures."""


class MalformedJSONError(StructuredError):
    """Raw model output is not valid JSON."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message or "Model output is not valid JSON."
        )


class MissingFieldError(StructuredError):
    """A required field is absent from the parsed JSON."""

    def __init__(self, field: str) -> None:
        super().__init__(f"Missing required field: '{field}'.")
        self.field = field


class TypeMismatchError(StructuredError):
    """A field has an unexpected type."""

    def __init__(self, field: str, expected: str, got: str) -> None:
        super().__init__(
            f"Field '{field}' expected {expected}, got {got}."
        )
        self.field = field
        self.expected = expected
        self.got = got


class ConstraintViolationError(StructuredError):
    """A field value violates a Pydantic constraint."""

    def __init__(self, field: str, detail: str) -> None:
        super().__init__(
            f"Constraint violation on '{field}': {detail}."
        )
        self.field = field
        self.detail = detail
