"""Structured generation result type.

Wraps the outcome of the parse → validate pipeline, separating
successful validation from observable failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from localbench.runtime.generation.failures import StructuredError


@dataclass
class StructuredResult:
    """Outcome of validating raw model output against a Pydantic schema.

    Attributes:
        valid: Whether the output passed validation.
        raw_text: The original model output, always preserved.
        data: The validated Pydantic model instance (None if invalid).
        errors: List of structured failures (empty if valid).
        extraction_ms: Time spent extracting JSON (None if skipped).
        validation_ms: Time spent on Pydantic validation (None if skipped).
    """

    valid: bool
    raw_text: str
    data: Any = None
    errors: list[StructuredError] = field(default_factory=list)
    extraction_ms: float | None = None
    validation_ms: float | None = None
