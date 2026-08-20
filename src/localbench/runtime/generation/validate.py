"""Generic Pydantic validation pipeline for structured model output.

The pipeline: raw text → JSON extraction → Pydantic validation → result.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ValidationError

from localbench.runtime.generation.extract import extract_json
from localbench.runtime.generation.failures import (
    ConstraintViolationError,
    MalformedJSONError,
    MissingFieldError,
    StructuredError,
    TypeMismatchError,
)
from localbench.runtime.generation.result import StructuredResult


def validate_structured(
    raw_text: str,
    schema: type[BaseModel],
) -> StructuredResult:
    """Parse raw model output and validate against a Pydantic schema.

    Args:
        raw_text: The raw text returned by the model.
        schema: A Pydantic BaseModel class to validate against.

    Returns:
        A StructuredResult with valid=True and data set if successful,
        or valid=False with errors list populated on failure.
    """
    # Step 1: Extract JSON.
    t0 = time.perf_counter()
    try:
        parsed = extract_json(raw_text)
    except MalformedJSONError as exc:
        return StructuredResult(
            valid=False,
            raw_text=raw_text,
            errors=[exc],
            extraction_ms=_ms_since(t0),
        )
    extraction_ms = _ms_since(t0)

    # Step 2: Validate against schema.
    t1 = time.perf_counter()
    try:
        instance = schema.model_validate(parsed)
    except ValidationError as exc:
        failures = _classify_validation_error(exc)
        return StructuredResult(
            valid=False,
            raw_text=raw_text,
            errors=failures,
            extraction_ms=extraction_ms,
            validation_ms=_ms_since(t1),
        )

    return StructuredResult(
        valid=True,
        raw_text=raw_text,
        data=instance,
        extraction_ms=extraction_ms,
        validation_ms=_ms_since(t1),
    )


def _classify_validation_error(exc: ValidationError) -> list[StructuredError]:
    """Convert a Pydantic ValidationError into structured failure objects."""
    failures: list[StructuredError] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        field_name = ".".join(str(part) for part in loc) or "<root>"
        code = err.get("type", "")
        msg = err.get("msg", "")

        if code == "missing":
            failures.append(MissingFieldError(field_name))
        elif code.startswith("type_error") or code.startswith(
            "int_parsing"
        ):
            failures.append(
                TypeMismatchError(field_name, _expected_type(code), msg)
            )
        else:
            failures.append(ConstraintViolationError(field_name, msg))
    return failures


def _expected_type(code: str) -> str:
    """Derive a human-readable expected type from Pydantic error code."""
    mapping = {
        "int_parsing": "int",
        "float_parsing": "float",
        "bool_parsing": "bool",
        "list_type": "list",
        "str_type": "str",
    }
    return mapping.get(code, "unknown")


def _ms_since(start: float) -> float:
    """Milliseconds elapsed since *start*."""
    return round((time.perf_counter() - start) * 1000, 2)
