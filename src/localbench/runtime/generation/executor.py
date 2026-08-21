"""Bounded retry executor for structured generation.

Orchestrates the generate → validate → retry loop. Each attempt is
recorded in full so callers can inspect the retry history. Retries
are bounded by RetryPolicy and only triggered for retryable failures.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from localbench.runtime.generation.attempt import AttemptRecord, AttemptStatus
from localbench.runtime.generation.failures import StructuredError
from localbench.runtime.generation.policy import RetryPolicy
from localbench.runtime.generation.result import StructuredResult
from localbench.runtime.generation.retry_context import (
    build_corrective_prompt,
)
from localbench.runtime.generation.validate import validate_structured


class RetryExhaustedError(Exception):
    """All retry attempts exhausted without a valid result."""

    def __init__(self, attempts: list[AttemptRecord]) -> None:
        self.attempts = attempts
        count = len(attempts)
        super().__init__(
            f"All {count} attempt(s) failed. "
            f"Last error: {attempts[-1].errors[0]}"
            if attempts
            else "No attempts recorded."
        )


# Type alias for the generate function.
# Takes a prompt string, returns raw text from the model.
GenerateFn = Callable[[str], str]


@dataclass
class RetryResult:
    """Outcome of a bounded retry loop.

    Attributes:
        success: Whether a valid result was obtained.
        result: The final StructuredResult (None if all failed).
        attempts: Ordered list of AttemptRecords.
        total_generation_ms: Sum of generation times across attempts.
        total_validation_ms: Sum of validation times across attempts.
    """

    success: bool
    result: StructuredResult | None
    attempts: list[AttemptRecord] = field(default_factory=list)
    total_generation_ms: float = 0.0
    total_validation_ms: float = 0.0


def run_with_retry(
    prompt: str,
    schema: type[BaseModel],
    generate_fn: GenerateFn,
    policy: RetryPolicy | None = None,
) -> RetryResult:
    """Execute generation with bounded retry and structured validation.

    Args:
        prompt: The user prompt to send to the model.
        schema: Pydantic model class to validate output against.
        generate_fn: Callable that takes a prompt and returns raw text.
        policy: Retry configuration (uses defaults if None).

    Returns:
        RetryResult with the final outcome and full attempt history.
    """
    if policy is None:
        policy = RetryPolicy()

    attempts: list[AttemptRecord] = []
    total_gen_ms = 0.0
    total_val_ms = 0.0
    current_prompt = prompt

    for attempt_num in range(1, policy.max_attempts + 1):
        is_last = attempt_num == policy.max_attempts

        # Generate
        gen_t0 = time.perf_counter()
        try:
            raw_text = generate_fn(current_prompt)
        except Exception as exc:
            gen_ms = _ms_since(gen_t0)
            total_gen_ms += gen_ms
            err = _wrap_generation_error(exc)
            rec = AttemptRecord(
                attempt_number=attempt_num,
                status=AttemptStatus.FAILED,
                raw_text="",
                errors=[err],
                will_retry=False,
                generation_ms=gen_ms,
            )
            attempts.append(rec)
            break

        gen_ms = _ms_since(gen_t0)
        total_gen_ms += gen_ms

        # Validate
        val_t0 = time.perf_counter()
        result = validate_structured(raw_text, schema)
        val_ms = _ms_since(val_t0)
        total_val_ms += val_ms

        if result.valid:
            rec = AttemptRecord(
                attempt_number=attempt_num,
                status=AttemptStatus.SUCCESS,
                raw_text=raw_text,
                generation_ms=gen_ms,
            )
            attempts.append(rec)
            return RetryResult(
                success=True,
                result=result,
                attempts=attempts,
                total_generation_ms=total_gen_ms,
                total_validation_ms=total_val_ms,
            )

        # Validation failed — check if retryable
        all_retryable = all(
            policy.is_retryable(e) for e in result.errors
        )
        will_retry = all_retryable and not is_last

        rec = AttemptRecord(
            attempt_number=attempt_num,
            status=AttemptStatus.FAILED,
            raw_text=raw_text,
            errors=result.errors,
            will_retry=will_retry,
            generation_ms=gen_ms,
        )
        attempts.append(rec)

        if not will_retry:
            break

        # Build corrective prompt for next attempt
        corrective = build_corrective_prompt(result.errors)
        current_prompt = f"{prompt}\n\n{corrective}"

    # All attempts exhausted
    return RetryResult(
        success=False,
        result=None,
        attempts=attempts,
        total_generation_ms=total_gen_ms,
        total_validation_ms=total_val_ms,
    )


def _ms_since(start: float) -> float:
    """Milliseconds elapsed since *start*."""
    return round((time.perf_counter() - start) * 1000, 2)


def _wrap_generation_error(exc: Exception) -> StructuredError:
    """Wrap a non-retryable generation error as a StructuredError.

    The original exception type name is kept in the message so callers
    can distinguish provider failures (unavailable, timeout, model
    error) from structured validation failures.
    """
    from localbench.runtime.generation.failures import (
        StructuredError as BaseStructuredError,
    )

    if isinstance(exc, BaseStructuredError):
        return exc
    return StructuredError(f"{type(exc).__name__}: {exc}")
