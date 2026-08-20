"""Bounded retry policy for structured output recovery.

Defines which failures are retryable, maximum attempt count,
and validation logic for the policy itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from localbench.runtime.generation.failures import (
    ConstraintViolationError,
    MalformedJSONError,
    MissingFieldError,
    StructuredError,
    TypeMismatchError,
)

# Failures the model can plausibly fix on a subsequent attempt.
RETRYABLE_FAILURES: tuple[type[StructuredError], ...] = (
    MalformedJSONError,
    MissingFieldError,
    TypeMismatchError,
    ConstraintViolationError,
)

DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for bounded retry behaviour.

    Attributes:
        max_attempts: Total generation attempts allowed (must be >= 1).
        retryable_failures: Failure types that trigger a retry.
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    retryable_failures: tuple[type[StructuredError], ...] = field(
        default_factory=lambda: RETRYABLE_FAILURES
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1, got {self.max_attempts}"
            )

    def is_retryable(self, error: StructuredError) -> bool:
        """Return True if *error* should trigger a retry."""
        return isinstance(error, self.retryable_failures)
