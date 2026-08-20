"""Attempt records for structured generation retries.

Every generation attempt — whether successful or failed — is captured
in an AttemptRecord so callers can inspect the full retry history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from localbench.runtime.generation.failures import StructuredError


class AttemptStatus:
    """Enum-like constants for attempt outcome."""

    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class AttemptRecord:
    """One generation attempt within a retry loop.

    Attributes:
        attempt_number: 1-indexed attempt count.
        status: AttemptStatus.SUCCESS or AttemptStatus.FAILED.
        raw_text: The raw model output for this attempt.
        errors: Structured failures (empty on success).
        will_retry: Whether another attempt will follow.
        generation_ms: Wall-clock time for the generate call.
    """

    attempt_number: int
    status: str
    raw_text: str
    errors: list[StructuredError] = field(default_factory=list)
    will_retry: bool = False
    generation_ms: float | None = None
