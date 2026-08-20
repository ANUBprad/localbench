"""Tests for retry policy and attempt records."""

import pytest

from localbench.runtime.generation.attempt import AttemptRecord, AttemptStatus
from localbench.runtime.generation.failures import (
    ConstraintViolationError,
    MalformedJSONError,
    MissingFieldError,
    TypeMismatchError,
)
from localbench.runtime.generation.policy import (
    DEFAULT_MAX_ATTEMPTS,
    RetryPolicy,
)


class TestRetryPolicyDefaults:
    def test_default_max_attempts(self):
        """Default policy allows 3 attempts."""
        policy = RetryPolicy()
        assert policy.max_attempts == DEFAULT_MAX_ATTEMPTS
        assert policy.max_attempts == 3

    def test_default_retryable_failures(self):
        """Default policy retries structured output failures."""
        policy = RetryPolicy()
        assert MalformedJSONError in policy.retryable_failures
        assert MissingFieldError in policy.retryable_failures
        assert TypeMismatchError in policy.retryable_failures
        assert ConstraintViolationError in policy.retryable_failures


class TestRetryPolicyCustom:
    def test_custom_max_attempts(self):
        """Custom max attempts are accepted."""
        policy = RetryPolicy(max_attempts=5)
        assert policy.max_attempts == 5

    def test_single_attempt_disables_retry(self):
        """max_attempts=1 means no retries."""
        policy = RetryPolicy(max_attempts=1)
        assert policy.max_attempts == 1

    def test_custom_retryable_failures(self):
        """Custom retryable failure set is accepted."""
        policy = RetryPolicy(retryable_failures=(MalformedJSONError,))
        assert policy.retryable_failures == (MalformedJSONError,)


class TestRetryPolicyValidation:
    def test_zero_max_attempts_raises(self):
        """max_attempts < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            RetryPolicy(max_attempts=0)

    def test_negative_max_attempts_raises(self):
        """Negative max_attempts raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            RetryPolicy(max_attempts=-1)


class TestRetryPolicyIsRetryable:
    def test_malformed_json_is_retryable(self):
        """MalformedJSONError is retryable by default."""
        policy = RetryPolicy()
        assert policy.is_retryable(MalformedJSONError()) is True

    def test_missing_field_is_retryable(self):
        """MissingFieldError is retryable by default."""
        policy = RetryPolicy()
        assert policy.is_retryable(MissingFieldError("x")) is True

    def test_type_mismatch_is_retryable(self):
        """TypeMismatchError is retryable by default."""
        policy = RetryPolicy()
        err = TypeMismatchError("x", "int", "str")
        assert policy.is_retryable(err) is True

    def test_constraint_violation_is_retryable(self):
        """ConstraintViolationError is retryable by default."""
        policy = RetryPolicy()
        assert policy.is_retryable(
            ConstraintViolationError("x", "too small")
        ) is True

    def test_custom_policy_restricts_retryable(self):
        """Only listed failures are retryable."""
        policy = RetryPolicy(retryable_failures=(MalformedJSONError,))
        assert policy.is_retryable(MalformedJSONError()) is True
        assert policy.is_retryable(MissingFieldError("x")) is False


class TestAttemptRecord:
    def test_success_record(self):
        """Successful attempt has correct fields."""
        rec = AttemptRecord(
            attempt_number=1,
            status=AttemptStatus.SUCCESS,
            raw_text='{"a": 1}',
        )
        assert rec.attempt_number == 1
        assert rec.status == AttemptStatus.SUCCESS
        assert rec.errors == []
        assert rec.will_retry is False

    def test_failed_record_with_errors(self):
        """Failed attempt records errors."""
        err = MalformedJSONError("bad json")
        rec = AttemptRecord(
            attempt_number=2,
            status=AttemptStatus.FAILED,
            raw_text="not json",
            errors=[err],
            will_retry=True,
        )
        assert rec.status == AttemptStatus.FAILED
        assert len(rec.errors) == 1
        assert rec.will_retry is True

    def test_attempt_numbering(self):
        """Attempt numbers are 1-indexed."""
        rec = AttemptRecord(
            attempt_number=1,
            status=AttemptStatus.SUCCESS,
            raw_text="ok",
        )
        assert rec.attempt_number == 1

    def test_generation_ms_recorded(self):
        """Generation timing is captured when provided."""
        rec = AttemptRecord(
            attempt_number=1,
            status=AttemptStatus.SUCCESS,
            raw_text="ok",
            generation_ms=123.45,
        )
        assert rec.generation_ms == 123.45

    def test_generation_ms_none_by_default(self):
        """Generation timing defaults to None."""
        rec = AttemptRecord(
            attempt_number=1,
            status=AttemptStatus.SUCCESS,
            raw_text="ok",
        )
        assert rec.generation_ms is None
