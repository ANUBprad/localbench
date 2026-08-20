"""Tests for retry context builder and bounded retry executor."""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel

from localbench.runtime.generation.attempt import AttemptStatus
from localbench.runtime.generation.executor import (
    RetryResult,
    run_with_retry,
)
from localbench.runtime.generation.failures import (
    MalformedJSONError,
    MissingFieldError,
    TypeMismatchError,
)
from localbench.runtime.generation.policy import RetryPolicy
from localbench.runtime.generation.retry_context import (
    build_corrective_prompt,
    build_retry_prompt,
)

# --- Fixtures ---


class SimpleSchema(BaseModel):
    name: str
    value: int


class SchemaWithDefault(BaseModel):
    name: str
    value: int = 0


# --- Tests: build_corrective_prompt ---


class TestBuildCorrectivePrompt:
    def test_empty_errors_returns_empty(self):
        """No errors → empty string."""
        assert build_corrective_prompt([]) == ""

    def test_single_error(self):
        """Single error produces a readable message."""
        err = MalformedJSONError("not valid JSON")
        prompt = build_corrective_prompt([err])
        assert "Previous attempt failed validation:" in prompt
        assert "- not valid JSON" in prompt
        assert "Please produce valid JSON" in prompt

    def test_multiple_errors(self):
        """Multiple errors are all listed."""
        errs = [
            MissingFieldError("name"),
            TypeMismatchError("value", "int", "str"),
        ]
        prompt = build_corrective_prompt(errs)
        assert "Missing required field: 'name'." in prompt
        assert "Field 'value' expected int, got str." in prompt

    def test_deterministic_output(self):
        """Same errors always produce the same prompt."""
        errs = [MalformedJSONError("bad")]
        p1 = build_corrective_prompt(errs)
        p2 = build_corrective_prompt(errs)
        assert p1 == p2


# --- Tests: build_retry_prompt ---


class TestBuildRetryPrompt:
    def test_no_errors_returns_original(self):
        """With no errors, the original prompt is returned unchanged."""
        result = build_retry_prompt("hello", [])
        assert result == "hello"

    def test_appends_corrective_suffix(self):
        """Errors cause the corrective prompt to be appended."""
        errs = [MalformedJSONError("bad")]
        result = build_retry_prompt("original", errs)
        assert result.startswith("original")
        assert "Previous attempt failed validation:" in result

    def test_preserves_original_prompt(self):
        """The original prompt content is fully preserved."""
        errs = [MalformedJSONError("bad")]
        result = build_retry_prompt("Do this task carefully.", errs)
        assert result.startswith("Do this task carefully.")


# --- Tests: run_with_retry ---


class TestRunWithRetrySuccess:
    def test_success_on_first_attempt(self):
        """Valid output succeeds immediately with no retries."""
        valid_json = '{"name": "test", "value": 42}'
        generate_fn = MagicMock(return_value=valid_json)

        result = run_with_retry("prompt", SimpleSchema, generate_fn)

        assert result.success is True
        assert result.result is not None
        assert result.result.valid is True
        assert result.result.data.name == "test"
        assert result.result.data.value == 42
        assert len(result.attempts) == 1
        assert result.attempts[0].status == AttemptStatus.SUCCESS
        assert result.attempts[0].will_retry is False
        generate_fn.assert_called_once_with("prompt")

    def test_success_on_retry(self):
        """Failed attempt then success on retry."""
        responses = [
            '{"name": "test"}',  # missing 'value'
            '{"name": "test", "value": 42}',
        ]
        generate_fn = MagicMock(side_effect=responses)

        result = run_with_retry("prompt", SimpleSchema, generate_fn)

        assert result.success is True
        assert len(result.attempts) == 2
        assert result.attempts[0].status == AttemptStatus.FAILED
        assert result.attempts[0].will_retry is True
        assert result.attempts[1].status == AttemptStatus.SUCCESS
        assert generate_fn.call_count == 2

    def test_retry_prompt_includes_corrective_context(self):
        """Second call gets the corrective prompt."""
        responses = [
            '{"name": "test"}',  # missing 'value'
            '{"name": "test", "value": 42}',
        ]
        generate_fn = MagicMock(side_effect=responses)

        result = run_with_retry("prompt", SimpleSchema, generate_fn)

        assert result.success is True
        # The second call should include the corrective prompt
        second_call = generate_fn.call_args_list[1][0][0]
        assert second_call.startswith("prompt")
        assert "Previous attempt failed validation:" in second_call

    def test_timing_is_recorded(self):
        """Generation and validation times are summed."""
        valid_json = '{"name": "a", "value": 1}'
        generate_fn = MagicMock(return_value=valid_json)

        result = run_with_retry("p", SimpleSchema, generate_fn)

        assert result.total_generation_ms >= 0
        assert result.total_validation_ms >= 0


class TestRunWithRetryExhaustion:
    def test_exhausted_after_max_attempts(self):
        """All retries fail → RetryResult with success=False."""
        bad_json = '{"name": "test"}'  # always missing 'value'
        generate_fn = MagicMock(return_value=bad_json)
        policy = RetryPolicy(max_attempts=3)

        result = run_with_retry(
            "prompt", SimpleSchema, generate_fn, policy=policy
        )

        assert result.success is False
        assert result.result is None
        assert len(result.attempts) == 3
        for rec in result.attempts:
            assert rec.status == AttemptStatus.FAILED
        # Last attempt should not be marked will_retry
        assert result.attempts[-1].will_retry is False
        assert generate_fn.call_count == 3

    def test_single_attempt_no_retry(self):
        """max_attempts=1 means no retries on failure."""
        bad_json = "not json"
        generate_fn = MagicMock(return_value=bad_json)
        policy = RetryPolicy(max_attempts=1)

        result = run_with_retry(
            "prompt", SimpleSchema, generate_fn, policy=policy
        )

        assert result.success is False
        assert len(result.attempts) == 1
        generate_fn.assert_called_once()

    def test_corrective_prompt_accumulates(self):
        """Each retry builds a corrective prompt from the original."""
        responses = [
            '{"name": "a"}',  # missing value
            '{"name": "a"}',  # still missing value
        ]
        generate_fn = MagicMock(side_effect=responses)
        policy = RetryPolicy(max_attempts=2)

        result = run_with_retry(
            "my prompt", SimpleSchema, generate_fn, policy=policy
        )

        assert result.success is False
        assert len(result.attempts) == 2
        # Second call should have corrective prompt
        second_call = generate_fn.call_args_list[1][0][0]
        assert second_call.startswith("my prompt")
        assert "Missing required field" in second_call


class TestRunWithRetryTerminalErrors:
    def test_non_retryable_error_stops_immediately(self):
        """A non-retryable error should not trigger a retry."""
        policy = RetryPolicy(
            max_attempts=3, retryable_failures=()
        )
        generate_fn = MagicMock(return_value="not important")

        result = run_with_retry(
            "prompt", SimpleSchema, generate_fn, policy=policy
        )

        # Should still attempt once but not retry
        assert len(result.attempts) == 1
        assert result.success is False

    def test_generate_exception_stops_immediately(self):
        """An exception from generate_fn stops the loop."""
        generate_fn = MagicMock(
            side_effect=RuntimeError("provider crashed")
        )

        result = run_with_retry("prompt", SimpleSchema, generate_fn)

        assert result.success is False
        assert len(result.attempts) == 1
        assert result.attempts[0].status == AttemptStatus.FAILED
        assert "provider crashed" in str(result.attempts[0].errors[0])


class TestRunWithRetryDefaults:
    def test_default_policy_is_used(self):
        """If no policy is provided, default (3 attempts) is used."""
        bad_json = "not json"
        generate_fn = MagicMock(return_value=bad_json)

        result = run_with_retry("prompt", SimpleSchema, generate_fn)

        assert len(result.attempts) == 3
        assert result.success is False

    def test_retry_result_dataclass(self):
        """RetryResult has correct default fields."""
        rr = RetryResult(success=False, result=None)
        assert rr.success is False
        assert rr.result is None
        assert rr.attempts == []
        assert rr.total_generation_ms == 0.0
        assert rr.total_validation_ms == 0.0
