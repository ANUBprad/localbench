"""Tests for Stage A: deterministic behavior extraction from source code.

All tests are deterministic, network-free, and use local fixtures.
No LLM is involved.
"""

from __future__ import annotations

from localbench.workloads.code_retrieval.behavior_extraction import (
    extract_behavior_facts,
)
from localbench.workloads.code_retrieval.schemas import StructuredBehaviorFacts


# ===========================================================================
# extract_behavior_facts
# ===========================================================================


class TestExtractBehaviorFacts:
    def test_returns_structured_behavior_facts(self) -> None:
        src = (
            "def retry_with_backoff(tid, max_attempts=3):\n"
            '    """Retry a failed transaction with backoff."""\n'
            "    attempts = 0\n"
            "    while attempts < max_attempts:\n"
            "        try:\n"
            "            return process(tid)\n"
            "        except Exception:\n"
            "            attempts += 1\n"
            '    raise RuntimeError("Failed")\n'
        )
        facts = extract_behavior_facts(src)
        assert isinstance(facts, StructuredBehaviorFacts)
        assert facts.primary_purpose
        assert facts.input_summary
        assert facts.output_summary

    def test_extracts_parameters(self) -> None:
        src = (
            "def fetch(url, timeout=30, retries=3):\n"
            "    pass\n"
        )
        facts = extract_behavior_facts(src)
        assert "3 parameters" in facts.input_summary

    def test_extracts_side_effects(self) -> None:
        src = (
            "def update_state(self):\n"
            "    self.counter += 1\n"
            "    self.log.append('updated')\n"
        )
        facts = extract_behavior_facts(src)
        assert any("instance" in se for se in facts.side_effects)

    def test_extracts_raises(self) -> None:
        src = (
            "def validate(x):\n"
            "    if x < 0:\n"
            '        raise ValueError("negative")\n'
        )
        facts = extract_behavior_facts(src)
        assert "ValueError" in facts.raises

    def test_extracts_error_handling(self) -> None:
        src = (
            "def safe_divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except ZeroDivisionError:\n"
            '        return 0\n'
        )
        facts = extract_behavior_facts(src)
        assert "ZeroDivisionError" in facts.error_handling

    def test_extracts_control_flow_loop(self) -> None:
        src = (
            "def count(items):\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        total += 1\n"
            "    return total\n"
        )
        facts = extract_behavior_facts(src)
        assert "for loop" in facts.control_flow

    def test_extracts_key_operations(self) -> None:
        src = (
            "def process(data):\n"
            "    result = transform(data)\n"
            "    return result\n"
        )
        facts = extract_behavior_facts(src)
        assert any("function call" in op for op in facts.key_operations)

    def test_invalid_syntax_returns_safe_defaults(self) -> None:
        facts = extract_behavior_facts("not valid python!!!")
        assert facts.primary_purpose
        assert facts.input_summary == "unknown"

    def test_no_side_effects(self) -> None:
        src = (
            "def pure_func(x):\n"
            "    return x * 2\n"
        )
        facts = extract_behavior_facts(src)
        assert facts.side_effects == []

    def test_no_raises(self) -> None:
        src = (
            "def safe_func(x):\n"
            "    return x + 1\n"
        )
        facts = extract_behavior_facts(src)
        assert facts.raises == []


class TestStructuredBehaviorFactsSchema:
    def test_fields_are_strings(self) -> None:
        facts = StructuredBehaviorFacts(
            primary_purpose="does something",
            input_summary="takes x",
            output_summary="returns y",
        )
        assert isinstance(facts.primary_purpose, str)
        assert isinstance(facts.input_summary, str)
        assert isinstance(facts.output_summary, str)

    def test_optional_fields_default_empty(self) -> None:
        facts = StructuredBehaviorFacts(
            primary_purpose="does something",
            input_summary="takes x",
            output_summary="returns y",
        )
        assert facts.side_effects == []
        assert facts.key_operations == []
        assert facts.error_handling == ""
        assert facts.control_flow == ""
        assert facts.raises == []

    def test_frozen_dataclass(self) -> None:
        facts = StructuredBehaviorFacts(
            primary_purpose="does something",
            input_summary="takes x",
            output_summary="returns y",
        )
        import pytest

        with pytest.raises(AttributeError):
            facts.primary_purpose = "changed"  # type: ignore[misc]


# ===========================================================================
# Provenance: no implementation identifiers in extracted facts
# ===========================================================================


import re

# Pattern matching Python identifiers (snake_case, camelCase, PascalCase)
_IDENTIFIER_PATTERN = re.compile(
    r"\b(self|cls|[A-Z][a-zA-Z0-9]+|"
    r"[a-z][a-z_]+_[a-z][a-z_]+|"
    r"_[a-z][a-z_]+)\b"
)


def _contains_identifier(text: str) -> bool:
    """Check if text contains a Python identifier (name-like string)."""
    return bool(_IDENTIFIER_PATTERN.search(text))


class TestNoIdentifiersInExtractedFacts:
    """Ensure extracted facts are free of implementation identifiers."""

    def test_primary_purpose_no_function_name(self) -> None:
        src = (
            "def process_retry(self, tid, max_attempts=3):\n"
            '    """Retry a failed transaction."""\n'
            "    attempts = 0\n"
            "    while attempts < max_attempts:\n"
            "        try:\n"
            "            return self._do_process(tid)\n"
            "        except Exception:\n"
            "            attempts += 1\n"
            '    raise RuntimeError("Failed")\n'
        )
        facts = extract_behavior_facts(src)
        assert "process_retry" not in facts.primary_purpose

    def test_input_summary_no_param_names(self) -> None:
        src = (
            "def fetch(url, timeout=30, retries=3):\n"
            "    pass\n"
        )
        facts = extract_behavior_facts(src)
        assert "url" not in facts.input_summary
        assert "timeout" not in facts.input_summary
        assert "retries" not in facts.input_summary

    def test_side_effects_no_attribute_names(self) -> None:
        src = (
            "def update_state(self):\n"
            "    self.counter += 1\n"
            "    self.log.append('updated')\n"
        )
        facts = extract_behavior_facts(src)
        for se in facts.side_effects:
            assert "self.counter" not in se
            assert "self.log" not in se

    def test_key_operations_no_call_names(self) -> None:
        src = (
            "def process(data):\n"
            "    result = transform(data)\n"
            "    return result\n"
        )
        facts = extract_behavior_facts(src)
        for op in facts.key_operations:
            assert "transform" not in op

    def test_raises_still_contains_exception_names(self) -> None:
        """Exception class names are domain identifiers, not implementation identifiers."""
        src = (
            "def validate(x):\n"
            "    if x < 0:\n"
            '        raise ValueError("negative")\n'
        )
        facts = extract_behavior_facts(src)
        assert "ValueError" in facts.raises

    def test_error_handling_still_contains_exception_names(self) -> None:
        """Exception class names in error handling are domain identifiers."""
        src = (
            "def safe_divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except ZeroDivisionError:\n"
            '        return 0\n'
        )
        facts = extract_behavior_facts(src)
        assert "ZeroDivisionError" in facts.error_handling
