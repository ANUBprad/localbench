"""Tests for retrieval query prompt builder.

All tests are deterministic, network-free, and use local fixtures.
"""

from __future__ import annotations

from localbench.workloads.code_retrieval.query_prompt import (
    QUERY_PROMPT_TEMPLATE_VERSION,
    build_query_generation_prompt,
    get_query_system_prompt,
)
from localbench.workloads.code_retrieval.schemas import QueryGenerationInput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_input(**overrides) -> QueryGenerationInput:
    """Return a QueryGenerationInput with sensible defaults."""
    defaults = {
        "source_code": (
            "def process_retry(self, tid, max_attempts=3):\n"
            '    """Retry a failed transaction."""\n'
            "    attempts = 0\n"
            "    while attempts < max_attempts:\n"
            "        try:\n"
            "            return self._do_process(tid)\n"
            "        except Exception:\n"
            "            attempts += 1\n"
            '    raise RuntimeError("Failed")\n'
        ),
        "docstring": "Retry a failed transaction.",
        "symbol_type": "method",
        "class_name": "PaymentProcessor",
        "module_docstring": "Payment processing module.",
        "imports": ["logging", "time"],
        "parent_methods": ["__init__", "validate"],
    }
    defaults.update(overrides)
    return QueryGenerationInput(**defaults)


# ===========================================================================
# QUERY_PROMPT_TEMPLATE_VERSION
# ===========================================================================


class TestQueryPromptTemplateVersion:
    def test_version_is_string(self) -> None:
        assert isinstance(QUERY_PROMPT_TEMPLATE_VERSION, str)

    def test_version_format(self) -> None:
        parts = QUERY_PROMPT_TEMPLATE_VERSION.split(".")
        assert len(parts) == 3

    def test_version_value(self) -> None:
        assert QUERY_PROMPT_TEMPLATE_VERSION == "1.0.0"


# ===========================================================================
# get_query_system_prompt
# ===========================================================================


class TestGetQuerySystemPrompt:
    def test_returns_string(self) -> None:
        result = get_query_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mentions_json(self) -> None:
        assert "JSON" in get_query_system_prompt()

    def test_mentions_query(self) -> None:
        assert "query" in get_query_system_prompt().lower()

    def test_mentions_no_identifiers(self) -> None:
        prompt = get_query_system_prompt()
        assert "Do NOT include" in prompt
        assert "file paths" in prompt
        assert "function names" in prompt

    def test_deterministic(self) -> None:
        assert get_query_system_prompt() == get_query_system_prompt()


# ===========================================================================
# build_query_generation_prompt
# ===========================================================================


class TestBuildQueryGenerationPrompt:
    def test_contains_source_code(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "process_retry" in prompt
        assert "def process_retry" in prompt

    def test_contains_docstring(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "Retry a failed transaction." in prompt

    def test_contains_symbol_type(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "method" in prompt

    def test_contains_class_context(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "Class: PaymentProcessor" in prompt

    def test_contains_module_docstring(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "Module docstring: Payment processing module." in prompt

    def test_contains_imports(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "Imports: logging, time" in prompt

    def test_contains_parent_methods(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "Sibling methods: __init__, validate" in prompt

    def test_no_repository(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "repo001" not in prompt
        assert "repository" not in prompt.lower()

    def test_no_file_path(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "payment/processor.py" not in prompt
        assert "file_path" not in prompt.lower()

    def test_no_symbol_path(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "PaymentProcessor.process_retry" not in prompt

    def test_no_source_url(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "source_url" not in prompt.lower()
        assert "github.com" not in prompt

    def test_no_semantic_label_content(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "concepts" not in prompt.lower()
        assert "side_effects" not in prompt.lower()
        assert "label_version" not in prompt.lower()

    def test_requests_json_output(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "JSON" in prompt

    def test_requests_query_fields(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "query" in prompt
        assert "query_style" in prompt
        assert "query_intent" in prompt

    def test_function_symbol_type(self) -> None:
        prompt = build_query_generation_prompt(
            _make_input(symbol_type="function")
        )
        assert "function" in prompt

    def test_no_docstring(self) -> None:
        prompt = build_query_generation_prompt(_make_input(docstring=""))
        assert "Docstring:" not in prompt

    def test_minimal_context(self) -> None:
        unit = QueryGenerationInput(
            source_code="def func():\n    return 1\n    pass\n",
            symbol_type="function",
        )
        prompt = build_query_generation_prompt(unit)
        assert "func" in prompt
        assert "Class:" not in prompt

    def test_deterministic(self) -> None:
        unit = _make_input()
        p1 = build_query_generation_prompt(unit)
        p2 = build_query_generation_prompt(unit)
        assert p1 == p2

    def test_no_retrieval_mention(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "retrieval" not in prompt.lower().split("retrieval query")[0]

    def test_no_embedding_mention(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "embedding" not in prompt.lower()
