"""Tests for retrieval query prompt builder.

All tests are deterministic, network-free, and use local fixtures.
"""

from __future__ import annotations

from localbench.workloads.code_retrieval.query_prompt import (
    QUERY_PROMPT_TEMPLATE_VERSION,
    build_query_generation_prompt,
    build_query_generation_prompt_v2,
    get_query_system_prompt,
)
from localbench.workloads.code_retrieval.schemas import (
    QueryGenerationInput,
    StructuredBehaviorFacts,
)

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
        assert QUERY_PROMPT_TEMPLATE_VERSION == "3.2.0"


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

    def test_mentions_critical_anti_leakage(self) -> None:
        prompt = get_query_system_prompt()
        assert "CRITICAL" in prompt
        assert "ANTI-LEAKAGE" in prompt

    def test_mentions_never_use_class_names(self) -> None:
        prompt = get_query_system_prompt()
        assert "NEVER use class names" in prompt

    def test_mentions_describe_behavior(self) -> None:
        prompt = get_query_system_prompt()
        assert "Describe WHAT the code does" in prompt

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

    def test_user_template_has_reminder(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "REMINDER: Do NOT use any class names" in prompt

    def test_user_template_reminds_behavior(self) -> None:
        prompt = build_query_generation_prompt(_make_input())
        assert "Describe the behavior in plain language only" in prompt


# ===========================================================================
# build_query_generation_prompt_v2 — docstring provenance tests
# ===========================================================================


def _make_facts(**overrides) -> StructuredBehaviorFacts:
    defaults = {
        "primary_purpose": "retries an operation with error handling",
        "input_summary": "takes 2 parameters, at least one with a default value",
        "output_summary": "returns a computed value",
        "side_effects": ["modifies instance state"],
        "key_operations": ["performs 1 method call(s)"],
        "error_handling": "catches Exception",
        "control_flow": "while loop",
        "raises": ["RuntimeError"],
    }
    defaults.update(overrides)
    return StructuredBehaviorFacts(**defaults)


class TestBuildQueryGenerationPromptV2:
    def test_docstring_absent_from_v2_prompt(self) -> None:
        facts = _make_facts()
        prompt = build_query_generation_prompt_v2(
            facts,
            source_code='def foo():\n    """THIS_UNIQUE_SENTENCE."""\n    pass',
            symbol_type="function",
        )
        assert "THIS_UNIQUE_SENTENCE" not in prompt

    def test_source_code_absent_from_v2_prompt(self) -> None:
        facts = _make_facts()
        prompt = build_query_generation_prompt_v2(
            facts,
            source_code="def secret_function(self, token):\n    pass",
            symbol_type="function",
        )
        assert "secret_function" not in prompt
        assert "def secret" not in prompt

    def test_module_docstring_absent_from_v2_prompt(self) -> None:
        facts = _make_facts()
        prompt = build_query_generation_prompt_v2(
            facts,
            source_code="def foo():\n    pass",
            symbol_type="function",
            module_docstring="SECRET_MODULE_DESCRIPTION",
        )
        assert "SECRET_MODULE_DESCRIPTION" not in prompt

    def test_v2_prompt_contains_behavioral_facts(self) -> None:
        facts = _make_facts()
        prompt = build_query_generation_prompt_v2(
            facts,
            source_code="def foo():\n    pass",
            symbol_type="function",
        )
        assert "retries an operation" in prompt
        assert "takes 2 parameters" in prompt
        assert "while loop" in prompt

    def test_v2_prompt_version_is_3(self) -> None:
        from localbench.workloads.code_retrieval.query_prompt import (
            QUERY_PROMPT_TEMPLATE_VERSION,
        )

        assert QUERY_PROMPT_TEMPLATE_VERSION == "3.2.0"

    def test_v2_context_class_name_preserved(self) -> None:
        facts = _make_facts()
        prompt = build_query_generation_prompt_v2(
            facts,
            source_code="def foo():\n    pass",
            symbol_type="method",
            class_name="MyClass",
        )
        assert "Class: MyClass" in prompt

    def test_v2_context_imports_preserved(self) -> None:
        facts = _make_facts()
        prompt = build_query_generation_prompt_v2(
            facts,
            source_code="def foo():\n    pass",
            symbol_type="function",
            imports=["logging", "os"],
        )
        assert "Imports: logging, os" in prompt

    def test_v2_context_parent_methods_preserved(self) -> None:
        facts = _make_facts()
        prompt = build_query_generation_prompt_v2(
            facts,
            source_code="def foo():\n    pass",
            symbol_type="method",
            parent_methods=["__init__", "close"],
        )
        assert "Sibling methods" in prompt

    def test_v2_deterministic(self) -> None:
        facts = _make_facts()
        p1 = build_query_generation_prompt_v2(
            facts, source_code="def foo():\n    pass"
        )
        p2 = build_query_generation_prompt_v2(
            facts, source_code="def foo():\n    pass"
        )
        assert p1 == p2

    def test_v2_no_docstring_keyword(self) -> None:
        facts = _make_facts()
        prompt = build_query_generation_prompt_v2(
            facts,
            source_code='def foo():\n    """Some docstring."""\n    pass',
            symbol_type="function",
        )
        assert "Some docstring" not in prompt

    def test_v2_no_identifiers_in_template(self) -> None:
        facts = _make_facts()
        prompt = build_query_generation_prompt_v2(
            facts,
            source_code="def process_data(self, token, timeout=30):\n    pass",
            symbol_type="method",
            class_name="DataProcessor",
        )
        assert "process_data" not in prompt
        assert "token" not in prompt
        assert "timeout" not in prompt


class TestQueryPromptDomainBlocks:
    """The v3.1 prompt surfaces identifier-free domain/effect facts."""

    def test_v2_domain_concepts_rendered(self) -> None:
        facts = _make_facts(
            domain_concepts=["terminal/console rendering", "progress display"],
            observable_effects=["renders a table of items to the console"],
        )
        prompt = build_query_generation_prompt_v2(
            facts, source_code="def foo():\n    pass", symbol_type="function"
        )
        assert "Domain concepts: terminal/console rendering, progress display" in prompt
        assert (
            "Observable effects: renders a table of items to the console" in prompt
        )

    def test_v2_empty_domain_blocks_omitted(self) -> None:
        facts = _make_facts()
        prompt = build_query_generation_prompt_v2(
            facts, source_code="def foo():\n    pass", symbol_type="function"
        )
        assert "Domain concepts:" not in prompt
        assert "Observable effects:" not in prompt

    def test_v2_domain_facts_stay_identifier_free_in_prompt(self) -> None:
        facts = _make_facts(
            domain_concepts=["terminal cursor positioning"],
            observable_effects=["moves the terminal cursor"],
        )
        prompt = build_query_generation_prompt_v2(
            facts,
            source_code=(
                "def SetConsoleCursorPosition(handle, pos):\n"
                "    ctypes.windll.kernel32._SetConsoleCursorPosition(handle, pos)\n"
            ),
            symbol_type="function",
        )
        assert "SetConsoleCursorPosition" not in prompt
        assert "handle" not in prompt

    def test_system_prompt_query_style_enum_guidance(self) -> None:
        sys_prompt = get_query_system_prompt()
        assert '"natural"' in sys_prompt
        assert '"technical"' in sys_prompt
        assert '"verbose"' in sys_prompt
        assert '"concise"' in sys_prompt

    def test_system_prompt_forbids_non_enum_style_labels(self) -> None:
        sys_prompt = get_query_system_prompt()
        assert "Descriptive" in sys_prompt or "descriptive" in sys_prompt
        assert "Expository" in sys_prompt or "expository" in sys_prompt


class TestQueryStyleEnum:
    """query_style accepts exactly the four documented literals."""

    def test_all_four_literals_valid(self) -> None:
        from localbench.workloads.code_retrieval.schemas import CandidateQuery

        for style in ("natural", "technical", "verbose", "concise"):
            q = CandidateQuery(
                query="is there a way to sort a list",
                query_style=style,  # type: ignore[arg-type]
                query_intent="find_implementation",
            )
            assert q.query_style == style

    def test_non_enum_style_rejected(self) -> None:
        import pydantic
        import pytest

        from localbench.workloads.code_retrieval.schemas import CandidateQuery

        with pytest.raises(pydantic.ValidationError):
            CandidateQuery(
                query="sorting helpers",
                query_style="Descriptive",  # type: ignore[arg-type]
                query_intent="find_implementation",
            )
