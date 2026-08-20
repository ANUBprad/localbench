"""Tests for semantic label prompt builder.

All tests are deterministic, network-free, and use local fixtures.
"""

from __future__ import annotations

from localbench.workloads.code_retrieval.extraction import (
    ExtractedCodeUnit,
)
from localbench.workloads.code_retrieval.semantic_prompt import (
    PROMPT_TEMPLATE_VERSION,
    build_context_block,
    build_semantic_label_prompt,
    get_system_prompt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_code_unit(**overrides) -> ExtractedCodeUnit:
    """Return an ExtractedCodeUnit with sensible defaults."""
    from localbench.workloads.code_retrieval.schemas import CodeUnitContext

    defaults = {
        "repository": "repo001",
        "language": "python",
        "file_path": "payment/processor.py",
        "symbol": "PaymentProcessor.process_retry",
        "symbol_type": "method",
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
        "source_url": "",
        "is_public": True,
        "docstring": "Retry a failed transaction.",
        "source_file_lines": 12,
        "content_hash": "abc123",
        "extracted_at": "2026-08-20T00:00:00Z",
        "context": CodeUnitContext(
            class_name="PaymentProcessor",
            module_docstring="Payment processing module.",
            imports=["logging", "time"],
            parent_methods=["__init__", "validate"],
        ),
    }
    defaults.update(overrides)
    return ExtractedCodeUnit(**defaults)


# ===========================================================================
# PROMPT_TEMPLATE_VERSION
# ===========================================================================


class TestPromptTemplateVersion:
    def test_version_is_string(self) -> None:
        assert isinstance(PROMPT_TEMPLATE_VERSION, str)

    def test_version_format(self) -> None:
        parts = PROMPT_TEMPLATE_VERSION.split(".")
        assert len(parts) == 3


# ===========================================================================
# get_system_prompt
# ===========================================================================


class TestGetSystemPrompt:
    def test_returns_string(self) -> None:
        result = get_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mentions_json(self) -> None:
        assert "JSON" in get_system_prompt()

    def test_mentions_description(self) -> None:
        assert "description" in get_system_prompt()

    def test_mentions_concepts(self) -> None:
        assert "concepts" in get_system_prompt()

    def test_deterministic(self) -> None:
        assert get_system_prompt() == get_system_prompt()


# ===========================================================================
# build_context_block
# ===========================================================================


class TestBuildContextBlock:
    def test_empty_context(self) -> None:
        assert build_context_block() == ""

    def test_class_name(self) -> None:
        result = build_context_block(class_name="MyClass")
        assert "Class: MyClass" in result

    def test_module_docstring(self) -> None:
        result = build_context_block(module_docstring="Module docs.")
        assert "Module docstring: Module docs." in result

    def test_imports(self) -> None:
        result = build_context_block(imports=["os", "sys"])
        assert "Imports: os, sys" in result

    def test_parent_methods(self) -> None:
        result = build_context_block(parent_methods=["__init__", "run"])
        assert "Sibling methods: __init__, run" in result

    def test_all_context(self) -> None:
        result = build_context_block(
            class_name="Foo",
            module_docstring="Bar.",
            imports=["x"],
            parent_methods=["y"],
        )
        assert "Class: Foo" in result
        assert "Module docstring: Bar." in result
        assert "Imports: x" in result
        assert "Sibling methods: y" in result

    def test_deterministic(self) -> None:
        kw = dict(
            class_name="C",
            module_docstring="M",
            imports=["a"],
            parent_methods=["b"],
        )
        assert build_context_block(**kw) == build_context_block(**kw)


# ===========================================================================
# build_semantic_label_prompt
# ===========================================================================


class TestBuildSemanticLabelPrompt:
    def test_contains_symbol(self) -> None:
        unit = _make_code_unit()
        prompt = build_semantic_label_prompt(unit)
        assert "PaymentProcessor.process_retry" in prompt

    def test_contains_source_code(self) -> None:
        unit = _make_code_unit()
        prompt = build_semantic_label_prompt(unit)
        assert "process_retry" in prompt
        assert "def process_retry" in prompt

    def test_contains_file_path(self) -> None:
        unit = _make_code_unit()
        prompt = build_semantic_label_prompt(unit)
        assert "payment/processor.py" in prompt

    def test_contains_symbol_type(self) -> None:
        unit = _make_code_unit()
        prompt = build_semantic_label_prompt(unit)
        assert "method" in prompt

    def test_contains_class_context(self) -> None:
        unit = _make_code_unit()
        prompt = build_semantic_label_prompt(unit)
        assert "PaymentProcessor" in prompt

    def test_contains_imports(self) -> None:
        unit = _make_code_unit()
        prompt = build_semantic_label_prompt(unit)
        assert "logging" in prompt
        assert "time" in prompt

    def test_contains_parent_methods(self) -> None:
        unit = _make_code_unit()
        prompt = build_semantic_label_prompt(unit)
        assert "__init__" in prompt
        assert "validate" in prompt

    def test_no_request_for_queries(self) -> None:
        unit = _make_code_unit()
        prompt = build_semantic_label_prompt(unit)
        assert "retrieval" not in prompt.lower()
        assert "query" not in prompt.lower()

    def test_no_request_for_embeddings(self) -> None:
        unit = _make_code_unit()
        prompt = build_semantic_label_prompt(unit)
        assert "embedding" not in prompt.lower()

    def test_requests_json_output(self) -> None:
        unit = _make_code_unit()
        prompt = build_semantic_label_prompt(unit)
        assert "JSON" in prompt

    def test_deterministic(self) -> None:
        unit = _make_code_unit()
        p1 = build_semantic_label_prompt(unit)
        p2 = build_semantic_label_prompt(unit)
        assert p1 == p2

    def test_function_symbol_type(self) -> None:
        prompt = build_semantic_label_prompt(
            _make_code_unit(symbol="greet", symbol_type="function")
        )
        assert "function" in prompt

    def test_minimal_context(self) -> None:
        from localbench.workloads.code_retrieval.schemas import (
            CodeUnitContext,
        )

        unit = ExtractedCodeUnit(
            repository="r",
            language="python",
            file_path="f.py",
            symbol="func",
            symbol_type="function",
            source_code="def func():\n    return 1\n",
            context=CodeUnitContext(),
            source_url="",
            is_public=True,
            docstring="",
            source_file_lines=2,
            content_hash="h",
            extracted_at="t",
        )
        prompt = build_semantic_label_prompt(unit)
        assert "func" in prompt
