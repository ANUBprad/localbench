"""Tests for retrieval query generation pipeline.

All tests use a fake LocalModel — no Ollama or network dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from localbench.runtime.generation.policy import RetryPolicy
from localbench.runtime.model import GenerationRequest, ModelInfo
from localbench.workloads.code_retrieval.extraction import ExtractedCodeUnit
from localbench.workloads.code_retrieval.query_generator import (
    LeakageCheckResult,
    ProvenanceCheckResult,
    QueryGenerationResult,
    QueryGenerator,
    _extract_identifier_parts,
    _extract_param_names,
    _extract_sphinx_ref_names,
    _extract_docstring_content,
    _make_identifier_pattern,
    check_meta_query,
    check_query_leakage,
    check_query_provenance,
    generate_query,
)
from localbench.workloads.code_retrieval.schemas import (
    CandidateQuery,
    CodeUnitContext,
)

# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------

_VALID_QUERY_JSON = json.dumps(
    {
        "query": (
            "Find the Python function that retries failed payments "
            "with exponential backoff."
        ),
        "query_style": "natural",
        "query_intent": "find_error_handling",
    }
)


@dataclass
class FakeModel:
    """Fake LocalModel for testing."""

    _name: str = "fake-model"
    _responses: list[str] = field(default_factory=lambda: [_VALID_QUERY_JSON])
    _call_index: int = 0
    _calls: list[GenerationRequest] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> ModelInfo:
        return ModelInfo(name=self._name)

    def generate(self, request: GenerationRequest) -> object:
        self._calls.append(request)
        if self._call_index < len(self._responses):
            text = self._responses[self._call_index]
            self._call_index += 1
        else:
            text = self._responses[-1]

        return type(
            "Result",
            (),
            {
                "model": self._name,
                "text": text,
                "duration_ms": 100.0,
                "total_duration_ms": 100.0,
                "eval_count": None,
                "eval_duration_ms": None,
                "done": True,
                "error": None,
            },
        )()


def _make_code_unit(**overrides) -> ExtractedCodeUnit:
    """Create a minimal ExtractedCodeUnit for testing."""
    defaults = {
        "repository": "repo001",
        "language": "python",
        "file_path": "payment/processor.py",
        "symbol": "PaymentProcessor.process_retry",
        "symbol_type": "method",
        "source_code": (
            "def process_retry(self, tid, max_attempts=3):\n"
            '    """Retry a failed transaction with backoff."""\n'
            "    attempts = 0\n"
            "    while attempts < max_attempts:\n"
            "        try:\n"
            "            return self._do_process(tid)\n"
            "        except Exception:\n"
            "            attempts += 1\n"
            '    raise RuntimeError("Failed")\n'
        ),
        "source_url": "https://github.com/example/repo/blob/main/payment/processor.py#L42",
        "is_public": True,
        "docstring": "Retry a failed transaction with backoff.",
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
# LeakageCheckResult
# ===========================================================================


class TestLeakageCheckResult:
    def test_passed_fields(self) -> None:
        result = LeakageCheckResult(passed=True)
        assert result.passed is True
        assert result.violations == []

    def test_failed_fields(self) -> None:
        result = LeakageCheckResult(
            passed=False, violations=["leaked path"]
        )
        assert result.passed is False
        assert len(result.violations) == 1


# ===========================================================================
# check_query_leakage
# ===========================================================================


class TestCheckQueryLeakage:
    def test_clean_query_passes(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "Find the Python function that retries failed payments.",
            unit,
        )
        assert result.passed is True

    def test_file_extension_detected(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "Find the retry logic in processor.py", unit
        )
        assert result.passed is False
        assert any(".py" in v for v in result.violations)

    def test_github_url_detected(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "Check github.com for the retry function", unit
        )
        assert result.passed is False

    def test_repository_detected(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "Find the function in repo001", unit
        )
        assert result.passed is False
        assert any("repo001" in v for v in result.violations)

    def test_file_path_detected(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "Find the function at payment/processor.py", unit
        )
        assert result.passed is False
        assert any("payment/processor.py" in v for v in result.violations)

    def test_symbol_detected(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "Find PaymentProcessor.process_retry", unit
        )
        assert result.passed is False
        assert any("PaymentProcessor.process_retry" in v for v in result.violations)

    def test_java_extension_detected(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "Find the handler in MyClass.java", unit
        )
        assert result.passed is False

    def test_gitlab_detected(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "Check gitlab.com for the function", unit
        )
        assert result.passed is False

    # ----------------------------------------------------------------
    # Regression: individual identifier-part leakage (Phase C6)
    # ----------------------------------------------------------------

    def test_class_name_detected_in_query(self) -> None:
        unit = _make_code_unit(
            symbol="Color.downgrade",
            context=CodeUnitContext(class_name="Color"),
        )
        result = check_query_leakage(
            "How to downgrade a color system in the Color class?", unit
        )
        assert result.passed is False
        assert any("Color" in v for v in result.violations)

    def test_method_name_detected_in_query(self) -> None:
        unit = _make_code_unit(
            symbol="PromptBase.render_default",
            context=CodeUnitContext(class_name="PromptBase"),
        )
        result = check_query_leakage(
            "How does the render_default method work?", unit
        )
        assert result.passed is False
        assert any("render_default" in v for v in result.violations)

    def test_class_name_from_symbol_detected(self) -> None:
        unit = _make_code_unit(
            symbol="ConsoleOptions.ascii_only",
            context=CodeUnitContext(),
        )
        result = check_query_leakage(
            "Is the ascii_only property of the ConsoleOptions class set to True?",
            unit,
        )
        assert result.passed is False

    def test_standalone_function_name_detected(self) -> None:
        unit = _make_code_unit(
            symbol="process_retry",
            context=CodeUnitContext(),
            symbol_type="function",
        )
        result = check_query_leakage(
            "Find the process_retry function that handles retries.",
            unit,
        )
        assert result.passed is False

    def test_dunder_method_detected(self) -> None:
        unit = _make_code_unit(
            symbol="Console.__enter__",
            context=CodeUnitContext(class_name="Console"),
        )
        result = check_query_leakage(
            "Implement a context manager method in the Console class "
            "to enter a buffer context.",
            unit,
        )
        assert result.passed is False

    def test_parameter_name_detected_in_query(self) -> None:
        unit = _make_code_unit(
            symbol="TimeRemainingColumn.__init__",
            context=CodeUnitContext(class_name="TimeRemainingColumn"),
            source_code=(
                "def __init__(self, compact, elapsed_when_finished,\n"
                "              table_column=None):\n"
                "    pass\n"
            ),
        )
        result = check_query_leakage(
            "Create a method with parameters compact and elapsed_when_finished.",
            unit,
        )
        assert result.passed is False
        assert any("compact" in v for v in result.violations)

    def test_clean_query_with_similar_words_passes(self) -> None:
        unit = _make_code_unit(
            symbol="SomeClass.some_method",
            context=CodeUnitContext(class_name="SomeClass"),
        )
        result = check_query_leakage(
            "Find the function that processes data efficiently.",
            unit,
        )
        assert result.passed is True

    def test_short_identifier_not_flagged(self) -> None:
        unit = _make_code_unit(
            symbol="Foo.bar",
            context=CodeUnitContext(),
            source_code="def bar(self, ok, id):\n    pass\n",
        )
        result = check_query_leakage(
            "Is the operation ok to proceed?", unit
        )
        assert result.passed is True

    def test_context_class_name_checked(self) -> None:
        unit = _make_code_unit(
            symbol="load",
            context=CodeUnitContext(class_name="LocalPath"),
            symbol_type="function",
        )
        result = check_query_leakage(
            "How can I unpickle an object using the LocalPath class?", unit
        )
        assert result.passed is False
        assert any("LocalPath" in v for v in result.violations)

    def test_docstring_class_name_detected(self) -> None:
        # Regression: Item #3 — query reproduces docstring text containing
        # a Sphinx :class: reference (pytest.Config).
        unit = _make_code_unit(
            symbol="pytest_cmdline_parse",
            context=CodeUnitContext(),
            symbol_type="function",
            docstring=(
                "Return an initialized :class:`~pytest.Config`, "
                "parsing the specified args.\n\n"
                "Stops at first non-None result."
            ),
            source_code=(
                "def pytest_cmdline_parse(pluginmanager, args):\n"
                '    """Return an initialized :class:`~pytest.Config`.\n'
                '    Stops at first non-None result."""\n'
            ),
        )
        result = check_query_leakage(
            "Return an initialized pytest.Config, parsing the "
            "specified args. Stops at first non-None result.",
            unit,
        )
        assert result.passed is False
        assert any("Config" in v for v in result.violations)

    def test_docstring_class_name_no_false_positive(self) -> None:
        # A clean query that does NOT reproduce docstring identifiers
        # should pass even when docstring contains :class: references.
        unit = _make_code_unit(
            symbol="pytest_cmdline_parse",
            context=CodeUnitContext(),
            symbol_type="function",
            docstring=(
                "Return an initialized :class:`~pytest.Config`, "
                "parsing the specified args."
            ),
            source_code=(
                "def pytest_cmdline_parse(pluginmanager, args):\n"
                "    pass\n"
            ),
        )
        result = check_query_leakage(
            "Parse command line arguments and return a config object.",
            unit,
        )
        assert result.passed is True

    # ----------------------------------------------------------------
    # Regression: meta-task query detection (Phase C8 remediation)
    # ----------------------------------------------------------------

    def test_meta_task_query_i_need_a_query_detected(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "I need a technical query to retrieve all issues from a paginated API",
            unit,
        )
        assert result.passed is False
        assert any("meta-task" in v for v in result.violations)

    def test_meta_task_query_generate_retrieval_detected(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "Generate a retrieval query for the payment processor",
            unit,
        )
        assert result.passed is False
        assert any("meta-task" in v for v in result.violations)

    def test_meta_task_query_pytest_collect_only_detected(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "pytest --collect-only",
            unit,
        )
        assert result.passed is False

    def test_clean_query_not_flagged_as_meta_task(self) -> None:
        unit = _make_code_unit()
        result = check_query_leakage(
            "How can I find the closest matching color in a palette?",
            unit,
        )
        assert result.passed is True

    # ----------------------------------------------------------------
    # Regression: docstring reproduction detection (Phase C8 remediation)
    # ----------------------------------------------------------------

    def test_docstring_reproduction_detected(self) -> None:
        unit = _make_code_unit(
            symbol="test_scope_module_uses_session",
            context=CodeUnitContext(class_name="TestFixtureMarker"),
            docstring=(
                "pytester makes a pyfile with a pytest fixture and tests "
                "that use the fixture, then pytester runs the tests and "
                "asserts that all tests passed."
            ),
            source_code=(
                "def test_scope_module_uses_session(self, pytester):\n"
                "    pass\n"
            ),
        )
        result = check_query_leakage(
            "pytester makes a pyfile with a pytest fixture and tests that "
            "use the fixture, then pytester runs the tests and asserts that "
            "all tests passed.",
            unit,
        )
        assert result.passed is False
        assert any("docstring" in v for v in result.violations)

    def test_docstring_reproduction_partial_detected(self) -> None:
        unit = _make_code_unit(
            symbol="test_funcarg",
            context=CodeUnitContext(class_name="TestRequestScopeAccess"),
            docstring=(
                "Overriding a parametrized fixture with a new parametrized "
                "fixture and requesting the overwritten fixture as a parameter "
                "yields the same value as request.param."
            ),
            source_code=(
                "def test_funcarg(self, pytester, scope, ok, error):\n"
                "    pass\n"
            ),
        )
        result = check_query_leakage(
            "Overriding a parametrized fixture with a new parametrized fixture "
            "and requesting the overwritten fixture as a parameter yields the "
            "same value as request.param.",
            unit,
        )
        assert result.passed is False
        assert any("docstring" in v for v in result.violations)

    def test_clean_query_not_flagged_as_docstring_reproduction(self) -> None:
        unit = _make_code_unit(
            symbol="test_something",
            context=CodeUnitContext(),
            docstring="A function that processes data efficiently.",
            source_code="def test_something():\n    pass\n",
        )
        result = check_query_leakage(
            "How can I process data efficiently in Python?",
            unit,
        )
        assert result.passed is True

    # ----------------------------------------------------------------
    # Regression: Sphinx :attr:/:meth:/:func: detection (Phase C8)
    # ----------------------------------------------------------------

    def test_sphinx_attr_reference_detected(self) -> None:
        unit = _make_code_unit(
            symbol="syspathinsert",
            context=CodeUnitContext(class_name="Pytester"),
            docstring=(
                "Prepend a directory to sys.path, defaults to "
                ":attr:`path`. This is undone automatically when this "
                "object dies at the end of each test."
            ),
            source_code=(
                "def syspathinsert(self, path=None):\n"
                "    pass\n"
            ),
        )
        result = check_query_leakage(
            "Prepend a directory to sys.path, defaults to :attr:`path`. "
            "This is undone automatically when this object dies at the end "
            "of each test.",
            unit,
        )
        assert result.passed is False
        # Detected either via Sphinx directive match or docstring reproduction
        assert any(
            "Sphinx directive" in v or "reference name" in v or "docstring" in v
            for v in result.violations
        )

    def test_clean_query_no_sphinx_false_positive(self) -> None:
        unit = _make_code_unit(
            symbol="some_method",
            context=CodeUnitContext(),
            docstring="Do something with :attr:`some_attr`.",
            source_code="def some_method():\n    pass\n",
        )
        result = check_query_leakage(
            "How can I do something with an attribute?",
            unit,
        )
        assert result.passed is True


# ===========================================================================
# _extract_identifier_parts
# ===========================================================================


class TestExtractIdentifierParts:
    def test_splits_symbol_on_dot(self) -> None:
        unit = _make_code_unit(
            symbol="PaymentProcessor.process_retry",
            context=CodeUnitContext(),
        )
        parts = _extract_identifier_parts(unit)
        assert "PaymentProcessor" in parts
        assert "process_retry" in parts

    def test_includes_context_class_name(self) -> None:
        unit = _make_code_unit(
            symbol="render_default",
            context=CodeUnitContext(class_name="PromptBase"),
            symbol_type="function",
        )
        parts = _extract_identifier_parts(unit)
        assert "PromptBase" in parts
        assert "render_default" in parts

    def test_deduplicates_class_from_symbol_and_context(self) -> None:
        unit = _make_code_unit(
            symbol="Color.downgrade",
            context=CodeUnitContext(class_name="Color"),
        )
        parts = _extract_identifier_parts(unit)
        assert parts.count("Color") == 1

    def test_excludes_short_parts(self) -> None:
        unit = _make_code_unit(
            symbol="AB.cd",
            context=CodeUnitContext(),
        )
        parts = _extract_identifier_parts(unit)
        assert "AB" not in parts
        assert "cd" not in parts

    def test_empty_symbol(self) -> None:
        unit = _make_code_unit(
            symbol="",
            context=CodeUnitContext(),
        )
        parts = _extract_identifier_parts(unit)
        assert parts == []


# ===========================================================================
# _extract_param_names
# ===========================================================================


class TestExtractParamNames:
    def test_extracts_parameters(self) -> None:
        src = "def foo(self, bar, baz_qux):\n    pass\n"
        params = _extract_param_names(src)
        assert "bar" not in params  # too short (< 4)
        assert "baz_qux" in params

    def test_ignores_self_cls(self) -> None:
        src = "def method(self, cls, max_retries):\n    pass\n"
        params = _extract_param_names(src)
        assert "self" not in params
        assert "cls" not in params
        assert "max_retries" in params  # valid param, kept

    def test_handles_default_values(self) -> None:
        src = "def foo(self, max_retries=3, timeout=None):\n    pass\n"
        params = _extract_param_names(src)
        assert "max_retries" in params
        assert "timeout" in params

    def test_handles_star_args(self) -> None:
        src = "def foo(self, *args, **kwargs):\n    pass\n"
        params = _extract_param_names(src)
        assert "args" not in params
        assert "kwargs" not in params

    def test_no_match_returns_empty(self) -> None:
        params = _extract_param_names("not a function def")
        assert params == []


# ===========================================================================
# _make_identifier_pattern
# ===========================================================================


class TestMakeIdentifierPattern:
    def test_matches_whole_word(self) -> None:
        pat = _make_identifier_pattern("Color")
        assert pat.search("the Color class")
        assert pat.search("the Color.")
        assert pat.search("the Color's")

    def test_no_partial_match(self) -> None:
        pat = _make_identifier_pattern("Color")
        assert pat.search("Colorful") is None
        assert pat.search("disColor") is None

    def test_case_sensitive(self) -> None:
        pat = _make_identifier_pattern("Color")
        assert pat.search("color") is None
        assert pat.search("COLOR") is None


# ===========================================================================
# QueryGenerationResult
# ===========================================================================


class TestQueryGenerationResult:
    def test_success_fields(self) -> None:
        candidate = CandidateQuery(
            query="Find the retry function",
            query_style="natural",
            query_intent="find_implementation",
        )
        result = QueryGenerationResult(success=True, candidate=candidate)
        assert result.success is True
        assert result.candidate is candidate

    def test_failure_fields(self) -> None:
        result = QueryGenerationResult(success=False)
        assert result.success is False
        assert result.candidate is None


# ===========================================================================
# QueryGenerator
# ===========================================================================


class TestQueryGenerator:
    def test_successful_generation(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is True
        assert result.candidate is not None
        assert "retr" in result.candidate.query.lower()
        assert result.candidate.query_style == "natural"
        assert result.candidate.query_intent == "find_error_handling"
        assert len(result.attempts) == 1
        assert result.model_name == "fake-model"

    def test_model_called_with_prompt(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        gen.generate(unit)

        assert len(model._calls) == 1
        assert "Behavioral Facts" in model._calls[0].prompt
        assert "process_retry" not in model._calls[0].prompt

    def test_model_uses_correct_config(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        gen.generate(unit)

        req = model._calls[0]
        assert req.model == "fake-model"
        assert req.temperature == 0.7
        assert req.max_tokens == 128

    def test_prompt_excludes_repository(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        gen.generate(unit)

        assert "repo001" not in model._calls[0].prompt

    def test_prompt_excludes_file_path(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        gen.generate(unit)

        assert "payment/processor.py" not in model._calls[0].prompt

    def test_prompt_excludes_symbol_path(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        gen.generate(unit)

        assert "PaymentProcessor.process_retry" not in model._calls[0].prompt

    def test_retry_on_malformed_json(self) -> None:
        bad_response = "This is not JSON at all"
        good_response = _VALID_QUERY_JSON
        model = FakeModel(_responses=[bad_response, good_response])
        policy = RetryPolicy(max_attempts=3)
        gen = QueryGenerator(model, policy=policy)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is True
        assert len(result.attempts) == 2

    def test_retry_on_validation_error(self) -> None:
        bad_schema = json.dumps(
            {
                "query": "x",
            }
        )
        model = FakeModel(_responses=[bad_schema, _VALID_QUERY_JSON])
        policy = RetryPolicy(max_attempts=3)
        gen = QueryGenerator(model, policy=policy)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is True
        assert len(result.attempts) == 2

    def test_retry_exhaustion(self) -> None:
        bad = "not json"
        model = FakeModel(_responses=[bad, bad, bad])
        policy = RetryPolicy(max_attempts=3)
        gen = QueryGenerator(model, policy=policy)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is False
        assert result.candidate is None
        assert len(result.attempts) == 3

    def test_provider_failure_recorded(self) -> None:
        model = FakeModel()
        original_generate = model.generate

        def failing_generate(request):
            raise ConnectionError("Ollama unreachable")

        model.generate = failing_generate  # type: ignore[assignment]
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is False
        model.generate = original_generate  # type: ignore[assignment]

    def test_batch_generation(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        units = [_make_code_unit() for _ in range(3)]
        results = gen.generate_batch(units)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_prompt_version_recorded(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.prompt_template_version == "3.0.0"

    def test_timing_recorded(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.total_generation_ms >= 0
        assert result.total_validation_ms >= 0

    def test_no_ollama_dependency(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is True
        assert isinstance(result.candidate, CandidateQuery)

    def test_leakage_rejects_query_with_identifier(self) -> None:
        leaked_query = json.dumps(
            {
                "query": "Find the process_retry function in repo001",
                "query_style": "natural",
                "query_intent": "find_implementation",
            }
        )
        model = FakeModel(_responses=[leaked_query])
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is False
        assert result.candidate is None

    def test_success_preserves_leakage_outcome(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is True
        assert result.leakage is not None
        assert result.leakage.passed is True
        assert result.leakage.violations == []

    def test_leakage_failure_preserves_violations(self) -> None:
        leaked_query = json.dumps(
            {
                "query": "Find PaymentProcessor.process_retry in repo001",
                "query_style": "natural",
                "query_intent": "find_implementation",
            }
        )
        model = FakeModel(_responses=[leaked_query])
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is False
        assert result.candidate is None
        assert result.leakage is not None
        assert result.leakage.passed is False
        assert len(result.leakage.violations) >= 1

    def test_validation_failure_has_no_leakage_outcome(self) -> None:
        model = FakeModel(_responses=["not json", "not json", "not json"])
        policy = RetryPolicy(max_attempts=3)
        gen = QueryGenerator(model, policy=policy)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is False
        assert result.leakage is None


# ===========================================================================
# generate_query convenience
# ===========================================================================


class TestGenerateQuery:
    def test_convenience_function(self) -> None:
        model = FakeModel()
        unit = _make_code_unit()
        result = generate_query(unit, model)

        assert result.success is True
        assert result.candidate is not None

    def test_with_custom_policy(self) -> None:
        model = FakeModel()
        unit = _make_code_unit()
        policy = RetryPolicy(max_attempts=5)
        result = generate_query(unit, model, policy=policy)

        assert result.success is True


# ===========================================================================
# Source isolation
# ===========================================================================


class TestSourceIsolation:
    def test_semantic_label_not_in_prompt(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        gen.generate(unit)

        prompt = model._calls[0].prompt
        assert "concepts" not in prompt.lower()
        assert "side_effects" not in prompt.lower()
        assert "label_version" not in prompt.lower()

    def test_source_url_not_in_prompt(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        gen.generate(unit)

        prompt = model._calls[0].prompt
        assert "github.com" not in prompt
        assert "source_url" not in prompt.lower()

    def test_content_hash_not_in_prompt(self) -> None:
        model = FakeModel()
        gen = QueryGenerator(model)
        unit = _make_code_unit()
        gen.generate(unit)

        prompt = model._calls[0].prompt
        assert "abc123" not in prompt


# ===========================================================================
# Reproducibility
# ===========================================================================


class TestReproducibility:
    def test_same_input_same_prompt(self) -> None:
        from localbench.workloads.code_retrieval.query_generator import (
            _to_query_input,
        )
        from localbench.workloads.code_retrieval.query_prompt import (
            build_query_generation_prompt,
        )

        unit = _make_code_unit()
        inp = _to_query_input(unit)
        p1 = build_query_generation_prompt(inp)
        p2 = build_query_generation_prompt(inp)
        assert p1 == p2

    def test_prompt_template_version_constant(self) -> None:
        from localbench.workloads.code_retrieval.query_prompt import (
            QUERY_PROMPT_TEMPLATE_VERSION,
        )

        assert QUERY_PROMPT_TEMPLATE_VERSION == "3.0.0"
