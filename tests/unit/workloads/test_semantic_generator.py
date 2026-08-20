"""Tests for semantic label generation pipeline.

All tests use a fake LocalModel — no Ollama or network dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from localbench.runtime.generation.policy import RetryPolicy
from localbench.runtime.model import GenerationRequest, ModelInfo
from localbench.workloads.code_retrieval.extraction import ExtractedCodeUnit
from localbench.workloads.code_retrieval.schemas import (
    CodeUnitContext,
    SemanticLabel,
)
from localbench.workloads.code_retrieval.semantic_generator import (
    LABEL_VERSION,
    SemanticLabelGenerator,
    SemanticLabelResult,
    generate_semantic_label,
)

# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------

_VALID_LABEL_JSON = json.dumps(
    {
        "code_unit_id": "repo001_py_func_greet",
        "description": (
            "Greets a person by name using an f-string template. "
            "Constructs a formatted greeting message, prints it to "
            "standard output, and returns the resulting string for "
            "further use by the caller."
        ),
        "summary": "Greets a person by name.",
        "concepts": ["string formatting", "output", "greeting"],
        "input_types": ["str (name)"],
        "output_type": "str",
        "side_effects": ["Prints to stdout"],
        "created_by": "model_generated",
        "label_version": "1.0.0",
    }
)


@dataclass
class FakeModel:
    """Fake LocalModel for testing."""

    _name: str = "fake-model"
    _responses: list[str] = field(default_factory=lambda: [_VALID_LABEL_JSON])
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


def _make_code_unit() -> ExtractedCodeUnit:
    """Create a minimal ExtractedCodeUnit for testing."""
    return ExtractedCodeUnit(
        repository="repo001",
        language="python",
        file_path="utils.py",
        symbol="greet",
        symbol_type="function",
        source_code=(
            "def greet(name):\n"
            '    """Greet someone."""\n'
            '    message = f"Hello, {name}!"\n'
            "    print(message)\n"
            "    return message\n"
        ),
        source_url="",
        is_public=True,
        docstring="Greet someone.",
        source_file_lines=6,
        content_hash="abc",
        extracted_at="2026-08-20T00:00:00Z",
        context=CodeUnitContext(
            module_docstring="Utilities module.",
            imports=["os"],
        ),
    )


# ===========================================================================
# SemanticLabelResult
# ===========================================================================


class TestSemanticLabelResult:
    def test_success_fields(self) -> None:
        label = SemanticLabel(
            code_unit_id="x",
            description=(
                "This function performs a specific computation that "
                "takes an input value, applies a transformation, and "
                "returns the result for further processing downstream."
            ),
            summary="s",
            concepts=["a", "b"],
            created_by="model_generated",
            label_version="1.0.0",
        )
        result = SemanticLabelResult(success=True, label=label)
        assert result.success is True
        assert result.label is label

    def test_failure_fields(self) -> None:
        result = SemanticLabelResult(success=False)
        assert result.success is False
        assert result.label is None


# ===========================================================================
# SemanticLabelGenerator
# ===========================================================================


class TestSemanticLabelGenerator:
    def test_successful_generation(self) -> None:
        model = FakeModel()
        gen = SemanticLabelGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is True
        assert result.label is not None
        assert result.label.code_unit_id == "repo001_py_func_greet"
        assert result.label.created_by == "model_generated"
        assert result.label.label_version == "1.0.0"
        assert len(result.attempts) == 1
        assert result.model_name == "fake-model"

    def test_model_called_with_prompt(self) -> None:
        model = FakeModel()
        gen = SemanticLabelGenerator(model)
        unit = _make_code_unit()
        gen.generate(unit)

        assert len(model._calls) == 1
        assert "greet" in model._calls[0].prompt
        assert "def greet" in model._calls[0].prompt

    def test_model_uses_correct_config(self) -> None:
        model = FakeModel()
        gen = SemanticLabelGenerator(model)
        unit = _make_code_unit()
        gen.generate(unit)

        req = model._calls[0]
        assert req.model == "fake-model"
        assert req.temperature == 0.3
        assert req.max_tokens == 256

    def test_retry_on_malformed_json(self) -> None:
        bad_response = "This is not JSON at all"
        good_response = _VALID_LABEL_JSON
        model = FakeModel(_responses=[bad_response, good_response])
        policy = RetryPolicy(max_attempts=3)
        gen = SemanticLabelGenerator(model, policy=policy)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is True
        assert len(result.attempts) == 2
        assert model._call_index == 2

    def test_retry_on_validation_error(self) -> None:
        short_desc = json.dumps(
            {
                "code_unit_id": "x",
                "description": "Too short.",
                "summary": "s",
                "concepts": ["a", "b"],
                "created_by": "model_generated",
                "label_version": "1.0.0",
            }
        )
        model = FakeModel(_responses=[short_desc, _VALID_LABEL_JSON])
        policy = RetryPolicy(max_attempts=3)
        gen = SemanticLabelGenerator(model, policy=policy)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is True
        assert len(result.attempts) == 2

    def test_retry_exhaustion(self) -> None:
        bad = "not json"
        model = FakeModel(_responses=[bad, bad, bad])
        policy = RetryPolicy(max_attempts=3)
        gen = SemanticLabelGenerator(model, policy=policy)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is False
        assert result.label is None
        assert len(result.attempts) == 3

    def test_provider_failure_recorded(self) -> None:
        model = FakeModel()
        original_generate = model.generate

        def failing_generate(request):
            raise ConnectionError("Ollama unreachable")

        model.generate = failing_generate  # type: ignore[assignment]
        gen = SemanticLabelGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is False
        model.generate = original_generate  # type: ignore[assignment]

    def test_batch_generation(self) -> None:
        model = FakeModel()
        gen = SemanticLabelGenerator(model)
        units = [_make_code_unit() for _ in range(3)]
        results = gen.generate_batch(units)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_prompt_version_recorded(self) -> None:
        model = FakeModel()
        gen = SemanticLabelGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.prompt_template_version == "1.0.0"

    def test_timing_recorded(self) -> None:
        model = FakeModel()
        gen = SemanticLabelGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.total_generation_ms >= 0
        assert result.total_validation_ms >= 0

    def test_no_ollama_dependency(self) -> None:
        model = FakeModel()
        gen = SemanticLabelGenerator(model)
        unit = _make_code_unit()
        result = gen.generate(unit)

        assert result.success is True
        assert isinstance(result.label, SemanticLabel)


# ===========================================================================
# generate_semantic_label convenience
# ===========================================================================


class TestGenerateSemanticLabel:
    def test_convenience_function(self) -> None:
        model = FakeModel()
        unit = _make_code_unit()
        result = generate_semantic_label(unit, model)

        assert result.success is True
        assert result.label is not None

    def test_with_custom_policy(self) -> None:
        model = FakeModel()
        unit = _make_code_unit()
        policy = RetryPolicy(max_attempts=5)
        result = generate_semantic_label(unit, model, policy=policy)

        assert result.success is True


# ===========================================================================
# Reproducibility
# ===========================================================================


class TestReproducibility:
    def test_same_input_same_prompt(self) -> None:
        from localbench.workloads.code_retrieval.semantic_prompt import (
            build_semantic_label_prompt,
        )

        unit = _make_code_unit()
        p1 = build_semantic_label_prompt(unit)
        p2 = build_semantic_label_prompt(unit)
        assert p1 == p2

    def test_label_version_constant(self) -> None:
        assert LABEL_VERSION == "1.0.0"
