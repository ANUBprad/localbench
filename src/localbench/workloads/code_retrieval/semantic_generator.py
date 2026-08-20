"""Semantic label generation pipeline.

Generates semantic labels for extracted Python code units using a
local model via the ``LocalModel`` protocol.  Reuses existing Phase 2
(structured validation) and Phase 3 (bounded retry) infrastructure.

Scope (Phase 4E):
- SemanticLabel generation for ExtractedCodeUnit objects
- Configurable model selection (no hardcoded model name)
- Prompt construction via semantic_prompt module
- Structured validation against SemanticLabel schema
- Bounded retry with attempt recording
- Generation metadata for reproducibility

Out of scope:
- Retrieval query generation
- Dataset splitting / assembly
- Benchmark evaluation
- Training / fine-tuning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from localbench.runtime.generation.attempt import AttemptRecord
from localbench.runtime.generation.executor import (
    GenerateFn,
    run_with_retry,
)
from localbench.runtime.generation.policy import RetryPolicy
from localbench.runtime.model import GenerationRequest, LocalModel
from localbench.workloads.code_retrieval.extraction import ExtractedCodeUnit
from localbench.workloads.code_retrieval.schemas import SemanticLabel
from localbench.workloads.code_retrieval.semantic_prompt import (
    PROMPT_TEMPLATE_VERSION,
    build_semantic_label_prompt,
)

logger = logging.getLogger(__name__)

LABEL_VERSION = "1.0.0"
"""Version of generated semantic labels (matches schema requirement)."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SemanticLabelResult:
    """Outcome of semantic label generation for a single code unit.

    Attributes:
        success: Whether a valid SemanticLabel was produced.
        label: The generated label (``None`` if generation failed).
        attempts: Full attempt history from the retry loop.
        model_name: Name of the model used for generation.
        prompt_template_version: Version of the prompt template.
        total_generation_ms: Total generation time across attempts.
        total_validation_ms: Total validation time across attempts.
    """

    success: bool
    label: SemanticLabel | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    model_name: str = ""
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION
    total_generation_ms: float = 0.0
    total_validation_ms: float = 0.0


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SemanticLabelGenerator:
    """Generates semantic labels for code units using a local model.

    Parameters
    ----------
    model:
        A provider-agnostic local model implementing the ``LocalModel``
        protocol.  Must be separate from all benchmark evaluation models
        per DATASET_SPECIFICATION.md section 4.4.
    policy:
        Retry configuration.  Uses defaults (3 attempts) if ``None``.
    """

    def __init__(
        self,
        model: LocalModel,
        policy: RetryPolicy | None = None,
    ) -> None:
        self.model = model
        self.policy = policy or RetryPolicy()

    def generate(self, code_unit: ExtractedCodeUnit) -> SemanticLabelResult:
        """Generate a semantic label for a single code unit.

        Pipeline: build prompt -> model generate -> validate -> retry

        Parameters
        ----------
        code_unit:
            The extracted code unit to describe.

        Returns
        -------
        A ``SemanticLabelResult`` with the outcome and full attempt history.
        """
        prompt = build_semantic_label_prompt(code_unit)
        generate_fn = self._make_generate_fn()

        retry_result = run_with_retry(
            prompt=prompt,
            schema=SemanticLabel,
            generate_fn=generate_fn,
            policy=self.policy,
        )

        if retry_result.success and retry_result.result is not None:
            label = retry_result.result.data
            assert isinstance(label, SemanticLabel)
            return SemanticLabelResult(
                success=True,
                label=label,
                attempts=retry_result.attempts,
                model_name=self.model.name,
                total_generation_ms=retry_result.total_generation_ms,
                total_validation_ms=retry_result.total_validation_ms,
            )

        return SemanticLabelResult(
            success=False,
            attempts=retry_result.attempts,
            model_name=self.model.name,
            total_generation_ms=retry_result.total_generation_ms,
            total_validation_ms=retry_result.total_validation_ms,
        )

    def generate_batch(
        self,
        code_units: list[ExtractedCodeUnit],
    ) -> list[SemanticLabelResult]:
        """Generate semantic labels for multiple code units.

        Processes each code unit independently.  Failures on individual
        units do not affect other units.

        Parameters
        ----------
        code_units:
            List of extracted code units to describe.

        Returns
        -------
        A list of ``SemanticLabelResult`` objects, one per input unit.
        """
        return [self.generate(unit) for unit in code_units]

    def _make_generate_fn(self) -> GenerateFn:
        """Create a generate function bound to this model and config."""
        model = self.model

        def _generate(prompt: str) -> str:
            request = GenerationRequest(
                prompt=prompt,
                model=model.name,
                temperature=0.3,
                max_tokens=256,
            )
            result = model.generate(request)
            if result.error:
                raise RuntimeError(result.error)
            return result.text

        return _generate


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def generate_semantic_label(
    code_unit: ExtractedCodeUnit,
    model: LocalModel,
    policy: RetryPolicy | None = None,
) -> SemanticLabelResult:
    """One-call convenience for semantic label generation."""
    generator = SemanticLabelGenerator(model, policy=policy)
    return generator.generate(code_unit)
