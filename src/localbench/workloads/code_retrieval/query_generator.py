"""Retrieval query generation pipeline.

Generates candidate retrieval queries for extracted Python code units
using a dedicated local model via the ``LocalModel`` protocol.  Reuses
existing Phase 2 (structured validation) and Phase 3 (bounded retry)
infrastructure.

Scope (Phase 4F):
- CandidateQuery generation for ExtractedCodeUnit objects
- Configurable model selection (no hardcoded model name)
- Source-only input contract (no SemanticLabel leakage)
- Prompt construction via query_prompt module
- Structured validation against CandidateQuery schema
- Query quality / leakage validation
- Bounded retry with attempt recording
- Generation metadata for reproducibility

Out of scope:
- Final 45-query dataset generation
- Human review UI
- Ground-truth relevance assignment
- Retrieval execution
- Benchmark evaluation
- Training / fine-tuning
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from localbench.runtime.generation.attempt import AttemptRecord
from localbench.runtime.generation.executor import (
    GenerateFn,
    run_with_retry,
)
from localbench.runtime.generation.policy import RetryPolicy
from localbench.runtime.model import GenerationRequest, LocalModel
from localbench.workloads.code_retrieval.extraction import ExtractedCodeUnit
from localbench.workloads.code_retrieval.query_prompt import (
    QUERY_PROMPT_TEMPLATE_VERSION,
    build_query_generation_prompt,
)
from localbench.workloads.code_retrieval.schemas import (
    CandidateQuery,
    QueryGenerationInput,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

_LEAKAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\.py\b", re.IGNORECASE),
    re.compile(r"\.java\b", re.IGNORECASE),
    re.compile(r"\.js\b", re.IGNORECASE),
    re.compile(r"\.ts\b", re.IGNORECASE),
    re.compile(r"github\.com", re.IGNORECASE),
    re.compile(r"gitlab\.com", re.IGNORECASE),
    re.compile(r"bitbucket\.org", re.IGNORECASE),
    re.compile(r"\brepo\d+\b", re.IGNORECASE),
    re.compile(r"\btest_\w+\.py\b", re.IGNORECASE),
    re.compile(r"\w+_test\.py\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class LeakageCheckResult:
    """Outcome of query leakage validation."""

    passed: bool
    violations: list[str] = field(default_factory=list)


def check_query_leakage(
    query: str,
    code_unit: ExtractedCodeUnit,
) -> LeakageCheckResult:
    """Check if a generated query leaks forbidden identifiers.

    Inspects the query text for common leakage patterns: file paths,
    repository references, file extensions, and URL patterns.

    This is a heuristic check.  It does NOT guarantee absence of all
    possible information leakage — that requires human review per the
    frozen methodology.
    """
    violations: list[str] = []

    for pattern in _LEAKAGE_PATTERNS:
        if pattern.search(query):
            violations.append(
                f"Query matches leakage pattern: {pattern.pattern}"
            )

    if code_unit.repository and code_unit.repository in query:
        violations.append(
            f"Query contains repository identifier: {code_unit.repository}"
        )

    if code_unit.file_path and code_unit.file_path in query:
        violations.append(
            f"Query contains file path: {code_unit.file_path}"
        )

    if code_unit.symbol and code_unit.symbol in query:
        violations.append(
            f"Query contains symbol path: {code_unit.symbol}"
        )

    return LeakageCheckResult(
        passed=len(violations) == 0,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class QueryGenerationResult:
    """Outcome of query generation for a single code unit.

    Attributes:
        success: Whether a valid CandidateQuery was produced.
        candidate: The generated candidate query (``None`` if failed).
        attempts: Full attempt history from the retry loop.
        model_name: Name of the model used for generation.
        prompt_template_version: Version of the prompt template.
        total_generation_ms: Total generation time across attempts.
        total_validation_ms: Total validation time across attempts.
    """

    success: bool
    candidate: CandidateQuery | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    model_name: str = ""
    prompt_template_version: str = QUERY_PROMPT_TEMPLATE_VERSION
    total_generation_ms: float = 0.0
    total_validation_ms: float = 0.0
    leakage: LeakageCheckResult | None = None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class QueryGenerator:
    """Generates candidate retrieval queries using a local model.

    The generator is provider-agnostic — it depends only on the
    ``LocalModel`` protocol.  The caller determines which model to use
    (e.g. ``qwen2.5-coder:7b``).

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
        top_p: float = 0.9,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.policy = policy or RetryPolicy()
        self.top_p = top_p
        self.seed = seed

    def generate(self, code_unit: ExtractedCodeUnit) -> QueryGenerationResult:
        """Generate a candidate retrieval query for a single code unit.

        Pipeline: convert to source-only input -> build prompt ->
        model generate -> validate -> leakage check -> retry

        Parameters
        ----------
        code_unit:
            The extracted code unit to generate a query for.

        Returns
        -------
        A ``QueryGenerationResult`` with the outcome and full attempt history.
        """
        gen_input = _to_query_input(code_unit)
        prompt = build_query_generation_prompt(gen_input)
        generate_fn = self._make_generate_fn()

        retry_result = run_with_retry(
            prompt=prompt,
            schema=CandidateQuery,
            generate_fn=generate_fn,
            policy=self.policy,
        )

        if retry_result.success and retry_result.result is not None:
            candidate = retry_result.result.data
            assert isinstance(candidate, CandidateQuery)

            leakage = check_query_leakage(candidate.query, code_unit)
            if not leakage.passed:
                logger.warning(
                    "Query failed leakage check: %s",
                    "; ".join(leakage.violations),
                )
                return QueryGenerationResult(
                    success=False,
                    attempts=retry_result.attempts,
                    model_name=self.model.name,
                    total_generation_ms=retry_result.total_generation_ms,
                    total_validation_ms=retry_result.total_validation_ms,
                    leakage=leakage,
                )

            return QueryGenerationResult(
                success=True,
                candidate=candidate,
                attempts=retry_result.attempts,
                model_name=self.model.name,
                total_generation_ms=retry_result.total_generation_ms,
                total_validation_ms=retry_result.total_validation_ms,
                leakage=leakage,
            )

        return QueryGenerationResult(
            success=False,
            attempts=retry_result.attempts,
            model_name=self.model.name,
            total_generation_ms=retry_result.total_generation_ms,
            total_validation_ms=retry_result.total_validation_ms,
        )

    def generate_batch(
        self,
        code_units: list[ExtractedCodeUnit],
    ) -> list[QueryGenerationResult]:
        """Generate candidate queries for multiple code units.

        Processes each code unit independently.  Failures on individual
        units do not affect other units.

        Parameters
        ----------
        code_units:
            List of extracted code units to generate queries for.

        Returns
        -------
        A list of ``QueryGenerationResult`` objects, one per input unit.
        """
        return [self.generate(unit) for unit in code_units]

    def _make_generate_fn(self) -> GenerateFn:
        """Create a generate function bound to this model and config."""
        model = self.model

        def _generate(prompt: str) -> str:
            request = GenerationRequest(
                prompt=prompt,
                model=model.name,
                temperature=0.7,
                max_tokens=128,
                top_p=self.top_p,
                seed=self.seed,
            )
            result = model.generate(request)
            if result.error:
                raise RuntimeError(result.error)
            return result.text

        return _generate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_query_input(code_unit: ExtractedCodeUnit) -> QueryGenerationInput:
    """Convert an ExtractedCodeUnit to a source-only QueryGenerationInput.

    Strips repository, file_path, symbol, source_url, and content_hash.
    Only source code and context information are preserved.
    """
    return QueryGenerationInput(
        source_code=code_unit.source_code,
        docstring=code_unit.docstring,
        symbol_type=code_unit.symbol_type,
        class_name=code_unit.context.class_name,
        module_docstring=code_unit.context.module_docstring,
        imports=code_unit.context.imports,
        parent_methods=code_unit.context.parent_methods,
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def generate_query(
    code_unit: ExtractedCodeUnit,
    model: LocalModel,
    policy: RetryPolicy | None = None,
) -> QueryGenerationResult:
    """One-call convenience for query generation."""
    generator = QueryGenerator(model, policy=policy)
    return generator.generate(code_unit)
