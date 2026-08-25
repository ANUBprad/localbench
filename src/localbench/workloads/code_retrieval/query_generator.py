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

_MIN_IDENTIFIER_LENGTH = 4
"""Minimum identifier length to check for word-boundary matches.

Short identifiers (e.g. ``id``, ``ok``, ``abs``) are too common in
natural language and cause unacceptable false-positive rates.
"""

_PARAM_PATTERN = re.compile(
    r"def\s+\w+\s*\(([^)]*)\)",
    re.DOTALL,
)
"""Matches a Python function/method signature to extract parameter names."""


@dataclass(frozen=True)
class LeakageCheckResult:
    """Outcome of query leakage validation."""

    passed: bool
    violations: list[str] = field(default_factory=list)


def _extract_identifier_parts(code_unit: ExtractedCodeUnit) -> list[str]:
    """Extract individual identifier parts from the code unit.

    Returns a deduplicated list of identifier parts extracted from:
    - The symbol path (split on ``.``)
    - The class name from context (if different from the first symbol part)

    Parts shorter than ``_MIN_IDENTIFIER_LENGTH`` are excluded.
    """
    parts: list[str] = []

    if code_unit.symbol:
        for segment in code_unit.symbol.split("."):
            if len(segment) >= _MIN_IDENTIFIER_LENGTH:
                parts.append(segment)

    ctx_class = code_unit.context.class_name if code_unit.context else None
    if ctx_class and len(ctx_class) >= _MIN_IDENTIFIER_LENGTH:
        if ctx_class not in parts:
            parts.append(ctx_class)

    return list(dict.fromkeys(parts))


_PARAM_NAME_BLACKLIST = frozenset({
    "self", "cls", "args", "kwargs",
    # Common English words that appear as parameter/attribute names
    # and cause unacceptable false-positive rates.
    "padding", "link", "size", "name", "type", "value", "key",
    "item", "data", "text", "path", "mode", "error", "result",
    "status", "option", "config", "format", "table", "column",
    "width", "height", "color", "level", "count", "index",
    "first", "last", "next", "prev", "start", "end",
})
"""Parameter names that are never flagged as leaked identifiers."""


_DUNDER_PATTERN = re.compile(r"\b__\w+__\b")
"""Matches dunder references (e.g. ``__init__``, ``__new__``) in query text."""

_SPHINX_CLASS_PATTERN = re.compile(
    r":class:`~?([^`]+)`",
)
"""Extracts class names from Sphinx ``:class:`` cross-references in docstrings."""


def _extract_param_names(source_code: str) -> list[str]:
    """Extract parameter names from a function/method signature.

    Returns parameter identifiers longer than ``_MIN_IDENTIFIER_LENGTH``
    that are not in ``_PARAM_NAME_BLACKLIST``.
    """
    match = _PARAM_PATTERN.search(source_code)
    if not match:
        return []

    raw = match.group(1)
    names: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        # Strip leading * or ** for *args / **kwargs
        stripped = token.lstrip("*")
        name = stripped.split("=")[0].strip()
        if name in _PARAM_NAME_BLACKLIST or not name:
            continue
        if len(name) >= _MIN_IDENTIFIER_LENGTH:
            names.append(name)
    return list(dict.fromkeys(names))


def _make_identifier_pattern(identifier: str) -> re.Pattern[str]:
    """Build a word-boundary regex for a case-sensitive identifier match."""
    escaped = re.escape(identifier)
    return re.compile(rf"\b{escaped}\b")


def _extract_docstring_class_names(docstring: str) -> list[str]:
    """Extract class names from Sphinx ``:class:`` cross-references.

    Returns the trailing class name portion (e.g. ``Config`` from
    ``:class:`~pytest.Config```), deduplicated, excluding short names.
    """
    names: list[str] = []
    for match in _SPHINX_CLASS_PATTERN.finditer(docstring):
        full_ref = match.group(1)
        # Take the last component after the final dot (e.g. "pytest.Config" -> "Config")
        short_name = full_ref.rsplit(".", 1)[-1]
        if len(short_name) >= _MIN_IDENTIFIER_LENGTH:
            names.append(short_name)
    return list(dict.fromkeys(names))


def check_query_leakage(
    query: str,
    code_unit: ExtractedCodeUnit,
) -> LeakageCheckResult:
    """Check if a generated query leaks forbidden identifiers.

    Inspects the query text for common leakage patterns: file paths,
    repository references, file extensions, URL patterns, and
    individual identifier parts (class names, method names,
    parameter names) extracted from the code unit.

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

    # Check individual identifier parts from symbol and context class name
    for part in _extract_identifier_parts(code_unit):
        if part in _PARAM_NAME_BLACKLIST:
            continue
        pat = _make_identifier_pattern(part)
        if pat.search(query):
            violations.append(
                f"Query contains identifier part: {part}"
            )

    # Check for dunder references (e.g. __init__, __new__) in the query.
    # Dunders are implementation-specific identifiers regardless of whether
    # they match the exact code unit symbol.
    if _DUNDER_PATTERN.search(query):
        violations.append(
            "Query contains dunder method reference"
        )

    # Check parameter names extracted from source code signature
    if code_unit.source_code:
        for param in _extract_param_names(code_unit.source_code):
            pat = _make_identifier_pattern(param)
            if pat.search(query):
                violations.append(
                    f"Query contains parameter name: {param}"
                )

    # Check class names from Sphinx :class: cross-references in the docstring.
    # These are implementation-specific type names that should not appear in
    # natural language queries (e.g. "pytest.Config" leaked from docstring).
    if code_unit.docstring:
        for cls_name in _extract_docstring_class_names(code_unit.docstring):
            if cls_name in _PARAM_NAME_BLACKLIST:
                continue
            pat = _make_identifier_pattern(cls_name)
            if pat.search(query):
                violations.append(
                    f"Query contains docstring class name: {cls_name}"
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
