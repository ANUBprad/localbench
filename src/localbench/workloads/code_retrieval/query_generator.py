"""Retrieval query generation pipeline.

Generates candidate retrieval queries for extracted Python code units
using a dedicated local model via the ``LocalModel`` protocol.  Reuses
existing Phase 2 (structured validation) and Phase 3 (bounded retry)
infrastructure.

Two-stage generation (v2):
  Stage A: Deterministic AST-based behavior extraction (no LLM)
  Stage B: Docstring-blind prompt using StructuredBehaviorFacts + source

This eliminates docstring-provenance leakage — the LLM never sees
the docstring during query generation.

Scope (Phase 4F):
- CandidateQuery generation for ExtractedCodeUnit objects
- Configurable model selection (no hardcoded model name)
- Source-only input contract (no SemanticLabel leakage)
- Prompt construction via query_prompt module
- Structured validation against CandidateQuery schema
- Query quality / leakage validation
- Provenance validation (docstring-provenance leakage detection)
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
from localbench.workloads.code_retrieval.behavior_extraction import (
    extract_behavior_facts,
)
from localbench.workloads.code_retrieval.extraction import ExtractedCodeUnit
from localbench.workloads.code_retrieval.query_prompt import (
    QUERY_PROMPT_TEMPLATE_VERSION,
    build_query_generation_prompt_v2,
)
from localbench.workloads.code_retrieval.schemas import (
    CandidateQuery,
    QueryGenerationInput,
    StructuredBehaviorFacts,
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
    # Common English words found in docstrings that are not identifiers
    "Refresh", "Overriding", "Return", "Check", "Create", "Find",
    "Get", "Set", "Add", "Remove", "Update", "Delete", "Insert",
    "Move", "Copy", "Split", "Join", "Merge", "Sort", "Filter",
    "Test", "Run", "Execute", "Parse", "Build", "Generate", "Write",
    "Read", "Load", "Save", "Import", "Export", "Convert", "Format",
    "Display", "Print", "Show", "Hide", "Enable", "Disable",
})
"""Parameter names that are never flagged as leaked identifiers."""


_DUNDER_PATTERN = re.compile(r"\b__\w+__\b")
"""Matches dunder references (e.g. ``__init__``, ``__new__``) in query text."""

_SPHINX_CLASS_PATTERN = re.compile(
    r":class:`~?([^`]+)`",
)
"""Extracts class names from Sphinx ``:class:`` cross-references in docstrings."""

_SPHINX_ATTR_PATTERN = re.compile(
    r":attr:`~?([^`]+)`",
)
"""Extracts attribute names from Sphinx ``:attr:`` cross-references in docstrings."""

_SPHINX_METH_PATTERN = re.compile(
    r":meth:`~?([^`]+)`",
)
"""Extracts method names from Sphinx ``:meth:`` cross-references in docstrings."""

_SPHINX_FUNC_PATTERN = re.compile(
    r":func:`~?([^`]+)`",
)
"""Extracts function names from Sphinx ``:func:`` cross-references in docstrings."""

_META_TASK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi need a (technical )?query\b", re.IGNORECASE),
    re.compile(r"\bgenerate (a )?(retrieval )?query\b", re.IGNORECASE),
    re.compile(r"\bretrieve the source code\b", re.IGNORECASE),
    re.compile(r"\bfind the (source )?code for\b", re.IGNORECASE),
    re.compile(r"\bwrite (a )?retrieval query\b", re.IGNORECASE),
    re.compile(r"\bcreate (a )?query\b", re.IGNORECASE),
    re.compile(r"\bquery to (find|retrieve|get)\b", re.IGNORECASE),
    re.compile(
        r"\bhow (can|do) i (write|create|generate|find) (a )?query\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bpytest\s+--collect-only\b", re.IGNORECASE),
    re.compile(r"\bpytest\s+-\s*-\s*collect\s*-\s*only\b", re.IGNORECASE),
)
"""Patterns that indicate the model generated a meta-task query
(requesting query generation) instead of an actual retrieval query."""


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


def _extract_sphinx_ref_names(docstring: str) -> list[str]:
    """Extract names from all Sphinx cross-reference types in the docstring.

    Checks ``:class:``, ``:attr:``, ``:meth:``, and ``:func:`` directives.
    Returns the trailing name portion (e.g. ``path`` from
    ``:attr:`~pytest.pytester.Pytester.path``), deduplicated, excluding
    short names.
    """
    patterns = [
        _SPHINX_CLASS_PATTERN,
        _SPHINX_ATTR_PATTERN,
        _SPHINX_METH_PATTERN,
        _SPHINX_FUNC_PATTERN,
    ]
    names: list[str] = []
    for pat in patterns:
        for match in pat.finditer(docstring):
            full_ref = match.group(1)
            short_name = full_ref.rsplit(".", 1)[-1]
            if len(short_name) >= _MIN_IDENTIFIER_LENGTH:
                names.append(short_name)
    return list(dict.fromkeys(names))


def _extract_docstring_content(docstring: str) -> list[str]:
    """Extract significant phrases from docstring for reproduction checks.

    Returns normalized sentences from the docstring that are long enough
    to be meaningful for matching against queries.  Excludes Sphinx
    directives and very short fragments.
    """
    if not docstring:
        return []
    phrases: list[str] = []
    for line in docstring.splitlines():
        stripped = line.strip()
        # Skip Sphinx directives, empty lines, very short lines
        if not stripped or stripped.startswith(":") or len(stripped) < 15:
            continue
        # Skip lines that are mostly punctuation or markup
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in stripped) / len(stripped)
        if alpha_ratio < 0.6:
            continue
        phrases.append(stripped.lower())
    return phrases


def _extract_source_identifiers(source_code: str) -> list[str]:
    """Extract significant identifiers from source code body.

    Looks for method calls, attribute accesses, and variable names
    that are longer than ``_MIN_IDENTIFIER_LENGTH`` and could be
    implementation-specific identifiers.
    """
    identifiers: list[str] = []
    # Match method calls like obj.method_name() or Class.method_name()
    for match in re.finditer(r"\b([A-Z]\w+)\.(\w+)\s*\(", source_code):
        for part in match.groups():
            if len(part) >= _MIN_IDENTIFIER_LENGTH:
                identifiers.append(part)
    # Match standalone function/method calls
    for match in re.finditer(r"\b(\w{4,})\s*\(", source_code):
        name = match.group(1)
        if name[0].isupper() or "_" in name:
            identifiers.append(name)
    return list(dict.fromkeys(identifiers))


def check_query_leakage(
    query: str,
    code_unit: ExtractedCodeUnit,
) -> LeakageCheckResult:
    """Check if a generated query leaks forbidden identifiers.

    Inspects the query text for common leakage patterns: file paths,
    repository references, file extensions, URL patterns, and
    individual identifier parts (class names, method names,
    parameter names) extracted from the code unit.

    Also detects meta-task queries (the model asking to generate a
    query) and docstring reproduction (the query reproducing
    docstring text verbatim).

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

        # Check for method/function names from the source code body
        # that appear in the query (e.g. runpytest, fnmatch_lines, formatTime)
        for ident in _extract_source_identifiers(code_unit.source_code):
            if ident in _PARAM_NAME_BLACKLIST:
                continue
            pat = _make_identifier_pattern(ident)
            if pat.search(query):
                violations.append(
                    f"Query contains source code identifier: {ident}"
                )

    # Check sibling method names from context.parent_methods.  These are
    # implementation identifiers the model may reproduce from the "Sibling
    # methods" prompt context (e.g. test_show_fixtures, runpytest_subprocess).
    # Only flag names that clearly look like code identifiers (underscored or
    # camel/Pascal-cased), reusing the same heuristic applied to source-body
    # names, so common English/method words (run, spawn, request) are not
    # falsely rejected.
    if code_unit.context and code_unit.context.parent_methods:
        for sibling in code_unit.context.parent_methods:
            if sibling in _PARAM_NAME_BLACKLIST:
                continue
            if len(sibling) < _MIN_IDENTIFIER_LENGTH:
                continue
            has_underscore = "_" in sibling
            has_camel = any(c.isupper() for c in sibling[1:]) and any(
                c.islower() for c in sibling
            )
            if not (has_underscore or has_camel):
                continue
            pat = _make_identifier_pattern(sibling)
            if pat.search(query):
                violations.append(
                    f"Query contains sibling method name: {sibling}"
                )
                break

    # Check all Sphinx cross-reference names (:class:, :attr:, :meth:, :func:)
    # from the docstring.  These are implementation-specific identifiers.
    if code_unit.docstring:
        for ref_name in _extract_sphinx_ref_names(code_unit.docstring):
            if ref_name in _PARAM_NAME_BLACKLIST:
                continue
            pat = _make_identifier_pattern(ref_name)
            if pat.search(query):
                violations.append(
                    f"Query contains docstring reference name: {ref_name}"
                )
            # Also check if the Sphinx directive itself appears in the query
            # (e.g. :attr:`path`, :meth:`formatTime`)
            for directive in (":attr:", ":meth:", ":func:", ":class:"):
                sphinx_ref = f"{directive}`{ref_name}`"
                if sphinx_ref in query:
                    violations.append(
                        f"Query contains Sphinx directive: {sphinx_ref}"
                    )
                    break

    # Check for meta-task queries (model asking to generate a query)
    for pattern in _META_TASK_PATTERNS:
        if pattern.search(query):
            violations.append(
                f"Query is a meta-task query: {pattern.pattern}"
            )
            break  # One meta-task violation is sufficient

    # Check for docstring reproduction: if the query contains a significant
    # portion of docstring text, or if the docstring contains the query text,
    # it's likely reproducing the docstring.
    if code_unit.docstring:
        query_lower = query.lower()
        doc_lower = code_unit.docstring.lower()
        # Check if query reproduces docstring text (query contains docstring phrase)
        for phrase in _extract_docstring_content(code_unit.docstring):
            if phrase in query_lower:
                violations.append(
                    "Query reproduces docstring text"
                )
                break
        # Check if docstring contains the query text (query is a subset of docstring)
        # This catches cases like "pytest.approx() should raise..." matching
        # docstring "pytest.approx() should raise an error on unordered sequences (#9692)."
        if not violations:
            query_stripped = query_lower.strip().rstrip(".")
            if len(query_stripped) > 20 and query_stripped in doc_lower:
                violations.append(
                    "Query reproduces docstring text"
                )

    # Check for function/method names mentioned in docstrings
    # (e.g. "getfuncargnames" in docstring "Check getfuncargnames for...")
    if code_unit.docstring and not violations:
        # Extract identifiers from docstring that look like function/method names
        # (contain underscore, camelCase, or are PascalCase)
        doc_identifiers = set()
        for word in re.findall(r"\b([a-zA-Z_]\w{3,})\b", code_unit.docstring):
            # Only flag words that look like code identifiers:
            # - snake_case (contains underscore)
            # - camelCase (contains lowercase followed by uppercase)
            # - PascalCase (starts with uppercase)
            # Skip common English words and Sphinx directive names
            if word.lower() in {"this", "that", "with", "from", "then", "when",
                                "class", "attr", "meth", "func", "param",
                                "returns", "raise", "value", "type", "args",
                                "true", "false", "none", "self", "cls",
                                "return", "check", "create", "find", "get",
                                "set", "add", "remove", "update", "delete",
                                "insert", "move", "copy", "split", "join",
                                "merge", "sort", "filter", "test", "run",
                                "execute", "parse", "build", "generate",
                                "write", "read", "load", "save", "import",
                                "export", "convert", "format", "display",
                                "print", "show", "hide", "enable", "disable",
                                "refresh", "overriding", "failed", "passed",
                                "error", "result", "item", "name", "type",
                                "value", "path", "mode", "data", "text",
                                "size", "status", "option", "config",
                                "table", "column", "width", "height",
                                "color", "level", "count", "index",
                                "first", "last", "next", "prev", "start",
                                "end", "padding", "link", "key"}:
                continue
            # Must look like a code identifier (snake_case or camelCase)
            has_underscore = "_" in word
            has_camel = any(c.isupper() for c in word[1:]) and any(c.islower() for c in word)
            if has_underscore or has_camel:
                doc_identifiers.add(word)
        for word in doc_identifiers:
            if word in _PARAM_NAME_BLACKLIST:
                continue
            pat = _make_identifier_pattern(word)
            if pat.search(query):
                violations.append(
                    f"Query contains docstring function name: {word}"
                )
                break

    return LeakageCheckResult(
        passed=len(violations) == 0,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Provenance validation
# ---------------------------------------------------------------------------

_MIN_PROVENANCE_NGRAM = 4
"""Minimum n-gram length for provenance overlap detection."""

_MIN_PROVENANCE_PHRASE_LENGTH = 20
"""Minimum character length for distinctive phrase matching."""


@dataclass(frozen=True)
class ProvenanceCheckResult:
    """Outcome of query provenance validation.

    Checks whether the generated query reproduces docstring text
    verbatim or via high n-gram overlap, indicating the model is
    paraphrasing the docstring rather than reasoning about code behavior.
    """

    passed: bool
    violations: list[str] = field(default_factory=list)


def _extract_ngrams(text: str, n: int) -> set[str]:
    """Extract character n-grams from text."""
    words = text.lower().split()
    ngrams: set[str] = set()
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        ngrams.add(ngram)
    return ngrams


def _extract_distinctive_phrases(docstring: str) -> list[str]:
    """Extract distinctive phrases from docstring for provenance checks.

    Returns phrases that are long enough and specific enough to
    indicate provenance copying if they appear in the query.
    """
    if not docstring:
        return []

    phrases: list[str] = []
    for sentence in re.split(r"[.!?\n]", docstring):
        stripped = sentence.strip()
        # Skip very short, Sphinx directives, and non-alphabetic
        if len(stripped) < _MIN_PROVENANCE_PHRASE_LENGTH:
            continue
        if stripped.startswith(":"):
            continue
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in stripped) / len(
            stripped
        )
        if alpha_ratio < 0.7:
            continue
        phrases.append(stripped.lower())
    return phrases


def check_query_provenance(
    query: str,
    docstring: str,
) -> ProvenanceCheckResult:
    """Check if a query reproduces docstring text (provenance leakage).

    Detects three types of provenance leakage:
    1. Exact copy: query contains a docstring sentence verbatim
    2. High n-gram overlap: query shares many n-grams with docstring
    3. Distinctive phrase: query contains a distinctive docstring phrase

    Parameters
    ----------
    query:
        The generated query text.
    docstring:
        The source code docstring.

    Returns
    -------
    A ``ProvenanceCheckResult`` indicating pass/fail and violations.
    """
    if not docstring:
        return ProvenanceCheckResult(passed=True)

    violations: list[str] = []
    query_lower = query.lower().strip().rstrip(".")
    doc_lower = docstring.lower().strip()

    # 1. Exact copy: query is a substring of docstring or vice versa
    if len(query_lower) > 20 and query_lower in doc_lower:
        violations.append("Query reproduces docstring text (exact substring)")
    elif len(query_lower) > 20 and doc_lower in query_lower:
        violations.append("Query contains full docstring text")

    # 2. Distinctive phrase matching
    for phrase in _extract_distinctive_phrases(docstring):
        if phrase in query_lower:
            violations.append(
                f"Query contains distinctive docstring phrase: "
                f"'{phrase[:50]}...'"
            )
            break

    # 3. N-gram overlap (4-gram)
    if not violations:
        query_ngrams = _extract_ngrams(query_lower, _MIN_PROVENANCE_NGRAM)
        doc_ngrams = _extract_ngrams(doc_lower, _MIN_PROVENANCE_NGRAM)
        if query_ngrams and doc_ngrams:
            overlap = len(query_ngrams & doc_ngrams)
            total = min(len(query_ngrams), len(doc_ngrams))
            if total > 0 and overlap / total > 0.6:
                violations.append(
                    f"High n-gram overlap with docstring "
                    f"({overlap}/{total} = {overlap / total:.0%})"
                )

    return ProvenanceCheckResult(
        passed=len(violations) == 0,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Meta-query validation (strengthened)
# ---------------------------------------------------------------------------

_META_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Direct meta-task requests
    re.compile(r"\bi need a (technical )?query\b", re.IGNORECASE),
    re.compile(r"\bgenerate (a )?(retrieval )?query\b", re.IGNORECASE),
    re.compile(r"\bretrieve the source code\b", re.IGNORECASE),
    re.compile(r"\bfind the (source )?code for\b", re.IGNORECASE),
    re.compile(r"\bwrite (a )?retrieval query\b", re.IGNORECASE),
    re.compile(r"\bcreate (a )?query\b", re.IGNORECASE),
    re.compile(r"\bquery to (find|retrieve|get)\b", re.IGNORECASE),
    re.compile(
        r"\bhow (can|do) i (write|create|generate|find) (a )?query\b",
        re.IGNORECASE,
    ),
    # pytest meta-commands
    re.compile(r"\bpytest\s+--collect-only\b", re.IGNORECASE),
    re.compile(r"\bpytest\s+-\s*-\s*collect\s*-\s*only\b", re.IGNORECASE),
    # Task-inappropriate patterns
    re.compile(r"\bwrite (a )?(python )?(function|script|code)\b", re.IGNORECASE),
    re.compile(r"\bimplement (a )?(function|class|method)\b", re.IGNORECASE),
    re.compile(r"\bdef\s+\w+\s*\(", re.IGNORECASE),
    re.compile(r"\bclass\s+\w+\s*[\(:]", re.IGNORECASE),
    # Instruction-following patterns (model explaining what it will do)
    re.compile(r"\bhere is (a )?(the )?query\b", re.IGNORECASE),
    re.compile(r"\bthe (retrieval )?query is\b", re.IGNORECASE),
    re.compile(r"\bsure,?\s*(here|i can|i'll)\b", re.IGNORECASE),
    re.compile(r"\bokay,?\s*(here|i can|i'll)\b", re.IGNORECASE),
    re.compile(r"\bcertainly,?\s*(here|i can|i'll)\b", re.IGNORECASE),
    # Self-referential patterns
    re.compile(r"\bas (an )?ai\b", re.IGNORECASE),
    re.compile(r"\bas a (language )?model\b", re.IGNORECASE),
    re.compile(r"\bi (cannot|can't|am unable to)\b", re.IGNORECASE),
)


def check_meta_query(query: str) -> bool:
    """Check if a query is a meta-query (not a real retrieval query).

    Returns True if the query matches any meta-query pattern,
    indicating the model generated a meta-task response instead
    of a real retrieval query.
    """
    for pattern in _META_QUERY_PATTERNS:
        if pattern.search(query):
            return True
    return False


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
    provenance: ProvenanceCheckResult | None = None
    behavior_facts: StructuredBehaviorFacts | None = None


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

        Two-stage pipeline:
          Stage A: Extract StructuredBehaviorFacts from source code (AST-based)
          Stage B: Build docstring-blind prompt, generate query, validate

        Parameters
        ----------
        code_unit:
            The extracted code unit to generate a query for.

        Returns
        -------
        A ``QueryGenerationResult`` with the outcome and full attempt history.
        """
        # Stage A: Deterministic behavior extraction (no LLM, no docstring)
        behavior_facts = extract_behavior_facts(
            code_unit.source_code,
            imports=code_unit.context.imports if code_unit.context else None,
        )

        # Stage B: Docstring-blind prompt using behavior facts
        context = code_unit.context
        prompt = build_query_generation_prompt_v2(
            facts=behavior_facts,
            source_code=code_unit.source_code,
            symbol_type=code_unit.symbol_type,
            class_name=context.class_name if context else None,
            module_docstring=context.module_docstring if context else None,
            imports=context.imports if context else None,
            parent_methods=context.parent_methods if context else None,
        )
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

            # Leakage check
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
                    behavior_facts=behavior_facts,
                )

            # Provenance check (docstring-provenance leakage)
            provenance = check_query_provenance(
                candidate.query, code_unit.docstring
            )
            if not provenance.passed:
                logger.warning(
                    "Query failed provenance check: %s",
                    "; ".join(provenance.violations),
                )
                return QueryGenerationResult(
                    success=False,
                    attempts=retry_result.attempts,
                    model_name=self.model.name,
                    total_generation_ms=retry_result.total_generation_ms,
                    total_validation_ms=retry_result.total_validation_ms,
                    provenance=provenance,
                    behavior_facts=behavior_facts,
                )

            # Meta-query check
            if check_meta_query(candidate.query):
                logger.warning("Query is a meta-query")
                return QueryGenerationResult(
                    success=False,
                    attempts=retry_result.attempts,
                    model_name=self.model.name,
                    total_generation_ms=retry_result.total_generation_ms,
                    total_validation_ms=retry_result.total_validation_ms,
                    behavior_facts=behavior_facts,
                )

            return QueryGenerationResult(
                success=True,
                candidate=candidate,
                attempts=retry_result.attempts,
                model_name=self.model.name,
                total_generation_ms=retry_result.total_generation_ms,
                total_validation_ms=retry_result.total_validation_ms,
                leakage=leakage,
                provenance=provenance,
                behavior_facts=behavior_facts,
            )

        return QueryGenerationResult(
            success=False,
            attempts=retry_result.attempts,
            model_name=self.model.name,
            total_generation_ms=retry_result.total_generation_ms,
            total_validation_ms=retry_result.total_validation_ms,
            behavior_facts=behavior_facts,
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
