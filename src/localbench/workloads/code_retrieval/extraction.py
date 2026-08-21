"""AST-based Python code unit extraction.

Parses Python source files using the standard-library ``ast`` module
and produces validated extraction results per
DATASET_SPECIFICATION.md section 4.2.

Scope (Phase 4D):
- Python source discovery with deterministic ordering
- AST parsing with error isolation per file
- Top-level function and class-method extraction
- Nested-function exclusion
- 3-100 source-line validation
- Deterministic symbol-path and code-unit-ID construction
- SHA-256 content hashing

Out of scope:
- Semantic label generation
- Query generation
- Dataset splitting
- Retrieval / embeddings
- Benchmark execution
"""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from localbench.workloads.code_retrieval.schemas import (
    CodeUnitContext,
    Language,
    SymbolType,
    _count_source_lines,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_SOURCE_LINES = 3
_MAX_SOURCE_LINES = 100

_EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        "tests",
        "test",
        "vendor",
        "generated",
        ".git",
        "__pycache__",
        ".tox",
        "node_modules",
        ".eggs",
    }
)

_EXCLUDED_DIR_SUFFIXES: tuple[str, ...] = (".egg-info",)

_TEST_FILE_PREFIXES: tuple[str, ...] = ("test_",)
_TEST_FILE_SUFFIXES: tuple[str, ...] = ("_test.py",)

# Async function nodes share the same extraction interface.
_FUNCTION_NODE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


# ---------------------------------------------------------------------------
# Intermediate result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedCodeUnit:
    """Code unit produced by AST extraction -- not yet assigned to a split.

    Contains everything needed to construct a final ``CodeUnit``, except
    ``split`` which is assigned during dataset assembly.
    """

    repository: str
    language: Language
    file_path: str
    symbol: str
    symbol_type: SymbolType
    source_code: str
    context: CodeUnitContext
    source_url: str
    is_public: bool
    docstring: str
    source_file_lines: int
    content_hash: str
    extracted_at: str


@dataclass(frozen=True)
class SkippedFile:
    """Record of a file excluded by filter rules."""

    file_path: str
    reason: str


@dataclass(frozen=True)
class ParseError:
    """Record of a file that failed AST parsing."""

    file_path: str
    error_type: str
    message: str


@dataclass
class ExtractionResult:
    """Categorized result of extracting code units from a repository.

    Allows future phases to distinguish:
    - successfully extracted code units
    - files skipped by exclusion
    - files with no eligible units
    - files with parse errors
    """

    code_units: list[ExtractedCodeUnit] = field(default_factory=list)
    skipped_files: list[SkippedFile] = field(default_factory=list)
    empty_files: list[str] = field(default_factory=list)
    parse_errors: list[ParseError] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source file discovery
# ---------------------------------------------------------------------------


def discover_python_files(root: Path) -> list[Path]:
    """Find all eligible Python source files under *root*.

    Applies exclusion rules from DATASET_SPECIFICATION.md section 4.2:
    - Skips test, vendor, generated directories
    - Skips ``test_*.py`` and ``*_test.py`` files
    - Returns deterministic sorted order (repository-relative paths)

    Parameters
    ----------
    root:
        Root directory to search (typically ``RepositorySnapshot.local_path``).

    Returns
    -------
    Sorted list of repository-relative ``.py`` paths.
    """
    if not root.is_dir():
        return []

    py_files: list[Path] = []
    for path in root.rglob("*.py"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if _is_excluded_by_directory(rel):
            continue
        if _is_excluded_by_filename(rel):
            continue
        py_files.append(rel)

    return sorted(py_files)


def _is_excluded_by_directory(rel_path: Path) -> bool:
    """Check if any parent directory is in the exclusion set."""
    for part in rel_path.parts[:-1]:
        if part in _EXCLUDED_DIR_NAMES:
            return True
        if part.endswith(_EXCLUDED_DIR_SUFFIXES):
            return True
    return False


def _is_excluded_by_filename(rel_path: Path) -> bool:
    """Check if filename matches test-file exclusion patterns."""
    name = rel_path.name
    if name.startswith(_TEST_FILE_PREFIXES):
        return True
    if name.endswith(_TEST_FILE_SUFFIXES):
        return True
    return False


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _parse_source(source: str) -> ast.Module | None:
    """Parse Python source with ``ast``.  Returns ``None`` on failure."""
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return None


def _extract_module_docstring(tree: ast.Module) -> str:
    """Extract module-level docstring, if present."""
    return ast.get_docstring(tree) or ""


def _extract_imports(tree: ast.Module) -> list[str]:
    """Extract top-level import names from the module AST."""
    imports: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _get_source_segment(source_lines: list[str], node: ast.FunctionDef) -> str:
    """Extract source code for a function/method node, including decorators.

    Uses ``lineno`` and ``end_lineno`` from AST location metadata.
    """
    if node.decorator_list:
        start_line = node.decorator_list[0].lineno - 1
    else:
        start_line = node.lineno - 1

    end_line = node.end_lineno or node.lineno
    segment_lines = source_lines[start_line:end_line]
    return "".join(segment_lines)


def _is_private(name: str) -> bool:
    """Return ``True`` if *name* is private (underscore prefix, not dunder)."""
    return name.startswith("_") and not name.startswith("__")


def _build_symbol_path(
    function_name: str,
    class_name: str | None = None,
) -> str:
    """Construct a deterministic symbol path."""
    if class_name:
        return f"{class_name}.{function_name}"
    return function_name


def _build_code_unit_id(
    repository: str,
    file_path: str,
    symbol_path: str,
) -> str:
    """Construct a deterministic, globally unique code-unit identifier.

    Format: ``{repo_id}_py_{file_path}__{symbol_path}``

    Path separators and dots collapse to underscores; the ``__``
    delimiter marks the file/symbol boundary so identical symbol names
    in different modules cannot collide.
    """
    normalized_path = (
        file_path.replace("\\", "_").replace("/", "_").replace(".", "_")
    )
    normalized_symbol = symbol_path.replace(".", "_")
    return f"{repository}_py_{normalized_path}__{normalized_symbol}"


def _hash_source(source_code: str) -> str:
    """SHA-256 hash of normalised source code."""
    normalised = source_code.strip().encode("utf-8")
    return hashlib.sha256(normalised).hexdigest()


# ---------------------------------------------------------------------------
# Per-function processing
# ---------------------------------------------------------------------------


def _process_function(  # noqa: C901 -- intentionally flat
    node: ast.FunctionDef,
    source_lines: list[str],
    file_path: str,
    repository_id: str,
    base_url: str,
    extracted_at: str,
    class_name: str | None,
    module_docstring: str,
    module_imports: list[str],
    parent_methods: list[str],
) -> ExtractedCodeUnit | None:
    """Process a single ``FunctionDef`` node.  Returns ``None`` if ineligible."""

    if _is_private(node.name):
        return None

    source_code = _get_source_segment(source_lines, node)

    line_count = _count_source_lines(source_code)
    if line_count < _MIN_SOURCE_LINES:
        return None
    if line_count > _MAX_SOURCE_LINES:
        return None

    symbol_path = _build_symbol_path(node.name, class_name)
    symbol_type: SymbolType = "method" if class_name else "function"

    docstring = ast.get_docstring(node) or ""

    context = CodeUnitContext(
        class_name=class_name,
        module_docstring=module_docstring or None,
        imports=module_imports,
        parent_methods=parent_methods,
    )

    content_hash = _hash_source(source_code)

    if node.decorator_list:
        start_line = node.decorator_list[0].lineno
    else:
        start_line = node.lineno
    source_url = (
        f"{base_url.rstrip('/')}/blob/main/{file_path}#L{start_line}"
        if base_url
        else ""
    )

    return ExtractedCodeUnit(
        repository=repository_id,
        language="python",
        file_path=file_path,
        symbol=symbol_path,
        symbol_type=symbol_type,
        source_code=source_code,
        context=context,
        source_url=source_url,
        is_public=True,
        docstring=docstring,
        source_file_lines=len(source_lines),
        content_hash=content_hash,
        extracted_at=extracted_at,
    )


# ---------------------------------------------------------------------------
# Module-level extraction
# ---------------------------------------------------------------------------


def _extract_from_module(
    tree: ast.Module,
    source_lines: list[str],
    file_path: str,
    repository_id: str,
    base_url: str,
    extracted_at: str,
) -> list[ExtractedCodeUnit]:
    """Extract eligible functions and methods from a parsed module."""
    units: list[ExtractedCodeUnit] = []
    module_docstring = _extract_module_docstring(tree)
    module_imports = _extract_imports(tree)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            sibling_names = [
                n.name
                for n in ast.iter_child_nodes(node)
                if isinstance(n, _FUNCTION_NODE_TYPES)
            ]
            for item in ast.iter_child_nodes(node):
                if isinstance(item, _FUNCTION_NODE_TYPES):
                    parent_methods = [n for n in sibling_names if n != item.name]
                    unit = _process_function(
                        node=item,
                        source_lines=source_lines,
                        file_path=file_path,
                        repository_id=repository_id,
                        base_url=base_url,
                        extracted_at=extracted_at,
                        class_name=class_name,
                        module_docstring=module_docstring,
                        module_imports=module_imports,
                        parent_methods=parent_methods,
                    )
                    if unit is not None:
                        units.append(unit)

        elif isinstance(node, _FUNCTION_NODE_TYPES):
            unit = _process_function(
                node=node,
                source_lines=source_lines,
                file_path=file_path,
                repository_id=repository_id,
                base_url=base_url,
                extracted_at=extracted_at,
                class_name=None,
                module_docstring=module_docstring,
                module_imports=module_imports,
                parent_methods=[],
            )
            if unit is not None:
                units.append(unit)

    return units


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_code_units(
    local_path: Path,
    repository_id: str,
    commit: str,
    base_url: str = "",
) -> ExtractionResult:
    """Extract all eligible code units from a repository checkout.

    Pipeline: discover -> parse -> extract -> filter -> validate -> hash

    Parameters
    ----------
    local_path:
        Root of the checked-out repository.
    repository_id:
        Dataset-level identifier (e.g. ``"repo001"``).
    commit:
        Exact commit SHA (reserved for provenance; currently unused in
        output but required by the interface contract).
    base_url:
        Base URL for source links (e.g. ``"https://github.com/owner/repo"``).

    Returns
    -------
    ``ExtractionResult`` with categorized outcomes.
    """
    result = ExtractionResult()
    extracted_at = datetime.now(timezone.utc).isoformat()

    py_files = discover_python_files(local_path)

    for rel_path in py_files:
        abs_path = local_path / rel_path
        file_str = str(rel_path)

        try:
            source = abs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            result.parse_errors.append(
                ParseError(
                    file_path=file_str,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue

        source_lines = source.splitlines(keepends=True)

        tree = _parse_source(source)
        if tree is None:
            result.parse_errors.append(
                ParseError(
                    file_path=file_str,
                    error_type="SyntaxError",
                    message="Failed to parse source file",
                )
            )
            continue

        units = _extract_from_module(
            tree=tree,
            source_lines=source_lines,
            file_path=file_str,
            repository_id=repository_id,
            base_url=base_url,
            extracted_at=extracted_at,
        )

        if units:
            result.code_units.extend(units)
        else:
            result.empty_files.append(file_str)

    return result
