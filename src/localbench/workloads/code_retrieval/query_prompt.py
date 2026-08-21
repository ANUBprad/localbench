"""Deterministic prompt builder for retrieval query generation.

Constructs a versioned, reproducible prompt that requests a natural
language retrieval query describing a Python code unit's behaviour.

The prompt is independent of any specific model and must produce output
compatible with the ``CandidateQuery`` Pydantic schema.

Scope (Phase 4F):
- Prompt template definition
- Deterministic prompt construction
- Version tracking for reproducibility

Out of scope:
- Model invocation
- Validation
- Retry logic
- Dataset assembly
"""

from __future__ import annotations

from localbench.workloads.code_retrieval.schemas import QueryGenerationInput

QUERY_PROMPT_TEMPLATE_VERSION = "1.0.0"
"""Version of the query prompt template.  Increment when wording changes."""

_SYSTEM_PROMPT = (
    "You are a code search assistant. "
    "Given a Python function or method, generate a single natural language "
    "retrieval query that a developer might use to find this code.\n"
    "\n"
    "Rules:\n"
    "- The query must describe WHAT the code does, not its name or location.\n"
    "- Do NOT include: file paths, repository names, function names, "
    "class names, variable names, or any implementation-specific identifiers.\n"
    "- Use natural language a developer would type into a search bar.\n"
    "- The query should be specific enough to distinguish this code from "
    "unrelated functions.\n"
    "\n"
    "Output ONLY valid JSON with these fields:\n"
    "- query: string (the natural language retrieval query)\n"
    "- query_style: one of \"natural\", \"technical\", \"verbose\", \"concise\"\n"
    "- query_intent: a short descriptor like \"find_implementation\", "
    "\"find_error_handling\", \"find_optimization\", \"understand_behavior\", "
    "\"find_usage\"\n"
    "\n"
    "No markdown fences, no commentary."
)

_USER_TEMPLATE = (
    "Generate a retrieval query for the following Python {symbol_type}.\n"
    "\n"
    "{docstring_block}"
    "Source code:\n"
    "```python\n"
    "{source_code}\n"
    "```\n"
    "{context_block}"
    "\n"
    "Return a JSON object with: query, query_style, query_intent."
)


def _build_docstring_block(docstring: str) -> str:
    """Build the docstring section of the prompt."""
    if not docstring:
        return ""
    return f"Docstring: {docstring}\n\n"


def _build_context_block(
    class_name: str | None = None,
    module_docstring: str | None = None,
    imports: list[str] | None = None,
    parent_methods: list[str] | None = None,
) -> str:
    """Build the context section of the prompt.

    Returns an empty string if no context is available.
    """
    parts: list[str] = []
    if class_name:
        parts.append(f"Class: {class_name}")
    if module_docstring:
        parts.append(f"Module docstring: {module_docstring}")
    if imports:
        parts.append(f"Imports: {', '.join(imports)}")
    if parent_methods:
        parts.append(f"Sibling methods: {', '.join(parent_methods)}")
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def build_query_generation_prompt(input: QueryGenerationInput) -> str:  # noqa: A002
    """Build a deterministic prompt for retrieval query generation.

    The prompt requests structured JSON output compatible with the
    ``CandidateQuery`` Pydantic schema.  The same input always produces
    the same prompt (deterministic, no randomness).

    The prompt does NOT include: repository ID, file path, symbol path,
    source URL, content hash, or any SemanticLabel fields.

    Parameters
    ----------
    input:
        Source-only code unit information for query generation.

    Returns
    -------
    A string prompt ready to send to a ``LocalModel``.
    """
    docstring_block = _build_docstring_block(input.docstring)
    context_block = _build_context_block(
        class_name=input.class_name,
        module_docstring=input.module_docstring,
        imports=input.imports or None,
        parent_methods=input.parent_methods or None,
    )

    return _USER_TEMPLATE.format(
        symbol_type=input.symbol_type,
        docstring_block=docstring_block,
        source_code=input.source_code,
        context_block=context_block,
    )


def get_query_system_prompt() -> str:
    """Return the system prompt for retrieval query generation."""
    return _SYSTEM_PROMPT
