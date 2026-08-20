"""Deterministic prompt builder for semantic label generation.

Constructs a versioned, reproducible prompt that requests a structured
semantic description of a Python code unit.  The prompt is independent
of any specific model and must produce output compatible with the
``SemanticLabel`` Pydantic schema.

Scope (Phase 4E):
- Prompt template definition
- Deterministic prompt construction
- Version tracking for reproducibility

Out of scope:
- Model invocation
- Validation
- Retry logic
- Retrieval query generation
"""

from __future__ import annotations

from localbench.workloads.code_retrieval.extraction import ExtractedCodeUnit

PROMPT_TEMPLATE_VERSION = "1.0.0"
"""Version of the prompt template.  Increment when wording changes."""

_SYSTEM_PROMPT = (
    "You are a code analysis assistant. "
    "Given a Python function or method, produce a JSON object "
    "with the following fields:\n"
    "- code_unit_id: string (the provided identifier)\n"
    "- description: string (20-256 words, detailed semantic explanation)\n"
    "- summary: string (one sentence)\n"
    "- concepts: list of strings (2-10 semantic tags)\n"
    "- input_types: list of strings (parameter descriptions)\n"
    "- output_type: string (return type description)\n"
    "- side_effects: list of strings (state modifications)\n"
    "- created_by: always \"model_generated\"\n"
    "- label_version: always \"1.0.0\"\n"
    "\n"
    "Output ONLY valid JSON.  No markdown fences, no commentary."
)

_USER_TEMPLATE = (
    "Analyze the following Python {symbol_type} and produce a semantic "
    "label as JSON.\n"
    "\n"
    "Symbol: {symbol}\n"
    "File: {file_path}\n"
    "{context_block}"
    "\n"
    "Source code:\n"
    "```python\n"
    "{source_code}\n"
    "```\n"
    "\n"
    "Return a JSON object with: code_unit_id, description, summary, "
    "concepts, input_types, output_type, side_effects, created_by, "
    "label_version."
)


def build_context_block(
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


def build_semantic_label_prompt(code_unit: ExtractedCodeUnit) -> str:
    """Build a deterministic prompt for semantic label generation.

    The prompt requests structured JSON output compatible with the
    ``SemanticLabel`` Pydantic schema.  The same input always produces
    the same prompt (deterministic, no randomness).

    Parameters
    ----------
    code_unit:
        The extracted code unit to describe.

    Returns
    -------
    A string prompt ready to send to a ``LocalModel``.
    """
    context_block = build_context_block(
        class_name=code_unit.context.class_name,
        module_docstring=code_unit.context.module_docstring,
        imports=code_unit.context.imports or None,
        parent_methods=code_unit.context.parent_methods or None,
    )

    return _USER_TEMPLATE.format(
        symbol_type=code_unit.symbol_type,
        symbol=code_unit.symbol,
        file_path=code_unit.file_path,
        context_block=context_block,
        source_code=code_unit.source_code,
    )


def get_system_prompt() -> str:
    """Return the system prompt for semantic label generation."""
    return _SYSTEM_PROMPT
