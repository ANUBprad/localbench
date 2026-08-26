"""Deterministic prompt builder for retrieval query generation.

Constructs a versioned, reproducible prompt that requests a natural
language retrieval query describing a Python code unit's behaviour.

Two-stage prompt design (v3):
- Stage A: Deterministic AST-based behavior extraction (no LLM)
- Stage B: Docstring-blind prompt using StructuredBehaviorFacts only

The prompt is independent of any specific model and must produce output
compatible with the ``CandidateQuery`` Pydantic schema.

Docstring provenance constraint (v3):
The final Stage B prompt must NOT contain the original docstring text.
Source code (which may embed docstrings) and module docstrings are
also excluded to prevent the LLM from paraphrasing docstring content.

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

from localbench.workloads.code_retrieval.schemas import (
    QueryGenerationInput,
    StructuredBehaviorFacts,
)

QUERY_PROMPT_TEMPLATE_VERSION = "3.0.0"
"""Version of the query prompt template.  Increment when wording changes.

v3: Removed source_code and module_docstring from Stage B prompt
to eliminate docstring-provenance leakage.
"""

# ---------------------------------------------------------------------------
# System prompt (shared by both legacy and v2 modes)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a code search assistant. "
    "Given behavioral facts about a Python function or method, generate a single "
    "natural language retrieval query that a developer might use to find this code.\n"
    "\n"
    "CRITICAL - ANTI-LEAKAGE RULE (MUST FOLLOW):\n"
    "- NEVER use class names, method names, function names, parameter names, "
    "variable names, or any identifiers from the source code in your query.\n"
    "- NEVER reference the code's location (file path, module, repository).\n"
    "- Describe WHAT the code does (behavior, purpose, effect), not WHO or WHERE "
    "it is.\n"
    "- If you include any identifier from the source, the query is INVALID.\n"
    "\n"
    "Rules:\n"
    "- The query must describe WHAT the code does, not its name or location.\n"
    "- Do NOT include: file paths, repository names, function names, "
    "class names, variable names, or any implementation-specific identifiers.\n"
    "- Use natural language a developer would type into a search bar.\n"
    "- The query should be specific enough to distinguish this code from "
    "unrelated functions.\n"
    "- Do NOT reproduce the behavioral facts verbatim — rephrase them as "
    "a natural developer question.\n"
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

# ---------------------------------------------------------------------------
# v2 user template (docstring-blind, uses StructuredBehaviorFacts)
# ---------------------------------------------------------------------------

_USER_TEMPLATE_V2 = (
    "Generate a retrieval query for the following Python {symbol_type}.\n"
    "\n"
    "Behavioral Facts (extracted from source code analysis):\n"
    "- Purpose: {primary_purpose}\n"
    "- Inputs: {input_summary}\n"
    "- Output: {output_summary}\n"
    "{side_effects_block}"
    "{error_handling_block}"
    "{control_flow_block}"
    "{key_operations_block}"
    "{raises_block}"
    "{context_block}"
    "\n"
    "REMINDER: Do NOT use any class names, method names, or identifiers from "
    "the context above. Describe the behavior in plain language only.\n"
    "Do NOT copy the behavioral facts — rephrase as a natural question.\n"
    "\n"
    "Return a JSON object with: query, query_style, query_intent."
)

# ---------------------------------------------------------------------------
# Legacy user template (v1, for backward compatibility)
# ---------------------------------------------------------------------------

_USER_TEMPLATE_LEGACY = (
    "Generate a retrieval query for the following Python {symbol_type}.\n"
    "\n"
    "{docstring_block}"
    "Source code:\n"
    "```python\n"
    "{source_code}\n"
    "```\n"
    "{context_block}"
    "\n"
    "REMINDER: Do NOT use any class names, method names, or identifiers from "
    "the code or context above. Describe the behavior in plain language only.\n"
    "\n"
    "Return a JSON object with: query, query_style, query_intent."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _build_behavior_block(label: str, value: str) -> str:
    """Build a single-line behavior fact block."""
    if not value:
        return ""
    return f"- {label}: {value}\n"


def _build_list_block(label: str, items: list[str]) -> str:
    """Build a list behavior fact block."""
    if not items:
        return ""
    return f"- {label}: {', '.join(items)}\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_query_generation_prompt_v2(
    facts: StructuredBehaviorFacts,
    source_code: str,
    symbol_type: str = "function",
    class_name: str | None = None,
    module_docstring: str | None = None,
    imports: list[str] | None = None,
    parent_methods: list[str] | None = None,
) -> str:
    """Build a docstring-blind prompt using StructuredBehaviorFacts.

    This is the Stage B prompt: the model receives behavioral facts
    extracted by Stage A (AST-based, no docstring).  Neither source
    code nor docstrings are included in the final prompt text.

    Parameters
    ----------
    facts:
        Behavioral facts extracted by Stage A (``extract_behavior_facts``).
    source_code:
        Retained for backward compatibility but NOT included in the prompt.
    symbol_type:
        ``"function"`` or ``"method"``.
    class_name:
        Class name for context (optional).
    module_docstring:
        Retained for backward compatibility but NOT included in the prompt.
    imports:
        Import list for context (optional).
    parent_methods:
        Sibling method names for context (optional).

    Returns
    -------
    A string prompt ready to send to a ``LocalModel``.
    """
    side_effects_block = _build_list_block("Side effects", facts.side_effects)
    error_handling_block = _build_behavior_block(
        "Error handling", facts.error_handling
    )
    control_flow_block = _build_behavior_block("Control flow", facts.control_flow)
    key_operations_block = _build_list_block("Key operations", facts.key_operations)
    raises_block = _build_list_block("Raises", facts.raises)

    context_block = _build_context_block(
        class_name=class_name,
        module_docstring=None,
        imports=imports,
        parent_methods=parent_methods,
    )

    return _USER_TEMPLATE_V2.format(
        symbol_type=symbol_type,
        primary_purpose=facts.primary_purpose,
        input_summary=facts.input_summary,
        output_summary=facts.output_summary,
        side_effects_block=side_effects_block,
        error_handling_block=error_handling_block,
        control_flow_block=control_flow_block,
        key_operations_block=key_operations_block,
        raises_block=raises_block,
        context_block=context_block,
    )


def build_query_generation_prompt(input: QueryGenerationInput) -> str:  # noqa: A002
    """Build a deterministic prompt for retrieval query generation.

    Legacy v1 prompt — includes docstring in the prompt.  Kept for
    backward compatibility and testing.

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

    return _USER_TEMPLATE_LEGACY.format(
        symbol_type=input.symbol_type,
        docstring_block=docstring_block,
        source_code=input.source_code,
        context_block=context_block,
    )


def get_query_system_prompt() -> str:
    """Return the system prompt for retrieval query generation."""
    return _SYSTEM_PROMPT
