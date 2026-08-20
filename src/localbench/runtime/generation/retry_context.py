"""Corrective prompt builder for structured generation retries.

When structured validation fails, this module constructs a corrective
prompt suffix that tells the model exactly what went wrong and asks it
to retry. The output is deterministic and testable.
"""

from __future__ import annotations

from localbench.runtime.generation.failures import StructuredError


def build_corrective_prompt(errors: list[StructuredError]) -> str:
    """Build a corrective prompt suffix from structured failure list.

    Produces a deterministic, model-facing message naming each error.
    """
    if not errors:
        return ""
    lines = ["Previous attempt failed validation:"]
    for err in errors:
        lines.append(f"- {err}")
    lines.append(
        "Please produce valid JSON matching the schema exactly."
    )
    return "\n".join(lines)


def build_retry_prompt(
    original_prompt: str,
    errors: list[StructuredError],
) -> str:
    """Combine the original prompt with corrective context for retry.

    Returns the full prompt to send to the model on a retry attempt.
    """
    suffix = build_corrective_prompt(errors)
    if not suffix:
        return original_prompt
    return f"{original_prompt}\n\n{suffix}"
