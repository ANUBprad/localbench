"""Extract JSON from raw model output.

Models frequently wrap JSON in markdown fences, prepend conversational
text, or append commentary. This module isolates all the messy parsing
logic in one place.
"""

from __future__ import annotations

import json
import re

from localbench.runtime.generation.failures import MalformedJSONError

# Matches ```json ... ``` or ``` ... ``` fenced blocks.
_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n(.*?)```", re.DOTALL
)


def extract_json(raw: str) -> dict | list:
    """Extract a JSON object or array from raw model output.

    Strategies, tried in order:
    1. Parse the entire string directly.
    2. Strip markdown fences and parse.
    3. Find the first ``{...}`` or ``[...]`` block via brace matching.

    Raises MalformedJSON if no valid JSON can be extracted.
    """
    stripped = raw.strip()
    if not stripped:
        raise MalformedJSONError("Model output is empty.")

    # Strategy 1: entire string is JSON.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences.
    fence_match = _FENCE_RE.search(stripped)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: find first {…} or […] via brace counting.
    result = _find_json_block(stripped)
    if result is not None:
        return result

    raise MalformedJSONError(
        "Could not extract valid JSON from model output."
    )


def _find_json_block(text: str) -> dict | list | None:
    """Locate and parse the first balanced JSON block."""
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start = text.find(open_char)
        if start == -1:
            continue

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None
