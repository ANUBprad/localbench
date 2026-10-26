"""Conservative, advisory checks for generated code-retrieval queries.

Flags are review hints, never evidence of correctness or relevance.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_GENERIC = (
    re.compile(r"\bpurpose,? inputs?,? and outputs?\b", re.I),
    re.compile(r"\bspecific domain concepts\b", re.I),
    re.compile(
        r"\bwhat (?:does|is) (?:this|the) (?:code|function|method) do\??$",
        re.I,
    ),
    re.compile(
        r"\bhow does (?:this|the) (?:code|function|method) work\??$",
        re.I,
    ),
)
_COUNT = re.compile(
    r"\b(?:exactly\s+)?(?:two|three|four|five|six|seven|eight|nine|ten|"
    r"\d+)\s+(?:method|function|api|serialization|database|network)\s+calls?\b",
    re.I,
)
_UNSUPPORTED = re.compile(
    r"\b(?:randomly|always|never|guarantees?|encrypts?|"
    r"authenticates?|serializ(?:es|ation))\b",
    re.I,
)


def assess_query(query: str) -> list[str]:
    """Return deterministic advisory flags, without rejecting the query."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    normalized = " ".join(query.split())
    flags: list[str] = []
    if not normalized:
        return ["empty_query"]
    words = re.findall(r"[A-Za-z0-9_]+", normalized)
    if len(words) < 8:
        flags.append("short_query")
    if any(pattern.search(normalized) for pattern in _GENERIC):
        flags.append("generic_template")
    if _COUNT.search(normalized):
        flags.append("exact_count_claim_review")
    if _UNSUPPORTED.search(normalized):
        flags.append("behavior_claim_review")
    return flags


def assess_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
    """Return an audit-friendly, non-mutating report row."""
    query = candidate.get("query")
    if not isinstance(query, str):
        raise ValueError("candidate.query must be a string")
    return {
        "candidate_id": candidate.get("candidate_id"),
        "code_unit_id": candidate.get("code_unit_id"),
        "flags": assess_query(query),
    }
