"""Conservative, advisory checks for generated code-retrieval queries.

Flags are review hints, never evidence of correctness or relevance.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher

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


def _normalize_for_comparison(query: str) -> str:
    """Ignore case, punctuation, and whitespace in duplicate comparisons."""
    return " ".join(re.findall(r"[a-z0-9_]+", query.casefold()))


def assess_batch(
    candidates: Sequence[Mapping[str, object]],
    *,
    similarity_threshold: float = 0.90,
) -> list[dict[str, object]]:
    """Assess candidates in order, with advisory duplicate flags.

    Each duplicate points to the most similar earlier candidate. Equal scores
    prefer the earliest candidate. This does not mutate or filter the input.
    """
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")

    rows: list[dict[str, object]] = []
    previous: list[tuple[object, str]] = []
    for candidate in candidates:
        row = assess_candidate(candidate)
        query = candidate["query"]
        assert isinstance(query, str)
        normalized = _normalize_for_comparison(query)

        best_index = -1
        best_score = -1.0
        if normalized:
            for index, (_, earlier) in enumerate(previous):
                if not earlier:
                    continue
                score = SequenceMatcher(
                    None, normalized, earlier, autojunk=False
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_index = index

        if best_index >= 0 and best_score >= similarity_threshold:
            row["flags"].append(
                "duplicate_query" if best_score == 1.0 else "near_duplicate_query"
            )
            row["similar_to_candidate_id"] = previous[best_index][0]
            row["similarity_score"] = round(best_score, 4)

        rows.append(row)
        previous.append((candidate.get("candidate_id"), normalized))
    return rows
