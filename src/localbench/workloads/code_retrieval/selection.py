"""Deterministic final-45 selection (DATASET_SPECIFICATION.md §4.4.5).

Implements the frozen methodology resolution:

    generation -> automated eligibility -> deterministic selection of 45
    -> benchmark-blind human review of the selected 45 -> freeze -> ground truth

The eligible pool consists of successful terminal candidates that pass schema
validation, leakage screening, non-trivial-query checks, belong to the
canonical test split, and carry a unique ``code_unit_id``. Selection sorts the
pool by ``code_unit_id`` ascending (Python string ordering) and draws exactly
45 entries with ``random.Random(42).sample`` — uniform, without replacement,
quota-free, and blind to any benchmark information.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

SEED = 42
FINAL_QUERY_COUNT = 45
PRNG_ID = "python.random.Random"
SAMPLING_METHOD = "sample_without_replacement"
CANONICAL_ORDER = "code_unit_id_lexicographic_ascending"

REQUIRED_RECORD_FIELDS = (
    "selection_version",
    "seed",
    "prng",
    "python_version",
    "sampling_method",
    "canonical_order",
    "eligible_candidate_count",
    "eligible_pool_sha256",
    "selected_count",
    "selected_code_unit_ids",
    "selected_candidate_ids",
    "selected_repository_distribution",
    "selected_query_style_distribution",
    "generation_source_commit",
    "selection_created_utc",
)


class SelectionError(ValueError):
    """Raised when the eligible pool violates a frozen §4.4.5 precondition."""


def _utc_now_iso() -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return stamp.replace("+00:00", "Z")


def _require_unique_code_unit_ids(candidates: Sequence[dict[str, Any]]) -> None:
    counts = Counter(c["code_unit_id"] for c in candidates)
    duplicates = sorted(unit_id for unit_id, n in counts.items() if n > 1)
    if duplicates:
        raise SelectionError(f"duplicate code_unit_id values encountered: {duplicates}")


def build_eligible_pool(
    candidates: Sequence[dict[str, Any]],
    test_code_unit_ids: set[str],
    train_code_unit_ids: set[str] | None = None,
    validation_code_unit_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return eligible candidates in canonical order (§4.4.5).

    Eligibility requires: successful terminal candidate, schema validation
    passed, leakage screening passed, non-trivial query text, canonical test
    split membership, and a unique ``code_unit_id`` across all candidate
    records. Duplicate IDs abort with :class:`SelectionError`; they are never
    silently deduplicated.
    """
    _require_unique_code_unit_ids(candidates)
    excluded = (train_code_unit_ids or set()) | (validation_code_unit_ids or set())
    pool = [
        candidate
        for candidate in candidates
        if candidate.get("success") is True
        and candidate.get("validation_passed") is True
        and candidate.get("leakage_passed") is True
        and str(candidate.get("query", "")).strip()
        and candidate["code_unit_id"] in test_code_unit_ids
        and candidate["code_unit_id"] not in excluded
    ]
    return sorted(pool, key=lambda candidate: candidate["code_unit_id"])


def pool_hash(ordered_candidates: Sequence[dict[str, Any]]) -> str:
    """SHA-256 of the canonical pool representation (frozen serialization).

    Records are sorted by ``code_unit_id`` ascending, serialized with
    ``sort_keys=True``, ``separators=(",", ":")``, ``ensure_ascii=False``,
    joined with ``"\\n"``, UTF-8 encoded; no trailing newline participates.
    """
    payload = "\n".join(
        json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for candidate in sorted(ordered_candidates, key=lambda c: c["code_unit_id"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_final_queries(
    ordered_candidates: Sequence[dict[str, Any]],
    count: int = FINAL_QUERY_COUNT,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Draw ``count`` candidates uniformly without replacement using seed 42."""
    if count < 0:
        raise SelectionError(f"requested count must be non-negative, got {count}")
    if len(ordered_candidates) < count:
        raise SelectionError(
            f"eligible pool has {len(ordered_candidates)} candidates; {count} required"
        )
    return random.Random(seed).sample(list(ordered_candidates), count)


def distribution(values: Sequence[str]) -> dict[str, int]:
    """Deterministic observation counter (recorded, never enforced)."""
    return dict(sorted(Counter(values).items()))


def build_selection_record(
    *,
    ordered_pool: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    repository_by_code_unit_id: dict[str, str],
    generation_source_commit: str,
    selection_version: str,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Assemble the reproducible selection record (§4.4.5 field contract)."""
    created = created_utc or _utc_now_iso()
    repositories = [repository_by_code_unit_id[c["code_unit_id"]] for c in selected]
    styles = [c["query_style"] for c in selected]
    return {
        "selection_version": selection_version,
        "seed": SEED,
        "prng": PRNG_ID,
        "python_version": platform.python_version(),
        "sampling_method": SAMPLING_METHOD,
        "canonical_order": CANONICAL_ORDER,
        "eligible_candidate_count": len(ordered_pool),
        "eligible_pool_sha256": pool_hash(ordered_pool),
        "selected_count": len(selected),
        "selected_code_unit_ids": [c["code_unit_id"] for c in selected],
        "selected_candidate_ids": [c["candidate_id"] for c in selected],
        "selected_repository_distribution": distribution(repositories),
        "selected_query_style_distribution": distribution(styles),
        "generation_source_commit": generation_source_commit,
        "selection_created_utc": created,
    }
