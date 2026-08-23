"""Benchmark-blind human-review workflow for the selected 45 (§4.4.4/§4.4.5).

Builds and validates the human-review artifact covering exactly the
deterministically selected queries.  The artifact is blind by construction:
its schema allowlist makes benchmark metrics, model rankings, and ground-truth
relevance structurally impossible to include.  Reviewer decisions are stored
per item as ``pending`` / ``accepted`` / ``rejected`` plus optional notes and
a metadata-only decision timestamp.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

FINAL_QUERY_COUNT = 45
REVIEW_VERSION = "1.0.0"

PENDING = "pending"
ACCEPTED = "accepted"
REJECTED = "rejected"
REVIEW_STATES = (PENDING, ACCEPTED, REJECTED)

_ITEM_KEYS = frozenset(
    {
        "position",
        "candidate_id",
        "code_unit_id",
        "query",
        "query_style",
        "query_intent",
        "target",
        "automated_validation",
        "review",
    }
)
_TARGET_KEYS = frozenset(
    {
        "repository",
        "file_path",
        "symbol",
        "symbol_type",
        "docstring",
        "source_code",
    }
)
_AUTOMATED_KEYS = frozenset({"validation_passed", "leakage_passed"})
_REVIEW_KEYS = frozenset({"state", "notes", "decided_utc"})
_TOP_LEVEL_KEYS = frozenset({"artifact", "review_version", "basis", "items"})

_FORBIDDEN_KEY_MARKERS = (
    "hit",
    "mrr",
    "latency",
    "throughput",
    "ranking",
    "benchmark",
    "relevanc",
    "ground_truth",
    "score",
)


class ReviewArtifactError(ValueError):
    """Raised when a review artifact violates the §4.4.4/§4.4.5 contract."""


def _reject_forbidden_keys(keys: Sequence[str], where: str) -> None:
    for key in keys:
        lowered = key.lower()
        if any(marker in lowered for marker in _FORBIDDEN_KEY_MARKERS):
            raise ReviewArtifactError(
                f"forbidden field '{key}' in {where}: benchmark or ground-truth "
                "information may not enter the review artifact"
            )


def build_review_item(
    position: int,
    candidate: dict[str, Any],
    unit: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one benchmark-blind review item in canonical order."""
    return {
        "position": position,
        "candidate_id": candidate["candidate_id"],
        "code_unit_id": candidate["code_unit_id"],
        "query": candidate["query"],
        "query_style": candidate["query_style"],
        "query_intent": candidate.get("query_intent", ""),
        "target": {
            "repository": unit["repository"],
            "file_path": unit["file_path"],
            "symbol": unit["symbol"],
            "symbol_type": unit["symbol_type"],
            "docstring": unit.get("docstring") or "",
            "source_code": unit["source_code"],
        },
        "automated_validation": {
            "validation_passed": bool(candidate["validation_passed"]),
            "leakage_passed": bool(candidate["leakage_passed"]),
        },
        "review": {
            "state": PENDING,
            "notes": "",
            "decided_utc": None,
        },
    }


def build_review_artifact(
    selection_record: dict[str, Any],
    candidates_by_id: dict[str, dict[str, Any]],
    units_by_id: dict[str, Any],
    test_code_unit_ids: set[str],
    train_code_unit_ids: set[str] | None = None,
    validation_code_unit_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build the review artifact for the selection record's 45 candidates.

    Item order equals the canonical selection order.  Raises
    :class:`ReviewArtifactError` when identities are missing/duplicated or any
    selected CodeUnit falls outside the canonical test split.
    """
    candidate_ids = selection_record["selected_candidate_ids"]
    code_unit_ids = selection_record["selected_code_unit_ids"]
    if (
        len(candidate_ids) != FINAL_QUERY_COUNT
        or len(code_unit_ids) != FINAL_QUERY_COUNT
    ):
        raise ReviewArtifactError(
            f"selection record must contain exactly {FINAL_QUERY_COUNT} entries, "
            f"got {len(candidate_ids)} candidate ids / {len(code_unit_ids)} unit ids"
        )
    excluded = (train_code_unit_ids or set()) | (validation_code_unit_ids or set())
    items: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    seen_units: set[str] = set()
    for position, (candidate_id, unit_id) in enumerate(
        zip(candidate_ids, code_unit_ids, strict=True), start=1
    ):
        if candidate_id in seen_candidates or unit_id in seen_units:
            raise ReviewArtifactError(
                f"duplicate identity at position {position}: {candidate_id}/{unit_id}"
            )
        seen_candidates.add(candidate_id)
        seen_units.add(unit_id)
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            raise ReviewArtifactError(f"candidate not found: {candidate_id}")
        mapped = candidate["code_unit_id"]
        if mapped != unit_id:
            raise ReviewArtifactError(
                f"candidate/unit mismatch at position {position}: "
                f"{candidate_id} maps to {mapped}, expected {unit_id}"
            )
        unit = units_by_id.get(unit_id)
        if unit is None:
            raise ReviewArtifactError(f"CodeUnit not found: {unit_id}")
        if unit_id not in test_code_unit_ids:
            raise ReviewArtifactError(
                f"CodeUnit outside canonical test split: {unit_id}"
            )
        if unit_id in excluded:
            raise ReviewArtifactError(
                f"CodeUnit also present in train/validation split: {unit_id}"
            )
        items.append(build_review_item(position, candidate, unit))
    basis = {
        "selection_version": selection_record["selection_version"],
        "eligible_pool_sha256": selection_record["eligible_pool_sha256"],
        "generation_source_commit": selection_record["generation_source_commit"],
        "selection_created_utc": selection_record["selection_created_utc"],
        "selected_count": len(items),
    }
    return {
        "artifact": "human_review",
        "review_version": REVIEW_VERSION,
        "basis": basis,
        "items": items,
    }


def validate_review_artifact(artifact: dict[str, Any]) -> list[str]:
    """Validate artifact structure; return a list of contract violations."""
    errors: list[str] = []
    _reject_forbidden_keys(sorted(artifact), "top level")
    if set(artifact) != _TOP_LEVEL_KEYS:
        unexpected = sorted(set(artifact) - _TOP_LEVEL_KEYS)
        missing = sorted(_TOP_LEVEL_KEYS - set(artifact))
        errors.append(
            f"top-level keys mismatch; unexpected={unexpected} missing={missing}"
        )
        return errors
    items = artifact["items"]
    if len(items) != FINAL_QUERY_COUNT:
        errors.append(f"expected {FINAL_QUERY_COUNT} review items, found {len(items)}")
    seen_units: set[str] = set()
    for index, item in enumerate(items, start=1):
        where = f"item {index}"
        _reject_forbidden_keys(sorted(item), where)
        if set(item) != _ITEM_KEYS:
            errors.append(f"{where}: key set mismatch: {sorted(set(item))}")
            continue
        if item["position"] != index:
            errors.append(
                f"{where}: position {item['position']} breaks canonical order"
            )
        if item["code_unit_id"] in seen_units:
            errors.append(f"{where}: duplicate code_unit_id {item['code_unit_id']}")
        seen_units.add(item["code_unit_id"])
        if set(item["target"]) != _TARGET_KEYS:
            mismatch = sorted(set(item["target"]))
            errors.append(f"{where}: target key mismatch: {mismatch}")
        automated = item["automated_validation"]
        if set(automated) != _AUTOMATED_KEYS:
            errors.append(f"{where}: automated_validation key mismatch")
        elif not (automated["validation_passed"] and automated["leakage_passed"]):
            errors.append(f"{where}: automated gates must have passed pre-selection")
        review = item["review"]
        if set(review) != _REVIEW_KEYS:
            errors.append(f"{where}: review key mismatch: {sorted(set(review))}")
            continue
        state = review["state"]
        if state not in REVIEW_STATES:
            errors.append(f"{where}: invalid review state '{state}'")
            continue
        notes = review["notes"]
        if not isinstance(notes, str):
            errors.append(f"{where}: notes must be a string")
        if state == REJECTED and not (isinstance(notes, str) and notes.strip()):
            errors.append(f"{where}: rejected items require an auditable reason")
        decided = review["decided_utc"]
        if state == PENDING and decided is not None:
            errors.append(f"{where}: pending item carries a decision timestamp")
        if decided is not None and not isinstance(decided, str):
            errors.append(f"{where}: decided_utc must be null or an ISO-8601 string")
    return errors


def review_progress(artifact: dict[str, Any]) -> dict[str, int]:
    """Count items per review state."""
    counts = {state: 0 for state in REVIEW_STATES}
    for item in artifact["items"]:
        counts[item["review"]["state"]] += 1
    counts["total"] = len(artifact["items"])
    return counts
