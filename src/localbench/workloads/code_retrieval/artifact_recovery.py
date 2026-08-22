"""Deterministic recovery planning for duplicated candidate artifacts.

Phase 4F-I-B generation ran in two overlapping processes before the
fail-fast generation lock existed. Both JSONL artifacts therefore hold
duplicate CodeUnit records, and four CodeUnits appear in both files.
This module turns such an artifact pair into either a loadable
checkpoint plan or an explicit conflict report — never a silent guess.

Canonical selection rule (per CodeUnit across both files):
- Successful records with identical semantic payloads (candidate_id,
  query, query_style, query_intent) are duplicate executions of one
  candidate; the earliest ``completed_utc`` wins because it is the
  original successful execution under the frozen methodology. Ties
  break on the canonical JSON form of the record, so selection never
  depends on JSONL ordering.
- Successful records whose semantic payloads differ cannot be resolved:
  temperature-0.7 executions are equally valid and no frozen rule
  prefers one, so planning refuses with ``semantic_candidate_conflict``.
- Duplicate failure records keep the latest execution (the final
  failure state).
- A failure that predates the latest success is historical audit
  trail; the success is canonical and the failure is archived as
  superseded.
- A failure executed after the latest success leaves the unit's
  current state ambiguous; planning refuses with
  ``failure_supersedes_success``.

Recovered output is sorted by ``code_unit_id``: identical logical input
always serializes to identical bytes regardless of input line order.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from localbench.workloads.code_retrieval.candidate_store import (
    CandidateArtifactError,
    validate_record,
)

SEMANTIC_FIELDS = (
    "code_unit_id",
    "candidate_id",
    "query",
    "query_style",
    "query_intent",
)

METHOD_FIELDS = (
    "model",
    "model_version",
    "prompt_version",
    "seed",
    "temperature",
    "top_p",
    "max_tokens",
)


class ArtifactRecoveryError(Exception):
    """Candidate artifacts cannot be safely recovered."""


@dataclass(frozen=True)
class Conflict:
    kind: str
    code_unit_id: str
    detail: str


@dataclass
class RecoveryPlan:
    candidates: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    superseded: list[dict] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)

    @property
    def recoverable(self) -> bool:
        return not self.conflicts


def read_jsonl(path: Path) -> list[dict]:
    """Strictly parse *path* into records; any defect aborts recovery."""
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactRecoveryError(
                    f"{path}:{number} is not valid JSON: {exc}"
                ) from exc
            try:
                validate_record(record)
            except CandidateArtifactError as exc:
                raise ArtifactRecoveryError(
                    f"{path}:{number}: {exc}"
                ) from exc
            records.append(record)
    return records


def _order_key(record: dict) -> tuple[str, str]:
    """Execution time first, then content — order-independent ranking."""
    return record.get("completed_utc", ""), json.dumps(
        record, sort_keys=True
    )


def _semantic_signature(record: dict) -> tuple:
    return tuple(record.get(name) for name in SEMANTIC_FIELDS)


def _method_signature(record: dict) -> tuple:
    return tuple(record.get(name) for name in METHOD_FIELDS)


def file_stats(records: list[dict]) -> dict:
    counts = Counter(record["code_unit_id"] for record in records)
    duplicates = {uid: n for uid, n in counts.items() if n > 1}
    return {
        "records": len(records),
        "unique_ids": len(counts),
        "duplicate_groups": len(duplicates),
        "max_multiplicity": max(duplicates.values(), default=0),
        "extra_records": len(records) - len(counts),
    }


def cross_file_overview(candidates: list[dict], failures: list[dict]) -> dict:
    """Classify every CodeUnit that appears in both artifacts."""
    candidate_ids = {record["code_unit_id"] for record in candidates}
    failure_ids = {record["code_unit_id"] for record in failures}
    verdicts = {}
    for unit_id in sorted(candidate_ids & failure_ids):
        latest_success = max(
            record.get("completed_utc", "")
            for record in candidates
            if record["code_unit_id"] == unit_id
        )
        latest_failure = max(
            record.get("completed_utc", "")
            for record in failures
            if record["code_unit_id"] == unit_id
        )
        verdicts[unit_id] = (
            "success_is_later"
            if latest_success > latest_failure
            else "failure_is_later"
        )
    return {
        "only_candidates": sorted(candidate_ids - failure_ids),
        "only_failures": sorted(failure_ids - candidate_ids),
        "both": sorted(candidate_ids & failure_ids),
        "verdicts": verdicts,
    }


def plan_recovery(
    candidates: list[dict], failures: list[dict]
) -> RecoveryPlan:
    """Build canonical replacement sets or an explicit conflict report."""
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"success": [], "failure": []}
    )
    for record in candidates:
        grouped[record["code_unit_id"]]["success"].append(record)
    for record in failures:
        grouped[record["code_unit_id"]]["failure"].append(record)

    plan = RecoveryPlan()
    for unit_id in sorted(grouped):
        successes = grouped[unit_id]["success"]
        fails = grouped[unit_id]["failure"]

        expected_candidate_id = f"candidate_{unit_id}"
        mismatched = [
            record
            for record in successes + fails
            if record.get("candidate_id") != expected_candidate_id
        ]
        if mismatched:
            plan.conflicts.append(
                Conflict(
                    kind="candidate_id_mismatch",
                    code_unit_id=unit_id,
                    detail=(
                        f"{len(mismatched)} record(s) disagree with "
                        f"'{expected_candidate_id}'"
                    ),
                )
            )
            continue

        payload_signatures = {_semantic_signature(r) for r in successes}
        method_signatures = {_method_signature(r) for r in successes}
        if len(payload_signatures) > 1:
            plan.conflicts.append(
                Conflict(
                    kind="semantic_candidate_conflict",
                    code_unit_id=unit_id,
                    detail=(
                        f"{len(payload_signatures)} distinct successful "
                        "payloads"
                    ),
                )
            )
            continue
        if len(method_signatures) > 1:
            plan.conflicts.append(
                Conflict(
                    kind="methodology_mismatch",
                    code_unit_id=unit_id,
                    detail=(
                        "duplicate executions disagree on frozen "
                        "generation parameters"
                    ),
                )
            )
            continue

        if not successes:
            canonical_failure = max(fails, key=_order_key)
            plan.failures.append(canonical_failure)
            plan.superseded.extend(
                record
                for record in fails
                if record is not canonical_failure
            )
            continue

        if fails:
            latest_success = max(
                record.get("completed_utc", "") for record in successes
            )
            latest_failure = max(
                record.get("completed_utc", "") for record in fails
            )
            if latest_failure > latest_success:
                plan.conflicts.append(
                    Conflict(
                        kind="failure_supersedes_success",
                        code_unit_id=unit_id,
                        detail=(
                            f"failure completed {latest_failure} after "
                            f"success {latest_success}"
                        ),
                    )
                )
                continue

        canonical_success = min(successes, key=_order_key)
        plan.candidates.append(canonical_success)
        plan.superseded.extend(
            record
            for record in successes
            if record is not canonical_success
        )
        plan.superseded.extend(fails)
    return plan


def serialize_records(records: list[dict]) -> str:
    """Deterministic JSONL text ordered by CodeUnit ID."""
    seen: set[str] = set()
    lines: list[str] = []
    for record in sorted(records, key=lambda r: r["code_unit_id"]):
        unit_id = record["code_unit_id"]
        if unit_id in seen:
            raise ArtifactRecoveryError(
                f"Internal error: recovered set still contains a "
                f"duplicate '{unit_id}'."
            )
        seen.add(unit_id)
        lines.append(json.dumps(record, ensure_ascii=False))
    return "".join(line + "\n" for line in lines)
