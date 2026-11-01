"""Independent source-code audit records, separate from blind human review."""

from __future__ import annotations

import hashlib
import json


def source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def audit_template(review: dict) -> dict:
    """Generate an undecided template; never auto-approve source matches."""
    return {
        "artifact": "source_code_audit",
        "version": 1,
        "items": [
            {
                "candidate_id": item["candidate_id"],
                "code_unit_id": item["code_unit_id"],
                "query": item["query"],
                "source_sha256": source_digest(item["target"]["source_code"]),
                "status": "pending",
                "notes": "",
            }
            for item in review["items"]
            if item["review"]["state"] == "accepted"
        ],
    }


def validated_audit_ids(audit: dict, review: dict) -> set[str]:
    """Fail closed on malformed, stale, or unsupported audit decisions."""
    if set(audit) != {"artifact", "version", "items"}:
        raise ValueError("Invalid source audit fields")
    if audit["artifact"] != "source_code_audit" or audit["version"] != 1:
        raise ValueError("Unsupported source audit version")
    if not isinstance(audit["items"], list):
        raise ValueError("Audit items must be a list")
    accepted = {
        item["candidate_id"]: item
        for item in review["items"]
        if item["review"]["state"] == "accepted"
    }
    seen = set()
    passed = set()
    for entry in audit["items"]:
        if not isinstance(entry, dict) or set(entry) != {
            "candidate_id",
            "code_unit_id",
            "query",
            "source_sha256",
            "status",
            "notes",
        }:
            raise ValueError("Invalid audit entry fields")
        cid = entry["candidate_id"]
        if not isinstance(cid, str) or cid not in accepted or cid in seen:
            raise ValueError(f"Unknown or duplicate audit candidate: {cid}")
        seen.add(cid)
        item = accepted[cid]
        if (
            entry["code_unit_id"] != item["code_unit_id"]
            or entry["query"] != item["query"]
            or entry["source_sha256"] != source_digest(item["target"]["source_code"])
        ):
            raise ValueError(f"Stale or mismatched source audit: {cid}")
        if entry["status"] not in {"pending", "passed", "failed"}:
            raise ValueError(f"Invalid source audit status: {cid}")
        if not isinstance(entry["notes"], str):
            raise ValueError(f"Audit notes must be text: {cid}")
        if entry["status"] in {"passed", "failed"} and not entry["notes"].strip():
            raise ValueError(f"Decided audit requires explanation: {cid}")
        if entry["status"] == "passed":
            passed.add(cid)
    if seen != set(accepted):
        raise ValueError(f"Missing audit entries: {sorted(set(accepted) - seen)}")
    return passed


def write_template(review: dict, path) -> None:
    from pathlib import Path

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(audit_template(review), indent=2) + "\n",
        encoding="utf-8",
    )


REVIEW_CRITERIA = (
    "understandable",
    "non_trivial",
    "self_contained",
    "specific",
    "non_identifying",
    "retrievable",
)


def prepare_batch_audit_rows(
    review_rows: list[dict[str, str]],
    review_key: dict[str, str],
    candidates: dict[str, dict],
    units: dict[str, dict],
) -> list[dict[str, str]]:
    """Map a completed blind CSV to source-audit rows; never decide correctness."""
    review_ids = [row.get("review_id", "") for row in review_rows]
    if not all(review_ids) or len(set(review_ids)) != len(review_ids):
        raise ValueError("Review IDs must be present and unique")
    if set(review_ids) != set(review_key):
        raise ValueError("Review IDs do not match blind-review key")

    rows = []
    seen_candidates = set()
    for review in review_rows:
        review_id = review["review_id"]
        if any(review.get(field) not in {"0", "1"} for field in REVIEW_CRITERIA):
            raise ValueError(f"Incomplete review criteria: {review_id}")
        expected_decision = (
            "accept"
            if all(review[field] == "1" for field in REVIEW_CRITERIA)
            else "reject"
        )
        if review.get("decision") != expected_decision:
            raise ValueError(f"Inconsistent review decision: {review_id}")
        candidate_id = review_key[review_id]
        if candidate_id in seen_candidates:
            raise ValueError(f"Duplicate candidate mapping: {candidate_id}")
        seen_candidates.add(candidate_id)
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate not found: {candidate_id}")
        if review.get("query") != candidate.get("query"):
            raise ValueError(f"Query mismatch for {review_id}")
        if expected_decision != "accept":
            continue

        code_unit_id = candidate.get("code_unit_id")
        unit = units.get(code_unit_id)
        if unit is None:
            raise ValueError(f"CodeUnit not found for {review_id}: {code_unit_id}")
        source = unit.get("source_code")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Missing source code for {review_id}")
        rows.append(
            {
                "review_id": review_id,
                "candidate_id": candidate_id,
                "code_unit_id": code_unit_id,
                "query": review["query"],
                "repository": str(unit.get("repository", "")),
                "file_path": str(unit.get("file_path", "")),
                "symbol": str(unit.get("symbol", "")),
                "source_sha256": source_digest(source),
                "source_code": source,
                "source_verified": "",
                "source_evidence": "",
                "audit_notes": "",
            }
        )
    return rows

QUERY_REVISION_FIELDS = (
    "review_id",
    "candidate_id",
    "code_unit_id",
    "repository",
    "file_path",
    "symbol",
    "query",
    "source_sha256",
    "source_code",
    "source_verified",
    "source_evidence",
    "audit_notes",
)


def prepare_query_revision_rows(audit_rows):
    """Prepare a revision worksheet from completed, hash-checked source audits.

    Only failed source-audit rows are returned. A human must write and assess
    each revised query; this function never generates or approves query text.
    """
    seen_reviews = set()
    seen_candidates = set()
    revision_rows = []

    for row in audit_rows:
        missing = [field for field in QUERY_REVISION_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Source audit row is missing fields: {missing}")

        review_id = row["review_id"]
        candidate_id = row["candidate_id"]
        if not review_id or review_id in seen_reviews:
            raise ValueError(f"Missing or duplicate review_id: {review_id}")
        if not candidate_id or candidate_id in seen_candidates:
            raise ValueError(f"Missing or duplicate candidate_id: {candidate_id}")
        seen_reviews.add(review_id)
        seen_candidates.add(candidate_id)

        source = row["source_code"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Missing source code: {review_id}")
        if row["source_sha256"] != source_digest(source):
            raise ValueError(f"Source hash mismatch: {review_id}")
        if row["source_verified"] not in {"0", "1"}:
            raise ValueError(f"Source audit decision must be 0 or 1: {review_id}")

        if row["source_verified"] == "1":
            if not row["source_evidence"].strip():
                raise ValueError(f"Verified audit requires source evidence: {review_id}")
            continue

        if not row["audit_notes"].strip():
            raise ValueError(f"Failed audit requires audit notes: {review_id}")

        revision_rows.append(
            {
                "review_id": review_id,
                "candidate_id": candidate_id,
                "code_unit_id": row["code_unit_id"],
                "repository": row["repository"],
                "file_path": row["file_path"],
                "symbol": row["symbol"],
                "original_query": row["query"],
                "source_sha256": row["source_sha256"],
                "source_code": source,
                "source_evidence": row["source_evidence"],
                "audit_notes": row["audit_notes"],
                "revised_query": "",
                "revision_rationale": "",
                "revision_status": "pending",
            }
        )

    return revision_rows

