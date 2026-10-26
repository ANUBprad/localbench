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
        item["candidate_id"]: item for item in review["items"]
        if item["review"]["state"] == "accepted"
    }
    seen = set()
    passed = set()
    for entry in audit["items"]:
        if not isinstance(entry, dict) or set(entry) != {
            "candidate_id", "code_unit_id", "query",
            "source_sha256", "status", "notes",
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
            or entry["source_sha256"]
            != source_digest(item["target"]["source_code"])
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
