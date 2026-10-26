"""Ensure human review and independent relevance labels gate evaluation."""

import pytest

from localbench.workloads.code_retrieval.source_audit import audit_template

from localbench.workloads.code_retrieval.ground_truth_export import (
    export_adjudicated_cases,
)


def make_review():
    return {
        "artifact": "human_review",
        "review_version": "1.0.0",
        "basis": {},
        "items": [
            {
                "position": i,
                "candidate_id": f"c{i}",
                "code_unit_id": f"u{i}",
                "query": f"query {i}",
                "query_style": "natural",
                "query_intent": "",
                "target": {
                    "repository": "r", "file_path": "f",
                    "symbol": "s", "symbol_type": "function",
                    "docstring": "", "source_code": "def f(): pass",
                },
                "automated_validation": {
                    "validation_passed": True, "leakage_passed": True
                },
                "review": {
                    "state": "accepted" if i == 1 else "pending",
                    "notes": "", "decided_utc": None,
                },
            }
            for i in range(1, 46)
        ],
    }


def audited(review):
    audit = audit_template(review)
    for item in audit["items"]:
        item["status"] = "passed" if item["candidate_id"] == "c1" else "pending"
        if item["status"] == "passed":
            item["notes"] = "Verified against source code"
    return audit


def test_accepted_and_labeled_export():
    result = export_adjudicated_cases(
        make_review(),
        [{"candidate_id": "c1", "relevant_code_units": ["u1"]}],
        {f"u{i}" for i in range(1, 46)},\n        audit=audited(make_review()),\n    )
    assert result == [{
        "id": "c1", "query": "query 1", "relevant_code_units": ["u1"]
    }]


def test_missing_labels_rejected():
    with pytest.raises(ValueError, match="missing labels"):
        export_adjudicated_cases(
            make_review(), [], {f"u{i}" for i in range(1, 46)},\n            audit=audited(make_review())
        )


def test_pending_labels_rejected():
    with pytest.raises(ValueError, match="non-accepted"):
        export_adjudicated_cases(
            make_review(),
            [{"candidate_id": "c2", "relevant_code_units": ["u2"]}],
            {f"u{i}" for i in range(1, 46)},\n            audit=audited(make_review()),\n        )


def test_unknown_label_rejected():
    with pytest.raises(ValueError, match="unselected"):
        export_adjudicated_cases(
            make_review(),
            [{"candidate_id": "unknown", "relevant_code_units": ["u1"]}],
            {f"u{i}" for i in range(1, 46)},
        )


def test_no_accepted_cases_rejected():
    review = make_review()
    review["items"][0]["review"]["state"] = "pending"
    with pytest.raises(ValueError, match="No accepted"):
        export_adjudicated_cases(\n            review, [], {f"u{i}" for i in range(1, 46)},\n            audit=audit_template(review),\n        )
