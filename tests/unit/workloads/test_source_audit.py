"""Independent source audit must reject stale, incomplete and undecided records."""

import pytest

from localbench.workloads.code_retrieval.source_audit import (
    audit_template,
    validated_audit_ids,
)


def review():
    return {
        "items": [{
            "candidate_id": "c1",
            "code_unit_id": "u1",
            "query": "Find color parser",
            "target": {"source_code": "def parse(): pass"},
            "review": {"state": "accepted"},
        }]
    }


def test_template_pending():
    audit = audit_template(review())
    assert audit["items"][0]["status"] == "pending"
    assert validated_audit_ids(audit, review()) == set()


def test_passed_requires_notes():
    audit = audit_template(review())
    audit["items"][0]["status"] = "passed"
    with pytest.raises(ValueError, match="explanation"):
        validated_audit_ids(audit, review())
    audit["items"][0]["notes"] = "Verified against code"
    assert validated_audit_ids(audit, review()) == {"c1"}


def test_stale_source_rejected():
    audit = audit_template(review())
    modified = review()
    modified["items"][0]["target"]["source_code"] = "different"
    with pytest.raises(ValueError, match="Stale"):
        validated_audit_ids(audit, modified)


def test_missing_entry_rejected():
    audit = audit_template(review())
    audit["items"].clear()
    with pytest.raises(ValueError, match="Missing"):
        validated_audit_ids(audit, review())
