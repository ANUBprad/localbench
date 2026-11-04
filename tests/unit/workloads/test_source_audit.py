"""Independent source audit must reject stale, incomplete and undecided records."""

import pytest

from localbench.workloads.code_retrieval.source_audit import (
    audit_template,
    validated_audit_ids,
)


def review():
    return {
        "items": [
            {
                "candidate_id": "c1",
                "code_unit_id": "u1",
                "query": "Find color parser",
                "target": {"source_code": "def parse(): pass"},
                "review": {"state": "accepted"},
            }
        ]
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


def test_prepare_batch_audit_rows_maps_only_accepted_without_verifying():
    from localbench.workloads.code_retrieval.source_audit import (
        prepare_batch_audit_rows,
    )

    criteria = {
        "understandable": "1",
        "non_trivial": "1",
        "self_contained": "1",
        "specific": "1",
        "non_identifying": "1",
        "retrievable": "1",
    }
    reviews = [
        {
            "review_id": "R001",
            "query": "Find color parser behavior",
            **criteria,
            "decision": "accept",
        },
        {
            "review_id": "R002",
            "query": "A vague query",
            **{key: "0" for key in criteria},
            "decision": "reject",
        },
    ]
    rows = prepare_batch_audit_rows(
        reviews,
        {"R001": "c1", "R002": "c2"},
        {
            "c1": {
                "candidate_id": "c1",
                "code_unit_id": "u1",
                "query": "Find color parser behavior",
            },
            "c2": {
                "candidate_id": "c2",
                "code_unit_id": "u2",
                "query": "A vague query",
            },
        },
        {
            "u1": {
                "repository": "repo",
                "file_path": "colors.py",
                "symbol": "parse",
                "source_code": "def parse_color(value): return value",
            }
        },
    )
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "c1"
    assert rows[0]["source_sha256"]
    assert rows[0]["source_verified"] == ""
    assert rows[0]["source_evidence"] == ""


def test_prepare_batch_audit_rows_rejects_query_mismatch():
    from localbench.workloads.code_retrieval.source_audit import (
        prepare_batch_audit_rows,
    )

    review = {
        "review_id": "R001",
        "query": "Altered query",
        "understandable": "1",
        "non_trivial": "1",
        "self_contained": "1",
        "specific": "1",
        "non_identifying": "1",
        "retrievable": "1",
        "decision": "accept",
    }
    with pytest.raises(ValueError, match="Query mismatch"):
        prepare_batch_audit_rows(
            [review],
            {"R001": "c1"},
            {"c1": {"query": "Original query", "code_unit_id": "u1"}},
            {"u1": {"source_code": "def f(): pass"}},
        )


def test_prepare_batch_audit_rows_fails_on_missing_source():
    from localbench.workloads.code_retrieval.source_audit import (
        prepare_batch_audit_rows,
    )

    review = {
        "review_id": "R001",
        "query": "Find color parser behavior",
        "understandable": "1",
        "non_trivial": "1",
        "self_contained": "1",
        "specific": "1",
        "non_identifying": "1",
        "retrievable": "1",
        "decision": "accept",
    }
    with pytest.raises(ValueError, match="Missing source code"):
        prepare_batch_audit_rows(
            [review],
            {"R001": "c1"},
            {"c1": {"query": review["query"], "code_unit_id": "u1"}},
            {"u1": {"source_code": ""}},
        )


def _revision_row(review_id="R001", candidate_id="c1", source_verified="0"):
    from localbench.workloads.code_retrieval.source_audit import source_digest

    source = "def parse_color(value): return value"
    return {
        "review_id": review_id,
        "candidate_id": candidate_id,
        "code_unit_id": "u1",
        "repository": "repo",
        "file_path": "colors.py",
        "symbol": "parse_color",
        "query": "What does this function do?",
        "source_sha256": source_digest(source),
        "source_code": source,
        "source_verified": source_verified,
        "source_evidence": "lines 1-1" if source_verified == "1" else "",
        "audit_notes": "Query overstates behavior" if source_verified == "0" else "",
    }


def test_prepare_query_revision_rows_includes_only_failed_audits():
    from localbench.workloads.code_retrieval.source_audit import (
        prepare_query_revision_rows,
    )

    failed = _revision_row()
    passed = _revision_row("R002", "c2", "1")
    rows = prepare_query_revision_rows([failed, passed])
    assert len(rows) == 1
    assert rows[0]["review_id"] == "R001"
    assert rows[0]["original_query"] == failed["query"]
    assert rows[0]["revised_query"] == ""
    assert rows[0]["revision_status"] == "pending"


def test_prepare_query_revision_rows_rejects_stale_source_hash():
    from localbench.workloads.code_retrieval.source_audit import (
        prepare_query_revision_rows,
    )

    row = _revision_row()
    row["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_query_revision_rows([row])


def test_prepare_query_revision_rows_requires_decision_and_failure_notes():
    from localbench.workloads.code_retrieval.source_audit import (
        prepare_query_revision_rows,
    )

    row = _revision_row()
    row["source_verified"] = ""
    with pytest.raises(ValueError, match="decision must be 0 or 1"):
        prepare_query_revision_rows([row])

    row = _revision_row()
    row["audit_notes"] = ""
    with pytest.raises(ValueError, match="requires audit notes"):
        prepare_query_revision_rows([row])


def test_prepare_query_revision_rows_requires_evidence_for_passed_rows():
    from localbench.workloads.code_retrieval.source_audit import (
        prepare_query_revision_rows,
    )

    row = _revision_row(source_verified="1")
    row["source_evidence"] = ""
    with pytest.raises(ValueError, match="requires source evidence"):
        prepare_query_revision_rows([row])
