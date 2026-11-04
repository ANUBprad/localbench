"""Tests for fail-closed query revision assessment."""

import pytest

from localbench.workloads.code_retrieval.query_revision import (
    assess_query_revision_rows,
)
from localbench.workloads.code_retrieval.source_audit import source_digest


def revision_row(**overrides):
    source = "def parse_color(value): return value"
    row = {
        "review_id": "R001",
        "candidate_id": "c1",
        "code_unit_id": "u1",
        "repository": "repo",
        "file_path": "colors.py",
        "symbol": "parse_color",
        "original_query": "Describe this function.",
        "source_sha256": source_digest(source),
        "source_code": source,
        "source_evidence": "colors.py:1",
        "audit_notes": "Original query was too vague.",
        "revised_query": (
            "How does parse_color convert the supplied value into its "
            "normalized color representation?"
        ),
        "revision_rationale": "Names the symbol and asks about its behavior.",
        "revision_status": "ready_for_review",
    }
    row.update(overrides)
    return row


def test_assesses_completed_revision_without_approving_it():
    result = assess_query_revision_rows([revision_row()])
    assert result[0]["review_status"] == "requires_manual_review"
    assert result[0]["quality_flags"] == []


def test_rejects_stale_source_hash():
    with pytest.raises(ValueError, match="hash mismatch"):
        assess_query_revision_rows([revision_row(source_sha256="0" * 64)])


def test_requires_explicit_ready_status():
    with pytest.raises(ValueError, match="ready_for_review"):
        assess_query_revision_rows([revision_row(revision_status="pending")])


def test_rejects_unchanged_query():
    with pytest.raises(ValueError, match="unchanged"):
        assess_query_revision_rows(
            [revision_row(revised_query="Describe this function.")]
        )


def test_flags_duplicate_revised_queries():
    first = revision_row()
    second = revision_row(
        review_id="R002",
        candidate_id="c2",
        revised_query=first["revised_query"],
    )
    result = assess_query_revision_rows([first, second])
    assert result[0]["quality_flags"] == []
    assert "duplicate_query" in result[1]["quality_flags"]
    assert result[1]["review_status"] == "requires_manual_review"
