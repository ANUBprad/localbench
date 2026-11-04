"""Validation and advisory assessment for manually revised retrieval queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .query_quality import assess_batch
from .source_audit import source_digest

REVISION_REQUIRED_FIELDS = (
    "review_id",
    "candidate_id",
    "code_unit_id",
    "repository",
    "file_path",
    "symbol",
    "original_query",
    "source_sha256",
    "source_code",
    "source_evidence",
    "audit_notes",
    "revised_query",
    "revision_rationale",
    "revision_status",
)


def assess_query_revision_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Validate completed revisions and attach advisory quality flags.

    Every row must be explicitly marked ready_for_review. This is not approval:
    source correctness and retrieval performance still require separate review.
    """
    if not rows:
        raise ValueError("Query revision worksheet is empty")

    seen_reviews: set[str] = set()
    seen_candidates: set[str] = set()
    candidates: list[dict[str, str]] = []

    for row in rows:
        missing = [field for field in REVISION_REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Revision row is missing fields: {missing}")

        review_id = row["review_id"].strip()
        candidate_id = row["candidate_id"].strip()
        if not review_id or review_id in seen_reviews:
            raise ValueError(f"Missing or duplicate review_id: {review_id}")
        if not candidate_id or candidate_id in seen_candidates:
            raise ValueError(f"Missing or duplicate candidate_id: {candidate_id}")
        seen_reviews.add(review_id)
        seen_candidates.add(candidate_id)

        source = row["source_code"]
        if not source.strip() or row["source_sha256"] != source_digest(source):
            raise ValueError(f"Missing source or source hash mismatch: {review_id}")
        if not row["source_evidence"].strip() or not row["audit_notes"].strip():
            raise ValueError(f"Missing source-audit provenance: {review_id}")
        if not row["original_query"].strip():
            raise ValueError(f"Missing original query: {review_id}")
        revised = row["revised_query"].strip()
        rationale = row["revision_rationale"].strip()
        if not revised:
            raise ValueError(f"Missing revised_query: {review_id}")
        if not rationale:
            raise ValueError(f"Missing revision_rationale: {review_id}")
        if revised.casefold() == row["original_query"].strip().casefold():
            raise ValueError(f"Revised query is unchanged: {review_id}")
        if row["revision_status"] != "ready_for_review":
            raise ValueError(f"Revision must be marked ready_for_review: {review_id}")

        candidates.append({"candidate_id": candidate_id, "query": revised})

    quality = assess_batch(candidates)
    assessed: list[dict[str, object]] = []
    for row, quality_row in zip(rows, quality, strict=True):
        assessed.append(
            {
                "review_id": row["review_id"],
                "candidate_id": row["candidate_id"],
                "code_unit_id": row["code_unit_id"],
                "original_query": row["original_query"],
                "revised_query": row["revised_query"],
                "revision_rationale": row["revision_rationale"],
                "source_sha256": row["source_sha256"],
                "source_evidence": row["source_evidence"],
                "audit_notes": row["audit_notes"],
                "quality_flags": quality_row["flags"],
                "similar_to_candidate_id": quality_row.get("similar_to_candidate_id"),
                "similarity_score": quality_row.get("similarity_score"),
                "review_status": "requires_manual_review",
            }
        )
    return assessed
