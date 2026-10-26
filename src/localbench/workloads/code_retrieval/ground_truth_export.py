"""Export adjudicated retrieval cases without modifying blind review artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .benchmark import read_jsonl
from .review import validate_review_artifact


def export_adjudicated_cases(
    review_artifact: dict,
    labels: list[dict],
    corpus_ids: set[str],
) -> list[dict]:
    """Require explicit labels and accepted review for every exported query."""
    errors = validate_review_artifact(review_artifact)
    if errors:
        raise ValueError("Invalid review artifact: " + "; ".join(errors))
    if not corpus_ids:
        raise ValueError("Corpus cannot be empty")
    by_candidate = {}
    for label in labels:
        if set(label) != {"candidate_id", "relevant_code_units"}:
            raise ValueError("Label rows require candidate_id and relevant_code_units")
        cid = label["candidate_id"]
        gold = label["relevant_code_units"]
        if not isinstance(cid, str) or not cid or cid in by_candidate:
            raise ValueError(f"Invalid or duplicate candidate label: {cid}")
        if (not isinstance(gold, list) or not gold
                or any(not isinstance(uid, str) or not uid for uid in gold)
                or len(gold) != len(set(gold))):
            raise ValueError(f"Invalid relevance labels for {cid}")
        if not set(gold) <= corpus_ids:
            raise ValueError(f"Relevance outside corpus for {cid}")
        by_candidate[cid] = gold

    selected = {item["candidate_id"]: item for item in review_artifact["items"]}
    unknown = set(by_candidate) - set(selected)
    if unknown:
        raise ValueError(f"Labels for unselected candidates: {sorted(unknown)}")
    result = []
    for item in review_artifact["items"]:
        cid = item["candidate_id"]
        if item["review"]["state"] != "accepted":
            if cid in by_candidate:
                raise ValueError(f"Labels for non-accepted candidate: {cid}")
            continue
        if cid not in by_candidate:
            raise ValueError(f"Accepted candidate missing labels: {cid}")
        if item["code_unit_id"] not in corpus_ids:
            raise ValueError(f"Selected target outside corpus: {cid}")
        result.append({
            "id": cid,
            "query": item["query"],
            "relevant_code_units": by_candidate[cid],
        })
    if not result:
        raise ValueError("No accepted, adjudicated queries available")
    return result


def export_from_files(
    review_path: Path, labels_path: Path, units_path: Path, output_path: Path
) -> int:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    units = read_jsonl(units_path)
    ids = [unit["id"] for unit in units]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate CodeUnit IDs in corpus")
    rows = export_adjudicated_cases(review, read_jsonl(labels_path), set(ids))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)
