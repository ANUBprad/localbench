"""Evaluate judged BM25 candidates from the docstring retrieval review CSV.

Labels: relevant=2, partially_relevant=1, not_relevant=0.
Blank and uncertain labels are unjudged, not negative judgments.

Usage:
  python scripts/evaluate_docstring_relevance.py --input reports/docstring_retrieval_v1/human_relevance_review.csv --output-dir reports/docstring_retrieval_v1/human_evaluation
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

LABELS = {"relevant": 2, "partially_relevant": 1, "not_relevant": 0}
REQUIRED = {"query_id", "candidate_rank", "candidate_code_unit_id", "relevance_label"}


def load_reviews(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Review CSV is missing required columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            unit_id = (row.get("candidate_code_unit_id") or "").strip()
            if not query_id or not unit_id:
                raise ValueError(f"Line {line}: query_id and candidate_code_unit_id are required")
            try:
                rank = int(row["candidate_rank"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Line {line}: candidate_rank must be an integer") from exc
            if rank < 1:
                raise ValueError(f"Line {line}: candidate_rank must be >= 1")
            key = (query_id, rank)
            if key in seen:
                raise ValueError(f"Line {line}: duplicate query/rank pair {key}")
            seen.add(key)
            label = (row.get("relevance_label") or "").strip().lower()
            if label not in LABELS and label not in {"", "uncertain"}:
                raise ValueError(f"Line {line}: unsupported relevance label {label!r}")
            row["candidate_rank"] = rank
            row["relevance_label"] = label
            grouped[query_id].append(row)
    for query_id, rows in grouped.items():
        rows.sort(key=lambda row: row["candidate_rank"])
        ids = [row["candidate_code_unit_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{query_id}: duplicate candidate CodeUnit IDs")
    return dict(grouped)


def dcg(gains: list[int], k: int) -> float:
    return sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains[:k], start=1))


def evaluate_query(rows: list[dict], k: int) -> dict:
    judged = [row for row in rows if row["relevance_label"] in LABELS]
    judged_at_k = [row for row in judged if row["candidate_rank"] <= k]
    strict_relevant = [row for row in judged if row["relevance_label"] == "relevant"]
    gains_by_rank = {row["candidate_rank"]: LABELS[row["relevance_label"]] for row in judged}
    ranked_gains = [gains_by_rank.get(rank, 0) for rank in range(1, k + 1)]
    ideal_dcg = dcg(sorted((LABELS[row["relevance_label"]] for row in judged), reverse=True), k)
    first_rank = min((row["candidate_rank"] for row in strict_relevant), default=None)
    return {
        "judged_candidates": len(judged),
        "uncertain_or_blank": len(rows) - len(judged),
        "judged_coverage_at_k": len(judged_at_k) / k,
        "strict_relevant_count": len(strict_relevant),
        "precision_at_k_judged": (
            sum(row["relevance_label"] == "relevant" for row in judged_at_k) / len(judged_at_k)
            if judged_at_k else None
        ),
        "hit_at_k_strict": int(first_rank is not None and first_rank <= k),
        "mrr_strict": 1 / first_rank if first_rank is not None else 0.0,
        "ndcg_at_k_judged_pool": dcg(ranked_gains, k) / ideal_dcg if ideal_dcg else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be >= 1")
    grouped = load_reviews(args.input)
    per_query = [
        {"query_id": query_id, **evaluate_query(rows, args.k)}
        for query_id, rows in sorted(grouped.items())
    ]
    evaluable = [row for row in per_query if row["judged_candidates"] > 0]

    def macro(key: str):
        values = [row[key] for row in evaluable if row[key] is not None]
        return mean(values) if values else None

    total = sum(len(rows) for rows in grouped.values())
    judged = sum(row["judged_candidates"] for row in per_query)
    summary = {
        "query_count": len(grouped),
        "query_count_with_judgments": len(evaluable),
        "candidate_rows": total,
        "judged_candidates": judged,
        "unjudged_candidates": total - judged,
        "judgment_coverage": judged / total if total else 0.0,
        "k": args.k,
        "label_mapping": LABELS,
        "aggregation": "Macro average over queries with at least one judgment; metric-specific nulls omitted.",
        "metrics": {
            "precision_at_k_judged": macro("precision_at_k_judged"),
            "hit_at_k_strict": macro("hit_at_k_strict"),
            "mrr_strict": macro("mrr_strict"),
            "ndcg_at_k_judged_pool": macro("ndcg_at_k_judged_pool"),
            "judged_coverage_at_k": macro("judged_coverage_at_k"),
        },
        "limitations": [
            "Only the frozen BM25 top-k pool is judged; corpus-wide recall cannot be estimated.",
            "Blank and uncertain labels are unjudged, not negative judgments.",
            "nDCG uses graded gains: not_relevant=0, partially_relevant=1, relevant=2.",
            "Strict Hit@K and MRR count only relevant; partially_relevant is not a strict hit.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = list(per_query[0]) if per_query else ["query_id"]
    with (args.output_dir / "per_query_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_query)
    print(json.dumps({"queries": len(grouped), "judged_candidates": judged, "judgment_coverage": summary["judgment_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
