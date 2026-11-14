"""Build a pooled human-review set for BM25, dense, and hybrid candidate rankings.

Existing BM25 labels/notes are preserved when the current review CSV is supplied.
Run this after local dense and hybrid result artifacts exist.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from prepare_docstring_retrieval_review import bm25_rankings, read_jsonl, strip_docstrings


def load_rankings(path: Path, method: str) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rankings = payload.get("rankings")
    if not isinstance(rankings, dict):
        raise ValueError(f"{path} does not contain a rankings object")
    result = {}
    for query_id, rows in rankings.items():
        result[query_id] = [
            row["code_unit_id"] if isinstance(row, dict) else row
            for row in rows
        ]
    return result


def load_existing(path: Path | None) -> dict[tuple[str, str], tuple[str, str]]:
    if path is None:
        return {}
    saved = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["query_id"], row["candidate_code_unit_id"])
            label = (row.get("relevance_label") or "").strip()
            notes = (row.get("reviewer_notes") or "").strip()
            if label or notes:
                saved[key] = (label, notes)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--dense-results", type=Path, required=True)
    parser.add_argument("--hybrid-results", type=Path, required=True)
    parser.add_argument("--existing-review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    units = read_jsonl(args.units)
    queries = read_jsonl(args.queries)
    by_id = {}
    documents = {}
    for unit in units:
        unit_id = unit["id"]
        if unit_id in by_id:
            raise ValueError(f"Duplicate CodeUnit ID: {unit_id}")
        by_id[unit_id] = unit
        documents[unit_id] = strip_docstrings(unit["source_code"])

    bm25 = bm25_rankings(documents, queries, top_k=args.top_k)
    dense = load_rankings(args.dense_results, "dense")
    hybrid = load_rankings(args.hybrid_results, "hybrid")
    existing = load_existing(args.existing_review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id", "query", "candidate_code_unit_id", "repository", "file_path",
        "symbol", "symbol_type", "candidate_rank_bm25", "candidate_rank_dense",
        "candidate_rank_hybrid", "candidate_source_without_docstrings",
        "relevance_label", "reviewer_notes",
    ]
    rows_written = 0
    preserved = 0
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query in queries:
            query_id = query["id"]
            method_lists = {
                "bm25": bm25.get(query_id, [])[: args.top_k],
                "dense": dense.get(query_id, [])[: args.top_k],
                "hybrid": hybrid.get(query_id, [])[: args.top_k],
            }
            for method, ranked in method_lists.items():
                if method != "bm25" and query_id not in (dense if method == "dense" else hybrid):
                    raise ValueError(f"{method} results missing query {query_id}")
            rank_maps = {
                method: {unit_id: rank for rank, unit_id in enumerate(ranked, start=1)}
                for method, ranked in method_lists.items()
            }
            candidate_ids = set().union(*(set(ranked) for ranked in method_lists.values()))
            for unit_id in sorted(candidate_ids):
                if unit_id not in by_id:
                    raise ValueError(f"Ranking references unknown CodeUnit ID: {unit_id}")
                unit = by_id[unit_id]
                label, notes = existing.get((query_id, unit_id), ("", ""))
                preserved += int(bool(label or notes))
                writer.writerow({
                    "query_id": query_id,
                    "query": query["query"],
                    "candidate_code_unit_id": unit_id,
                    "repository": unit.get("repository", ""),
                    "file_path": unit.get("file_path", ""),
                    "symbol": unit.get("symbol", ""),
                    "symbol_type": unit.get("symbol_type", ""),
                    "candidate_rank_bm25": rank_maps["bm25"].get(unit_id, ""),
                    "candidate_rank_dense": rank_maps["dense"].get(unit_id, ""),
                    "candidate_rank_hybrid": rank_maps["hybrid"].get(unit_id, ""),
                    "candidate_source_without_docstrings": documents[unit_id],
                    "relevance_label": label,
                    "reviewer_notes": notes,
                })
                rows_written += 1
    print(json.dumps({
        "output": str(args.output),
        "query_count": len(queries),
        "pooled_candidate_rows": rows_written,
        "existing_judgments_or_notes_preserved": preserved,
        "top_k_per_method": args.top_k,
    }, indent=2))


if __name__ == "__main__":
    main()
