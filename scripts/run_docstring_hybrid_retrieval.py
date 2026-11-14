"""Fuse docstring-stripped BM25 and local dense rankings with Reciprocal Rank Fusion.

The dense input must be produced by run_docstring_dense_retrieval.py. This script
recomputes BM25 using the frozen helper and applies RRF without tuning on labels.

Usage:
  python scripts/run_docstring_hybrid_retrieval.py --units dataset/splits/test.jsonl --queries reports/docstring_retrieval_v1/queries.jsonl --dense-results reports/docstring_retrieval_v1/dense/dense_results.json --output-dir reports/docstring_retrieval_v1/hybrid
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_docstring_retrieval_review import bm25_rankings, read_jsonl, strip_docstrings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(rankings: dict[str, list[str]], queries: list[dict], top_k: int) -> dict:
    rows = []
    for query in queries:
        query_id = query["id"]
        targets = set(query.get("relevant_code_units", []))
        ranked = rankings.get(query_id, [])
        ranks = [rank for rank, unit_id in enumerate(ranked, start=1) if unit_id in targets]
        first = min(ranks, default=None)
        rows.append({
            "query_id": query_id,
            "target_rank": first,
            "hit_at_1": int(first is not None and first <= 1),
            "hit_at_5": int(first is not None and first <= 5),
            "hit_at_10": int(first is not None and first <= 10),
            "reciprocal_rank": 1 / first if first is not None else 0.0,
        })
    count = len(rows)
    summary = {
        "query_count": count,
        "top_k": top_k,
        "hit_at_1": sum(row["hit_at_1"] for row in rows) / count if count else 0.0,
        "hit_at_5": sum(row["hit_at_5"] for row in rows) / count if count else 0.0,
        "hit_at_10": sum(row["hit_at_10"] for row in rows) / count if count else 0.0,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / count if count else 0.0,
        "per_query": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--dense-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()
    if args.top_k < 1 or args.rrf_k < 1:
        parser.error("--top-k and --rrf-k must be >= 1")

    units = read_jsonl(args.units)
    queries = read_jsonl(args.queries)
    documents = {}
    for unit in units:
        unit_id = unit["id"]
        if unit_id in documents:
            raise ValueError(f"Duplicate CodeUnit ID: {unit_id}")
        documents[unit_id] = strip_docstrings(unit["source_code"])
    bm25 = bm25_rankings(documents, queries, top_k=max(args.top_k, 100))
    dense_payload = json.loads(args.dense_results.read_text(encoding="utf-8"))
    dense = dense_payload.get("rankings")
    if not isinstance(dense, dict):
        raise ValueError("Dense results must contain a rankings object")

    fused = {}
    for query in queries:
        query_id = query["id"]
        if query_id not in dense:
            raise ValueError(f"Dense results missing query {query_id}")
        dense_rows = dense[query_id]
        dense_ids = [
            item["code_unit_id"] if isinstance(item, dict) else item
            for item in dense_rows
        ]
        bm25_ids = bm25.get(query_id, [])
        rrf_scores = {}
        for ranking in (bm25_ids, dense_ids):
            for rank, unit_id in enumerate(ranking, start=1):
                rrf_scores[unit_id] = rrf_scores.get(unit_id, 0.0) + 1.0 / (args.rrf_k + rank)
        ordered = sorted(rrf_scores, key=lambda unit_id: (-rrf_scores[unit_id], unit_id))
        fused[query_id] = [
            {"code_unit_id": unit_id, "rrf_score": rrf_scores[unit_id]}
            for unit_id in ordered[: args.top_k]
        ]

    ids_only = {query_id: [item["code_unit_id"] for item in rows] for query_id, rows in fused.items()}
    result = {
        "fusion": "reciprocal_rank_fusion",
        "rrf_k": args.rrf_k,
        "top_k": args.top_k,
        "bm25": {"k1": 1.5, "b": 0.75, "top_k_for_fusion": max(args.top_k, 100)},
        "dense_input": str(args.dense_results),
        "dense_input_sha256": sha256_file(args.dense_results),
        "query_sha256": sha256_file(args.queries),
        "corpus_sha256": sha256_file(args.units),
        "rankings": fused,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hybrid_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = metrics(ids_only, queries, args.top_k)
    summary.update({"fusion": "RRF", "rrf_k": args.rrf_k, "bm25_dense_fusion": True})
    (args.output_dir / "hybrid_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
