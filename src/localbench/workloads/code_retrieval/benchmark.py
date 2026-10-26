"""Reproducible retrieval comparison on explicitly labeled query cases."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .dense import DenseIndex
from .metrics import evaluate_rankings
from .retrieval import BM25Index


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
    return records


def run_comparison(
    documents: Mapping[str, str],
    queries: Mapping[str, str],
    relevant: Mapping[str, Sequence[str]],
    *,
    encoder=None,
    k: int = 10,
) -> dict:
    """Compare methods on the same corpus and labels.

    The caller must supply adjudicated relevance; generated candidate targets
    are not automatically treated as human-verified ground truth.
    """
    if k < 10:
        raise ValueError("k must be >= 10 for Hit@10 evaluation")
    if set(queries) != set(relevant):
        raise ValueError("Query IDs and relevance IDs must match exactly")
    if any(not query_id or not isinstance(query, str) or not query.strip()
           for query_id, query in queries.items()):
        raise ValueError("Queries require nonempty IDs and text")
    if any(not set(gold) <= set(documents) for gold in relevant.values()):
        raise ValueError("Relevance contains CodeUnit IDs outside the corpus")

    methods = {"bm25": BM25Index(documents)}
    if encoder is not None:
        methods["dense"] = DenseIndex(documents, encoder)
    result = {"query_count": len(queries), "methods": {}}
    for name, index in methods.items():
        rankings = {
            query_id: [hit.code_unit_id for hit in index.search(query, k=k)]
            for query_id, query in sorted(queries.items())
        }
        scores = evaluate_rankings(rankings, relevant)
        result["methods"][name] = {
            "metrics": {
                "hit_at_1": scores.hit_at_1,
                "hit_at_5": scores.hit_at_5,
                "hit_at_10": scores.hit_at_10,
                "mrr": scores.mrr,
            },
            "rankings": rankings,
        }
    return result
