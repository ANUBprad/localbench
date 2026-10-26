"""Retrieval metrics with explicit query denominators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    query_count: int
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    mrr: float


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Sequence[str]],
) -> RetrievalMetrics:
    """Evaluate every labeled query; absent rankings count as misses.

    Rankings may not include unknown query IDs or repeated CodeUnit IDs.
    The relevance map is the evaluation denominator and must be nonempty.
    """
    if not relevant:
        raise ValueError("At least one labeled query is required")
    unknown = set(rankings) - set(relevant)
    if unknown:
        raise ValueError(f"Unknown query IDs: {sorted(unknown)}")
    totals = {1: 0, 5: 0, 10: 0}
    reciprocal_sum = 0.0
    for query_id, gold in relevant.items():
        if not gold:
            raise ValueError(f"Missing relevance labels for {query_id}")
        ranking = list(rankings.get(query_id, []))
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"Duplicate ranked CodeUnit IDs for {query_id}")
        matches = set(gold)
        first_rank = next(
            (rank for rank, item in enumerate(ranking, start=1) if item in matches),
            None,
        )
        if first_rank is not None:
            reciprocal_sum += 1 / first_rank
            for k in totals:
                totals[k] += first_rank <= k
    count = len(relevant)
    return RetrievalMetrics(
        query_count=count,
        hit_at_1=totals[1] / count,
        hit_at_5=totals[5] / count,
        hit_at_10=totals[10] / count,
        mrr=reciprocal_sum / count,
    )
