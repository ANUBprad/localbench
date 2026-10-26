"""Offline dense-vector retrieval with a caller-supplied embedding encoder.

The encoder is responsible for producing vectors; this module never downloads
models, accesses relevance labels, or sends source code to a remote service.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

from .retrieval import SearchHit

Encoder = Callable[[list[str]], Sequence[Sequence[float]]]


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Embedding vectors must be nonempty and finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("Embedding vectors must have a finite nonzero norm")
    return tuple(value / norm for value in values)


class DenseIndex:
    """Cosine retrieval using a fixed local encoder and deterministic ranking."""

    def __init__(self, documents: Mapping[str, str], encoder: Encoder):
        if not documents:
            raise ValueError("At least one document is required")
        if any(not isinstance(key, str) or not key or not isinstance(value, str)
               for key, value in documents.items()):
            raise ValueError("Documents require nonempty string IDs and text")
        self._encoder = encoder
        self._ids = sorted(documents)
        raw = list(encoder([documents[key] for key in self._ids]))
        if len(raw) != len(self._ids):
            raise ValueError("Encoder returned an incorrect number of vectors")
        self._vectors = [_normalize(vector) for vector in raw]
        self._dimension = len(self._vectors[0])
        if any(len(vector) != self._dimension for vector in self._vectors):
            raise ValueError("Encoder returned inconsistent dimensions")

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        if not query.strip():
            return []
        encoded = list(self._encoder([query]))
        if len(encoded) != 1:
            raise ValueError("Encoder must return exactly one query vector")
        vector = _normalize(encoded[0])
        if len(vector) != self._dimension:
            raise ValueError("Query embedding dimension differs from index")
        hits = [
            SearchHit(key, sum(a * b for a, b in zip(vector, document_vector)))
            for key, document_vector in zip(self._ids, self._vectors)
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.code_unit_id))[:k]
