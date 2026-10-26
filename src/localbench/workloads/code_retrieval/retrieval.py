"""Deterministic, dependency-free BM25 retrieval baseline.

The index is deliberately built from explicit CodeUnit IDs and text; it never
uses test relevance labels, model-generated answers, or external services.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Split camelCase and snake_case identifiers for code-oriented retrieval."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [token.lower() for token in _TOKEN.findall(text.replace("_", " "))]


@dataclass(frozen=True)
class SearchHit:
    code_unit_id: str
    score: float


class BM25Index:
    """In-memory BM25 with stable ID tie-breaking and unique-document validation."""

    def __init__(self, documents: Mapping[str, str], *, k1: float = 1.5, b: float = 0.75):
        if not (k1 > 0 and 0 <= b <= 1):
            raise ValueError("Expected k1 > 0 and 0 <= b <= 1")
        if not documents:
            raise ValueError("At least one document is required")
        if any(not key or not isinstance(value, str) for key, value in documents.items()):
            raise ValueError("Documents require nonempty IDs and string text")
        self.k1, self.b = k1, b
        self._counts = {key: Counter(tokenize(value)) for key, value in documents.items()}
        self._lengths = {key: sum(counts.values()) for key, counts in self._counts.items()}
        self._avgdl = sum(self._lengths.values()) / len(documents)
        self._df = Counter()
        for counts in self._counts.values():
            self._df.update(counts.keys())
        self._n = len(documents)

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        terms = set(tokenize(query))
        if not terms:
            return []
        scores = []
        for key, counts in self._counts.items():
            length = self._lengths[key]
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self._df[term]
                idf = math.log1p((self._n - df + 0.5) / (df + 0.5))
                norm = 1 - self.b + self.b * length / self._avgdl if self._avgdl else 1
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            if score > 0:
                scores.append(SearchHit(key, score))
        return sorted(scores, key=lambda hit: (-hit.score, hit.code_unit_id))[:k]


def code_unit_documents(units: list, *, representation: str = "source") -> dict[str, str]:
    """Build a source-only index from canonical CodeUnit objects.

    No ground-truth relevance or held-out query text is included.
    """
    if representation != "source":
        raise ValueError("Only the source representation is supported")
    documents = {}
    for unit in units:
        if unit.id in documents:
            raise ValueError(f"Duplicate CodeUnit ID: {unit.id}")
        documents[unit.id] = unit.source_code
    return documents
