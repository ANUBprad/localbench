"""Build a blinded top-10 BM25 relevance-review worksheet.

Usage:
  python scripts/prepare_docstring_retrieval_review.py \
      --units dataset/splits/test.jsonl \
      --queries reports/docstring_retrieval_v1/queries.jsonl \
      --output reports/docstring_retrieval_v1/human_relevance_review.csv
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [token.lower() for token in _TOKEN.findall(text.replace("_", " "))]


def strip_docstrings(source: str) -> str:
    """Remove leading module/class/function/async-function docstring spans."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                spans.append((first.lineno, first.end_lineno))
    lines = source.splitlines()
    for start, end in sorted(spans, reverse=True):
        for index in range(start - 1, min(end, len(lines))):
            lines[index] = ""
    return "\n".join(lines).strip()


def bm25_rankings(documents: dict[str, str], queries: list[dict], top_k: int = 10) -> dict[str, list[str]]:
    counts = {key: Counter(tokenize(value)) for key, value in documents.items()}
    lengths = {key: sum(value.values()) for key, value in counts.items()}
    avgdl = sum(lengths.values()) / len(documents)
    df = Counter()
    for value in counts.values():
        df.update(value.keys())
    n = len(documents)
    rankings = {}
    for query in queries:
        terms = set(tokenize(query["query"]))
        scored = []
        for key, frequencies in counts.items():
            length = lengths[key]
            norm = 1 - 0.75 + 0.75 * length / avgdl if avgdl else 1
            score = 0.0
            for term in terms:
                tf = frequencies.get(term, 0)
                if tf:
                    idf = math.log1p((n - df[term] + 0.5) / (df[term] + 0.5))
                    score += idf * tf * 2.5 / (tf + 1.5 * norm)
            if score > 0:
                scored.append((key, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        rankings[query["id"]] = [key for key, _ in scored[:top_k]]
    return rankings


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    units = read_jsonl(args.units)
    queries = read_jsonl(args.queries)
    by_id = {}
    documents = {}
    for unit in units:
        uid = unit["id"]
        if uid in by_id:
            raise ValueError(f"Duplicate CodeUnit ID: {uid}")
        by_id[uid] = unit
        documents[uid] = strip_docstrings(unit["source_code"])

    rankings = bm25_rankings(documents, queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id", "query", "candidate_rank", "candidate_code_unit_id",
        "repository", "file_path", "symbol", "symbol_type",
        "candidate_source_without_docstrings", "relevance_label", "reviewer_notes",
    ]
    rows = 0
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query in queries:
            for rank, uid in enumerate(rankings[query["id"]], start=1):
                unit = by_id[uid]
                writer.writerow({
                    "query_id": query["id"],
                    "query": query["query"],
                    "candidate_rank": rank,
                    "candidate_code_unit_id": uid,
                    "repository": unit.get("repository", ""),
                    "file_path": unit.get("file_path", ""),
                    "symbol": unit.get("symbol", ""),
                    "symbol_type": unit.get("symbol_type", ""),
                    "candidate_source_without_docstrings": documents[uid],
                    "relevance_label": "",
                    "reviewer_notes": "",
                })
                rows += 1
    print(f"Wrote {rows} blinded query-candidate rows to {args.output}")


if __name__ == "__main__":
    main()
