"""Run reproducible dense retrieval with a local Ollama embedding model.

No cloud fallback is used. Embeddings are cached in SQLite by model, CodeUnit ID,
and source hash so interrupted runs can resume safely.

Usage:
  python scripts/run_docstring_dense_retrieval.py --units dataset/splits/test.jsonl --queries reports/docstring_retrieval_v1/queries.jsonl --output-dir reports/docstring_retrieval_v1/dense --model nomic-embed-text:latest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

from prepare_docstring_retrieval_review import strip_docstrings


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def post_embeddings(base_url: str, model: str, texts: list[str], timeout: int) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            vectors = body.get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                raise RuntimeError("Ollama returned an unexpected embedding response shape")
            if any(not isinstance(vector, list) or not vector for vector in vectors):
                raise RuntimeError("Ollama returned an empty or malformed embedding")
            dimensions = {len(vector) for vector in vectors}
            if len(dimensions) != 1:
                raise RuntimeError("Ollama returned inconsistent embedding dimensions")
            return vectors
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Embedding request failed after 4 attempts: {last_error}") from last_error


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimension mismatch")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def get_or_embed(
    conn: sqlite3.Connection,
    base_url: str,
    model: str,
    records: list[tuple[str, str]],
    batch_size: int,
    timeout: int,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    pending = []
    for record_id, text in records:
        digest = sha256_text(text)
        cached = conn.execute(
            "SELECT vector_json FROM embeddings WHERE model=? AND record_id=? AND text_sha256=?",
            (model, record_id, digest),
        ).fetchone()
        if cached:
            result[record_id] = json.loads(cached[0])
        else:
            pending.append((record_id, text, digest))
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        vectors = post_embeddings(base_url, model, [item[1] for item in batch], timeout)
        for (record_id, _text, digest), vector in zip(batch, vectors):
            encoded = json.dumps(vector, separators=(",", ":"))
            conn.execute(
                "INSERT OR REPLACE INTO embeddings(model,record_id,text_sha256,vector_json) VALUES(?,?,?,?)",
                (model, record_id, digest, encoded),
            )
            result[record_id] = vector
        conn.commit()
        print(f"Embedded {min(offset + len(batch), len(pending))}/{len(pending)} uncached records")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="nomic-embed-text:latest")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if min(args.batch_size, args.timeout, args.top_k) < 1:
        parser.error("--batch-size, --timeout, and --top-k must be >= 1")

    units = read_jsonl(args.units)
    queries = read_jsonl(args.queries)
    documents = []
    seen_ids = set()
    skipped_empty_documents = 0
    for unit in units:
        unit_id = unit["id"]
        if unit_id in seen_ids:
            raise ValueError(f"Duplicate CodeUnit ID: {unit_id}")
        seen_ids.add(unit_id)
        source = strip_docstrings(unit["source_code"])
        if not source.strip():
            skipped_empty_documents += 1
            continue
        documents.append((unit_id, source))
    if not documents:
        raise ValueError("No non-empty CodeUnits remain after docstring removal")
    query_texts = [(query["id"], query["query"]) for query in queries]
    if len({query_id for query_id, _ in query_texts}) != len(query_texts):
        raise ValueError("Duplicate query IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "embedding_cache.sqlite3"
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (model TEXT NOT NULL, record_id TEXT NOT NULL, "
            "text_sha256 TEXT NOT NULL, vector_json TEXT NOT NULL, PRIMARY KEY(model,record_id,text_sha256))"
        )
        doc_vectors = get_or_embed(conn, args.ollama_url, args.model, documents, args.batch_size, args.timeout)
        query_vectors = get_or_embed(conn, args.ollama_url, args.model, query_texts, args.batch_size, args.timeout)

    rankings = {}
    metric_rows = []
    for query in queries:
        query_id = query["id"]
        scored = [
            (unit_id, cosine(query_vectors[query_id], doc_vectors[unit_id]))
            for unit_id, _ in documents
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        rankings[query_id] = [
            {"code_unit_id": unit_id, "score": score}
            for unit_id, score in scored[: args.top_k]
        ]
        targets = set(query.get("relevant_code_units", []))
        ranks = [rank for rank, (unit_id, _score) in enumerate(scored, start=1) if unit_id in targets]
        first_rank = min(ranks, default=None)
        metric_rows.append({
            "query_id": query_id,
            "target_rank": first_rank,
            "hit_at_1": int(first_rank is not None and first_rank <= 1),
            "hit_at_5": int(first_rank is not None and first_rank <= 5),
            "hit_at_10": int(first_rank is not None and first_rank <= 10),
            "reciprocal_rank": 1 / first_rank if first_rank is not None else 0.0,
        })

    n = len(metric_rows)
    summary = {
        "model": args.model,
        "ollama_url": args.ollama_url,
        "embedding_dimensions": len(next(iter(doc_vectors.values()))),
        "corpus_count": len(documents),
        "source_rows": len(units),
        "skipped_empty_documents_after_docstring_removal": skipped_empty_documents,
        "source_file_sha256": sha256_file(args.units),
        "query_file_sha256": sha256_file(args.queries),
        "query_count": n,
        "top_k": args.top_k,
        "metrics": {
            "hit_at_1": sum(row["hit_at_1"] for row in metric_rows) / n if n else 0.0,
            "hit_at_5": sum(row["hit_at_5"] for row in metric_rows) / n if n else 0.0,
            "hit_at_10": sum(row["hit_at_10"] for row in metric_rows) / n if n else 0.0,
            "mrr": sum(row["reciprocal_rank"] for row in metric_rows) / n if n else 0.0,
        },
        "limitations": [
            "Single-positive silver target association; other relevant CodeUnits are not exhaustively labeled.",
            "Docstrings are removed from indexed source before embedding; query text remains natural language.",
            "This run uses the local Ollama endpoint only and has no cloud fallback.",
        ],
    }
    (args.output_dir / "dense_results.json").write_text(
        json.dumps({"rankings": rankings}, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "dense_metrics.json").write_text(
        json.dumps({"summary": summary, "per_query": metric_rows}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
