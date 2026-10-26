"""Compare retrieval methods on explicit, adjudicated JSONL inputs.

Usage:
  python -m scripts.run_retrieval_comparison --units corpus.jsonl \\
      --queries labeled_queries.jsonl --output results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from localbench.workloads.code_retrieval.benchmark import (
    read_jsonl,
    run_comparison,
    sha256_file,
)
from localbench.workloads.code_retrieval.ollama_embeddings import OllamaEmbeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="Installed Ollama embedding model")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    units = read_jsonl(args.units)
    cases = read_jsonl(args.queries)
    documents = {}
    for unit in units:
        uid = unit["id"]
        if uid in documents:
            raise ValueError(f"Duplicate CodeUnit ID: {uid}")
        documents[uid] = unit["source_code"]
    queries, relevant = {}, {}
    for case in cases:
        qid = case["id"]
        if qid in queries:
            raise ValueError(f"Duplicate query ID: {qid}")
        queries[qid] = case["query"]
        relevant[qid] = case["relevant_code_units"]

    encoder = (
        OllamaEmbeddings(args.model, base_url=args.ollama_url)
        if args.model else None
    )
    results = run_comparison(documents, queries, relevant, encoder=encoder)
    results["provenance"] = {
        "units_sha256": sha256_file(args.units),
        "queries_sha256": sha256_file(args.queries),
        "embedding_model": args.model,
        "ollama_url": args.ollama_url if args.model else None,
        "representation": "source_code",
        "top_k": 10,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
