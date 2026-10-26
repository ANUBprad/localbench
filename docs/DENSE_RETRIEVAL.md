# Local dense retrieval baseline

`DenseIndex` implements exact cosine similarity over source-only documents.
It accepts an **explicit local encoder callable** rather than downloading or
selecting an embedding model implicitly. No relevance labels are used to build
the index.

```python
from localbench.workloads.code_retrieval.dense import DenseIndex

# Replace with a locally installed, pinned embedding model.
def encoder(texts: list[str]) -> list[list[float]]:
    return [[float("retry" in text), float("email" in text)] for text in texts]

index = DenseIndex(
    {"unit-1": "retry failed payment", "unit-2": "validate email"},
    encoder,
)
print(index.search("retry payment", k=2))
```

The example encoder is only a deterministic illustration, **not** a semantic
embedding model. For meaningful results, supply a real local embedding model,
pin its exact model name and revision, and use the same encoder for indexing
and queries. Do not compare runs made with different embedding models as if
they were the same baseline.

Vectors must be finite, nonzero and dimensionally consistent. Ties are
broken by CodeUnit ID. Unlike BM25, dense search returns the top k documents
even if cosine similarity is zero or negative.

This module does not yet provide persistent embedding caches, a model adapter,
or a dataset runner. Those are separate steps required for a reproducible
BM25-versus-dense benchmark.
