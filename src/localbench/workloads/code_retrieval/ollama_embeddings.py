"""Local Ollama embedding adapter for DenseIndex (no external API)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import httpx


class OllamaEmbeddingError(RuntimeError):
    """Embedding endpoint, response, or model failure."""


class OllamaEmbeddings:
    """Callable batch encoder using Ollama's /api/embed endpoint.

    Model installation is explicit; this adapter never pulls a model.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        if not model.strip():
            raise ValueError("An installed Ollama embedding model is required")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be positive and finite")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def __call__(self, texts: list[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("Embedding inputs must be strings")
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts, "truncate": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaEmbeddingError(
                f"Ollama embedding request failed for {self.model}: {exc}"
            ) from exc
        embeddings = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaEmbeddingError("Ollama returned an invalid embedding batch")
        if any(not isinstance(vector, list) for vector in embeddings):
            raise OllamaEmbeddingError("Ollama returned malformed embedding vectors")
        return embeddings
