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
        batch_size: int = 8,
    ) -> None:
        if not model.strip():
            raise ValueError("An installed Ollama embedding model is required")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be positive and finite")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def __call__(self, texts: list[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("Embedding inputs must be strings")
        result: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                response = httpx.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": batch, "truncate": False},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:500]
                raise OllamaEmbeddingError(
                    f"Ollama rejected embedding batch starting at {start} "
                    f"({len(batch)} inputs): HTTP {exc.response.status_code}: "
                    f"{detail}"
                ) from exc
            except (httpx.HTTPError, ValueError) as exc:
                raise OllamaEmbeddingError(
                    f"Ollama embedding batch starting at {start} failed "
                    f"for {self.model}: {exc}"
                ) from exc
            embeddings = data.get("embeddings") if isinstance(data, dict) else None
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise OllamaEmbeddingError(
                    f"Ollama returned an invalid embedding batch at {start}"
                )
            if any(not isinstance(vector, list) for vector in embeddings):
                raise OllamaEmbeddingError("Ollama returned malformed vectors")
            result.extend(embeddings)
        return result
