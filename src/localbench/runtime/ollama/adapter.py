"""Ollama HTTP client adapter implementing the LocalModel protocol."""

from __future__ import annotations

import httpx

from localbench.errors import (
    GenerationError,
    ModelNotFoundError,
    OllamaUnavailableError,
)
from localbench.runtime.model import (
    GenerationRequest,
    GenerationResult,
    ModelInfo,
)


class OllamaAdapter:
    """Ollama-specific inference adapter.

    Translates between the provider-agnostic LocalModel contract and
    Ollama's HTTP API. All Ollama-specific logic is isolated here.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=60.0)

    def _get(self, path: str) -> dict:
        """Make a GET request to Ollama.

        Raises OllamaUnavailableError on connection or timeout failure.
        """
        try:
            response = self._client.get(path)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError() from exc
        except httpx.TimeoutException as exc:
            raise OllamaUnavailableError(
                "Ollama connection timed out. "
                "The server may be overloaded."
            ) from exc

    def _post(self, path: str, json: dict) -> dict:
        """Make a POST request to Ollama.

        Raises on connection or generation failure.
        """
        try:
            response = self._client.post(path, json=json)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError() from exc
        except httpx.TimeoutException as exc:
            raise OllamaUnavailableError(
                "Ollama request timed out. "
                "The model may be loading or the prompt too long."
            ) from exc

    def health_check(self) -> bool:
        """Check if Ollama is reachable and responsive."""
        try:
            self._get("/")
            return True
        except OllamaUnavailableError:
            return False

    def discover_models(self) -> list[ModelInfo]:
        """Discover all models installed in Ollama."""
        data = self._get("/api/tags")
        models: list[ModelInfo] = []
        for model in data.get("models", []):
            name = model.get("name", "")
            # Strip the :latest tag for cleaner display
            if name.endswith(":latest"):
                name = name[: -len(":latest")]
            models.append(
                ModelInfo(
                    name=name,
                    size_bytes=model.get("size"),
                    modified_at=model.get("modified_at"),
                )
            )
        return models

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute a generation request against Ollama."""
        try:
            response = self._post(
                "/api/generate",
                json={
                    "model": request.model,
                    "prompt": request.prompt,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
                },
            )
        except OllamaUnavailableError:
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ModelNotFoundError(request.model) from exc
            raise GenerationError(
                f"Generation failed: {exc}"
            ) from exc
        except Exception as exc:
            raise GenerationError(
                f"Generation failed: {exc}"
            ) from exc

        # Check for model-not-found in the response body
        if "error" in response:
            error_msg = response["error"]
            not_found = (
                "not found" in error_msg.lower()
                or "does not exist" in error_msg.lower()
            )
            if not_found:
                raise ModelNotFoundError(request.model)
            raise GenerationError(error_msg)

        total_ns = response.get("total_duration", 0)
        eval_ns = response.get("eval_duration", 0)

        return GenerationResult(
            model=response.get("model", request.model),
            text=response.get("response", ""),
            duration_ms=round(total_ns / 1_000_000, 1),
            total_duration_ms=round(total_ns / 1_000_000, 1),
            eval_count=response.get("eval_count"),
            eval_duration_ms=(
                round(eval_ns / 1_000_000, 1) if eval_ns else None
            ),
            done=response.get("done", True),
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
