"""Ollama HTTP client adapter implementing the LocalModel protocol."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as ConcurrentTimeoutError

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

#: Default hard upper bound (seconds) for ONE generation request. This is
#: enforced by a dedicated deadline, not just an httpx read timeout, so a
#: hung or slow-to-trickle provider can never stall the caller indefinitely.
DEFAULT_REQUEST_TIMEOUT = 60.0


class OllamaAdapter:
    """Ollama-specific inference adapter.

    Translates between the provider-agnostic LocalModel contract and
    Ollama's HTTP API. All Ollama-specific logic is isolated here.

    ``generate`` has a hard wall-clock deadline (``request_timeout``): the
    HTTP call runs on a dedicated worker thread and the caller waits at most
    ``request_timeout`` before an :class:`OllamaUnavailableError` is raised.
    This makes provider hangs a bounded failure rather than an unbounded
    stall, regardless of how Ollama (or the OS socket) behaves. Each
    generation request also uses its own short-lived ``httpx.Client`` so that
    concurrent worker threads never share a connection pool.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "",
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._request_timeout = request_timeout
        self._client = httpx.Client(base_url=self._base_url, timeout=60.0)

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def metadata(self) -> ModelInfo:
        return ModelInfo(name=self._model_name)

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

    def _http_post(self, client: httpx.Client, path: str, json: dict) -> dict:
        """POST *json* with a specific *client*, mapping errors idempotently."""
        try:
            response = client.post(path, json=json)
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
        """Check if Ollama is reachable and responsive.

        Only verifies HTTP reachability. The root endpoint returns a
        plain-text body, so the response is not parsed as JSON.
        """
        try:
            response = self._client.get("/")
            response.raise_for_status()
            return True
        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.HTTPStatusError,
        ):
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
        """Execute a generation request with a hard wall-clock deadline.

        The HTTP call runs on a dedicated daemon worker thread and the caller
        waits at most ``request_timeout`` seconds. If the deadline expires
        (provider hung / not streaming), :class:`OllamaUnavailableError` is
        raised so the caller sees a bounded failure instead of an indefinite
        stall. Each request uses its own short-lived ``httpx.Client``.
        """
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ollama-generate"
        )
        try:
            future = executor.submit(self._generate_http, request)
            return future.result(timeout=self._request_timeout)
        except ConcurrentTimeoutError:
            raise OllamaUnavailableError(
                "Ollama request timed out. "
                "The model may be loading or the prompt too long."
            ) from None
        finally:
            # Never block exit on a (possibly hung) request thread; the io
            # thread is daemonized and dies with the interpreter.
            executor.shutdown(wait=False)

    def _generate_http(self, request: GenerationRequest) -> GenerationResult:
        """Run the HTTP POST for one generation request, then map the result.

        Executed on a dedicated worker thread. Uses a short-lived client per
        request so concurrent worker threads never contend on one pool.
        """
        options = {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
            "top_p": request.top_p,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        with httpx.Client(
            base_url=self._base_url, timeout=self._request_timeout
        ) as client:
            try:
                response = self._http_post(
                    client,
                    "/api/generate",
                    {
                        "model": request.model,
                        "prompt": request.prompt,
                        "stream": False,
                        "options": options,
                    },
                )
            except OllamaUnavailableError:
                raise
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise ModelNotFoundError(request.model) from exc
                raise GenerationError(f"Generation failed: {exc}") from exc
            except Exception as exc:
                raise GenerationError(f"Generation failed: {exc}") from exc

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
