"""Shared exception hierarchy for LocalBench."""


class LocalBenchError(Exception):
    """Base exception for all LocalBench errors."""


class OllamaError(LocalBenchError):
    """Base exception for Ollama-related errors."""


class OllamaUnavailableError(OllamaError):
    """Ollama server is not reachable."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "Ollama is not running or not reachable. "
            "Start Ollama with: ollama serve"
        )


class ModelNotFoundError(OllamaError):
    """Requested model is not installed in Ollama."""

    def __init__(self, model: str) -> None:
        super().__init__(
            f"Model '{model}' is not installed. "
            f"Pull it with: ollama pull {model}"
        )
        self.model = model


class GenerationError(OllamaError):
    """Generation request failed."""
