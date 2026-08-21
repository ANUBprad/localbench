"""Provider-agnostic model protocol and core data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ModelInfo:
    """Static metadata for a discovered local model."""

    name: str
    parameter_count: str | None = None
    quantization: str | None = None
    size_bytes: int | None = None
    modified_at: str | None = None

    @property
    def size_gb(self) -> float | None:
        if self.size_bytes is None:
            return None
        return round(self.size_bytes / (1024**3), 2)


@dataclass
class GenerationRequest:
    """Input for a single inference call."""

    prompt: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 256
    top_p: float = 0.9
    seed: int | None = None


@dataclass
class GenerationResult:
    """Output from a single inference call."""

    model: str
    text: str
    duration_ms: float
    total_duration_ms: float | None = None
    eval_count: int | None = None
    eval_duration_ms: float | None = None
    done: bool = True
    error: str | None = None


@runtime_checkable
class LocalModel(Protocol):
    """Provider-agnostic inference contract.

    All runtime providers implement this protocol. The rest of LocalBench
    depends only on this abstraction, never on provider-specific details.
    """

    @property
    def name(self) -> str:
        """Unique model identifier."""

    @property
    def metadata(self) -> ModelInfo:
        """Static metadata: parameters, quantization, footprint."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute generation with timing and token info."""
