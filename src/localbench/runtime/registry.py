"""Local model registry for discovered/available models."""

from __future__ import annotations

from localbench.errors import ModelNotFoundError
from localbench.runtime.model import ModelInfo


class ModelRegistry:
    """Central registry for discovered local models.

    Supports listing, lookup, and normalized metadata access.
    Does not score or recommend models — that belongs to later phases.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}

    def add(self, model: ModelInfo) -> None:
        """Register a discovered model."""
        self._models[model.name] = model

    def get(self, name: str) -> ModelInfo:
        """Look up a model by identifier. Raises ModelNotFoundError if absent."""
        model = self._models.get(name)
        if model is None:
            raise ModelNotFoundError(name)
        return model

    def has(self, name: str) -> bool:
        """Check if a model is registered."""
        return name in self._models

    def list_models(self) -> list[ModelInfo]:
        """Return all registered models, sorted by name."""
        return sorted(self._models.values(), key=lambda m: m.name)

    def clear(self) -> None:
        """Remove all registered models."""
        self._models.clear()
