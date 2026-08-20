"""Tests for model registry."""

import pytest

from localbench.errors import ModelNotFoundError
from localbench.runtime.model import ModelInfo
from localbench.runtime.registry import ModelRegistry


@pytest.fixture
def registry():
    return ModelRegistry()


@pytest.fixture
def sample_models():
    return [
        ModelInfo(name="phi-3-mini", size_bytes=2048000000),
        ModelInfo(name="mistral-7b", size_bytes=3800000000),
        ModelInfo(name="gemma-2", size_bytes=5400000000),
    ]


class TestAdd:
    def test_add_model(self, registry):
        """Adding a model makes it retrievable."""
        model = ModelInfo(name="phi-3-mini", size_bytes=1000)
        registry.add(model)
        assert registry.has("phi-3-mini")

    def test_add_multiple_models(self, registry, sample_models):
        """Multiple models can be added."""
        for m in sample_models:
            registry.add(m)
        assert len(registry.list_models()) == 3


class TestGet:
    def test_get_existing_model(self, registry):
        """Getting an existing model returns it."""
        model = ModelInfo(name="phi-3-mini", size_bytes=1000)
        registry.add(model)
        result = registry.get("phi-3-mini")
        assert result.name == "phi-3-mini"

    def test_get_nonexistent_model_raises(self, registry):
        """Getting a nonexistent model raises ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError):
            registry.get("nonexistent")


class TestHas:
    def test_has_returns_true_for_existing(self, registry):
        """has() returns True for registered models."""
        registry.add(ModelInfo(name="test", size_bytes=100))
        assert registry.has("test") is True

    def test_has_returns_false_for_missing(self, registry):
        """has() returns False for unregistered models."""
        assert registry.has("missing") is False


class TestListModels:
    def test_list_models_sorted_by_name(self, registry, sample_models):
        """Models are listed sorted by name."""
        for m in sample_models:
            registry.add(m)
        names = [m.name for m in registry.list_models()]
        assert names == ["gemma-2", "mistral-7b", "phi-3-mini"]

    def test_list_models_empty_registry(self, registry):
        """Empty registry returns empty list."""
        assert registry.list_models() == []


class TestClear:
    def test_clear_removes_all_models(self, registry, sample_models):
        """clear() removes all registered models."""
        for m in sample_models:
            registry.add(m)
        registry.clear()
        assert registry.list_models() == []
        assert registry.has("phi-3-mini") is False
