"""Tests for Ollama adapter (mocked HTTP)."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from localbench.errors import (
    GenerationError,
    ModelNotFoundError,
    OllamaUnavailableError,
)
from localbench.runtime.model import GenerationRequest
from localbench.runtime.ollama.adapter import OllamaAdapter


@pytest.fixture
def adapter():
    """Create an Ollama adapter with a mock HTTP client."""
    with patch(
        "localbench.runtime.ollama.adapter.httpx.Client"
    ) as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        adapter = OllamaAdapter(base_url="http://localhost:11434")
        adapter._client = mock_client
        yield adapter, mock_client


def _mock_response(json_data, status_code=200):
    """Create a mock httpx.Response."""
    response = MagicMock()
    response.json.return_value = json_data
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=response,
        )
    return response


class TestHealthCheck:
    def test_health_check_returns_true_when_ollama_running(
        self, adapter
    ):
        """Health check returns True when Ollama responds."""
        a, mock_client = adapter
        mock_client.get.return_value = _mock_response({})
        assert a.health_check() is True

    def test_health_check_accepts_plain_text_body(self, adapter):
        """Ollama's root endpoint returns text/plain, not JSON."""
        a, mock_client = adapter
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.side_effect = json.JSONDecodeError(
            "Expecting value", "Ollama is running", 0
        )
        mock_client.get.return_value = response
        assert a.health_check() is True

    def test_health_check_returns_false_when_ollama_unavailable(
        self, adapter
    ):
        """Health check returns False when Ollama is not reachable."""
        a, mock_client = adapter
        mock_client.get.side_effect = httpx.ConnectError(
            "Connection refused"
        )
        assert a.health_check() is False


class TestDiscoverModels:
    def test_discover_models_returns_model_list(self, adapter):
        """Model discovery returns ModelInfo objects."""
        a, mock_client = adapter
        mock_client.get.return_value = _mock_response(
            {
                "models": [
                    {
                        "name": "phi-3-mini:latest",
                        "size": 2048000000,
                    },
                    {
                        "name": "mistral-7b:latest",
                        "size": 3800000000,
                    },
                ]
            }
        )
        models = a.discover_models()
        assert len(models) == 2
        assert models[0].name == "phi-3-mini"
        assert models[1].name == "mistral-7b"

    def test_discover_models_strips_latest_tag(self, adapter):
        """The :latest tag is stripped from model names."""
        a, mock_client = adapter
        mock_client.get.return_value = _mock_response(
            {
                "models": [
                    {"name": "llama2:latest", "size": 1000}
                ]
            }
        )
        models = a.discover_models()
        assert models[0].name == "llama2"

    def test_discover_models_handles_empty_list(self, adapter):
        """Empty model list is handled gracefully."""
        a, mock_client = adapter
        mock_client.get.return_value = _mock_response(
            {"models": []}
        )
        models = a.discover_models()
        assert models == []

    def test_discover_models_raises_on_connection_error(
        self, adapter
    ):
        """Connection error raises OllamaUnavailableError."""
        a, mock_client = adapter
        mock_client.get.side_effect = httpx.ConnectError(
            "Connection refused"
        )
        with pytest.raises(OllamaUnavailableError):
            a.discover_models()


class TestGenerate:
    def test_generate_returns_result(self, adapter):
        """Successful generation returns a GenerationResult."""
        a, mock_client = adapter
        mock_client.post.return_value = _mock_response(
            {
                "model": "phi-3-mini",
                "response": "Hello, world!",
                "done": True,
                "total_duration": 500000000,
                "eval_count": 10,
                "eval_duration": 300000000,
            }
        )
        req = GenerationRequest(
            prompt="Say hello", model="phi-3-mini"
        )
        result = a.generate(req)
        assert result.text == "Hello, world!"
        assert result.model == "phi-3-mini"
        assert result.duration_ms == 500.0
        assert result.done is True

    def test_generate_maps_model_not_found(self, adapter):
        """Ollama 'not found' error raises ModelNotFoundError."""
        a, mock_client = adapter
        mock_client.post.return_value = _mock_response(
            {"error": "model 'nonexistent' not found"}
        )
        with pytest.raises(ModelNotFoundError):
            req = GenerationRequest(
                prompt="test", model="nonexistent"
            )
            a.generate(req)

    def test_generate_maps_404_to_model_not_found(self, adapter):
        """HTTP 404 from Ollama maps to ModelNotFoundError."""
        a, mock_client = adapter
        mock_client.post.return_value = _mock_response(
            {}, status_code=404
        )
        with pytest.raises(ModelNotFoundError):
            req = GenerationRequest(
                prompt="test", model="missing-model"
            )
            a.generate(req)

    def test_generate_raises_on_connection_error(self, adapter):
        """Connection error raises OllamaUnavailableError."""
        a, mock_client = adapter
        mock_client.post.side_effect = httpx.ConnectError(
            "Connection refused"
        )
        with pytest.raises(OllamaUnavailableError):
            req = GenerationRequest(
                prompt="test", model="phi-3-mini"
            )
            a.generate(req)

    def test_generate_raises_generation_error(self, adapter):
        """Unexpected failures raise GenerationError."""
        a, mock_client = adapter
        mock_client.post.side_effect = ValueError("Unexpected")
        with pytest.raises(GenerationError):
            req = GenerationRequest(
                prompt="test", model="phi-3-mini"
            )
            a.generate(req)


class TestClose:
    def test_close_closes_http_client(self, adapter):
        """close() closes the underlying HTTP client."""
        a, mock_client = adapter
        a.close()
        mock_client.close.assert_called_once()
