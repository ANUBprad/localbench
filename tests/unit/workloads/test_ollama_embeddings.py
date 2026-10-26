"""Offline tests for the Ollama embedding transport."""
from unittest.mock import Mock

import httpx
import pytest

from localbench.workloads.code_retrieval.ollama_embeddings import (
    OllamaEmbeddingError,
    OllamaEmbeddings,
)


def test_batch_request(monkeypatch):
    calls = []

    def fake_post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return Mock(json=lambda: {"embeddings": [[1.0, 0.0], [0.0, 1.0]]},
                    raise_for_status=lambda: None)

    monkeypatch.setattr(httpx, "post", fake_post)
    encoder = OllamaEmbeddings("nomic-embed-text")
    assert encoder(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert calls[0][1] == {
        "model": "nomic-embed-text", "input": ["a", "b"], "truncate": False
    }


def test_empty_batch_makes_no_request(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("unexpected network request")
    monkeypatch.setattr(httpx, "post", fail)
    assert OllamaEmbeddings("model")([]) == []


@pytest.mark.parametrize("payload", [{}, {"embeddings": []},
                                    {"embeddings": ["bad"]}])
def test_bad_response(monkeypatch, payload):
    monkeypatch.setattr(
        httpx, "post",
        lambda *args, **kwargs: Mock(
            json=lambda: payload, raise_for_status=lambda: None
        ),
    )
    with pytest.raises(OllamaEmbeddingError):
        OllamaEmbeddings("model")(["a"])


def test_connection_error(monkeypatch):
    def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "post", fail)
    with pytest.raises(OllamaEmbeddingError):
        OllamaEmbeddings("model")(["a"])


def test_model_and_timeout_validation():
    with pytest.raises(ValueError):
        OllamaEmbeddings("")
    with pytest.raises(ValueError):
        OllamaEmbeddings("model", timeout=0)
