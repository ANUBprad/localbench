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


def test_large_input_is_split_into_ordered_batches(monkeypatch):
    calls = []

    def fake_post(url, *, json, timeout):
        calls.append(json["input"])
        return Mock(
            json=lambda: {"embeddings": [[float(int(x)), 1.0] for x in json["input"]]},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    encoder = OllamaEmbeddings("model", batch_size=2)
    assert encoder(["1", "2", "3"]) == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert calls == [["1", "2"], ["3"]]


def test_http_400_includes_response_detail(monkeypatch):
    def fake_post(url, *, json, timeout):
        request = httpx.Request("POST", url)
        response = httpx.Response(
            400, text='{"error":"input too long"}', request=request
        )
        response.raise_for_status()

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(OllamaEmbeddingError, match="input too long"):
        OllamaEmbeddings("model")(["example"])


def test_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        OllamaEmbeddings("model", batch_size=0)
