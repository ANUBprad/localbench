"""Dense retrieval tests with a deterministic offline fake encoder."""
import pytest

from localbench.workloads.code_retrieval.dense import DenseIndex


def encode(texts):
    lookup = {
        "payment implementation": [1.0, 0.0],
        "email validation": [0.0, 1.0],
        "find payment": [2.0, 0.0],
        "tie": [1.0, 1.0],
    }
    return [lookup[text] for text in texts]


def test_dense_ranks_semantic_match():
    index = DenseIndex(
        {"email": "email validation", "pay": "payment implementation"},
        encode,
    )
    hits = index.search("find payment")
    assert [hit.code_unit_id for hit in hits] == ["pay", "email"]
    assert hits[0].score == pytest.approx(1.0)


def test_dense_ties_are_stable():
    index = DenseIndex({"z": "tie", "a": "tie"}, encode)
    assert [hit.code_unit_id for hit in index.search("tie")] == ["a", "z"]


def test_dense_validates_query_and_k():
    index = DenseIndex({"a": "tie"}, encode)
    assert index.search("  ") == []
    with pytest.raises(ValueError):
        index.search("tie", k=0)
    with pytest.raises(ValueError):
        index.search(123)


@pytest.mark.parametrize("vectors", [
    [], [[0.0, 0.0]], [[float("nan"), 1.0]], [[float("inf"), 1.0]],
])
def test_dense_rejects_bad_document_vectors(vectors):
    with pytest.raises(ValueError):
        DenseIndex({"a": "anything"}, lambda texts: vectors)


def test_dense_rejects_wrong_document_count():
    with pytest.raises(ValueError):
        DenseIndex({"a": "anything"}, lambda texts: [])


def test_dense_rejects_inconsistent_document_dimensions():
    with pytest.raises(ValueError):
        DenseIndex({"a": "one", "b": "two"}, lambda texts: [[1.0], [1.0, 0.0]])


def test_dense_rejects_query_dimension_mismatch():
    def encoder(texts):
        return [[1.0, 0.0] if text == "doc" else [1.0] for text in texts]
    index = DenseIndex({"a": "doc"}, encoder)
    with pytest.raises(ValueError):
        index.search("query")


def test_dense_rejects_duplicate_or_missing_query_vector():
    def encoder(texts):
        if texts == ["query"]:
            return []
        return [[1.0] for _ in texts]
    index = DenseIndex({"a": "doc"}, encoder)
    with pytest.raises(ValueError):
        index.search("query")
