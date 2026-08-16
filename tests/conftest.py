"""Pytest configuration and fixtures for LocalBench tests."""

import pytest


@pytest.fixture
def sample_fixture():
    """Sample fixture for future tests."""
    return {"test": "data"}
