"""Tests for LocalBench version."""

from localbench import __version__


def test_version_is_string():
    """Version should be a string."""
    assert isinstance(__version__, str)


def test_version_format():
    """Version should follow semver-like format."""
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts)
