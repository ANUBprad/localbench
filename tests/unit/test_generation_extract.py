"""Tests for JSON extraction from raw model output."""

import pytest

from localbench.runtime.generation.extract import extract_json
from localbench.runtime.generation.failures import MalformedJSONError


class TestExtractCleanJSON:
    def test_extracts_plain_object(self):
        """Clean JSON object is parsed directly."""
        raw = '{"name": "Alice", "age": 30}'
        result = extract_json(raw)
        assert result == {"name": "Alice", "age": 30}

    def test_extracts_plain_array(self):
        """Clean JSON array is parsed directly."""
        raw = '[1, 2, 3]'
        result = extract_json(raw)
        assert result == [1, 2, 3]

    def test_extracts_nested_object(self):
        """Nested JSON structures are handled."""
        raw = '{"a": {"b": [1, 2]}}'
        result = extract_json(raw)
        assert result == {"a": {"b": [1, 2]}}


class TestExtractFencedJSON:
    def test_extracts_json_fenced_with_json_lang(self):
        """```json ... ``` fences are stripped."""
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_extracts_json_fenced_without_lang(self):
        """``` ... ``` fences without lang tag are stripped."""
        raw = '```\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_fenced_with_surrounding_text(self):
        """Text before and after fences is ignored."""
        raw = 'Here is the result:\n```json\n{"x": 1}\n```\nDone.'
        result = extract_json(raw)
        assert result == {"x": 1}


class TestExtractBraceMatching:
    def test_extracts_json_with_prefix_text(self):
        """JSON preceded by conversational text is extracted."""
        raw = 'Sure! Here is the data:\n{"field": "value"}'
        result = extract_json(raw)
        assert result == {"field": "value"}

    def test_extracts_json_with_suffix_text(self):
        """JSON followed by commentary is extracted."""
        raw = '{"field": "value"}\nLet me know if you need more.'
        result = extract_json(raw)
        assert result == {"field": "value"}

    def test_extracts_first_json_when_multiple(self):
        """When multiple JSON blocks exist, the first is returned."""
        raw = '{"first": 1} some text {"second": 2}'
        result = extract_json(raw)
        assert result == {"first": 1}

    def test_extracts_array_with_surrounding_text(self):
        """Arrays are extracted via brace matching."""
        raw = 'Result: [1, 2, 3] done'
        result = extract_json(raw)
        assert result == [1, 2, 3]


class TestExtractErrors:
    def test_empty_string_raises(self):
        """Empty input raises MalformedJSON."""
        with pytest.raises(MalformedJSONError, match="empty"):
            extract_json("")

    def test_whitespace_only_raises(self):
        """Whitespace-only input raises MalformedJSON."""
        with pytest.raises(MalformedJSONError, match="empty"):
            extract_json("   \n  \t  ")

    def test_pure_text_raises(self):
        """Non-JSON text raises MalformedJSON."""
        with pytest.raises(MalformedJSONError):
            extract_json("Hello, how are you?")

    def test_truncated_json_raises(self):
        """Truncated JSON raises MalformedJSON."""
        with pytest.raises(MalformedJSONError):
            extract_json('{"name": "Alice", "age":')

    def test_invalid_fence_content_raises(self):
        """Fenced block with invalid JSON raises MalformedJSON."""
        raw = '```json\n{not valid json}\n```'
        with pytest.raises(MalformedJSONError):
            extract_json(raw)


class TestExtractEdgeCases:
    def test_json_with_string_containing_braces(self):
        """JSON with braces inside string values is handled."""
        raw = '{"template": "Hello {name}!"}'
        result = extract_json(raw)
        assert result["template"] == "Hello {name}!"

    def test_json_with_escaped_quotes(self):
        """JSON with escaped quotes in strings is handled."""
        raw = r'{"msg": "He said \"hello\""}'
        result = extract_json(raw)
        assert result["msg"] == 'He said "hello"'

    def test_json_with_newlines_in_strings(self):
        """JSON with newlines in string values is handled."""
        raw = '{"text": "line1\\nline2"}'
        result = extract_json(raw)
        assert result["text"] == "line1\nline2"
