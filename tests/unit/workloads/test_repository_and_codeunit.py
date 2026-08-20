"""Tests for repository snapshot and code unit contracts."""

import pytest
from pydantic import ValidationError

from localbench.workloads.code_retrieval.schemas import (
    CodeUnit,
    CodeUnitContext,
    SourceRepositorySnapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_SOURCE_3_LINES = "line one\nline two\nline three"
_VALID_SOURCE_10_LINES = "\n".join(f"line {i}" for i in range(1, 11))


def _make_code_unit(**overrides) -> dict:
    """Return a dict that satisfies all CodeUnit required fields."""
    base = {
        "id": "repo001_py_func_add",
        "repository": "repo001",
        "language": "python",
        "file_path": "math_utils.py",
        "symbol": "add",
        "symbol_type": "function",
        "source_code": _VALID_SOURCE_3_LINES,
        "source_url": "https://github.com/example/repo/blob/main/math_utils.py#L1",
        "split": "train",
        "is_public": True,
        "source_file_lines": 3,
        "extracted_at": "2026-08-19T10:30:00Z",
    }
    base.update(overrides)
    return base


# ===========================================================================
# SourceRepositorySnapshot
# ===========================================================================


class TestSourceRepositorySnapshot:
    def test_valid(self):
        snap = SourceRepositorySnapshot(
            repository="repo001",
            commit="abc1234",
        )
        assert snap.repository == "repo001"
        assert snap.commit == "abc1234"
        assert snap.content_hash == ""

    def test_with_content_hash(self):
        snap = SourceRepositorySnapshot(
            repository="repo001",
            commit="abc1234",
            content_hash="sha256:abcdef",
        )
        assert snap.content_hash == "sha256:abcdef"

    def test_missing_repository(self):
        with pytest.raises(ValidationError):
            SourceRepositorySnapshot(commit="abc1234")

    def test_missing_commit(self):
        with pytest.raises(ValidationError):
            SourceRepositorySnapshot(repository="repo001")


# ===========================================================================
# CodeUnitContext
# ===========================================================================


class TestCodeUnitContext:
    def test_defaults(self):
        ctx = CodeUnitContext()
        assert ctx.class_name is None
        assert ctx.module_docstring is None
        assert ctx.imports == []
        assert ctx.parent_methods == []

    def test_full_context(self):
        ctx = CodeUnitContext(
            class_name="PaymentProcessor",
            module_docstring="Payment processing module.",
            imports=["time", "logging"],
            parent_methods=["__init__", "validate"],
        )
        assert ctx.class_name == "PaymentProcessor"
        assert len(ctx.imports) == 2


# ===========================================================================
# CodeUnit — valid cases
# ===========================================================================


class TestCodeUnitValid:
    def test_valid_function(self):
        cu = CodeUnit(**_make_code_unit())
        assert cu.id == "repo001_py_func_add"
        assert cu.language == "python"
        assert cu.split == "train"

    def test_valid_class_method(self):
        data = _make_code_unit(
            id="repo001_py_class_Pay_method_process",
            symbol="PaymentProcessor.process",
            symbol_type="method",
            context=CodeUnitContext(
                class_name="PaymentProcessor",
                imports=["time"],
            ),
        )
        cu = CodeUnit(**data)
        assert cu.symbol_type == "method"
        assert cu.context.class_name == "PaymentProcessor"

    def test_test_split(self):
        cu = CodeUnit(**_make_code_unit(split="test"))
        assert cu.split == "test"

    def test_validation_split(self):
        cu = CodeUnit(**_make_code_unit(split="validation"))
        assert cu.split == "validation"

    def test_boundary_3_lines(self):
        cu = CodeUnit(**_make_code_unit(source_code=_VALID_SOURCE_3_LINES))
        assert cu.source_code == _VALID_SOURCE_3_LINES

    def test_boundary_100_lines(self):
        source = "\n".join(f"line {i}" for i in range(1, 101))
        cu = CodeUnit(**_make_code_unit(source_code=source))
        assert "\n" in cu.source_code

    def test_optional_docstring(self):
        cu = CodeUnit(**_make_code_unit(docstring="Add two numbers."))
        assert cu.docstring == "Add two numbers."

    def test_empty_docstring_default(self):
        cu = CodeUnit(**_make_code_unit())
        assert cu.docstring == ""

    def test_is_public_false(self):
        cu = CodeUnit(**_make_code_unit(is_public=False))
        assert cu.is_public is False


# ===========================================================================
# CodeUnit — invalid cases
# ===========================================================================


class TestCodeUnitInvalid:
    def test_too_few_lines(self):
        with pytest.raises(Exception, match="minimum"):
            CodeUnit(**_make_code_unit(source_code="x = 1"))

    def test_too_many_lines(self):
        source = "\n".join(f"line {i}" for i in range(1, 102))
        with pytest.raises(Exception, match="maximum"):
            CodeUnit(**_make_code_unit(source_code=source))

    def test_blank_lines_not_counted(self):
        """Blank lines should not count toward the minimum."""
        source = "line one\n\n\nline two\n\nline three"
        cu = CodeUnit(**_make_code_unit(source_code=source))
        assert cu.source_code == source

    def test_missing_id(self):
        with pytest.raises(ValidationError):
            CodeUnit(**{k: v for k, v in _make_code_unit().items() if k != "id"})

    def test_missing_repository(self):
        with pytest.raises(ValidationError):
            CodeUnit(**{
                k: v for k, v in _make_code_unit().items()
                if k != "repository"
            })

    def test_missing_source_code(self):
        with pytest.raises(ValidationError):
            CodeUnit(**{
                k: v for k, v in _make_code_unit().items()
                if k != "source_code"
            })

    def test_invalid_language(self):
        with pytest.raises(ValidationError):
            CodeUnit(**_make_code_unit(language="rust"))

    def test_invalid_split(self):
        with pytest.raises(ValidationError):
            CodeUnit(**_make_code_unit(split="dev"))

    def test_invalid_symbol_type(self):
        with pytest.raises(ValidationError):
            CodeUnit(**_make_code_unit(symbol_type="class"))

    def test_missing_source_url(self):
        with pytest.raises(ValidationError):
            CodeUnit(**{
                k: v for k, v in _make_code_unit().items()
                if k != "source_url"
            })

    def test_missing_extracted_at(self):
        with pytest.raises(ValidationError):
            CodeUnit(**{
                k: v for k, v in _make_code_unit().items()
                if k != "extracted_at"
            })

    def test_missing_source_file_lines(self):
        with pytest.raises(ValidationError):
            CodeUnit(**{
                k: v for k, v in _make_code_unit().items()
                if k != "source_file_lines"
            })
