"""Tests for Python AST code unit extraction.

All tests are deterministic, network-free, and use local fixtures.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from localbench.workloads.code_retrieval.extraction import (
    _build_code_unit_id,
    _build_symbol_path,
    _extract_imports,
    _extract_module_docstring,
    _get_source_segment,
    _hash_source,
    _is_excluded_by_directory,
    _is_excluded_by_filename,
    _is_private,
    _parse_source,
    discover_python_files,
    extract_code_units,
)

# ---------------------------------------------------------------------------
# Fixtures (source code strings)
# ---------------------------------------------------------------------------

SIMPLE_FUNC = (
    "def greet(name):\n"
    '    """Greet someone."""\n'
    '    message = f"Hello, {name}!"\n'
    "    print(message)\n"
    "    return message\n"
)

SHORT_FUNC = "def add(a, b):\n    return a + b\n"

LONG_FUNC = "def big():\n" + "    x = 1\n" * 105

CLASS_METHODS = (
    'class PaymentProcessor:\n'
    '    """Process payments."""\n'
    "\n"
    "    def __init__(self, config):\n"
    '        """Initialise processor."""\n'
    "        self.config = config\n"
    "        self.logger = None\n"
    "\n"
    "    def process_retry(self, tid, max_attempts=3):\n"
    '        """Retry a failed payment."""\n'
    "        attempts = 0\n"
    "        while attempts < max_attempts:\n"
    "            try:\n"
    "                return self._do_process(tid)\n"
    "            except Exception:\n"
    "                attempts += 1\n"
    "        raise RuntimeError(\"Failed\")\n"
    "\n"
    "    def _do_process(self, tid):\n"
    '        """Internal processing."""\n'
    "        return True\n"
)

NESTED_FUNC = (
    "def outer():\n"
    '    """Outer function."""\n'
    "    x = 1\n"
    "\n"
    "    def inner():\n"
    '        """Nested."""\n'
    "        return x\n"
    "\n"
    "    return inner()\n"
)

PRIVATE_FUNC = (
    "def _helper():\n"
    '    """Private helper."""\n'
    "    return 42\n"
)

DUNDER_CLASS = (
    'class MyClass:\n'
    '    """A class."""\n'
    "\n"
    "    def __init__(self, value):\n"
    '        """Initialise."""\n'
    "        self.value = value\n"
    "\n"
    "    def __repr__(self):\n"
    '        """String repr."""\n'
    '        return f"MyClass({self.value})"\n'
    "\n"
    "    def public_method(self, x):\n"
    '        """A public method."""\n'
    "        result = self.value + x\n"
    "        return result\n"
)

ASYNC_FUNC = (
    "async def fetch_data(url):\n"
    '    """Fetch data asynchronously."""\n'
    "    import aiohttp\n"
    "    async with aiohttp.ClientSession() as session:\n"
    "        async with session.get(url) as resp:\n"
    "            return await resp.json()\n"
)

DECORATED_FUNC = (
    "from functools import lru_cache\n"
    "\n"
    "@lru_cache(maxsize=128)\n"
    "def cached_compute(x):\n"
    '    """Compute with caching."""\n'
    "    result = x * 2\n"
    "    return result\n"
)

STATIC_CLASS = (
    'class Utilities:\n'
    '    """Utility class."""\n'
    "\n"
    "    @staticmethod\n"
    "    def add(a, b):\n"
    '        """Add two numbers."""\n'
    "        return a + b\n"
    "\n"
    "    @classmethod\n"
    "    def from_config(cls, config):\n"
    '        """Create from config."""\n'
    "        instance = cls()\n"
    "        return instance\n"
)

MODULE_IMPORTS = (
    "import os\n"
    "import sys\n"
    "from pathlib import Path\n"
    "from typing import List\n"
    "\n"
    "def main():\n"
    '    """Entry point."""\n'
    '    p = Path(".")\n'
    "    return p\n"
)


# ===========================================================================
# discover_python_files
# ===========================================================================


class TestDiscoverPythonFiles:
    def test_finds_python_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("pass")
        (tmp_path / "b.py").write_text("pass")
        result = discover_python_files(tmp_path)
        assert result == [Path("a.py"), Path("b.py")]

    def test_ignores_non_python(self, tmp_path: Path) -> None:
        (tmp_path / "readme.md").write_text("hi")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "code.py").write_text("pass")
        result = discover_python_files(tmp_path)
        assert result == [Path("code.py")]

    def test_deterministic_ordering(self, tmp_path: Path) -> None:
        for name in ["z.py", "a.py", "m.py"]:
            (tmp_path / name).write_text("pass")
        first = discover_python_files(tmp_path)
        second = discover_python_files(tmp_path)
        assert first == second
        assert first == sorted(first)

    def test_excludes_test_directory(self, tmp_path: Path) -> None:
        (tmp_path / "src.py").write_text("pass")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_something.py").write_text("pass")
        result = discover_python_files(tmp_path)
        assert result == [Path("src.py")]

    def test_excludes_vendor_directory(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("pass")
        (tmp_path / "vendor").mkdir()
        (tmp_path / "vendor" / "lib.py").write_text("pass")
        result = discover_python_files(tmp_path)
        assert result == [Path("app.py")]

    def test_excludes_generated_directory(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / "generated").mkdir()
        (tmp_path / "generated" / "auto.py").write_text("pass")
        result = discover_python_files(tmp_path)
        assert result == [Path("main.py")]

    def test_excludes_pycache(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("pass")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("pass")
        result = discover_python_files(tmp_path)
        assert result == [Path("app.py")]

    def test_excludes_egg_info(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("pass")
        (tmp_path / "mypackage.egg-info").mkdir()
        result = discover_python_files(tmp_path)
        assert result == [Path("app.py")]

    def test_excludes_test_prefixed_files(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("pass")
        (tmp_path / "test_app.py").write_text("pass")
        result = discover_python_files(tmp_path)
        assert result == [Path("app.py")]

    def test_excludes_test_suffixed_files(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("pass")
        (tmp_path / "app_test.py").write_text("pass")
        result = discover_python_files(tmp_path)
        assert result == [Path("app.py")]

    def test_subdirectory_files(self, tmp_path: Path) -> None:
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "mod.py").write_text("pass")
        (tmp_path / "root.py").write_text("pass")
        result = discover_python_files(tmp_path)
        assert result == [Path("pkg/mod.py"), Path("root.py")]

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        result = discover_python_files(tmp_path / "nope")
        assert result == []


# ===========================================================================
# Exclusion helpers
# ===========================================================================


class TestExclusionHelpers:
    def test_directory_excluded(self) -> None:
        assert _is_excluded_by_directory(Path("tests/test_foo.py"))

    def test_directory_not_excluded(self) -> None:
        assert not _is_excluded_by_directory(Path("src/module.py"))

    def test_filename_test_prefix(self) -> None:
        assert _is_excluded_by_filename(Path("test_module.py"))

    def test_filename_test_suffix(self) -> None:
        assert _is_excluded_by_filename(Path("module_test.py"))

    def test_filename_normal(self) -> None:
        assert not _is_excluded_by_filename(Path("module.py"))


# ===========================================================================
# _parse_source
# ===========================================================================


class TestParseSource:
    def test_valid_python(self) -> None:
        tree = _parse_source("def foo(): pass\n")
        assert tree is not None
        assert isinstance(tree, ast.Module)

    def test_syntax_error(self) -> None:
        tree = _parse_source("def broken(:\n    bad\n")
        assert tree is None

    def test_empty_file(self) -> None:
        tree = _parse_source("")
        assert tree is not None


# ===========================================================================
# _is_private
# ===========================================================================


class TestIsPrivate:
    def test_single_underscore(self) -> None:
        assert _is_private("_helper")

    def test_no_underscore(self) -> None:
        assert not _is_private("public")

    def test_dunder_not_private(self) -> None:
        assert not _is_private("__init__")

    def test_double_not_dunder(self) -> None:
        assert not _is_private("__foo")


# ===========================================================================
# Symbol path / ID / Hash
# ===========================================================================


class TestSymbolPath:
    def test_function(self) -> None:
        assert _build_symbol_path("add") == "add"

    def test_method(self) -> None:
        assert _build_symbol_path("process", "Processor") == "Processor.process"


class TestCodeUnitId:
    def test_function(self) -> None:
        result = _build_code_unit_id("repo001", "src/calc.py", "add", "a" * 64)
        assert result == "repo001_py_src_calc_py__add_" + "a" * 12

    def test_method(self) -> None:
        result = _build_code_unit_id(
            "repo001",
            "pkg/payments.py",
            "PaymentProcessor.process",
            "b" * 64,
        )
        expected = (
            "repo001_py_pkg_payments_py__PaymentProcessor_process_" + "b" * 12
        )
        assert result == expected

    def test_same_symbol_different_modules_do_not_collide(self) -> None:
        a = _build_code_unit_id(
            "repo002", "examples/aliases/aliases.py", "cli", "a" * 64
        )
        b = _build_code_unit_id(
            "repo002", "examples/repo/repo.py", "cli", "a" * 64
        )
        assert a != b

    def test_repeated_definition_different_bodies_do_not_collide(self) -> None:
        a = _build_code_unit_id(
            "repo002", "src/click/core.py", "Context.invoke", "a" * 64
        )
        b = _build_code_unit_id(
            "repo002", "src/click/core.py", "Context.invoke", "b" * 64
        )
        assert a != b

    def test_identical_identity_fields_produce_identical_id(self) -> None:
        a = _build_code_unit_id(
            "repo003", "src/rich/console.py", "Console.print", "c" * 64
        )
        b = _build_code_unit_id(
            "repo003", "src/rich/console.py", "Console.print", "c" * 64
        )
        assert a == b

    def test_distinct_content_hashes_produce_distinct_ids(self) -> None:
        a = _build_code_unit_id("repo001", "m.py", "f", "d" * 64)
        b = _build_code_unit_id("repo001", "m.py", "f", "e" * 64)
        assert a != b

    def test_windows_separators_normalized(self) -> None:
        a = _build_code_unit_id(
            "repo001", "src\\click\\cli.py", "cli", "a" * 64
        )
        b = _build_code_unit_id("repo001", "src/click/cli.py", "cli", "a" * 64)
        assert a == b == "repo001_py_src_click_cli_py__cli_" + "a" * 12


class TestHashSource:
    def test_same_source_same_hash(self) -> None:
        src = "def foo():\n    return 1\n"
        assert _hash_source(src) == _hash_source(src)

    def test_different_source_different_hash(self) -> None:
        h1 = _hash_source("def foo(): return 1\n")
        h2 = _hash_source("def foo(): return 2\n")
        assert h1 != h2

    def test_strips_whitespace(self) -> None:
        h1 = _hash_source("  def foo(): pass  ")
        h2 = _hash_source("def foo(): pass")
        assert h1 == h2

    def test_deterministic(self) -> None:
        src = "x = 42\n"
        assert len({_hash_source(src) for _ in range(10)}) == 1


# ===========================================================================
# Module docstring / imports
# ===========================================================================


class TestModuleDocstring:
    def test_with_docstring(self) -> None:
        tree = ast.parse('"""Module doc."""\n')
        assert _extract_module_docstring(tree) == "Module doc."

    def test_without_docstring(self) -> None:
        tree = ast.parse("x = 1\n")
        assert _extract_module_docstring(tree) == ""


class TestExtractImports:
    def test_import_statements(self) -> None:
        tree = ast.parse("import os\nimport sys\n")
        assert _extract_imports(tree) == ["os", "sys"]

    def test_from_import(self) -> None:
        tree = ast.parse("from pathlib import Path\n")
        assert _extract_imports(tree) == ["pathlib"]

    def test_no_imports(self) -> None:
        tree = ast.parse("x = 1\n")
        assert _extract_imports(tree) == []


# ===========================================================================
# _get_source_segment
# ===========================================================================


class TestGetSourceSegment:
    def test_simple_function(self) -> None:
        src = "def foo():\n    return 1\n"
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src)
        seg = _get_source_segment(lines, tree.body[0])
        assert "def foo():" in seg
        assert "return 1" in seg

    def test_includes_decorators(self) -> None:
        src = "@decorator\ndef foo():\n    pass\n"
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src)
        seg = _get_source_segment(lines, tree.body[0])
        assert "@decorator" in seg
        assert "def foo():" in seg


# ===========================================================================
# Full extraction: extract_code_units
# ===========================================================================


class TestExtractCodeUnits:
    def _write(self, tmp: Path, name: str, content: str) -> None:
        (tmp / name).write_text(content, encoding="utf-8")

    def test_top_level_function(self, tmp_path: Path) -> None:
        self._write(tmp_path, "greet.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "repo001", "abc123")
        assert len(result.code_units) == 1
        u = result.code_units[0]
        assert u.symbol == "greet"
        assert u.symbol_type == "function"
        assert u.repository == "repo001"
        assert u.language == "python"
        assert u.file_path == "greet.py"
        assert "Greet someone" in u.docstring
        assert u.is_public is True

    def test_short_function_rejected(self, tmp_path: Path) -> None:
        self._write(tmp_path, "add.py", SHORT_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert len(result.code_units) == 0
        assert len(result.empty_files) == 1

    def test_long_function_rejected(self, tmp_path: Path) -> None:
        self._write(tmp_path, "big.py", LONG_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert len(result.code_units) == 0
        assert len(result.empty_files) == 1

    def test_class_methods(self, tmp_path: Path) -> None:
        self._write(tmp_path, "proc.py", CLASS_METHODS)
        result = extract_code_units(tmp_path, "repo001", "abc")
        symbols = sorted(u.symbol for u in result.code_units)
        assert "PaymentProcessor.__init__" in symbols
        assert "PaymentProcessor.process_retry" in symbols
        for u in result.code_units:
            assert u.symbol_type == "method"
            assert u.context.class_name == "PaymentProcessor"

    def test_private_methods_excluded(self, tmp_path: Path) -> None:
        self._write(tmp_path, "proc.py", CLASS_METHODS)
        result = extract_code_units(tmp_path, "r", "c")
        symbols = [u.symbol for u in result.code_units]
        assert "PaymentProcessor._do_process" not in symbols

    def test_private_function_excluded(self, tmp_path: Path) -> None:
        self._write(tmp_path, "priv.py", PRIVATE_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert len(result.code_units) == 0

    def test_nested_function_excluded(self, tmp_path: Path) -> None:
        self._write(tmp_path, "nest.py", NESTED_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert len(result.code_units) == 1
        assert result.code_units[0].symbol == "outer"
        # inner must not appear
        assert all("inner" not in u.symbol for u in result.code_units)

    def test_dunder_methods_extracted(self, tmp_path: Path) -> None:
        self._write(tmp_path, "cls.py", DUNDER_CLASS)
        result = extract_code_units(tmp_path, "r", "c")
        symbols = sorted(u.symbol for u in result.code_units)
        assert "MyClass.__init__" in symbols
        assert "MyClass.__repr__" in symbols
        assert "MyClass.public_method" in symbols

    def test_async_function(self, tmp_path: Path) -> None:
        self._write(tmp_path, "async_mod.py", ASYNC_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert len(result.code_units) == 1
        assert result.code_units[0].symbol == "fetch_data"

    def test_decorated_function(self, tmp_path: Path) -> None:
        self._write(tmp_path, "dec.py", DECORATED_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert len(result.code_units) == 1
        assert "@lru_cache" in result.code_units[0].source_code

    def test_static_and_classmethod(self, tmp_path: Path) -> None:
        self._write(tmp_path, "util.py", STATIC_CLASS)
        result = extract_code_units(tmp_path, "r", "c")
        symbols = sorted(u.symbol for u in result.code_units)
        assert "Utilities.add" in symbols
        assert "Utilities.from_config" in symbols

    def test_syntax_error_recorded(self, tmp_path: Path) -> None:
        self._write(tmp_path, "bad.py", "def broken(:\n    bad\n")
        result = extract_code_units(tmp_path, "r", "c")
        assert len(result.parse_errors) == 1
        assert result.parse_errors[0].file_path == "bad.py"
        assert result.parse_errors[0].error_type == "SyntaxError"

    def test_empty_file_categorized(self, tmp_path: Path) -> None:
        self._write(tmp_path, "empty.py", "")
        result = extract_code_units(tmp_path, "r", "c")
        assert len(result.empty_files) == 1
        assert len(result.parse_errors) == 0

    def test_context_populated(self, tmp_path: Path) -> None:
        self._write(tmp_path, "mod.py", MODULE_IMPORTS)
        result = extract_code_units(tmp_path, "r", "c")
        assert len(result.code_units) == 1
        ctx = result.code_units[0].context
        assert ctx.module_docstring is None or isinstance(
            ctx.module_docstring, str
        )
        assert "os" in ctx.imports
        assert "sys" in ctx.imports
        assert "pathlib" in ctx.imports

    def test_parent_methods_populated(self, tmp_path: Path) -> None:
        self._write(tmp_path, "cls.py", CLASS_METHODS)
        result = extract_code_units(tmp_path, "r", "c")
        for u in result.code_units:
            if u.symbol == "PaymentProcessor.process_retry":
                assert "__init__" in u.context.parent_methods
                assert "_do_process" in u.context.parent_methods
                break
        else:
            pytest.fail("process_retry not found")

    def test_source_url_construction(self, tmp_path: Path) -> None:
        self._write(tmp_path, "g.py", SIMPLE_FUNC)
        result = extract_code_units(
            tmp_path, "r", "c", base_url="https://github.com/o/r"
        )
        assert len(result.code_units) == 1
        url = result.code_units[0].source_url
        assert url.startswith("https://github.com/o/r/blob/main/")
        assert "#L" in url

    def test_source_url_empty_when_no_base(self, tmp_path: Path) -> None:
        self._write(tmp_path, "g.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert result.code_units[0].source_url == ""

    def test_is_public_true_for_extracted(self, tmp_path: Path) -> None:
        self._write(tmp_path, "g.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert result.code_units[0].is_public is True

    def test_docstring_captured(self, tmp_path: Path) -> None:
        self._write(tmp_path, "g.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert "Greet someone" in result.code_units[0].docstring

    def test_source_file_lines(self, tmp_path: Path) -> None:
        self._write(tmp_path, "g.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        expected = len(SIMPLE_FUNC.splitlines())
        assert result.code_units[0].source_file_lines == expected

    def test_extracted_at_set(self, tmp_path: Path) -> None:
        self._write(tmp_path, "g.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert result.code_units[0].extracted_at != ""

    def test_deterministic_results(self, tmp_path: Path) -> None:
        self._write(tmp_path, "g.py", SIMPLE_FUNC)
        r1 = extract_code_units(tmp_path, "r", "c")
        r2 = extract_code_units(tmp_path, "r", "c")
        assert len(r1.code_units) == len(r2.code_units)
        for u1, u2 in zip(r1.code_units, r2.code_units, strict=True):
            assert u1.content_hash == u2.content_hash
            assert u1.symbol == u2.symbol

    def test_test_files_excluded(self, tmp_path: Path) -> None:
        self._write(tmp_path, "app.py", SIMPLE_FUNC)
        self._write(tmp_path, "test_app.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert all(u.file_path == "app.py" for u in result.code_units)

    def test_test_dir_excluded(self, tmp_path: Path) -> None:
        self._write(tmp_path, "app.py", SIMPLE_FUNC)
        (tmp_path / "tests").mkdir()
        self._write(tmp_path / "tests", "test_core.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert all(u.file_path == "app.py" for u in result.code_units)

    def test_vendor_dir_excluded(self, tmp_path: Path) -> None:
        self._write(tmp_path, "app.py", SIMPLE_FUNC)
        (tmp_path / "vendor").mkdir()
        self._write(tmp_path / "vendor", "lib.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert all(u.file_path == "app.py" for u in result.code_units)

    def test_generated_dir_excluded(self, tmp_path: Path) -> None:
        self._write(tmp_path, "app.py", SIMPLE_FUNC)
        (tmp_path / "generated").mkdir()
        self._write(tmp_path / "generated", "auto.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        assert all(u.file_path == "app.py" for u in result.code_units)

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        result = extract_code_units(tmp_path / "nope", "r", "c")
        assert len(result.code_units) == 0
        assert len(result.parse_errors) == 0
        assert len(result.skipped_files) == 0

    def test_code_unit_id_deterministic(self, tmp_path: Path) -> None:
        self._write(tmp_path, "g.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "repo001", "c")
        u = result.code_units[0]
        expected_id = _build_code_unit_id(
            "repo001", u.file_path, u.symbol, u.content_hash
        )
        assert _build_code_unit_id(
            "repo001", u.file_path, u.symbol, u.content_hash
        ) == expected_id

    def test_content_hash_matches_direct(self, tmp_path: Path) -> None:
        self._write(tmp_path, "g.py", SIMPLE_FUNC)
        result = extract_code_units(tmp_path, "r", "c")
        direct_hash = _hash_source(result.code_units[0].source_code)
        assert result.code_units[0].content_hash == direct_hash
