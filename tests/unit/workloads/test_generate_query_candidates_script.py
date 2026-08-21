"""Tests for the candidate generation script's resume filtering."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from localbench.workloads.code_retrieval.extraction import ExtractedCodeUnit
from localbench.workloads.code_retrieval.schemas import CodeUnitContext

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "generate_query_candidates.py"
)
_spec = importlib.util.spec_from_file_location(
    "generate_query_candidates", _SCRIPT_PATH
)
_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("generate_query_candidates", _script)
_spec.loader.exec_module(_script)

filter_pending = _script.filter_pending


def _unit(repo: str, symbol: str) -> ExtractedCodeUnit:
    return ExtractedCodeUnit(
        repository=repo,
        language="python",
        file_path="module.py",
        symbol=symbol,
        symbol_type="function",
        source_code="def f():\n    return 1\n",
        context=CodeUnitContext(),
        source_url="",
        is_public=True,
        docstring="",
        source_file_lines=3,
        content_hash=f"hash-{repo}-{symbol}",
        extracted_at="2026-08-21T00:00:00Z",
    )


def _unit_ids(units):
    return {
        _script._build_code_unit_id(
            u.repository, u.file_path, u.symbol, u.content_hash
        )
        for u in units
    }


class TestFilterPending:
    def test_empty_checkpoint_keeps_all_units(self):
        units = [_unit("repo003", "alpha"), _unit("repo006", "beta")]
        pending = filter_pending(units, set())
        assert pending == units

    def test_restart_skips_successful_record(self):
        units = [_unit("repo003", "alpha"), _unit("repo006", "beta")]
        completed = _unit_ids([units[0]])
        pending = filter_pending(units, completed)
        assert pending == [units[1]]

    def test_restart_skips_failed_record(self):
        units = [_unit("repo003", "alpha"), _unit("repo006", "beta")]
        completed = _unit_ids([units[1]])
        pending = filter_pending(units, completed)
        assert pending == [units[0]]

    def test_fully_completed_checkpoint_leaves_nothing(self):
        units = [_unit("repo003", "alpha"), _unit("repo006", "beta")]
        pending = filter_pending(units, _unit_ids(units))
        assert pending == []

    def test_resume_preserves_input_order(self):
        units = [
            _unit("repo006", "a"),
            _unit("repo003", "b"),
            _unit("repo006", "c"),
        ]
        completed = _unit_ids([units[1]])
        pending = filter_pending(units, completed)
        assert pending == [units[0], units[2]]
