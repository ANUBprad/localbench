"""Integration coverage: the fail-fast generation lock around the generator."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from localbench.workloads.code_retrieval.candidate_store import (
    CandidateStore,
)
from localbench.workloads.code_retrieval.run_lock import (
    LOCK_FILENAME,
    generation_run_lock,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "generate_query_candidates.py"
)
_spec = importlib.util.spec_from_file_location(
    "generate_query_candidates_lock_integration", _SCRIPT_PATH
)
_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(
    "generate_query_candidates_lock_integration", _script
)
_spec.loader.exec_module(_script)

_UNITS = [("repo003", "alpha"), ("repo006", "beta")]


def _split_row(repo: str, symbol: str) -> dict:
    return {
        "repository": repo,
        "language": "python",
        "file_path": "pkg/module.py",
        "symbol": symbol,
        "symbol_type": "function",
        "source_code": "def f():\n    return 1\n",
        "context": {},
        "source_url": "",
        "is_public": True,
        "docstring": "",
        "source_file_lines": 2,
        "content_hash": f"hash-{repo}-{symbol}",
        "extracted_at": "2026-08-21T00:00:00Z",
    }


def _write_split(path: Path, units) -> None:
    lines = [json.dumps(_split_row(r, s)) for r, s in units]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Script wired to fake model infrastructure; Ollama is never touched."""
    split = tmp_path / "test_split.jsonl"
    outdir = tmp_path / "out"
    _write_split(split, _UNITS)
    state = SimpleNamespace(
        generated=[],
        adapters_built=0,
        store_saw_lock=[],
        fail_next=False,
    )

    class FakeAdapter:
        def __init__(self, model_name=None):
            state.adapters_built += 1
            self.model_name = model_name

        def health_check(self):
            return True

        def discover_models(self):
            return []

        def close(self):
            pass

    class FakeQueryGenerator:
        def __init__(self, model=None, policy=None, top_p=None, seed=None):
            pass

        def generate(self, unit):
            if state.fail_next:
                raise RuntimeError("simulated generation crash")
            state.generated.append(unit)
            return SimpleNamespace(
                success=True,
                candidate=SimpleNamespace(
                    query=f"locate the {unit.symbol} implementation",
                    query_style="natural",
                    query_intent="find_implementation",
                ),
                attempts=[],
                total_generation_ms=1.0,
                total_validation_ms=0.5,
                leakage=None,
            )

    class RecordingStore(CandidateStore):
        def __init__(self, candidates_path, failures_path):
            state.store_saw_lock.append(
                (candidates_path.parent / LOCK_FILENAME).exists()
            )
            super().__init__(candidates_path, failures_path)

    monkeypatch.setattr(_script, "TEST_SPLIT_PATH", split)
    monkeypatch.setattr(_script, "OllamaAdapter", FakeAdapter)
    monkeypatch.setattr(_script, "QueryGenerator", FakeQueryGenerator)
    monkeypatch.setattr(_script, "CandidateStore", RecordingStore)
    return SimpleNamespace(tmp=tmp_path, split=split, outdir=outdir, state=state)


def _run(harness, output_dir=None, extra_args=()):
    argv = ["--output-dir", str(output_dir or harness.outdir), *extra_args]
    return _script.main(argv)


def _artifact_state(output_dir: Path) -> dict[str, bytes]:
    return {
        p.name: p.read_bytes()
        for p in sorted(output_dir.iterdir())
        if p.is_file()
    }


class TestLockCoverage:
    def test_lock_held_during_checkpoint_load_and_released_after(
        self, harness
    ):
        assert _run(harness) == 0
        # Checkpoint loading ran while the directory was locked...
        assert harness.state.store_saw_lock == [True]
        # ...and normal completion released it.
        assert not (harness.outdir / LOCK_FILENAME).exists()

    def test_second_invocation_fails_fast_with_message(self, harness, caplog):
        assert _run(harness) == 0
        with generation_run_lock(harness.outdir):
            with caplog.at_level(logging.ERROR):
                exit_code = _run(harness)
        assert exit_code == 4
        assert "already using" in caplog.text

    def test_refused_invocation_contacts_no_model(self, harness):
        assert _run(harness) == 0
        with generation_run_lock(harness.outdir):
            assert _run(harness) == 4
        # Refusal happens before adapter construction: no health check,
        # no model discovery, zero generations.
        assert harness.state.adapters_built == 1
        assert len(harness.state.generated) == len(_UNITS)

    def test_artifacts_untouched_after_refusal(self, harness):
        assert _run(harness) == 0
        before = _artifact_state(harness.outdir)
        with generation_run_lock(harness.outdir):
            assert _run(harness) == 4
        assert _artifact_state(harness.outdir) == before

    def test_different_output_directories_run_independently(self, harness):
        other = harness.tmp / "other-out"
        other.mkdir()
        with generation_run_lock(other):
            held_before = (other / LOCK_FILENAME).read_bytes()
            assert _run(harness) == 0
            # The unrelated run neither blocked nor disturbed this lock.
            assert (other / LOCK_FILENAME).read_bytes() == held_before
        assert not (other / LOCK_FILENAME).exists()

    def test_lock_released_after_normal_completion(self, harness):
        assert _run(harness) == 0
        with generation_run_lock(harness.outdir):
            pass

    def test_lock_released_when_generation_raises(self, harness):
        harness.state.fail_next = True
        with pytest.raises(RuntimeError, match="simulated"):
            _run(harness)
        assert not (harness.outdir / LOCK_FILENAME).exists()
        with generation_run_lock(harness.outdir):
            pass


class TestResumeUnderLockIntegration:
    def test_checkpoint_resume_skips_completed_units(self, harness):
        assert _run(harness) == 0
        first_pass = list(harness.state.generated)
        assert len(first_pass) == len(_UNITS)

        assert _run(harness) == 0
        # Second invocation resumed from the checkpoint: nothing new.
        assert harness.state.generated == first_pass

        successful, failed = CandidateStore(
            harness.outdir / "candidates.jsonl",
            harness.outdir / "candidate_failures.jsonl",
        ).load()
        assert len(successful) == len(_UNITS)
        assert failed == []
