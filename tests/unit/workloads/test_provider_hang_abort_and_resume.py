"""Provider hang / abort / bounded-shutdown coverage for the generator.

These tests drive ``generate_all`` directly with a scripted fake provider so
the abort and bounded-termination behavior can be verified deterministically
without a real Ollama outage. Each test completes in well under a second: a
provider outage must produce a bounded failure, not an indefinite stall.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from localbench.runtime.generation.attempt import AttemptRecord, AttemptStatus
from localbench.runtime.generation.failures import StructuredError
from localbench.workloads.code_retrieval.candidate_store import CandidateStore
from localbench.workloads.code_retrieval.extraction import ExtractedCodeUnit
from localbench.workloads.code_retrieval.schemas import CodeUnitContext

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "generate_query_candidates.py"
)
_spec = importlib.util.spec_from_file_location(
    "generate_query_candidates_provider_hang", _SCRIPT_PATH
)
_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(
    "generate_query_candidates_provider_hang", _script
)
_spec.loader.exec_module(_script)


def _unit(repo: str, symbol: str) -> ExtractedCodeUnit:
    return ExtractedCodeUnit(
        repository=repo,
        language="python",
        file_path="pkg/module.py",
        symbol=symbol,
        symbol_type="function",
        source_code="def f():\n    return 1\n",
        context=CodeUnitContext(),
        source_url="",
        is_public=True,
        docstring="",
        source_file_lines=2,
        content_hash=f"hash-{repo}-{symbol}",
        extracted_at="2026-08-21T00:00:00Z",
    )


def _timeout_error() -> StructuredError:
    """A provider-timeout error shaped like the real adapter's (message)."""
    return StructuredError(
        "OllamaUnavailableError: Ollama request timed out. "
        "The model may be loading or the prompt too long."
    )


def _success_result(query: str) -> SimpleNamespace:
    return SimpleNamespace(
        success=True,
        candidate=SimpleNamespace(
            query=query,
            query_style="natural",
            query_intent="find_implementation",
        ),
        attempts=[
            AttemptRecord(
                attempt_number=1,
                status=AttemptStatus.SUCCESS,
                will_retry=False,
                generation_ms=5.0,
                errors=[],
                raw_text="{}",
            )
        ],
        total_generation_ms=5.0,
        total_validation_ms=0.5,
        leakage=None,
    )


def _timeout_result() -> SimpleNamespace:
    return SimpleNamespace(
        success=False,
        candidate=None,
        attempts=[
            AttemptRecord(
                attempt_number=1,
                status=AttemptStatus.FAILED,
                will_retry=False,
                generation_ms=10.0,
                errors=[_timeout_error()],
                raw_text="",
            )
        ],
        total_generation_ms=10.0,
        total_validation_ms=0.0,
        leakage=None,
    )


def _build_generate_all_harness(tmp_path, monkeypatch, unit_symbols):
    """Monkeypatch the script so ``generate_all`` runs against a scripted fake.

    Returns ``(run_fn, state)`` where ``run_fn(units, **kwargs)`` calls
    ``generate_all`` and ``state`` records what was generated.
    """
    outdir = tmp_path / "out"
    units = [_unit("repo003", s) for s in unit_symbols]

    state = SimpleNamespace(
        scripted=[],
        adapters_built=0,
        generated_symbols=[],
    )

    class ScriptedGenerator:
        def __init__(self, model=None, policy=None, top_p=None, seed=None):
            pass

        def generate(self, unit):
            outcome = state.scripted.pop(0)
            state.generated_symbols.append(unit.symbol)
            if outcome == "fail":
                return _timeout_result()
            if outcome == "hang":
                time.sleep(30)
            return _success_result(f"locate the {unit.symbol} implementation")

    class FakeAdapter:
        def __init__(self, model_name=None):
            state.adapters_built += 1
            self.name = "fake-model"

        def health_check(self):
            return True

        def discover_models(self):
            return []

        def close(self):
            pass

    monkeypatch.setattr(_script, "OllamaAdapter", FakeAdapter)
    monkeypatch.setattr(_script, "QueryGenerator", ScriptedGenerator)

    def run(units_to_run=None, workers=1, failure_limit=3, update_meta=False):
        target = units_to_run if units_to_run is not None else units
        return _script.generate_all(
            test_units=target,
            output_dir=outdir,
            provider_failure_limit=failure_limit,
            update_meta=update_meta,
            workers=workers,
        )

    return run, state, outdir, units


class TestProviderHangAbort:
    def test_all_success_generates_everything(self, tmp_path, monkeypatch):
        run, state, outdir, units = _build_generate_all_harness(
            tmp_path, monkeypatch, ["alpha", "beta", "gamma"]
        )
        state.scripted = ["ok", "ok", "ok"]

        exit_code = run()

        assert exit_code == 0
        assert state.generated_symbols == ["alpha", "beta", "gamma"]
        successful, failed = CandidateStore(
            outdir / "candidates.jsonl",
            outdir / "candidate_failures.jsonl",
        ).load()
        assert len(successful) == 3
        assert failed == []

    def test_provider_failure_threshold_aborts(self, tmp_path, monkeypatch):
        """3 consecutive provider failures exceed the limit -> bounded abort."""
        run, state, outdir, units = _build_generate_all_harness(
            tmp_path, monkeypatch, ["a", "b", "c", "d", "e"]
        )
        state.scripted = ["ok", "fail", "fail", "fail", "ok"]

        start = time.monotonic()
        exit_code = run(failure_limit=3)
        elapsed = time.monotonic() - start

        # Aborts after the 3rd provider failure; the run stays bounded and
        # does not process the whole queue (any post-abort unit claimed by an
        # idle worker is not double-persisted — resume regenerates it).
        assert exit_code == 3
        assert elapsed < 2.0  # bounded: must not hang

    def test_single_transient_failure_does_not_abort(
        self, tmp_path, monkeypatch
    ):
        """A lone provider failure is recorded, then generation continues."""
        run, state, outdir, units = _build_generate_all_harness(
            tmp_path, monkeypatch, ["a", "b", "c"]
        )
        state.scripted = ["ok", "fail", "ok"]

        exit_code = run(failure_limit=3)

        assert exit_code == 0
        assert state.generated_symbols == ["a", "b", "c"]
        successful, failed = CandidateStore(
            outdir / "candidates.jsonl",
            outdir / "candidate_failures.jsonl",
        ).load()
        assert len(successful) == 2
        assert len(failed) == 1
        assert failed[0]["failure_category"] == "timeout"

    def test_abort_persists_completed_records_and_no_hang(
        self, tmp_path, monkeypatch
    ):
        run, state, outdir, units = _build_generate_all_harness(
            tmp_path, monkeypatch, ["a", "b", "c", "d"]
        )
        state.scripted = ["ok", "fail", "fail", "fail"]

        start = time.monotonic()
        exit_code = run(failure_limit=3)
        elapsed = time.monotonic() - start

        assert exit_code == 3
        assert elapsed < 2.0
        successful, failed = CandidateStore(
            outdir / "candidates.jsonl",
            outdir / "candidate_failures.jsonl",
        ).load()
        # a succeeded; b,c,d failed as provider timeouts and were recorded.
        assert len(successful) == 1
        assert len(failed) == 3

    def test_hung_provider_aborts_via_stall_guard(self, tmp_path, monkeypatch):
        """A request that never returns is a bounded stall, not a hang."""
        # Crash the stall window down so the test stays fast.
        monkeypatch.setattr(_script, "STALL_ABORT_SECONDS", 0.3)
        run, state, outdir, units = _build_generate_all_harness(
            tmp_path, monkeypatch, ["a", "b"]
        )
        state.scripted = ["hang", "hang"]

        start = time.monotonic()
        exit_code = run(failure_limit=3)
        elapsed = time.monotonic() - start

        # The worker request sleeps 30s; the run must abort well before that
        # instead of blocking for the hung request.
        assert exit_code == 3
        assert elapsed < 10
        assert len(state.generated_symbols) <= 2


class TestResumeAfterAbort:
    def test_resume_skips_completed_and_no_duplicates(
        self, tmp_path, monkeypatch
    ):
        """After an abort, rerunning from the same checkpoint has no dupes."""
        run, state, outdir, units = _build_generate_all_harness(
            tmp_path, monkeypatch, ["a", "b", "c", "d", "e"]
        )

        # First pass: a succeeds, then 3 provider failures abort the run.
        state.scripted = ["ok", "fail", "fail", "fail", "ok"]
        assert run(failure_limit=3) == 3

        # Second pass: provider recovered; everything remaining succeeds.
        state.scripted = ["ok"]
        exit_code = run(failure_limit=3)
        assert exit_code == 0

        successful, failed = CandidateStore(
            outdir / "candidates.jsonl",
            outdir / "candidate_failures.jsonl",
        ).load()
        # a succeeded in pass 1 (not regenerated); e succeeded in pass 2.
        assert len(successful) == 2
        assert len(failed) == 3
        all_ids = [r["code_unit_id"] for r in successful + failed]
        assert len(all_ids) == len(set(all_ids))  # no duplicates
