"""Unit coverage: deterministic recovery of duplicated candidate artifacts."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from localbench.workloads.code_retrieval.artifact_recovery import (
    ArtifactRecoveryError,
    cross_file_overview,
    file_stats,
    plan_recovery,
    read_jsonl,
    serialize_records,
)
from localbench.workloads.code_retrieval.candidate_store import (
    CandidateStore,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "recover_query_candidates.py"
)
_spec = importlib.util.spec_from_file_location(
    "recover_query_candidates_test", _SCRIPT_PATH
)
_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("recover_query_candidates_test", _script)
_spec.loader.exec_module(_script)


def _base_record(unit_id: str, completed: str) -> dict:
    return {
        "code_unit_id": unit_id,
        "candidate_id": f"candidate_{unit_id}",
        "query": f"locate the {unit_id} implementation",
        "query_style": "natural",
        "query_intent": "find_implementation",
        "model": "qwen2.5-coder:7b",
        "model_version": "7b",
        "prompt_version": "1.0.0",
        "seed": 42,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 128,
        "attempt_count": 1,
        "attempts": [],
        "generation_ms": 10.0,
        "validation_ms": 1.0,
        "validation_passed": True,
        "leakage_passed": True,
        "leakage_violations": [],
        "success": True,
        "failure_category": None,
        "failure_reason": None,
        "completed_utc": completed,
    }


def success(unit_id: str, completed: str, query=None, **overrides) -> dict:
    record = _base_record(unit_id, completed)
    if query is not None:
        record["query"] = query
    record.update(overrides)
    return record


def failure(unit_id: str, completed: str, **overrides) -> dict:
    record = _base_record(unit_id, completed)
    record.update(
        query="",
        query_style="",
        query_intent="",
        attempt_count=2,
        attempts=[{"utc": completed}],
        generation_ms=5.0,
        validation_ms=0.5,
        validation_passed=False,
        leakage_passed=False,
        success=False,
        failure_category="malformed_json",
        failure_reason="synthetic failure",
    )
    record.update(overrides)
    return record


class TestReadJsonl:
    def test_missing_file_returns_empty(self, tmp_path):
        assert read_jsonl(tmp_path / "absent.jsonl") == []

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "art.jsonl"
        path.write_text(
            "\n" + json.dumps(success("u1", "t1")) + "\n\n", encoding="utf-8"
        )
        assert len(read_jsonl(path)) == 1

    def test_malformed_json_aborts_with_location(self, tmp_path):
        path = tmp_path / "art.jsonl"
        path.write_text('{"code_unit_id": "u1"\n', encoding="utf-8")
        with pytest.raises(ArtifactRecoveryError, match="art.jsonl:1"):
            read_jsonl(path)

    def test_ill_shaped_record_aborts(self, tmp_path):
        path = tmp_path / "art.jsonl"
        path.write_text(
            json.dumps({"code_unit_id": "u1"}) + "\n", encoding="utf-8"
        )
        with pytest.raises(ArtifactRecoveryError, match="success"):
            read_jsonl(path)


class TestPlanRecoveryRules:
    def test_identical_duplicates_keep_earliest_success(self):
        early = success("u1", "2026-08-22T04:00:00+00:00")
        late = success("u1", "2026-08-22T05:00:00+00:00")
        plan = plan_recovery([late, early], [])
        assert plan.recoverable
        assert [r["completed_utc"] for r in plan.candidates] == [
            "2026-08-22T04:00:00+00:00"
        ]
        assert [r["completed_utc"] for r in plan.superseded] == [
            "2026-08-22T05:00:00+00:00"
        ]

    def test_telemetry_only_differences_collapse(self):
        first = success("u1", "2026-08-22T04:00:00+00:00")
        slower = success(
            "u1", "2026-08-22T05:00:00+00:00", generation_ms=99.9
        )
        plan = plan_recovery([first, slower], [])
        assert plan.recoverable and len(plan.candidates) == 1

    def test_triple_execution_leaves_two_superseded(self):
        records = [
            success("u1", f"2026-08-22T0{n}:00:00+00:00") for n in (3, 1, 2)
        ]
        plan = plan_recovery(records, [])
        assert plan.recoverable
        assert len(plan.candidates) == 1 and len(plan.superseded) == 2

    def test_duplicate_failures_keep_latest(self):
        older = failure("u2", "2026-08-22T06:00:00+00:00")
        newer = failure("u2", "2026-08-22T07:00:00+00:00")
        plan = plan_recovery([], [older, newer])
        assert plan.recoverable
        assert [r["completed_utc"] for r in plan.failures] == [
            "2026-08-22T07:00:00+00:00"
        ]

    def test_historical_failure_archived_behind_later_success(self):
        victory = success("u3", "2026-08-22T08:00:00+00:00")
        history = failure("u3", "2026-08-22T06:30:00+00:00")
        plan = plan_recovery([victory], [history])
        assert plan.recoverable
        assert len(plan.candidates) == 1 and plan.failures == []
        assert plan.superseded == [history]

    def test_failure_after_success_is_refused_as_regressed(self):
        plan = plan_recovery(
            [success("u4", "2026-08-22T05:50:46+00:00")],
            [failure("u4", "2026-08-22T05:51:35+00:00")],
        )
        assert not plan.recoverable
        kinds = {c.kind for c in plan.conflicts}
        assert kinds == {"failure_supersedes_success"}
        assert not plan.candidates and not plan.failures

    def test_divergent_successful_queries_are_never_auto_resolved(self):
        plan = plan_recovery(
            [
                success("u5", "2026-08-22T01:00:00+00:00", query="query X"),
                success("u5", "2026-08-22T02:00:00+00:00", query="query Y"),
            ],
            [],
        )
        assert not plan.recoverable
        assert plan.conflicts[0].kind == "semantic_candidate_conflict"

    def test_methodology_drift_among_identical_payloads_is_refused(self):
        plan = plan_recovery(
            [
                success("u6", "2026-08-22T01:00:00+00:00", seed=42),
                success("u6", "2026-08-22T02:00:00+00:00", seed=43),
            ],
            [],
        )
        assert not plan.recoverable
        assert plan.conflicts[0].kind == "methodology_mismatch"

    def test_candidate_id_mismatch_is_refused(self):
        rogue = success("u7", "2026-08-22T01:00:00+00:00")
        rogue["candidate_id"] = "candidate_someone_else"
        plan = plan_recovery([rogue], [])
        assert not plan.recoverable
        assert plan.conflicts[0].kind == "candidate_id_mismatch"

    def test_selection_is_independent_of_input_ordering(self):
        base_candidates = [
            success("d2", "2026-08-22T02:00:00+00:00"),
            success("d1", "2026-08-22T03:00:00+00:00"),
            success("d2", "2026-08-22T01:00:00+00:00"),
        ]
        base_failures = [failure("d3", "2026-08-22T04:00:00+00:00")]
        rng = random.Random(11)
        reference = None
        for _ in range(8):
            cands, fails = base_candidates[:], base_failures[:]
            rng.shuffle(cands)
            rng.shuffle(fails)
            plan = plan_recovery(cands, fails)
            outcome = (
                serialize_records(plan.candidates),
                serialize_records(plan.failures),
            )
            if reference is None:
                reference = outcome
            assert outcome == reference


class TestSerialization:
    def test_output_is_sorted_by_code_unit_id(self):
        records = [
            success("zz", "t2"),
            success("aa", "t1"),
        ]
        lines = serialize_records(records).splitlines()
        ids = [json.loads(line)["code_unit_id"] for line in lines]
        assert ids == ["aa", "zz"]

    def test_leftover_duplicate_ids_raise(self):
        records = [success("u1", "t1"), success("u1", "t2")]
        with pytest.raises(ArtifactRecoveryError, match="duplicate 'u1'"):
            serialize_records(records)


class TestStatsAndOverview:
    def test_file_stats_summarize_duplication(self):
        stats = file_stats(
            [success("a", "t1"), success("a", "t2"), failure("b", "t3")]
        )
        assert stats == {
            "records": 3,
            "unique_ids": 2,
            "duplicate_groups": 1,
            "max_multiplicity": 2,
            "extra_records": 1,
        }

    def test_cross_file_verdicts_classify_overlap(self):
        overview = cross_file_overview(
            [success("x", "2026-08-22T06:00:00+00:00")],
            [failure("x", "2026-08-22T05:00:00+00:00"), failure("y", "t")],
        )
        assert overview["only_failures"] == ["y"]
        assert overview["both"] == ["x"]
        assert overview["verdicts"]["x"] == "success_is_later"


def _write_pair(directory: Path, candidates, failures) -> tuple[Path, Path]:
    cand_path = directory / "candidates.jsonl"
    fail_path = directory / "candidate_failures.jsonl"
    cand_path.write_text(
        "".join(json.dumps(r) + "\n" for r in candidates), encoding="utf-8"
    )
    fail_path.write_text(
        "".join(json.dumps(r) + "\n" for r in failures), encoding="utf-8"
    )
    return cand_path, fail_path


class TestRecoveryScript:
    @pytest.fixture
    def corrupted_dir(self, tmp_path):
        directory = tmp_path / "queries"
        directory.mkdir()
        candidates = [
            success("u1", "2026-08-22T05:00:00+00:00"),
            success("u1", "2026-08-22T04:00:00+00:00"),
            success("u2", "2026-08-22T03:00:00+00:00"),
        ]
        failures = [
            failure("u1", "2026-08-22T02:00:00+00:00"),
            failure("u3", "2026-08-22T06:00:00+00:00"),
            failure("u3", "2026-08-22T07:00:00+00:00"),
        ]
        _write_pair(directory, candidates, failures)
        return directory

    def _run(self, directory, extra_args=()):
        return _script.main(
            ["--queries-dir", str(directory), *extra_args]
        )

    def test_recovered_checkpoint_loads_through_candidate_store(
        self, corrupted_dir
    ):
        assert self._run(corrupted_dir) == 0
        successful, failed = CandidateStore(
            corrupted_dir / "candidates.jsonl",
            corrupted_dir / "candidate_failures.jsonl",
        ).load()
        assert sorted(r["code_unit_id"] for r in successful) == [
            "u1",
            "u2",
        ]
        assert [r["code_unit_id"] for r in failed] == ["u3"]

    def test_originals_backed_up_before_rewrite(self, corrupted_dir):
        before = (
            (corrupted_dir / "candidates.jsonl").read_bytes(),
            (corrupted_dir / "candidate_failures.jsonl").read_bytes(),
        )
        assert self._run(corrupted_dir) == 0
        backup_cand = corrupted_dir / "candidates.jsonl.pre-recovery.bak"
        backup_fail = (
            corrupted_dir / "candidate_failures.jsonl.pre-recovery.bak"
        )
        assert backup_cand.read_bytes() == before[0]
        assert backup_fail.read_bytes() == before[1]

    def test_second_run_is_a_noop(self, corrupted_dir):
        assert self._run(corrupted_dir) == 0
        after_first = (
            (corrupted_dir / "candidates.jsonl").read_bytes(),
            (corrupted_dir / "candidate_failures.jsonl").read_bytes(),
        )
        assert self._run(corrupted_dir) == 0
        after_second = (
            (corrupted_dir / "candidates.jsonl").read_bytes(),
            (corrupted_dir / "candidate_failures.jsonl").read_bytes(),
        )
        assert after_second == after_first

    def test_dry_run_reports_plan_without_touching_files(
        self, corrupted_dir
    ):
        before = {
            p.name: p.read_bytes() for p in corrupted_dir.iterdir()
        }
        assert self._run(corrupted_dir, ["--dry-run"]) == 0
        after = {p.name: p.read_bytes() for p in corrupted_dir.iterdir()}
        assert after == before

    def test_conflicts_exit_without_writing_anything(self, tmp_path):
        directory = tmp_path / "conflicted"
        directory.mkdir()
        cand_path, fail_path = _write_pair(
            directory,
            [
                success("x1", "2026-08-22T01:00:00+00:00", query="X"),
                success("x1", "2026-08-22T02:00:00+00:00", query="Y"),
            ],
            [],
        )
        snapshot = (cand_path.read_bytes(), fail_path.read_bytes())
        assert self._run(directory) == 2
        assert (cand_path.read_bytes(), fail_path.read_bytes()) == snapshot
        assert not list(directory.glob("*.bak"))
        assert not list(directory.glob("*recover-tmp*"))

    def test_existing_backup_blocks_the_run(self, corrupted_dir):
        guard = corrupted_dir / "candidates.jsonl.pre-recovery.bak"
        guard.write_bytes(b"previous recovery snapshot")
        original = (corrupted_dir / "candidates.jsonl").read_bytes()
        assert self._run(corrupted_dir) == _script.EXIT_ERROR
        assert (corrupted_dir / "candidates.jsonl").read_bytes() == original
