"""Tests for the development-only query-representation A/B harness.

No Ollama, no network: a fake ``LocalModel`` stands in for the provider.  The
harness is a script, so it is loaded by path the same way the other script
tests in this package do it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from localbench.runtime.model import GenerationRequest, GenerationResult
from localbench.workloads.code_retrieval.schemas import CandidateQuery

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "run_query_representation_ab.py"
)
_spec = importlib.util.spec_from_file_location(
    "run_query_representation_ab", _SCRIPT_PATH
)
ab = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("run_query_representation_ab", ab)
_spec.loader.exec_module(ab)

SPLITS = ("train", "validation")


# ---------------------------------------------------------------------------
# Fixtures and doubles
# ---------------------------------------------------------------------------


class FakeModel:
    """Records every request and replays canned replies in order."""

    def __init__(self, replies: list[str], name: str = "fake-model") -> None:
        self._replies = list(replies)
        self._name = name
        self.requests: list[GenerationRequest] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self):  # pragma: no cover - not used by the harness
        return None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        text = self._replies.pop(0) if self._replies else "{}"
        return GenerationResult(
            model=request.model, text=text, duration_ms=1.0, done=True
        )


def good_reply(query: str = "utility that reads a table and renders it") -> str:
    return json.dumps(
        {
            "query": query,
            "query_style": "natural",
            "query_intent": "find the table rendering helper",
        }
    )


def rows(count: int, split: str = "train") -> list[dict]:
    """Minimal split rows, alternating the stratification strata."""
    out: list[dict] = []
    for index in range(count):
        out.append(
            {
                "id": f"u{index:04d}",
                "split": split,
                "repository": f"repo{index % 3:03d}",
                "language": "python",
                "file_path": f"pkg/mod{index % 5}.py",
                "symbol": f"thing_{index}",
                "symbol_type": "method" if index % 2 else "function",
                "source_code": "def thing_0(a, b):\n    return a + b\n",
                "context": {
                    "class_name": None,
                    "module_docstring": "secret module docs",
                    "imports": ["json", "logging"],
                    "parent_methods": ["other_method"],
                },
                "source_url": "https://example.invalid/x",
                "is_public": index % 3 == 0,
                "docstring": "unique docstring marker zqjx",
                "source_file_lines": 3,
                "content_hash": f"hash{index}",
                "extracted_at": "2026-01-01T00:00:00Z",
            }
        )
    return out


@pytest.fixture
def sample_rows(monkeypatch):
    """Point the harness at a small deterministic split."""
    bank = {"train": rows(40, "train"), "validation": rows(30, "validation")}

    def fake_load_split(split: str):
        return list(bank[split])

    monkeypatch.setattr(ab, "load_split", fake_load_split)
    return bank


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_same_seed_gives_identical_sample(self, sample_rows):
        first = ab.select_sample(sample_rows["train"], 10, 42, "train")
        second = ab.select_sample(sample_rows["train"], 10, 42, "train")
        assert [row["id"] for row in first] == [row["id"] for row in second]

    def test_different_seed_gives_different_sample(self, sample_rows):
        first = ab.select_sample(sample_rows["train"], 10, 42, "train")
        other = ab.select_sample(sample_rows["train"], 10, 7, "train")
        assert [row["id"] for row in first] != [row["id"] for row in other]

    def test_returns_exactly_requested_count(self, sample_rows):
        assert len(ab.select_sample(sample_rows["train"], 10, 42, "train")) == 10

    def test_allocation_sums_to_total(self):
        counts = {("function", True): 50, ("method", False): 30, ("method", True): 20}
        allocated = ab._allocate(counts, 17)
        assert sum(allocated.values()) == 17
        assert all(value >= 0 for value in allocated.values())

    def test_sampling_preserves_stratum_mix(self, sample_rows):
        picked = ab.select_sample(sample_rows["train"], 12, 42, "train")
        strata = {ab._stratum(row) for row in picked}
        assert strata == {ab._stratum(row) for row in sample_rows["train"]}

    def test_every_stratum_is_covered(self, sample_rows):
        picked = ab.select_sample(sample_rows["train"], 12, 42, "train")
        assert len({ab._stratum(row) for row in picked}) == len(
            {ab._stratum(row) for row in sample_rows["train"]}
        )

    def test_build_sample_merges_both_splits(self, sample_rows):
        sample = ab.build_sample(3, 4, 42)
        assert len(sample) == 7
        assert {entry["split"] for entry in sample} == set(SPLITS)

    def test_build_sample_excludes_source_code(self, sample_rows):
        sample = ab.build_sample(3, 4, 42)
        for entry in sample:
            assert "source_code" not in entry
            assert entry["source_code_sha256"]

    def test_sample_entry_keys_are_known(self, sample_rows):
        allowed = {
            "code_unit_id",
            "split",
            "symbol_type",
            "is_public",
            "repository",
            "file_path",
            "language",
            "symbol",
            "source_code_sha256",
            "source_lines",
        }
        for entry in ab.build_sample(2, 2, 42):
            assert set(entry) == allowed


# ---------------------------------------------------------------------------
# Representations and prompt
# ---------------------------------------------------------------------------


class TestRepresentations:
    def test_arm_a_returns_production_facts_keys(self, sample_rows):
        unit = ab.to_code_unit(sample_rows["train"][0])
        assert set(ab.representation_a(unit)) == {
            "primary_purpose",
            "input_summary",
            "output_summary",
            "side_effects",
            "key_operations",
            "error_handling",
            "control_flow",
            "raises",
            "domain_concepts",
            "observable_effects",
        }

    def test_arm_b_returns_semantic_ir_keys(self, sample_rows):
        unit = ab.to_code_unit(sample_rows["train"][0])
        payload = ab.representation_b(unit)
        assert set(payload) == {
            "input_roles",
            "output_behavior",
            "domain_signals",
            "transformations",
            "relations",
            "conditions",
            "state_changes",
            "side_effects",
            "error_behavior",
            "observable_effects",
            "parse_ok",
            "anchor_count",
        }

    def test_both_arms_see_the_same_imports(self):
        # The guard rebuilds without imports, so both arms must drop them.
        assert ab.USE_IMPORTS is False
        assert ab._arm_imports(object()) is None

    def test_arm_b_passes_the_existing_guard(self, sample_rows):
        from localbench.workloads.code_retrieval.semantic_ir_experiment import (
            assert_identifier_free,
            build_semantic_ir,
        )

        unit = ab.to_code_unit(sample_rows["train"][0])
        ir = build_semantic_ir(
            unit.source_code, unit.language, unit_kind=unit.symbol_type
        )
        assert_identifier_free(ir, unit.source_code, unit.language)
        assert ab.representation_b(unit)["parse_ok"] is True

    def test_arm_b_is_guarded_not_guessing(self, sample_rows, monkeypatch):
        """A guard failure must surface, not be swallowed by a weaker check."""
        def boom(*_args, **_kwargs):
            raise AssertionError("leaked: domain_signals")

        monkeypatch.setattr(ab, "assert_identifier_free", boom)
        unit = ab.to_code_unit(sample_rows["train"][0])
        with pytest.raises(AssertionError):
            ab.representation_b(unit)

    def test_guard_failure_is_recorded_as_a_failure(self, sample_rows, monkeypatch):
        def boom(*_args, **_kwargs):
            raise AssertionError("leaked")

        monkeypatch.setattr(ab, "assert_identifier_free", boom)
        entry = {
            "code_unit_id": "u0000",
            "split": "train",
            "symbol_type": "function",
            "is_public": True,
        }
        model = FakeModel([good_reply()])
        record = ab.run_one(
            "B", entry, sample_rows["train"][0], model, ab.RetryPolicy(), 42
        )
        assert record["success"] is False
        assert record["stage"] == "representation_guard_failed"
        assert "leaked" in record["failure_reason"]
        assert model.requests == []

    def test_payload_schemas_intentionally_differ(self, sample_rows):
        """The two arms are the variable under test, so their keys must differ."""
        unit = ab.to_code_unit(sample_rows["train"][0])
        assert set(ab.representation_a(unit)) != set(ab.representation_b(unit))


class TestPrompt:
    def test_prompts_share_identical_shell(self, sample_rows):
        unit = ab.to_code_unit(sample_rows["train"][0])
        prompt_a = ab.build_prompt("A", ab.representation_a(unit))
        prompt_b = ab.build_prompt("B", ab.representation_b(unit))
        shell = ab.prompt_shell()
        assert prompt_a.startswith(shell)
        assert prompt_b.startswith(shell)
        assert len(prompt_a) > len(shell)
        assert len(prompt_b) > len(shell)

    def test_only_the_payload_region_differs(self, sample_rows):
        unit = ab.to_code_unit(sample_rows["train"][0])
        shell = ab.prompt_shell()
        prompt_a = ab.build_prompt("A", ab.representation_a(unit))
        prompt_b = ab.build_prompt("B", ab.representation_b(unit))
        assert prompt_a[: len(shell)] == prompt_b[: len(shell)] == shell

    def test_shared_system_instruction_is_reused(self, sample_rows):
        from localbench.workloads.code_retrieval.query_prompt import (
            get_query_system_prompt,
        )

        unit = ab.to_code_unit(sample_rows["train"][0])
        prompt = ab.build_prompt("A", ab.representation_a(unit))
        assert get_query_system_prompt() in prompt

    def test_prompt_excludes_forbidden_fields(self, sample_rows):
        unit = ab.to_code_unit(sample_rows["train"][0])
        for arm in ("A", "B"):
            payload = (
                ab.representation_a(unit) if arm == "A" else ab.representation_b(unit)
            )
            region = ab.build_prompt(arm, payload)[len(ab.prompt_shell()):]
            for token in ab._FORBIDDEN_PROMPT_TOKENS:
                assert token not in region, f"{arm} leaked {token}"

    def test_prompt_excludes_unit_identifiers(self, sample_rows):
        row = sample_rows["train"][0]
        unit = ab.to_code_unit(row)
        for arm in ("A", "B"):
            payload = (
                ab.representation_a(unit) if arm == "A" else ab.representation_b(unit)
            )
            prompt = ab.build_prompt(arm, payload)
            for secret in (row["symbol"], row["file_path"], row["repository"]):
                assert secret not in prompt
            assert "secret module docs" not in prompt
            assert "unique docstring marker zqjx" not in prompt
            assert "other_method" not in prompt

    def test_prompt_excludes_source_body(self, sample_rows):
        unit = ab.to_code_unit(sample_rows["train"][0])
        prompt = ab.build_prompt("A", ab.representation_a(unit))
        assert "return a + b" not in prompt

    def test_unknown_arm_is_rejected(self):
        with pytest.raises(ValueError):
            ab.build_prompt("C", {"primary_purpose": "x"})

    def test_ir_diagnostics_never_reach_the_prompt(self, sample_rows):
        unit = ab.to_code_unit(sample_rows["train"][0])
        payload = ab.representation_b(unit)
        region = ab.build_prompt("B", payload)[len(ab.prompt_shell()):]
        assert "anchor_count" not in region
        assert "parse_ok" not in region

    def test_prompt_hash_is_recorded(self, sample_rows):
        entry = {
            "code_unit_id": "u0000",
            "split": "train",
            "symbol_type": "function",
            "is_public": True,
        }
        model = FakeModel([good_reply()])
        record = ab.run_one(
            "A", entry, sample_rows["train"][0], model, ab.RetryPolicy(), 42
        )
        assert len(record["payload_sha256"]) == 64


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class TestGeneration:
    def _entry(self):
        return {
            "code_unit_id": "u0000",
            "split": "train",
            "symbol_type": "function",
            "is_public": True,
        }

    def test_request_matches_production_sampling(self, sample_rows):
        model = FakeModel([good_reply()])
        ab.run_one(
            "A", self._entry(), sample_rows["train"][0], model, ab.RetryPolicy(), 42
        )
        request = model.requests[0]
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.max_tokens == 128
        assert request.seed == 42

    def test_request_uses_the_model_name(self, sample_rows):
        model = FakeModel([good_reply()], name="qwen2.5-coder:7b")
        ab.run_one(
            "A", self._entry(), sample_rows["train"][0], model, ab.RetryPolicy(), 42
        )
        assert model.requests[0].model == "qwen2.5-coder:7b"

    def test_successful_generation_is_recorded(self, sample_rows):
        model = FakeModel([good_reply("utility that renders a table")])
        record = ab.run_one(
            "A", self._entry(), sample_rows["train"][0], model, ab.RetryPolicy(), 42
        )
        assert record["success"] is True
        assert record["stage"] == "complete"
        assert record["schema_valid"] is True
        assert record["query"] == "utility that renders a table"
        assert record["leakage_passed"] is True
        assert record["provenance_passed"] is True
        assert record["meta_passed"] is True

    def test_raw_output_is_persisted_exactly(self, sample_rows):
        raw = good_reply("exact raw text preserved")
        model = FakeModel([raw])
        record = ab.run_one(
            "A", self._entry(), sample_rows["train"][0], model, ab.RetryPolicy(), 42
        )
        assert record["attempts"][0]["raw_text"] == raw

    def test_schema_failure_is_recorded_and_retried(self, sample_rows):
        model = FakeModel(["not json at all", good_reply()])
        record = ab.run_one(
            "A", self._entry(), sample_rows["train"][0], model, ab.RetryPolicy(), 42
        )
        assert len(record["attempts"]) == 2
        assert record["attempts"][0]["status"] == "failed"
        assert record["attempts"][0]["errors"]
        assert record["attempts"][0]["will_retry"] is True
        assert record["success"] is True

    def test_every_failed_attempt_is_retained(self, sample_rows):
        model = FakeModel(["nope", "still nope", "never json"])
        record = ab.run_one(
            "A", self._entry(), sample_rows["train"][0], model, ab.RetryPolicy(), 42
        )
        assert record["success"] is False
        assert record["stage"] == "generation_exhausted"
        assert len(record["attempts"]) == 3
        assert all(attempt["raw_text"] for attempt in record["attempts"])

    def test_retry_count_follows_the_policy(self, sample_rows):
        model = FakeModel(["x"] * 6)
        record = ab.run_one(
            "A",
            self._entry(),
            sample_rows["train"][0],
            model,
            ab.RetryPolicy(max_attempts=2),
            42,
        )
        assert len(record["attempts"]) == 2
        assert record["max_attempts"] == 2

    def test_leakage_failure_is_caught(self, sample_rows):
        row = sample_rows["train"][0]
        model = FakeModel([good_reply(f"call {row['file_path']} in the module")])
        record = ab.run_one(
            "A", self._entry(), row, model, ab.RetryPolicy(), 42
        )
        assert record["success"] is False
        assert record["stage"] == "leakage"
        assert record["leakage_passed"] is False

    def test_meta_query_is_caught(self, sample_rows):
        # check_meta_query has a wider pattern set than the meta patterns the
        # leakage check applies, so a query can pass leakage and still be a
        # meta-response.
        model = FakeModel(
            [good_reply("here is the query that locates the table renderer")]
        )
        record = ab.run_one(
            "A", self._entry(), sample_rows["train"][0], model, ab.RetryPolicy(), 42
        )
        assert record["success"] is False
        assert record["leakage_passed"] is True
        assert record["meta_passed"] is False
        assert record["stage"] == "meta_query"

    def test_docstring_copied_text_is_rejected(self, sample_rows):
        row = dict(sample_rows["train"][0])
        row["docstring"] = (
            "Render a tabulated summary of collected rows with aligned "
            "columns and trailing separators"
        )
        model = FakeModel(
            [
                good_reply(
                    "render a tabulated summary of collected rows with aligned columns"
                )
            ]
        )
        record = ab.run_one("A", self._entry(), row, model, ab.RetryPolicy(), 42)
        assert record["success"] is False
        assert record["stage"] in {"leakage", "provenance"}

    def test_provenance_failure_is_caught(self, sample_rows, monkeypatch):
        """The chain must consult provenance and report its own stage.

        A real docstring copy is already caught by the leakage stage, so this
        forces the provenance check to fail to prove it is wired in rather
        than silently skipped.
        """
        from localbench.workloads.code_retrieval.query_generator import (
            ProvenanceCheckResult,
        )

        monkeypatch.setattr(
            ab,
            "check_query_provenance",
            lambda query, docstring: ProvenanceCheckResult(
                passed=False, violations=["copied from documentation"]
            ),
        )
        model = FakeModel([good_reply("utility that renders collected rows")])
        record = ab.run_one(
            "A", self._entry(), sample_rows["train"][0], model, ab.RetryPolicy(), 42
        )
        assert record["success"] is False
        assert record["stage"] == "provenance"
        assert record["leakage_passed"] is True
        assert record["provenance_passed"] is False
        assert "copied from documentation" in record["failure_reason"]

    def test_candidate_schema_unchanged(self):
        assert set(CandidateQuery.model_fields) == {
            "query",
            "query_style",
            "query_intent",
        }


# ---------------------------------------------------------------------------
# Pairing, summary, blind review
# ---------------------------------------------------------------------------


def _records(sample_rows, train_n=3, validation_n=2, replies_per_arm=10):
    """Build paired records for a small run without touching a model."""
    sample = ab.build_sample(train_n, validation_n, 42)
    bank = {
        row["id"]: row
        for split in SPLITS
        for row in sample_rows[split]
    }
    records = []
    for entry in sample:
        for arm in ab.ARMS:
            raw = good_reply(f"{arm} query for {entry['symbol_type']} behaviour")
            model = FakeModel([raw])
            records.append(
                ab.run_one(
                    arm, entry, bank[entry["code_unit_id"]], model,
                    ab.RetryPolicy(), 42,
                )
            )
    return sample, records


class TestPairingAndSummary:
    def test_both_arms_run_on_the_same_units(self, sample_rows):
        sample, records = _records(sample_rows)
        by_arm = {}
        for record in records:
            by_arm.setdefault(record["arm"], []).append(record["code_unit_id"])
        assert sorted(by_arm["A"]) == sorted(by_arm["B"])
        assert len(by_arm["A"]) == len(sample)

    def test_pair_count_matches_unit_count(self, sample_rows):
        _sample, records = _records(sample_rows)
        assert ab.paired_summary(records)["pairs"] == len(records) // 2

    def test_success_counts_are_per_arm(self, sample_rows):
        _sample, records = _records(sample_rows)
        summary = ab.build_summary(records)["overall"]
        assert summary["A"]["units"] == summary["B"]["units"]
        assert summary["A"]["success"] == summary["A"]["units"]

    def test_summary_is_split_aware(self, sample_rows):
        _sample, records = _records(sample_rows)
        summary = ab.build_summary(records)
        assert set(summary["by_split"]) == set(SPLITS)
        assert summary["by_split"]["train"]["A"]["units"] == 3

    def test_retry_rate_is_computed(self, sample_rows):
        _sample, records = _records(sample_rows)
        arm = ab.arm_summary([r for r in records if r["arm"] == "A"])
        assert arm["retry_rate"] == 0.0
        assert arm["generation_attempts"] == arm["units"]

    def test_distinct_n_counts_vocabulary(self):
        queries = ["read a table", "read a file", "render a table"]
        assert ab.distinct_n(queries, 1) == len(
            {"read", "a", "table", "file", "render"}
        )
        assert ab.distinct_n(queries, 2) == len(
            {"read a", "a table", "a file", "render a"}
        )

    def test_arm_summary_handles_no_queries(self):
        stats = ab.arm_summary([])
        assert stats["units"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["mean_query_words"] == 0.0

    def test_summary_declines_to_name_a_winner(self, sample_rows):
        """The summary reports counts; it must not rank the arms.

        Checked structurally (no verdict-shaped keys) rather than by scanning
        text, so the explicit disclaimer in the summary is not a false hit.
        """
        _sample, records = _records(sample_rows)
        summary = ab.build_summary(records)
        assert "no winner is computed" in summary["interpretation"].lower()

        banned = {
            "winner",
            "verdict",
            "recommendation",
            "preferred_arm",
            "better_arm",
            "ranking",
            "score",
        }

        def walk(node, trail=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    assert key.lower() not in banned, f"verdict key at {trail}{key}"
                    walk(value, f"{trail}{key}.")
            elif isinstance(node, list):
                for value in node:
                    walk(value, trail)

        walk(summary)

    def test_both_arms_report_the_same_metric_keys(self, sample_rows):
        _sample, records = _records(sample_rows)
        overall = ab.build_summary(records)["overall"]
        assert set(overall["A"]) == set(overall["B"])

    def test_failure_stages_are_tallied(self, sample_rows):
        _sample, records = _records(sample_rows)
        records[0].update(success=False, stage="leakage")
        stats = ab.arm_summary([records[0]])
        assert stats["failure_stages"] == {"leakage": 1}


class TestBlindReview:
    def test_input_hides_every_arm_label(self, sample_rows):
        _sample, records = _records(sample_rows)
        review = ab.build_blind_review(records, 42)["input"]
        for item in review["items"]:
            assert not (ab.BLIND_FORBIDDEN_KEYS & set(item))
            assert set(item) == {"review_id", "pair_id", "query"}

    def test_input_carries_no_arm_string(self, sample_rows):
        _sample, records = _records(sample_rows)
        review = ab.build_blind_review(records, 42)["input"]
        blob = json.dumps(review)
        assert '"arm"' not in blob
        assert "StructuredBehaviorFacts" not in blob
        assert "Semantic IR" not in blob

    def test_both_arms_present_and_balanced(self, sample_rows):
        _sample, records = _records(sample_rows)
        key = ab.build_blind_review(records, 42)["key"]
        arms = [entry["arm"] for entry in key["mapping"].values()]
        assert arms.count("A") == arms.count("B")
        assert len(arms) == len(records)

    def test_arms_are_not_segregated_by_position(self, sample_rows):
        """All-A-then-all-B would reveal the label through ordering alone."""
        _sample, records = _records(sample_rows)
        key = ab.build_blind_review(records, 42)["key"]
        sequence = [
            entry["arm"]
            for _slot, entry in sorted(key["mapping"].items())
        ]
        runs = [
            arm
            for i, arm in enumerate(sequence)
            if i == 0 or arm != sequence[i - 1]
        ]
        assert len(runs) > 2

    def test_review_ids_are_unique(self, sample_rows):
        _sample, records = _records(sample_rows)
        review = ab.build_blind_review(records, 42)["input"]
        ids = [item["review_id"] for item in review["items"]]
        assert len(ids) == len(set(ids))

    def test_review_order_is_deterministic(self, sample_rows):
        _sample, records = _records(sample_rows)
        first = ab.build_blind_review(records, 42)["input"]
        second = ab.build_blind_review(records, 42)["input"]
        assert [item["review_id"] for item in first["items"]] == [
            item["review_id"] for item in second["items"]
        ]
        assert [item["query"] for item in first["items"]] == [
            item["query"] for item in second["items"]
        ]

    def test_different_seed_reorders_the_blind_input(self, sample_rows):
        _sample, records = _records(sample_rows)
        first = ab.build_blind_review(records, 42)["input"]
        other = ab.build_blind_review(records, 99)["input"]
        assert [i["query"] for i in first["items"]] != [
            i["query"] for i in other["items"]
        ]

    def test_key_maps_every_review_id_back(self, sample_rows):
        _sample, records = _records(sample_rows)
        review = ab.build_blind_review(records, 42)
        for item in review["input"]["items"]:
            entry = review["key"]["mapping"][item["review_id"]]
            assert entry["pair_id"] == item["pair_id"]
            assert entry["arm"] in ab.ARMS

    def test_only_fully_successful_pairs_are_reviewed(self, sample_rows):
        _sample, records = _records(sample_rows)
        records[1].update(success=False)
        review = ab.build_blind_review(records, 42)
        assert review["key"]["eligible_pairs"] == len(records) // 2 - 1

    def test_criteria_are_the_frozen_six(self, sample_rows):
        _sample, records = _records(sample_rows)
        review = ab.build_blind_review(records, 42)["input"]
        assert [c["name"] for c in review["criteria"]] == list(ab.REVIEW_CRITERIA)
        assert len(review["criteria"]) == 6
        assert all(c["definition"] for c in review["criteria"])

    def test_requested_pair_count(self):
        assert ab.BLIND_REVIEW_PAIRS == 60


# ---------------------------------------------------------------------------
# Artifacts, isolation, CLI
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_protected_artifacts_are_listed(self):
        assert "dataset/queries/review_artifact.json" in ab.PROTECTED_ARTIFACTS
        assert "dataset/queries/final_45_selection.json" in ab.PROTECTED_ARTIFACTS
        assert len(ab.PROTECTED_ARTIFACTS) == 7

    def test_protected_hashes_are_recorded(self):
        hashes = ab.protected_hashes()
        assert set(hashes) == set(ab.PROTECTED_ARTIFACTS)
        assert all(len(value) == 64 or value == "missing" for value in hashes.values())

    def test_no_artifact_name_targets_a_protected_path(self):
        for name in ab.ARTIFACT_NAMES:
            assert "dataset/" not in name
            assert not name.startswith("..")

    def test_manifest_records_deviations(self, sample_rows):
        manifest = ab.build_manifest(2, 2, 42, "fake-model", ab.RetryPolicy())
        deviations = manifest["deviations_from_production"]
        assert deviations["imports_dropped"]["value"] is True
        assert deviations["prompt_context_omitted"]["value"] is True
        assert manifest["model"]["temperature"] == 0.7
        assert manifest["model"]["max_tokens"] == 128
        assert manifest["model"]["seed"] == 42

    def test_manifest_records_prompt_and_inputs(self, sample_rows):
        manifest = ab.build_manifest(2, 2, 42, "fake-model", ab.RetryPolicy())
        assert len(manifest["prompt"]["system_instruction_sha256"]) == 64
        assert set(manifest["inputs"]) == set(SPLITS)
        assert len(manifest["inputs"]["train"]["sha256"]) == 64
        assert "protected_artifacts" in manifest

    def test_experiment_writes_only_into_out_dir(self, sample_rows, tmp_path):
        before = ab.protected_hashes()
        out = tmp_path / "ab"
        ab.run_experiment(
            out, 2, 1, 42, FakeModel([good_reply()] * 20), ab.RetryPolicy()
        )
        written = {path.name for path in out.iterdir()}
        assert written == set(ab.ARTIFACT_NAMES)
        assert ab.protected_hashes() == before

    def test_generations_file_holds_both_arms(self, sample_rows, tmp_path):
        out = tmp_path / "ab"
        outcome = ab.run_experiment(
            out, 2, 1, 42, FakeModel([good_reply()] * 20), ab.RetryPolicy()
        )
        lines = (out / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert len(records) == len(outcome["records"])
        assert {record["arm"] for record in records} == set(ab.ARMS)

    def test_sample_file_holds_the_shared_sample(self, sample_rows, tmp_path):
        out = tmp_path / "ab"
        ab.run_experiment(
            out, 2, 1, 42, FakeModel([good_reply()] * 20), ab.RetryPolicy()
        )
        rows = [
            json.loads(line)
            for line in (out / "sample.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 3
        assert "source_code" not in rows[0]

    def test_default_out_dir_is_under_ignored_artifacts(self):
        args = ab.parse_args([])
        assert "artifacts" in str(args.out_dir).replace("\\", "/")
        assert ab.EXPERIMENT_ID in str(args.out_dir)

    def test_default_sample_sizes(self):
        args = ab.parse_args([])
        assert args.train_sample == 100
        assert args.validation_sample == 100
        assert args.seed == 42
        assert args.model == "qwen2.5-coder:7b"

    def test_smoke_sizes_are_accepted(self):
        args = ab.parse_args(["--train-sample", "5", "--validation-sample", "5"])
        assert args.train_sample == 5
        assert args.validation_sample == 5

    def test_sample_flag_writes_only_sample_and_manifest(self, sample_rows, tmp_path):
        out = tmp_path / "sample-only"
        assert ab.main(["--sample", "--out-dir", str(out), "--train-sample", "2",
                        "--validation-sample", "1"]) == 0
        assert {path.name for path in out.iterdir()} == {
            "manifest.json",
            "sample.jsonl",
        }

    def test_protected_change_aborts_the_run(self, sample_rows, tmp_path, monkeypatch):
        """A drift between the manifest snapshot and the end-of-run check aborts."""
        real = dict(ab.protected_hashes())
        calls = {"n": 0}

        def drifting():
            calls["n"] += 1
            if calls["n"] == 1:
                return real
            return {name: "0" * 64 for name in ab.PROTECTED_ARTIFACTS}

        monkeypatch.setattr(ab, "protected_hashes", drifting)
        with pytest.raises(RuntimeError, match="protected artifacts changed"):
            ab.run_experiment(
                tmp_path / "ab",
                1,
                1,
                42,
                FakeModel([good_reply()] * 4),
                ab.RetryPolicy(),
            )
