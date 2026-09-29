"""Tests for the development-only blind query A/B review CLI.

Every fixture is synthetic: a 120-item blind input (plus a key and manifest
used *only* to prove schema compatibility with the analyzer) written to
``tmp_path``.  The reviewer tool itself never receives the key path.

No Ollama, no network.  The reviewer is a script, so it is loaded by path the
same way the other script tests in this package do it.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "run_blind_query_ab_review.py"
)
_spec = importlib.util.spec_from_file_location(
    "run_blind_query_ab_review", _SCRIPT_PATH
)
rq = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("run_blind_query_ab_review", rq)
_spec.loader.exec_module(rq)

CRITERIA = rq.REVIEW_CRITERIA
N_PAIRS = rq.EXPECTED_PAIRS
N_ITEMS = rq.EXPECTED_REVIEW_ITEMS
DEFINITIONS = {name: f"{name} definition, frozen." for name in CRITERIA}


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def build_blind_input(
    root: Path, n_pairs: int = N_PAIRS, **overrides
) -> tuple[Path, dict, dict]:
    """Synthetic blind input plus the key/manifest used for compat checks.

    Returns ``(input_path, key, manifest)``.  Only ``input_path`` is ever given
    to the reviewer tool.
    """
    experiment_dir = root / "v1"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    mapping: dict[str, dict] = {}
    for index in range(1, n_pairs + 1):
        pair_id = f"p{index:03d}"
        code_unit_id = f"c{index:03d}"
        for offset, arm in enumerate(("A", "B"), start=1):
            review_id = f"r{(index - 1) * 2 + offset:03d}"
            items.append(
                {
                    "review_id": review_id,
                    "pair_id": pair_id,
                    "query": f"synthetic query number {review_id}",
                }
            )
            mapping[review_id] = {
                "pair_id": pair_id,
                "code_unit_id": code_unit_id,
                "arm": arm,
            }

    blind_input = {
        "experiment_id": "query_representation_ab",
        "version": "v1",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "instructions": "Judge each query against every criterion independently.",
        "criteria": [
            {"name": name, "definition": DEFINITIONS[name]} for name in CRITERIA
        ],
        "items": items,
    }
    blind_input.update(overrides)

    input_path = experiment_dir / "blind_review_input.json"
    input_path.write_text(json.dumps(blind_input, indent=2) + "\n", encoding="utf-8")
    key = {
        "experiment_id": "query_representation_ab",
        "seed": 42,
        "pairs": n_pairs,
        "eligible_pairs": 3 * n_pairs,
        "mapping": mapping,
    }
    manifest = {
        "experiment_id": "query_representation_ab",
        "version": "v1",
        "code_revision": "synthetic000",
        "review": {"criteria": list(CRITERIA), "criteria_source": "synthetic"},
    }
    return input_path, key, manifest


class Scripted:
    """Stands in for ``input()``: replays canned answers, records prompts."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"unexpected extra prompt: {prompt!r}")
        return self.answers.pop(0)


def item_script(
    answers: tuple[str, ...] = ("y",) * 6, note: str = "", action: str = ""
) -> list[str]:
    """One complete item: six decisions, the note, then the save action."""
    return [*answers, note, action]


def items(
    count: int = 1, answers: tuple[str, ...] = ("y",) * 6, note: str = ""
) -> list[str]:
    """Script ``count`` complete items, then quit cleanly at the next item."""
    return item_script(answers=answers, note=note) * count + ["q"]


def write_results(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"experiment_id": "query_representation_ab", "reviews": records}),
        encoding="utf-8",
    )


def record_for(
    blind, index: int, *, passed: bool = True, note: str | None = None
) -> dict:
    item = blind.items[index]
    out = {
        "review_id": item["review_id"],
        "pair_id": item["pair_id"],
        "criteria": {name: passed for name in CRITERIA},
        "accepted": passed,
    }
    if note is not None:
        out["note"] = note
    return out


def load_scripted_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["reviews"]


# ---------------------------------------------------------------------------
# 1. Input loading and the frozen criteria
# ---------------------------------------------------------------------------


def test_input_with_120_items_is_loaded(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    assert len(blind.items) == N_ITEMS
    assert len(blind.review_ids) == N_ITEMS
    assert len(set(blind.review_ids)) == N_ITEMS
    assert blind.experiment_id == "query_representation_ab"
    assert blind.version == "v1"


def test_frozen_six_criteria_are_preserved_with_definitions(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    assert [name for name, _ in blind.criteria] == list(CRITERIA)
    assert len(CRITERIA) == 6
    for name, definition in blind.criteria:
        assert definition == DEFINITIONS[name]
    # the reviewer tool borrows the analyzer's list, so it cannot drift
    assert CRITERIA == rq._analyzer().REVIEW_CRITERIA


def test_input_order_is_preserved(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    expected = [f"r{index:03d}" for index in range(1, N_ITEMS + 1)]
    assert blind.review_ids == expected
    # the tool must not shuffle: pair_ids are already interleaved in the file
    assert [blind.items[i]["pair_id"] for i in range(4)] == [
        "p001",
        "p001",
        "p002",
        "p002",
    ]


def test_wrong_item_count_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path, n_pairs=10)
    with pytest.raises(rq.ReviewDataError, match="expected 120 review items"):
        rq.load_blind_input(input_path)


def test_criteria_mismatch_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    data["criteria"] = data["criteria"][:5]
    input_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(rq.ReviewDataError, match="frozen six"):
        rq.load_blind_input(input_path)


def test_duplicate_review_id_in_input_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    data["items"][1]["review_id"] = data["items"][0]["review_id"]
    input_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(rq.ReviewDataError, match="duplicate review_id"):
        rq.load_blind_input(input_path)


def test_invalid_pair_id_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    data["items"][0]["pair_id"] = "not-a-pair"
    input_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(rq.ReviewDataError, match="invalid pair_id"):
        rq.load_blind_input(input_path)


def test_unpaired_pair_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    # keep 60 pairs, but p001 loses an item to p002
    data["items"][1]["pair_id"] = "p002"
    input_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(rq.ReviewDataError, match="without exactly two items"):
        rq.load_blind_input(input_path)


def test_wrong_pair_count_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    # invent a 61st pair, so the pair count is wrong before the per-pair check
    data["items"][0]["pair_id"] = "p099"
    input_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(rq.ReviewDataError, match="expected 60 pairs"):
        rq.load_blind_input(input_path)


def test_arm_field_in_input_item_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    data["items"][0]["arm"] = "A"
    input_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(rq.ReviewDataError, match="unexpected field"):
        rq.load_blind_input(input_path)


# ---------------------------------------------------------------------------
# 2. Resume, ordering, and saving
# ---------------------------------------------------------------------------


def test_first_missing_review_is_selected_correctly(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    assert rq.first_missing_index(blind, []) == 0
    assert rq.first_missing_index(blind, [record_for(blind, i) for i in range(5)]) == 5
    # a hole is found, and the rest after it are not skipped over
    holey = [record_for(blind, i) for i in (0, 1, 2, 4)]
    assert rq.first_missing_index(blind, holey) == 3
    assert (
        rq.first_missing_index(blind, [record_for(blind, i) for i in range(N_ITEMS)])
        == N_ITEMS
    )


def test_completed_reviews_are_skipped_on_resume(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    script = Scripted(items(2, answers=("y", "y", "y", "y", "y", "n")))
    assert rq.main(["--input", str(input_path), "--output", str(out)], ask=script) == 0
    records = load_scripted_results(out)
    assert [r["review_id"] for r in records] == ["r001", "r002"]
    # one "n" means not all six passed
    assert records[0]["accepted"] is False
    assert records[1]["accepted"] is False


def test_run_starts_at_the_first_missing_item(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    write_results(out, [record_for(blind, i) for i in range(3)])

    script = Scripted(item_script() + ["q"])
    assert rq.main(["--input", str(input_path), "--output", str(out)], ask=script) == 0
    records = load_scripted_results(out)
    assert [r["review_id"] for r in records] == ["r001", "r002", "r003", "r004"]
    printed = capsys.readouterr().out
    # it resumed at the fourth item, not the first
    assert "Blind Query Review 4 / 120" in printed
    assert "Blind Query Review 1 / 120" not in printed
    assert "Completed: 3 / 120" in printed
    assert "Remaining: 117" in printed


def test_resume_preserves_existing_records(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    prior = [record_for(blind, i, note="original note") for i in range(3)]
    write_results(out, prior)

    rq.main(
        ["--input", str(input_path), "--output", str(out)],
        ask=Scripted(item_script() + ["q"]),
    )
    records = load_scripted_results(out)
    assert records[:3] == prior
    assert all(r["note"] == "original note" for r in records[:3])
    assert len(records) == 4


def test_quitting_preserves_completed_records(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    script = Scripted(item_script() * 3 + ["q"])
    assert rq.main(["--input", str(input_path), "--output", str(out)], ask=script) == 0
    records = load_scripted_results(out)
    assert [r["review_id"] for r in records] == ["r001", "r002", "r003"]
    # the unfinished item was not written
    assert all(r["review_id"] != "r004" for r in records)


def test_quit_from_the_action_prompt_does_not_save_the_item(
    tmp_path: Path,
) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    script = Scripted(item_script(answers=("n",) * 6, action="q"))
    rq.main(["--input", str(input_path), "--output", str(out)], ask=script)
    # the item was answered but never confirmed, so nothing was written
    assert not out.exists()


def test_back_restarts_the_current_item(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    # first pass all "n", then b, then a corrected all-"y" pass
    script = Scripted(["n"] * 6 + ["", "b"] + ["y"] * 6 + ["corrected", "", "q"])
    assert rq.main(["--input", str(input_path), "--output", str(out)], ask=script) == 0
    records = load_scripted_results(out)
    assert len(records) == 1
    assert records[0]["accepted"] is True
    assert records[0]["note"] == "corrected"


def test_invalid_answer_is_re_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    script = Scripted(["maybe", "y", "y", "y", "y", "y", "y", "", "", "q"])
    rq.main(["--input", str(input_path), "--output", str(out)], ask=script)
    assert load_scripted_results(out)[0]["accepted"] is True
    assert "Please answer y or n." in capsys.readouterr().out


def test_note_is_optional_and_preserved(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    rq.main(
        ["--input", str(input_path), "--output", str(out)],
        ask=Scripted(items(1, note="  readable but vague  ")),
    )
    record = load_scripted_results(out)[0]
    assert record["note"] == "readable but vague"


def test_blank_note_is_omitted(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    rq.main(
        ["--input", str(input_path), "--output", str(out)],
        ask=Scripted(items(1, note="   ")),
    )
    assert "note" not in load_scripted_results(out)[0]


def test_full_review_completes_all_120(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    script = Scripted(item_script() * N_ITEMS)
    assert rq.main(["--input", str(input_path), "--output", str(out)], ask=script) == 0
    records = load_scripted_results(out)
    assert len(records) == N_ITEMS
    assert [r["review_id"] for r in records] == [
        f"r{index:03d}" for index in range(1, N_ITEMS + 1)
    ]


# ---------------------------------------------------------------------------
# 3. Atomic saving
# ---------------------------------------------------------------------------


def test_no_temp_file_survives_a_successful_write(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    rq.main(
        ["--input", str(input_path), "--output", str(out)],
        ask=Scripted(items(1)),
    )
    assert out.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_interrupted_write_leaves_the_previous_file_intact(
    tmp_path: Path, monkeypatch
) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    original = {"experiment_id": "query_representation_ab", "reviews": []}
    write_results(out, [])
    before = out.read_bytes()

    def boom(_src, _dst):  # pragma: no cover - must be reached
        raise OSError("simulated interrupt during replace")

    monkeypatch.setattr(rq.os, "replace", boom)
    with pytest.raises(OSError):
        rq.atomic_write_results(out, blind, [record_for(blind, 0)])
    assert out.read_bytes() == before
    # the truncated temp never became the results file
    assert json.loads(out.read_text(encoding="utf-8")) == original


def test_each_item_is_saved_immediately(tmp_path: Path) -> None:
    """An exception mid-session must not lose earlier items."""
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    script = Scripted(item_script() * 2)  # no answers for the third item
    with pytest.raises(AssertionError):
        rq.main(["--input", str(input_path), "--output", str(out)], ask=script)
    records = load_scripted_results(out)
    assert [r["review_id"] for r in records] == ["r001", "r002"]


# ---------------------------------------------------------------------------
# 4. Fail-closed validation of an existing results file
# ---------------------------------------------------------------------------


def test_duplicate_review_ids_in_results_are_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    write_results(out, [record_for(blind, 0), record_for(blind, 0)])
    with pytest.raises(rq.ReviewDataError, match="duplicate review_id"):
        rq.load_results(out, blind)


def test_unknown_review_id_in_results_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    record = record_for(blind, 0)
    record["review_id"] = "r999"
    write_results(out, [record])
    with pytest.raises(rq.ReviewDataError, match="not present in the blind review"):
        rq.load_results(out, blind)


def test_missing_criterion_in_results_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    record = record_for(blind, 0)
    del record["criteria"][CRITERIA[3]]
    record["accepted"] = False
    write_results(out, [record])
    with pytest.raises(rq.ReviewDataError, match="missing criterion"):
        rq.load_results(out, blind)


def test_unknown_criterion_in_results_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    record = record_for(blind, 0)
    record["criteria"]["Sounds good"] = True
    write_results(out, [record])
    with pytest.raises(rq.ReviewDataError, match="unknown criterion"):
        rq.load_results(out, blind)


@pytest.mark.parametrize("bad", [1, 0, "true", None, 1.0])
def test_non_boolean_criterion_value_is_rejected(bad: object, tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    record = record_for(blind, 0)
    record["criteria"][CRITERIA[0]] = bad
    record["accepted"] = False
    write_results(out, [record])
    with pytest.raises(rq.ReviewDataError, match="must be a boolean"):
        rq.load_results(out, blind)


def test_explicit_inconsistent_accepted_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    record = record_for(blind, 0, passed=False)
    record["accepted"] = True
    write_results(out, [record])
    with pytest.raises(rq.ReviewDataError, match="contradicts the frozen rule"):
        rq.load_results(out, blind)


def test_arm_field_in_results_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    for field in ("arm", "representation", "code_unit_id", "split"):
        record = record_for(blind, 0)
        record[field] = "leak"
        write_results(out, [record])
        with pytest.raises(rq.ReviewDataError, match="unblinding field"):
            rq.load_results(out, blind)


def test_malformed_results_json_does_not_overwrite_work(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    out.write_text("{not json", encoding="utf-8")
    before = out.read_bytes()
    code = rq.main(["--input", str(input_path), "--output", str(out)])
    assert code == 2
    assert out.read_bytes() == before


def test_mismatched_pair_id_in_results_is_rejected(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    record = record_for(blind, 0)
    record["pair_id"] = "p060"
    write_results(out, [record])
    with pytest.raises(rq.ReviewDataError, match="does not match the blind input"):
        rq.load_results(out, blind)


# ---------------------------------------------------------------------------
# 5. Acceptance is derived, never asked
# ---------------------------------------------------------------------------


def test_accepted_is_derived_from_all_six_criteria(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    # five y and one n
    rq.main(
        ["--input", str(input_path), "--output", str(out)],
        ask=Scripted(items(1, answers=("y", "y", "y", "y", "y", "n"))),
    )
    record = load_scripted_results(out)[0]
    assert record["criteria"][CRITERIA[5]] is False
    assert record["accepted"] is False

    out2 = tmp_path / "results2.json"
    rq.main(
        ["--input", str(input_path), "--output", str(out2)],
        ask=Scripted(items(1)),
    )
    assert load_scripted_results(out2)[0]["accepted"] is True


def test_accepted_is_never_prompted_for() -> None:
    """The reviewer is asked exactly six decisions, a note, and one action."""
    item = {"review_id": "r001", "pair_id": "p001", "query": "synthetic query"}
    criteria = tuple((name, DEFINITIONS[name]) for name in CRITERIA)
    ask = Scripted(item_script())
    result = rq._collect(item, criteria, ask, 1, N_ITEMS)
    assert result is not None
    answers, note = result
    assert set(answers) == set(CRITERIA)
    assert len(answers) == 6
    assert note == ""
    # exactly 8 prompts: 6 decisions + note + action. No accepted question.
    assert len(ask.prompts) == 8
    assert not any("accept" in prompt.lower() for prompt in ask.prompts)
    assert ask.answers == []


# ---------------------------------------------------------------------------
# 6. Output shape and blindness
# ---------------------------------------------------------------------------


def test_output_contains_no_arm_or_representation_fields(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    rq.main(
        ["--input", str(input_path), "--output", str(out)],
        ask=Scripted(items(3)),
    )
    records = load_scripted_results(out)
    for record in records:
        assert set(record) <= {"review_id", "pair_id", "criteria", "accepted", "note"}
        assert set(record["criteria"]) == set(CRITERIA)
        assert not set(record) & rq.BLIND_FORBIDDEN_KEYS
    # scoped to the records: the top-level experiment_id is legitimately named
    # query_representation_ab, so only the reviewer data can be checked
    reviews_blob = json.dumps(records).lower()
    for banned in ("arm", "representation", "code_unit", "generation", "a_only"):
        assert banned not in reviews_blob


def test_results_are_accepted_by_the_analyzer(tmp_path: Path) -> None:
    """Schema compatibility is checked by the analyzer itself, not by hand."""
    input_path, key, manifest = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    rq.main(
        ["--input", str(input_path), "--output", str(out)],
        ask=Scripted(item_script(answers=("y", "n", "y", "y", "y", "y")) * N_ITEMS),
    )
    analyzer = rq._analyzer()
    report, cleaned = analyzer.validate_reviews(
        json.loads(input_path.read_text(encoding="utf-8")),
        key,
        manifest,
        json.loads(out.read_text(encoding="utf-8")),
    )
    assert report.complete, report.errors
    assert report.valid == N_ITEMS
    assert len(cleaned) == N_ITEMS
    rows = analyzer.build_summary(
        analyzer.unblind(
            key,
            cleaned,
            {
                (f"c{index:03d}", arm): {"split": "train", "success": True}
                for index in range(1, N_PAIRS + 1)
                for arm in ("A", "B")
            },
        ),
        cleaned,
        key,
    )
    assert len(rows["rows"]) == 1 + len(CRITERIA)


def test_source_never_references_the_unblinding_artifacts() -> None:
    source = _SCRIPT_PATH.read_text(encoding="utf-8")
    start = source.index("FORBIDDEN_ARTIFACTS")
    open_paren = source.index("(", start)
    close_paren = source.index(")", open_paren)
    body = source[:open_paren] + source[close_paren:]
    for name in rq.FORBIDDEN_ARTIFACTS:
        assert name not in body, f"{name} is referenced outside the ban list"


def test_source_imports_no_model_or_network_client() -> None:
    source = _SCRIPT_PATH.read_text(encoding="utf-8")
    for banned in ("socket", "requests", "httpx", "urllib", "ollama", "subprocess"):
        assert banned not in source


def test_no_network_calls_during_a_full_review(tmp_path: Path, monkeypatch) -> None:
    def explode(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("the reviewer must not open a socket")

    monkeypatch.setattr(socket, "socket", explode)
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    assert (
        rq.main(
            ["--input", str(input_path), "--output", str(out)],
            ask=Scripted(items(2)),
        )
        == 0
    )


# ---------------------------------------------------------------------------
# 7. Restart safety and validate-only
# ---------------------------------------------------------------------------


def test_restart_requires_explicit_yes(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    write_results(out, [record_for(blind, i) for i in range(3)])
    before = out.read_bytes()

    script = Scripted(["no"])
    assert (
        rq.main(
            ["--input", str(input_path), "--output", str(out), "--restart"], ask=script
        )
        == 1
    )
    assert out.read_bytes() == before


def test_restart_with_yes_clears_and_resumes(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    blind = rq.load_blind_input(input_path)
    out = tmp_path / "results.json"
    write_results(out, [record_for(blind, i) for i in range(3)])

    script = Scripted(["yes"] + items(1))
    assert (
        rq.main(
            ["--input", str(input_path), "--output", str(out), "--restart"], ask=script
        )
        == 0
    )
    records = load_scripted_results(out)
    assert len(records) == 1
    assert records[0]["review_id"] == "r001"


def test_validate_only_writes_nothing(tmp_path: Path) -> None:
    input_path, _, _ = build_blind_input(tmp_path)
    out = tmp_path / "results.json"
    assert (
        rq.main(["--input", str(input_path), "--output", str(out), "--validate-only"])
        == 0
    )
    assert not out.exists()


def test_default_paths_point_at_the_v1_experiment() -> None:
    args = rq.parse_args([])
    assert args.review_pairs == rq.EXPECTED_PAIRS
    input_path, output_path = rq.review_paths(
        rq.DEFAULT_EXPERIMENT_DIR, args.review_pairs
    )
    assert input_path == rq.DEFAULT_EXPERIMENT_DIR / "blind_review_input.json"
    assert output_path == rq.DEFAULT_EXPERIMENT_DIR / "blind_review_results.json"
    assert rq.DEFAULT_EXPERIMENT_DIR.as_posix().endswith("query_representation_ab/v1")


# ---------------------------------------------------------------------------
# Reduced review samples (--review-pairs)
# ---------------------------------------------------------------------------

SUBSET_PAIRS = 20


def derived(experiment_dir: Path) -> tuple[Path, Path]:
    return rq.review_paths(experiment_dir, SUBSET_PAIRS)


def reduced_argv(input_path: Path, *extra: str) -> list[str]:
    return [
        "--input",
        str(input_path),
        "--review-pairs",
        str(SUBSET_PAIRS),
        *extra,
    ]


def test_default_is_all_sixty_pairs() -> None:
    """1. Omitting the flag must keep the full frozen sample."""
    assert rq.parse_args([]).review_pairs == 60 == N_PAIRS


def test_twenty_pairs_selects_exactly_twenty_pairs(tmp_path: Path) -> None:
    """2. Exactly 20 distinct pair_ids come back."""
    input_path, _key, _manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    selected = rq.select_pair_ids(rq.load_blind_input(input_path), SUBSET_PAIRS)
    assert len(selected) == len(set(selected)) == SUBSET_PAIRS
    assert set(selected) <= {
        item["pair_id"] for item in rq.load_blind_input(input_path).items
    }


def test_selection_is_deterministic(tmp_path: Path) -> None:
    """3. Same input, same pairs, every time and on any machine."""
    input_path, _key, _manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    first = rq.select_pair_ids(rq.load_blind_input(input_path), SUBSET_PAIRS)
    assert rq.select_pair_ids(rq.load_blind_input(input_path), SUBSET_PAIRS) == first
    other_input, _k, _m = build_blind_input(tmp_path / "elsewhere", n_pairs=N_PAIRS)
    assert rq.select_pair_ids(rq.load_blind_input(other_input), SUBSET_PAIRS) == first


def test_selection_ignores_item_ordering(tmp_path: Path) -> None:
    """4. Reversing the presentation order cannot change which pairs are chosen."""
    input_path, _key, _manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    baseline = rq.select_pair_ids(rq.load_blind_input(input_path), SUBSET_PAIRS)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["items"] = list(reversed(payload["items"]))
    reordered = input_path.with_name("reordered.json")
    reordered.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert rq.select_pair_ids(rq.load_blind_input(reordered), SUBSET_PAIRS) == baseline


def test_both_arms_are_kept_for_every_selected_pair(tmp_path: Path) -> None:
    """5. Selection is at pair level, so each pair keeps both of its queries."""
    input_path, key, _manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    derived_input = rq.derive_blind_input(rq.load_blind_input(input_path), SUBSET_PAIRS)
    by_pair: dict[str, list[str]] = {}
    for item in derived_input.items:
        by_pair.setdefault(item["pair_id"], []).append(item["review_id"])
    assert len(by_pair) == SUBSET_PAIRS
    for pair_id, review_ids in by_pair.items():
        assert len(set(review_ids)) == 2, pair_id
        arms = sorted(key["mapping"][rid]["arm"] for rid in review_ids)
        assert arms == ["A", "B"], pair_id


def test_selected_review_ids_are_exactly_forty(tmp_path: Path) -> None:
    """6. 20 pairs x 2 queries = 40 unique review_ids."""
    input_path, _key, _manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    derived_input = rq.derive_blind_input(rq.load_blind_input(input_path), SUBSET_PAIRS)
    assert len(derived_input.items) == 40
    assert len(set(derived_input.review_ids)) == 40


def test_derived_input_is_still_fully_blinded(tmp_path: Path) -> None:
    """7. The reviewer sees the same blinded fields, nothing that unblinds."""
    input_path, key, _manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    assert rq.main(reduced_argv(input_path, "--validate-only"), ask=Scripted([])) == 0
    derived_input, _results = derived(input_path.parent)
    payload = json.loads(derived_input.read_text(encoding="utf-8"))
    assert [c["name"] for c in payload["criteria"]] == list(CRITERIA)
    assert len(payload["items"]) == 40
    for item in payload["items"]:
        assert set(item) == {"review_id", "pair_id", "query"}
        assert not set(item) & rq.BLIND_FORBIDDEN_KEYS
        arm = key["mapping"][item["review_id"]]["arm"]
        assert arm not in item["query"]
        assert arm not in item["review_id"]
    text = derived_input.read_text(encoding="utf-8")
    for forbidden in rq.FORBIDDEN_ARTIFACTS:
        assert forbidden not in text


def test_frozen_sixty_pair_input_is_never_rewritten(tmp_path: Path) -> None:
    """8. The frozen input and the full-size results file are left alone."""
    input_path, _key, _manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    before = input_path.read_bytes()
    rq.main(reduced_argv(input_path, "--validate-only"), ask=Scripted([]))
    assert input_path.read_bytes() == before
    assert not (input_path.parent / "blind_review_results.json").exists()
    assert rq.review_paths(input_path.parent, N_PAIRS) == (
        input_path,
        input_path.parent / "blind_review_results.json",
    )
    with pytest.raises(rq.ReviewDataError):
        rq.write_blind_input(input_path, rq.load_blind_input(input_path))


def test_twenty_pair_results_are_accepted_by_the_analyzer(tmp_path: Path) -> None:
    """9. The analyzer takes the derived input and clears the smaller set."""
    input_path, key, manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    answers = ("y", "n", "y", "y", "y", "y")
    rq.main(reduced_argv(input_path), ask=Scripted(item_script(answers=answers) * 40))
    derived_input, results_path = derived(input_path.parent)
    analyzer = rq._analyzer()
    report, cleaned = analyzer.validate_reviews(
        json.loads(derived_input.read_text(encoding="utf-8")),
        key,
        manifest,
        json.loads(results_path.read_text(encoding="utf-8")),
    )
    assert report.complete, report.errors
    assert (report.expected_pairs, report.expected_items) == (SUBSET_PAIRS, 40)
    assert report.valid == len(cleaned) == 40
    rows = analyzer.build_summary(
        analyzer.unblind(
            key,
            cleaned,
            {
                (f"c{index:03d}", arm): {"split": "train", "success": True}
                for index in range(1, N_PAIRS + 1)
                for arm in ("A", "B")
            },
        ),
        cleaned,
        key,
    )
    assert len(rows["rows"]) == 1 + len(CRITERIA)


def test_incomplete_twenty_pair_review_fails_closed(tmp_path: Path) -> None:
    """10. 39 of 40 is incomplete, and the missing id is named."""
    input_path, key, manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    rq.main(reduced_argv(input_path), ask=Scripted(item_script() * 39 + ["q"]))
    derived_input, results_path = derived(input_path.parent)
    analyzer = rq._analyzer()
    report, _cleaned = analyzer.validate_reviews(
        json.loads(derived_input.read_text(encoding="utf-8")),
        key,
        manifest,
        json.loads(results_path.read_text(encoding="utf-8")),
    )
    assert not report.complete
    assert report.valid == 39
    assert report.missing == 1
    assert len(report.missing_review_ids) == 1


def test_full_size_workflow_is_unchanged(tmp_path: Path) -> None:
    """11. No --review-pairs still reviews all 120 in the frozen order."""
    input_path, _key, _manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    out = tmp_path / "results.json"
    rq.main(
        ["--input", str(input_path), "--output", str(out)],
        ask=Scripted(item_script() * N_ITEMS),
    )
    records = json.loads(out.read_text(encoding="utf-8"))["reviews"]
    assert len(records) == N_ITEMS == 120
    assert [r["review_id"] for r in records] == rq.load_blind_input(
        input_path
    ).review_ids
    rq.main(
        ["--input", str(input_path), "--review-pairs", str(N_PAIRS), "--validate-only"],
        ask=Scripted([]),
    )
    assert not (input_path.parent / "blind_review_input_60pairs.json").exists()


def test_resume_works_for_the_twenty_pair_run(tmp_path: Path) -> None:
    """12. The reduced run saves, stops, and resumes where it left off."""
    input_path, _key, _manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    argv = reduced_argv(input_path)
    rq.main(argv, ask=Scripted(item_script(note="early") * 2 + ["q"]))
    derived_input, results_path = derived(input_path.parent)
    saved = json.loads(results_path.read_text(encoding="utf-8"))["reviews"]
    assert len(saved) == 2
    rq.main(argv, ask=Scripted(item_script() * 38))
    records = json.loads(results_path.read_text(encoding="utf-8"))["reviews"]
    assert len(records) == 40
    assert [r["note"] for r in records[:2]] == ["early", "early"]
    assert [r["review_id"] for r in records] == rq.load_blind_input(
        derived_input, expected_pairs=SUBSET_PAIRS
    ).review_ids


@pytest.mark.parametrize("bad", [0, -1, 61, 1.5])
def test_pair_count_outside_the_frozen_range_is_rejected(
    tmp_path: Path, bad: object
) -> None:
    input_path, _key, _manifest = build_blind_input(tmp_path, n_pairs=N_PAIRS)
    with pytest.raises(rq.ReviewDataError):
        rq.select_pair_ids(rq.load_blind_input(input_path), bad)  # type: ignore[arg-type]
