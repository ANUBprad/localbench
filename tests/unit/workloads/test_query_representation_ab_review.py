"""Tests for the development-only blind-review analyzer.

No Ollama, no network.  The analyzer is a script, so it is loaded by path the
same way the other script tests in this package do it.

Every fixture is synthetic: a full 60-pair experiment plus a matching review
file, written to ``tmp_path``.  Nothing here reads or writes the real frozen
artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import re
import socket
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_query_representation_ab_review.py"
)
_spec = importlib.util.spec_from_file_location(
    "analyze_query_representation_ab_review", _SCRIPT_PATH
)
az = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("analyze_query_representation_ab_review", az)
_spec.loader.exec_module(az)

CRITERIA = az.REVIEW_CRITERIA
N_PAIRS = az.EXPECTED_PAIRS
N_ITEMS = az.EXPECTED_REVIEW_ITEMS


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_experiment(root: Path, n_pairs: int = N_PAIRS) -> Path:
    """A complete synthetic experiment with all six frozen artifacts."""
    experiment_dir = root / "v1"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    mapping: dict[str, dict] = {}
    generations: list[dict] = []
    sample: list[dict] = []

    for index in range(1, n_pairs + 1):
        pair_id = f"p{index:03d}"
        code_unit_id = f"c{index:03d}"
        split = "train" if index % 2 else "validation"
        sample.append({"code_unit_id": code_unit_id, "split": split})
        for offset, arm in enumerate(("A", "B"), start=1):
            review_id = f"r{(index - 1) * 2 + offset:03d}"
            items.append(
                {
                    "review_id": review_id,
                    "pair_id": pair_id,
                    "query": f"secret query text for {code_unit_id} {arm}",
                }
            )
            mapping[review_id] = {
                "pair_id": pair_id,
                "code_unit_id": code_unit_id,
                "arm": arm,
            }
            generations.append(
                {
                    "code_unit_id": code_unit_id,
                    "arm": arm,
                    "split": split,
                    "symbol_type": "function",
                    "success": True,
                }
            )

    _write_json(
        experiment_dir / "manifest.json",
        {
            "experiment_id": "query_representation_ab",
            "version": "v1",
            "code_revision": "synthetic000",
            "model": {"name": "synthetic-model", "temperature": 0.7},
            "review": {
                "criteria": list(CRITERIA),
                "criteria_source": "synthetic",
                "blind_pairs": n_pairs,
                "blinded": True,
            },
        },
    )
    _write_jsonl(experiment_dir / "sample.jsonl", sample)
    _write_jsonl(experiment_dir / "generations.jsonl", generations)
    _write_json(
        experiment_dir / "automatic_summary.json",
        {
            "overall": {
                "A": {"success": n_pairs, "units": n_pairs},
                "B": {"success": n_pairs, "units": n_pairs},
                "paired": {"pairs": n_pairs, "fully_successful": n_pairs},
            }
        },
    )
    _write_json(
        experiment_dir / "blind_review_input.json",
        {
            "experiment_id": "query_representation_ab",
            "version": "v1",
            "criteria": [
                {"name": name, "definition": f"{name} must hold."} for name in CRITERIA
            ],
            "items": items,
        },
    )
    _write_json(
        experiment_dir / "blind_review_key.json",
        {
            "experiment_id": "query_representation_ab",
            "seed": 42,
            "pairs": n_pairs,
            "eligible_pairs": 3 * n_pairs,
            "mapping": mapping,
        },
    )
    return experiment_dir


def make_pattern(
    both: int, a_only: int, b_only: int, neither: int
) -> tuple[list[bool], list[bool]]:
    """A 60-pair boolean pattern, pair-ordered."""
    a_values: list[bool] = []
    b_values: list[bool] = []
    for passed in (True, False):
        for _ in range(both if passed else neither):
            a_values.append(passed)
            b_values.append(passed)
    for _ in range(a_only):
        a_values.append(True)
        b_values.append(False)
    for _ in range(b_only):
        a_values.append(False)
        b_values.append(True)
    return a_values, b_values


def make_reviews(
    key: dict,
    a_values: list[bool],
    b_values: list[bool],
    *,
    with_accepted: bool = False,
    notes: dict[str, str] | None = None,
    overrides: dict[str, dict] | None = None,
) -> dict:
    """A review payload whose every record passes all six criteria."""
    mapping = key["mapping"]
    arm_of = {rid: entry["arm"] for rid, entry in mapping.items()}
    by_pair: dict[str, dict[str, str]] = {}
    for review_id in sorted(mapping):
        by_pair.setdefault(mapping[review_id]["pair_id"], {})[arm_of[review_id]] = (
            review_id
        )

    reviews: list[dict] = []
    for index, pair_id in enumerate(sorted(by_pair)):
        for arm, passed in (("A", a_values[index]), ("B", b_values[index])):
            review_id = by_pair[pair_id][arm]
            criteria = {name: bool(passed) for name in CRITERIA}
            record: dict = {
                "review_id": review_id,
                "pair_id": pair_id,
                "criteria": criteria,
            }
            if with_accepted:
                record["accepted"] = passed
            if notes and review_id in notes:
                record["note"] = notes[review_id]
            if overrides and review_id in overrides:
                record.update(overrides[review_id])
            reviews.append(record)
    return {"experiment_id": "query_representation_ab", "reviews": reviews}


def write_review(
    experiment_dir: Path, payload: dict | list, name: str = "reviews.json"
) -> Path:
    path = experiment_dir / name
    _write_json(path, payload)
    return path


def run_cli(
    experiment_dir: Path,
    review_file: Path | None = None,
    *extra: str,
) -> int:
    argv = ["--experiment-dir", str(experiment_dir)]
    if review_file is not None:
        argv += ["--review-file", str(review_file)]
    return az.main([*argv, *extra])


def analysis_dir(experiment_dir: Path) -> Path:
    return experiment_dir / "analysis"


def final_outputs(experiment_dir: Path) -> set[str]:
    directory = analysis_dir(experiment_dir)
    if not directory.exists():
        return set()
    return {path.name for path in directory.iterdir() if path.is_file()}


def complete_review(experiment_dir: Path, **kwargs) -> Path:
    """A valid 60-pair review; A slightly ahead, so counts are asymmetric."""
    key = json.loads(
        (experiment_dir / "blind_review_key.json").read_text(encoding="utf-8")
    )
    a_values, b_values = make_pattern(both=50, a_only=10, b_only=0, neither=0)
    return write_review(experiment_dir, make_reviews(key, a_values, b_values, **kwargs))


# ---------------------------------------------------------------------------
# Acceptance derivation
# ---------------------------------------------------------------------------


def test_acceptance_requires_all_six_criteria() -> None:
    assert az.derive_accepted({name: True for name in CRITERIA}) is True
    five = {name: True for name in CRITERIA}
    five[CRITERIA[2]] = False
    assert az.derive_accepted(five) is False
    assert az.derive_accepted({name: False for name in CRITERIA}) is False


def test_partial_criteria_map_is_never_accepted() -> None:
    assert az.derive_accepted({CRITERIA[0]: True}) is False
    assert az.derive_accepted({}) is False
    assert az.derive_accepted("not a mapping") is False  # type: ignore[arg-type]


def test_non_boolean_values_do_not_count_as_passing() -> None:
    assert az.derive_accepted({name: 1 for name in CRITERIA}) is False


# ---------------------------------------------------------------------------
# Exact McNemar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a_only", "b_only", "expected"),
    [
        (0, 0, 1.0),
        (1, 0, 1.0),
        (3, 1, 0.625),
        (8, 2, 0.109375),
        (5, 5, 1.0),
        (10, 0, 0.001953125),
        (12, 2, 0.012939453125),
    ],
)
def test_exact_mcnemar_p_values(a_only: int, b_only: int, expected: float) -> None:
    assert az.exact_mcnemar_p(a_only, b_only) == pytest.approx(expected, abs=1e-9)


def test_exact_mcnemar_is_symmetric_and_capped() -> None:
    assert az.exact_mcnemar_p(3, 7) == az.exact_mcnemar_p(7, 3)
    assert az.exact_mcnemar_p(6, 6) == 1.0
    assert 0.0 <= az.exact_mcnemar_p(0, 30) <= 1.0


def test_paired_table_partitions_every_pair() -> None:
    a_values = [True, True, False, False]
    b_values = [True, False, True, False]
    table = az.paired_table(a_values, b_values)
    assert table == {"both": 1, "a_only": 1, "b_only": 1, "neither": 1}
    assert sum(table.values()) == len(a_values)


def test_paired_table_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        az.paired_table([True, False], [True])


# ---------------------------------------------------------------------------
# Record-level validation
# ---------------------------------------------------------------------------


def _validate(experiment_dir: Path, payload: object):
    blind_input = json.loads(
        (experiment_dir / "blind_review_input.json").read_text(encoding="utf-8")
    )
    key = json.loads(
        (experiment_dir / "blind_review_key.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (experiment_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return az.validate_reviews(blind_input, key, manifest, payload)


def test_complete_review_validates(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    report, cleaned = _validate(experiment_dir, payload)
    assert report.complete
    assert report.errors == []
    assert report.valid == N_ITEMS
    assert report.missing == 0
    assert len(cleaned) == N_ITEMS


def test_bare_list_payload_is_accepted(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    key = json.loads(
        (experiment_dir / "blind_review_key.json").read_text(encoding="utf-8")
    )
    a_values, b_values = make_pattern(50, 10, 0, 0)
    payload = make_reviews(key, a_values, b_values)["reviews"]
    report, _ = _validate(experiment_dir, payload)
    assert report.complete


def test_partial_review_names_every_missing_id(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    dropped = {record["review_id"] for record in payload["reviews"][:83]}
    payload["reviews"] = [
        record for record in payload["reviews"] if record["review_id"] not in dropped
    ]
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete
    assert report.received == 37
    assert report.missing == 83
    assert set(report.missing_review_ids) == dropped


def test_duplicate_review_ids_are_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"].append(dict(payload["reviews"][0]))
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete
    assert report.duplicate_review_ids
    assert "duplicated" in " ".join(report.errors).lower()


def test_unknown_review_id_is_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"][0]["review_id"] = "r999"
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete
    assert report.unknown_review_ids == ["r999"]


def test_arm_field_in_a_review_record_is_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"][0]["arm"] = "A"
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete
    assert any("unblinding" in error for error in report.errors)


def test_code_unit_id_in_a_review_record_is_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"][0]["code_unit_id"] = "c001"
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete


def test_unexpected_field_is_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"][0]["confidence"] = 3
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete
    assert any("unexpected field" in error for error in report.errors)


def test_wrong_pair_id_is_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"][0]["pair_id"] = "p060"
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete
    assert any("does not match" in error for error in report.errors)


def test_missing_criterion_is_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    del payload["reviews"][0]["criteria"][CRITERIA[4]]
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete
    assert any("missing criterion" in error for error in report.errors)


def test_unknown_criterion_is_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"][0]["criteria"]["Feels nice"] = True
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete
    assert any("unknown criterion" in error for error in report.errors)


@pytest.mark.parametrize("bad", [1, 0, "true", None, 1.0, []])
def test_non_boolean_criterion_value_is_rejected(bad: object) -> None:
    criteria = {name: True for name in CRITERIA}
    criteria[CRITERIA[0]] = bad
    key_entry = {"pair_id": "p001", "code_unit_id": "c001", "arm": "A"}
    record = {"review_id": "r001", "pair_id": "p001", "criteria": criteria}
    clean, errors = az._validate_record(record, key_entry)
    assert clean is None
    assert any("must be a boolean" in error for error in errors)


def test_explicit_accepted_contradiction_is_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    key = json.loads(
        (experiment_dir / "blind_review_key.json").read_text(encoding="utf-8")
    )
    a_values, b_values = make_pattern(50, 10, 0, 0)
    payload = make_reviews(key, a_values, b_values, with_accepted=True)
    payload["reviews"][0]["accepted"] = not payload["reviews"][0]["accepted"]
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete
    assert any("contradicts the frozen rule" in error for error in report.errors)


def test_explicit_accepted_agreement_is_accepted(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    key = json.loads(
        (experiment_dir / "blind_review_key.json").read_text(encoding="utf-8")
    )
    a_values, b_values = make_pattern(50, 10, 0, 0)
    payload = make_reviews(key, a_values, b_values, with_accepted=True)
    report, _ = _validate(experiment_dir, payload)
    assert report.complete


def test_non_string_note_is_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    key = json.loads(
        (experiment_dir / "blind_review_key.json").read_text(encoding="utf-8")
    )
    a_values, b_values = make_pattern(50, 10, 0, 0)
    payload = make_reviews(key, a_values, b_values, notes={"r001": "clear"})
    payload["reviews"][0]["note"] = 7
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete
    assert any("'note' must be a string" in error for error in report.errors)


def test_one_sided_pair_is_rejected(tmp_path: Path) -> None:
    """Both items of a pair claim the same arm, so the pair cannot unblind."""
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    blind_input = json.loads(
        (experiment_dir / "blind_review_input.json").read_text(encoding="utf-8")
    )
    key = json.loads(
        (experiment_dir / "blind_review_key.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (experiment_dir / "manifest.json").read_text(encoding="utf-8")
    )
    key["mapping"]["r002"] = dict(key["mapping"]["r002"], arm="A")
    report, _ = az.validate_reviews(blind_input, key, manifest, payload)
    assert not report.complete
    assert any("one A and one B" in error for error in report.errors)


def test_whole_pair_missing_is_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"] = [
        record for record in payload["reviews"] if record["pair_id"] != "p001"
    ]
    report, _ = _validate(experiment_dir, payload)
    assert not report.complete
    assert report.missing == 2


def test_non_object_payload_is_rejected(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    report, _ = _validate(experiment_dir, "not a review")
    assert not report.complete
    assert report.errors


# ---------------------------------------------------------------------------
# Unblinding
# ---------------------------------------------------------------------------


def test_unblinding_uses_the_key_not_record_order(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"].reverse()
    blind_input = json.loads(
        (experiment_dir / "blind_review_input.json").read_text(encoding="utf-8")
    )
    key = json.loads(
        (experiment_dir / "blind_review_key.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (experiment_dir / "manifest.json").read_text(encoding="utf-8")
    )
    _, cleaned = az.validate_reviews(blind_input, key, manifest, payload)
    generations = {
        (row["code_unit_id"], row["arm"]): row
        for row in [
            json.loads(line)
            for line in (experiment_dir / "generations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    }
    pairs = az.unblind(key, cleaned, generations)
    assert len(pairs) == N_PAIRS
    assert [pair["pair_id"] for pair in pairs] == sorted(
        pair["pair_id"] for pair in pairs
    )
    # Pattern was 50 both, 10 A-only: the counts must survive the unblinding.
    assert sum(1 for pair in pairs if pair["overall_outcome"] == "a_only") == 10
    assert sum(1 for pair in pairs if pair["overall_outcome"] == "both") == 50
    for pair in pairs:
        assert pair["a_review_id"] != pair["b_review_id"]
        assert key["mapping"][pair["a_review_id"]]["arm"] == "A"
        assert key["mapping"][pair["b_review_id"]]["arm"] == "B"


def test_split_metadata_is_recovered_from_generations(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    blind_input = json.loads(
        (experiment_dir / "blind_review_input.json").read_text(encoding="utf-8")
    )
    key = json.loads(
        (experiment_dir / "blind_review_key.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (experiment_dir / "manifest.json").read_text(encoding="utf-8")
    )
    _, cleaned = az.validate_reviews(blind_input, key, manifest, payload)
    generations = {
        (row["code_unit_id"], row["arm"]): row
        for row in [
            json.loads(line)
            for line in (experiment_dir / "generations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    }
    pairs = az.unblind(key, cleaned, generations)
    assert {pair["split"] for pair in pairs} == {"train", "validation"}
    assert pairs[0]["split"] == "train"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_cli_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        az.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_cli_defaults_derive_from_experiment_dir() -> None:
    args = az.parse_args(["--experiment-dir", "some/dir"])
    assert args.review_file == Path("some/dir") / "blind_review_results.json"
    assert args.analysis_dir == Path("some/dir") / "analysis"
    assert args.validate_only is False


def test_missing_review_file_reports_incomplete(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    assert run_cli(experiment_dir) == 1
    assert final_outputs(experiment_dir) == {"review_validation.json"}
    report = json.loads(
        (analysis_dir(experiment_dir) / "review_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["complete"] is False
    assert report["received"] == 0
    assert report["missing"] == N_ITEMS
    assert len(report["missing_review_ids"]) == N_ITEMS


def test_incomplete_review_writes_no_analysis(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"] = payload["reviews"][:37]
    write_review(experiment_dir, payload)
    assert run_cli(experiment_dir, review) == 1
    assert final_outputs(experiment_dir) == {"review_validation.json"}


def test_stale_analysis_is_removed_when_a_review_regresses(
    tmp_path: Path,
) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    assert run_cli(experiment_dir, review) == 0
    assert "blind_review_analysis.json" in final_outputs(experiment_dir)

    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"] = payload["reviews"][:10]
    write_review(experiment_dir, payload)
    assert run_cli(experiment_dir, review) == 1
    assert final_outputs(experiment_dir) == {"review_validation.json"}


def test_validate_only_writes_no_analysis(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    assert run_cli(experiment_dir, review, "--validate-only") == 0
    assert final_outputs(experiment_dir) == {"review_validation.json"}


def test_validate_only_still_fails_on_an_incomplete_review(
    tmp_path: Path,
) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviews"] = payload["reviews"][:50]
    write_review(experiment_dir, payload)
    assert run_cli(experiment_dir, review, "--validate-only") == 1


def test_complete_review_produces_exactly_four_outputs(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    assert run_cli(experiment_dir, review) == 0
    assert final_outputs(experiment_dir) == set(az.ANALYSIS_OUTPUTS)


def test_summary_counts_match_the_known_pattern(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    run_cli(experiment_dir, review)
    summary = json.loads(
        (analysis_dir(experiment_dir) / "review_summary.json").read_text(
            encoding="utf-8"
        )
    )
    overall = summary["review"]["rows"][0]
    assert overall["outcome"] == "Overall acceptance"
    assert overall["a_pass"] == 60
    assert overall["b_pass"] == 50
    assert overall["both"] == 50
    assert overall["a_only"] == 10
    assert overall["b_only"] == 0
    assert overall["neither"] == 0
    assert overall["discordant_pairs"] == 10
    assert overall["exact_mcnemar_p"] == pytest.approx(0.001953, abs=1e-6)
    # every row must partition all 60 pairs
    for row in summary["review"]["rows"]:
        assert row["both"] + row["a_only"] + row["b_only"] + row["neither"] == 60
        assert row["a_total"] == row["b_total"] == 60
    assert len(summary["review"]["rows"]) == 1 + len(CRITERIA)
    assert [row["outcome"] for row in summary["review"]["rows"]] == [
        "Overall acceptance",
        *CRITERIA,
    ]


def test_summary_keeps_generation_context_separate(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    run_cli(experiment_dir, review)
    summary = json.loads(
        (analysis_dir(experiment_dir) / "review_summary.json").read_text(
            encoding="utf-8"
        )
    )
    context = summary["generation_context"]
    assert context["overall"]["paired"]["pairs"] == N_PAIRS
    assert "separately" in context["note"]
    # no combined metric anywhere in the payload
    text = json.dumps(summary).lower()
    for banned in ("combined_score", "quality_score", "weighted"):
        assert banned not in text


def test_analysis_is_deterministic_across_runs(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    run_cli(experiment_dir, review)
    first = {
        name: (analysis_dir(experiment_dir) / name).read_bytes()
        for name in az.ANALYSIS_OUTPUTS
    }
    run_cli(experiment_dir, review)
    second = {
        name: (analysis_dir(experiment_dir) / name).read_bytes()
        for name in az.ANALYSIS_OUTPUTS
    }
    assert first == second


def test_no_wall_clock_timestamps_in_any_output(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    run_cli(experiment_dir, review)
    stamp = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
    for name in az.ANALYSIS_OUTPUTS:
        text = (analysis_dir(experiment_dir) / name).read_text(encoding="utf-8")
        assert not stamp.search(text), f"{name} embeds a wall-clock timestamp"
    # marker keys are only checked in the JSON outputs: the report legitimately
    # contains filesystem paths, and a tmp path can contain any substring.
    for name in set(az.ANALYSIS_OUTPUTS) - {"review_report.txt"}:
        text = (analysis_dir(experiment_dir) / name).read_text(encoding="utf-8")
        for marker in ("generated_at", "analyzed_at", "created_at", "timestamp"):
            assert marker not in text


def test_query_text_is_not_duplicated_into_analysis(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    run_cli(experiment_dir, review)
    for name in az.ANALYSIS_OUTPUTS:
        text = (analysis_dir(experiment_dir) / name).read_text(encoding="utf-8")
        assert "secret query text" not in text
    analysis = json.loads(
        (analysis_dir(experiment_dir) / "blind_review_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    assert "query" not in analysis
    required = {
        "pair_id",
        "code_unit_id",
        "split",
        "a_review_id",
        "b_review_id",
        "a_accepted",
        "b_accepted",
        "a_criteria",
        "b_criteria",
        "overall_outcome",
        "criterion_outcomes",
        "a_generation_success",
        "b_generation_success",
    }
    for pair in analysis["pairs"]:
        assert set(pair) in (
            required,
            required | {"a_note"},
            required | {"b_note"},
            required | {"a_note", "b_note"},
        )


def test_review_ids_stay_traceable_to_pairs(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    run_cli(experiment_dir, review)
    analysis = json.loads(
        (analysis_dir(experiment_dir) / "blind_review_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    key = json.loads(
        (experiment_dir / "blind_review_key.json").read_text(encoding="utf-8")
    )
    seen: set[str] = set()
    for pair in analysis["pairs"]:
        assert key["mapping"][pair["a_review_id"]]["pair_id"] == pair["pair_id"]
        assert key["mapping"][pair["b_review_id"]]["pair_id"] == pair["pair_id"]
        seen |= {pair["a_review_id"], pair["b_review_id"]}
    assert seen == set(key["mapping"])


def test_notes_are_persisted_and_counted_but_not_interpreted(
    tmp_path: Path,
) -> None:
    experiment_dir = build_experiment(tmp_path)
    key = json.loads(
        (experiment_dir / "blind_review_key.json").read_text(encoding="utf-8")
    )
    a_values, b_values = make_pattern(50, 10, 0, 0)
    notes = {"r001": "too vague to use", "r002": "clear and specific"}
    payload = make_reviews(key, a_values, b_values, notes=notes)
    review = write_review(experiment_dir, payload)
    run_cli(experiment_dir, review)
    analysis = json.loads(
        (analysis_dir(experiment_dir) / "blind_review_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(
        (analysis_dir(experiment_dir) / "review_summary.json").read_text(
            encoding="utf-8"
        )
    )
    note_block = summary["review"]["notes"]
    assert note_block["records_with_note"] == 2
    assert note_block["by_arm_after_unblinding"] == {"A": 1, "B": 1}
    # the prose is preserved verbatim against the right review ids ...
    assert analysis["pairs"][0]["a_note"] == "too vague to use"
    assert analysis["pairs"][0]["b_note"] == "clear and specific"
    assert analysis["pairs"][0]["a_review_id"] == "r001"
    assert analysis["pairs"][0]["b_review_id"] == "r002"
    # ... and counted only: no sentiment or classification is derived from it
    dumped = json.dumps(summary).lower().replace("not classified", "")
    assert "classification" not in dumped
    assert "sentiment" not in dumped


def test_report_has_all_eight_sections(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    run_cli(experiment_dir, review)
    report = (analysis_dir(experiment_dir) / "review_report.txt").read_text(
        encoding="utf-8"
    )
    for heading in (
        "1. EXPERIMENT IDENTITY",
        "2. REVIEW COMPLETENESS",
        "3. OVERALL ACCEPTANCE",
        "4. CRITERION RESULTS",
        "5. REVIEW NOTES",
        "6. GENERATION CONTEXT",
        "7. PROTECTED STATE",
        "8. ARTIFACT LOCATIONS",
    ):
        assert heading in report


def _report_body(text: str) -> str:
    """Report text without the artifact-locations section.

    That section embeds filesystem paths, and a pytest tmp directory is named
    after the test, so it can contain any substring.  Only the prose is
    checked for verdict language.
    """
    return text.split("8. ARTIFACT LOCATIONS")[0]


def test_no_output_uses_verdict_language(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    run_cli(experiment_dir, review)
    for name in set(az.ANALYSIS_OUTPUTS) - {"review_report.txt"}:
        text = (analysis_dir(experiment_dir) / name).read_text(encoding="utf-8")
        for word in az.FORBIDDEN_OUTPUT_WORDS:
            assert word not in text.lower(), f"{name} contains {word!r}"
    report = _report_body(
        (analysis_dir(experiment_dir) / "review_report.txt").read_text(encoding="utf-8")
    ).lower()
    for word in az.FORBIDDEN_OUTPUT_WORDS:
        assert word not in report, f"review_report.txt contains {word!r}"


def test_frozen_inputs_are_unchanged_by_the_run(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    before = az.snapshot_hashes(experiment_dir)
    review = complete_review(experiment_dir)
    run_cli(experiment_dir, review)
    assert az.snapshot_hashes(experiment_dir) == before


def test_writes_stay_inside_the_analysis_directory(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    before = {path.name for path in experiment_dir.iterdir()}
    run_cli(experiment_dir, review)
    after = {path.name for path in experiment_dir.iterdir()}
    assert after - before == {"analysis"}
    assert final_outputs(experiment_dir) == set(az.ANALYSIS_OUTPUTS)


def test_never_touches_the_network(tmp_path: Path, monkeypatch) -> None:
    def explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the analyzer must not open a socket")

    monkeypatch.setattr(socket, "socket", explode)
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    assert run_cli(experiment_dir, review) == 0


def test_benchmark_protection_set_is_the_experiments_own(
    tmp_path: Path,
) -> None:
    experiment_dir = build_experiment(tmp_path)
    hashes = az.snapshot_hashes(experiment_dir)
    protected = {key for key in hashes if key.startswith("protected/")}
    assert protected == {f"protected/{relative}" for relative in az.PROTECTED_ARTIFACTS}
    assert len(az.PROTECTED_ARTIFACTS) == 7
    assert all(value != "missing" for value in hashes.values())


def test_summary_records_every_frozen_input_hash(tmp_path: Path) -> None:
    experiment_dir = build_experiment(tmp_path)
    review = complete_review(experiment_dir)
    run_cli(experiment_dir, review)
    summary = json.loads(
        (analysis_dir(experiment_dir) / "review_summary.json").read_text(
            encoding="utf-8"
        )
    )
    recorded = summary["frozen_input_sha256"]
    assert recorded == az.snapshot_hashes(experiment_dir)
    assert len([key for key in recorded if key.startswith("frozen/")]) == 6


def test_criteria_come_from_the_experiment_not_a_local_copy() -> None:
    assert az.REVIEW_CRITERIA == az._experiment().REVIEW_CRITERIA
    assert len(CRITERIA) == 6
    assert az.EXPECTED_PAIRS == az._experiment().BLIND_REVIEW_PAIRS
    assert az.EXPECTED_REVIEW_ITEMS == 2 * az.EXPECTED_PAIRS
