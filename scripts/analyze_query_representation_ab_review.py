"""Development-only analysis of the frozen A/B blind human review.

The frozen query-representation experiment produced a blinded review set
(``blind_review_input.json`` + ``blind_review_key.json``).  A human reviewer
fills in decisions in a *separate* file, ``blind_review_results.json``.  This
tool turns those decisions into neutral, paired, auditable evidence.

Order of operations is load-bearing:

    1. hash the frozen inputs and the benchmark-protection set
    2. validate the review file completely (fail closed)
    3. only then unblind through the key
    4. write analysis outputs under ``<experiment-dir>/analysis/``
    5. re-hash the frozen inputs and abort if anything moved

A/B identity is never exposed before validation passes, and a half-finished
review never produces analysis output: the tool reports exactly what is
missing and stops.

Acceptance rule, reproduced from the frozen artifacts rather than invented:
``blind_review_input.json`` states "A query is accepted only if all criteria
pass", and the repository's review workflow uses the same rule
(``scripts/apply_human_review.py``).  So acceptance is *derived* from the six
criteria.  If a results file also carries an explicit ``accepted`` boolean it
is cross-checked against the derived value and a mismatch fails closed.

Statistics: paired binary outcomes are summarised by a 2x2 paired table and
an exact two-sided McNemar (binomial) p-value computed with ``math.comb``.
At 60 pairs a large-sample chi-square approximation is not appropriate.  No
SciPy, no model calls, no network.

The tool reports counts and effect sizes only.  It computes no score, no
ranking, and no recommendation; the decision is made by a human afterwards.

Every expected count, the criteria list, and the protected-artifact set are
imported from the frozen experiment rather than restated here, so the
analyzer cannot disagree with the run that produced the data.

Usage:
    python scripts/analyze_query_representation_ab_review.py --help
    python scripts/analyze_query_representation_ab_review.py --validate-only
    python scripts/analyze_query_representation_ab_review.py \
        --experiment-dir artifacts/experiments/query_representation_ab/v1 \
        --review-file <experiment-dir>/blind_review_results.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _experiment() -> Any:
    """Import the frozen A/B script so its constants are the single source.

    Restating the criteria, the artifact list, or the protected set here would
    let the analyzer and the run drift apart, and the whole point of the
    protection check is that it means the same thing in both places.
    """
    path = REPO_ROOT / "scripts" / "run_query_representation_ab.py"
    if not path.exists():
        raise SystemExit(f"cannot find the frozen experiment script at {path}")
    spec = importlib.util.spec_from_file_location("run_query_representation_ab", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Reused verbatim from the frozen experiment.
ARMS: tuple[str, str] = _experiment().ARMS
REVIEW_CRITERIA: tuple[str, ...] = _experiment().REVIEW_CRITERIA
FROZEN_INPUTS: tuple[str, ...] = _experiment().ARTIFACT_NAMES
PROTECTED_ARTIFACTS: tuple[str, ...] = _experiment().PROTECTED_ARTIFACTS
BLIND_FORBIDDEN_KEYS: frozenset[str] = _experiment().BLIND_FORBIDDEN_KEYS
sha256_file = _experiment().sha256_file

EXPECTED_PAIRS: int = _experiment().BLIND_REVIEW_PAIRS
EXPECTED_REVIEW_ITEMS: int = 2 * EXPECTED_PAIRS

ANALYZER_VERSION = "review-analysis-1.0.0"

#: Analysis outputs, all under the analysis directory.  The report is a .txt on
#: purpose: it is written to be read by a person, not parsed.  Note that
#: .gitignore covers *.json and *.jsonl but not *.txt, so the report is an
#: untracked file and is never staged.
ANALYSIS_OUTPUTS: tuple[str, ...] = (
    "review_validation.json",
    "blind_review_analysis.json",
    "review_summary.json",
    "review_report.txt",
)

#: Fields a review record may carry.  Anything else is rejected, and the subset
#: that would reveal an arm gets a specific unblinding error.
ALLOWED_REVIEW_KEYS: frozenset[str] = frozenset(
    {
        "review_id",
        "pair_id",
        "criteria",
        "accepted",
        "note",
    }
)

#: Words that would turn a measurement into a judgement.  Guarded by a test.
FORBIDDEN_OUTPUT_WORDS: frozenset[str] = frozenset(
    {
        "winner",
        "loser",
        "better",
        "worse",
        "best",
        "superior",
        "preferred",
        "recommendation",
        "recommend",
        "verdict",
        "ranking",
        "ranked",
    }
)


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Hash protection
# ---------------------------------------------------------------------------


def snapshot_hashes(experiment_dir: Path) -> dict[str, str]:
    """SHA-256 of every frozen input and every protected benchmark artifact."""
    out: dict[str, str] = {}
    for name in FROZEN_INPUTS:
        path = experiment_dir / name
        out[f"frozen/{name}"] = sha256_file(path) if path.exists() else "missing"
    for relative in PROTECTED_ARTIFACTS:
        path = REPO_ROOT / relative
        out[f"protected/{relative}"] = sha256_file(path) if path.exists() else "missing"
    return out


def changed_hashes(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    )


# ---------------------------------------------------------------------------
# Validation (fail closed)
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Completeness and correctness of a review file.  Fails closed."""

    expected_items: int = EXPECTED_REVIEW_ITEMS
    expected_pairs: int = EXPECTED_PAIRS
    received: int = 0
    valid: int = 0
    invalid: int = 0
    missing: int = 0
    duplicated: int = 0
    missing_review_ids: list[str] = field(default_factory=list)
    duplicate_review_ids: list[str] = field(default_factory=list)
    unknown_review_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "expected_items": self.expected_items,
            "expected_pairs": self.expected_pairs,
            "received": self.received,
            "valid": self.valid,
            "invalid": self.invalid,
            "missing": self.missing,
            "duplicated": self.duplicated,
            "missing_review_ids": sorted(self.missing_review_ids),
            "duplicate_review_ids": sorted(self.duplicate_review_ids),
            "unknown_review_ids": sorted(self.unknown_review_ids),
            "errors": list(self.errors),
        }


def derive_accepted(criteria: dict[str, Any]) -> bool:
    """Acceptance per the frozen rule: all six criteria must pass.

    Returns False for a partial criteria mapping, so a half-filled record can
    never be read as accepted.
    """
    if not isinstance(criteria, dict) or set(criteria) != set(REVIEW_CRITERIA):
        return False
    return all(criteria[name] is True for name in REVIEW_CRITERIA)


def _extract_records(payload: Any) -> list[Any]:
    """Accept either ``{"reviews": [...]}`` or a bare list of records."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        records = payload.get("reviews")
        if records is None:
            raise ValueError("review file has no 'reviews' array")
        if not isinstance(records, list):
            raise ValueError("'reviews' must be an array")
        return records
    raise ValueError("review file must be an object with 'reviews' or an array")


def _validate_record(
    record: Any,
    key_entry: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate one review record against the frozen key and criteria."""
    if not isinstance(record, dict):
        return None, [f"review {record!r}: record is not an object"]

    review_id = record.get("review_id")
    if not isinstance(review_id, str):
        return None, [f"review {record!r}: review_id must be a string"]

    errors: list[str] = []
    forbidden = sorted(set(record) & BLIND_FORBIDDEN_KEYS)
    if forbidden:
        errors.append(
            f"{review_id}: record carries unblinding field(s) {forbidden}; "
            "the reviewer must stay blind"
        )
    unexpected = sorted(set(record) - ALLOWED_REVIEW_KEYS)
    if unexpected:
        errors.append(f"{review_id}: unexpected field(s) {unexpected}")

    if record.get("pair_id") != key_entry["pair_id"]:
        errors.append(
            f"{review_id}: pair_id {record.get('pair_id')!r} does not match "
            f"the frozen key ({key_entry['pair_id']!r})"
        )

    criteria = record.get("criteria")
    if not isinstance(criteria, dict):
        errors.append(f"{review_id}: 'criteria' must be an object")
        return None, errors

    missing = [name for name in REVIEW_CRITERIA if name not in criteria]
    extra = sorted(set(criteria) - set(REVIEW_CRITERIA))
    if missing:
        errors.append(f"{review_id}: missing criterion {missing}")
    if extra:
        errors.append(f"{review_id}: unknown criterion {extra}")
    for name in sorted(set(criteria) & set(REVIEW_CRITERIA)):
        # bool is a subclass of int, so the type must be checked exactly: 1 and
        # "true" are not acceptable decisions.
        if not isinstance(criteria[name], bool):
            errors.append(
                f"{review_id}: criterion {name!r} must be a boolean, got "
                f"{type(criteria[name]).__name__}"
            )

    note = record.get("note")
    if note is not None and not isinstance(note, str):
        errors.append(f"{review_id}: 'note' must be a string when present")

    accepted = record.get("accepted")
    if accepted is not None:
        if not isinstance(accepted, bool):
            errors.append(
                f"{review_id}: 'accepted' must be a boolean, got "
                f"{type(accepted).__name__}"
            )
        elif not missing and not extra and accepted != derive_accepted(criteria):
            errors.append(
                f"{review_id}: explicit accepted={accepted} contradicts the "
                f"frozen rule (all criteria pass) which gives "
                f"{derive_accepted(criteria)}"
            )

    if errors:
        return None, errors

    clean: dict[str, Any] = {
        "review_id": review_id,
        "pair_id": key_entry["pair_id"],
        "criteria": {name: criteria[name] for name in REVIEW_CRITERIA},
        "accepted": derive_accepted(criteria),
    }
    if isinstance(note, str) and note.strip():
        clean["note"] = note
    return clean, []


def validate_reviews(
    blind_input: dict[str, Any],
    key: dict[str, Any],
    manifest: dict[str, Any],
    payload: Any,
) -> tuple[ValidationReport, dict[str, dict[str, Any]]]:
    """Validate a review payload; return the report and cleaned records."""
    report = ValidationReport()

    # The frozen criteria must agree across the blind input and the manifest,
    # otherwise the two artifacts that define the review disagree.
    input_names = [item.get("name") for item in blind_input.get("criteria", [])]
    if tuple(input_names) != REVIEW_CRITERIA:
        report.errors.append(
            "blind_review_input.json criteria do not match the frozen six: "
            f"{input_names}"
        )
    manifest_names = tuple(manifest.get("review", {}).get("criteria", ()))
    if manifest_names != REVIEW_CRITERIA:
        report.errors.append(
            f"manifest criteria do not match the frozen six: {list(manifest_names)}"
        )

    # The review sample size is whatever the supplied input defines, so a
    # derived 20-pair input is analyzed on its own terms.  The frozen key is
    # still checked against the frozen constants: that validates the key
    # itself, not the size of the review sample drawn from it.
    input_ids = {item.get("review_id") for item in blind_input.get("items", [])}
    input_pair_ids = {item.get("pair_id") for item in blind_input.get("items", [])}
    report.expected_items = len(input_ids)
    report.expected_pairs = len(input_pair_ids)
    if report.expected_items != 2 * report.expected_pairs:
        report.errors.append(
            f"blind review input must hold exactly two items per pair: "
            f"{report.expected_items} items across {report.expected_pairs} pairs"
        )

    key_mapping = key.get("mapping", {})
    if len(key_mapping) != EXPECTED_REVIEW_ITEMS:
        report.errors.append(
            f"expected {EXPECTED_REVIEW_ITEMS} key mappings, found {len(key_mapping)}"
        )
    key_pairs = {entry.get("pair_id") for entry in key_mapping.values()}
    if len(key_pairs) != EXPECTED_PAIRS:
        report.errors.append(
            f"expected {EXPECTED_PAIRS} pairs in the blind key, found {len(key_pairs)}"
        )

    try:
        records = _extract_records(payload)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report, {}

    report.received = len(records)

    first_seen: dict[str, Any] = {}
    duplicates: set[str] = set()
    for record in records:
        review_id = record.get("review_id") if isinstance(record, dict) else None
        if not isinstance(review_id, str) or review_id not in key_mapping:
            continue
        if review_id in first_seen:
            duplicates.add(review_id)
        else:
            first_seen[review_id] = record
    report.duplicate_review_ids = sorted(duplicates)

    known = set(first_seen) - duplicates
    report.unknown_review_ids = sorted(
        record["review_id"]
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("review_id"), str)
        and record["review_id"] not in key_mapping
    )

    cleaned: dict[str, dict[str, Any]] = {}
    for review_id in sorted(first_seen):
        if review_id in duplicates:
            report.invalid += 1
            continue
        clean, errors = _validate_record(first_seen[review_id], key_mapping[review_id])
        if errors:
            report.invalid += 1
            report.errors.extend(errors)
            continue
        cleaned[review_id] = clean
    report.valid = len(cleaned)

    # Missing is measured against the supplied review input, not the full
    # frozen key: a 20-pair review is complete at 40 records, not 120.
    report.missing_review_ids = sorted(input_ids - known - duplicates)
    report.missing = len(report.missing_review_ids)
    if report.unknown_review_ids:
        report.invalid += len(report.unknown_review_ids)
        report.errors.append(
            f"unknown review_id(s) absent from the blind key: "
            f"{report.unknown_review_ids}"
        )
    if report.duplicate_review_ids:
        report.errors.append(f"duplicated review_id(s): {report.duplicate_review_ids}")
    if report.missing:
        report.errors.append(
            f"missing {report.missing} review(s): expected "
            f"{report.expected_items}, received {report.received}"
        )

    arms_by_pair: dict[str, list[str]] = {}
    for review_id, record in cleaned.items():
        arms_by_pair.setdefault(record["pair_id"], []).append(
            key_mapping[review_id]["arm"]
        )
    for pair_id in sorted(arms_by_pair):
        if sorted(arms_by_pair[pair_id]) != sorted(ARMS):
            report.errors.append(
                f"{pair_id}: reviewed items do not map to one A and one B "
                f"(arms={sorted(arms_by_pair[pair_id])})"
            )
    if len(arms_by_pair) != report.expected_pairs:
        report.errors.append(
            f"expected {report.expected_pairs} reviewed pairs, "
            f"found {len(arms_by_pair)}"
        )
    return report, cleaned


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


def paired_table(a_values: Sequence[bool], b_values: Sequence[bool]) -> dict[str, int]:
    """2x2 paired counts: both / A only / B only / neither."""
    both = a_only = b_only = neither = 0
    for a_value, b_value in zip(a_values, b_values, strict=True):
        if a_value and b_value:
            both += 1
        elif a_value:
            a_only += 1
        elif b_value:
            b_only += 1
        else:
            neither += 1
    return {
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
    }


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar (binomial) p-value for paired binary outcomes.

    Under the null the discordant pairs split evenly, so the count landing on
    either side is Binomial(a_only + b_only, 0.5).  The two-sided p-value is
    twice the lower tail, capped at 1.  Exact rather than chi-square because 60
    pairs with few discordant observations violates the large-sample
    assumption badly.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    smaller = min(a_only, b_only)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _rate(passed: int, total: int) -> float:
    return round(passed / total, 6) if total else 0.0


def outcome_row(
    label: str, a_values: Sequence[bool], b_values: Sequence[bool]
) -> dict[str, Any]:
    """One criterion, or overall acceptance, as a neutral paired summary."""
    total = len(a_values)
    a_pass = sum(1 for value in a_values if value)
    b_pass = sum(1 for value in b_values if value)
    table = paired_table(a_values, b_values)
    return {
        "outcome": label,
        "a_pass": a_pass,
        "a_total": total,
        "a_rate": _rate(a_pass, total),
        "b_pass": b_pass,
        "b_total": total,
        "b_rate": _rate(b_pass, total),
        "a_minus_b_percentage_points": (
            round((a_pass - b_pass) / total * 100, 4) if total else 0.0
        ),
        "both": table["both"],
        "a_only": table["a_only"],
        "b_only": table["b_only"],
        "neither": table["neither"],
        "discordant_pairs": table["a_only"] + table["b_only"],
        "exact_mcnemar_p": round(exact_mcnemar_p(table["a_only"], table["b_only"]), 6),
    }


def cell(a_value: bool, b_value: bool) -> str:
    """Neutral label for one pair's outcome, e.g. "a_only"."""
    if a_value and b_value:
        return "both"
    if a_value:
        return "a_only"
    if b_value:
        return "b_only"
    return "neither"


# ---------------------------------------------------------------------------
# Unblinding
# ---------------------------------------------------------------------------


def unblind(
    key: dict[str, Any],
    cleaned: dict[str, dict[str, Any]],
    generations: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct pair-level records through the frozen key.

    Runs only after validation passes.  Returns pairs sorted by ``pair_id``,
    each with exactly one A record and one B record.  Query text is not
    copied: the decision is the evidence, and the input artifact already holds
    the text against ``review_id``.
    """
    key_mapping = key["mapping"]
    by_pair: dict[str, list[str]] = {}
    for review_id in sorted(cleaned):
        by_pair.setdefault(key_mapping[review_id]["pair_id"], []).append(review_id)

    pairs: list[dict[str, Any]] = []
    for pair_id in sorted(by_pair):
        arms = {key_mapping[rid]["arm"]: rid for rid in sorted(by_pair[pair_id])}
        if sorted(arms) != sorted(ARMS):
            raise ValueError(f"{pair_id}: cannot map exactly one A and one B")
        a_record = cleaned[arms["A"]]
        b_record = cleaned[arms["B"]]
        code_unit_id = key_mapping[arms["A"]]["code_unit_id"]
        gen_a = generations.get((code_unit_id, "A"), {})
        gen_b = generations.get((code_unit_id, "B"), {})
        record = {
            "pair_id": pair_id,
            "code_unit_id": code_unit_id,
            "split": gen_a.get("split"),
            "a_review_id": arms["A"],
            "b_review_id": arms["B"],
            "a_accepted": a_record["accepted"],
            "b_accepted": b_record["accepted"],
            "a_criteria": a_record["criteria"],
            "b_criteria": b_record["criteria"],
            "overall_outcome": cell(a_record["accepted"], b_record["accepted"]),
            "criterion_outcomes": {
                name: cell(a_record["criteria"][name], b_record["criteria"][name])
                for name in REVIEW_CRITERIA
            },
            "a_generation_success": gen_a.get("success"),
            "b_generation_success": gen_b.get("success"),
        }
        # Reviewer prose is preserved verbatim so the decision is auditable;
        # it is never classified or summarised (that happens in build_summary).
        if "note" in a_record:
            record["a_note"] = a_record["note"]
        if "note" in b_record:
            record["b_note"] = b_record["note"]
        pairs.append(record)
    return pairs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_summary(
    pairs: list[dict[str, Any]],
    cleaned: dict[str, dict[str, Any]],
    key: dict[str, Any],
) -> dict[str, Any]:
    """Neutral aggregate.  No score, no ranking, no combined metric."""
    rows = [
        outcome_row(
            "Overall acceptance",
            [pair["a_accepted"] for pair in pairs],
            [pair["b_accepted"] for pair in pairs],
        )
    ]
    for name in REVIEW_CRITERIA:
        rows.append(
            outcome_row(
                name,
                [pair["a_criteria"][name] for pair in pairs],
                [pair["b_criteria"][name] for pair in pairs],
            )
        )

    arm_of = {
        review_id: entry["arm"] for review_id, entry in key.get("mapping", {}).items()
    }
    with_note = [rid for rid, record in cleaned.items() if "note" in record]
    notes_by_arm = {arm: 0 for arm in ARMS}
    for review_id in with_note:
        arm = arm_of.get(review_id)
        if arm in notes_by_arm:
            notes_by_arm[arm] += 1

    return {
        "analyzer_version": ANALYZER_VERSION,
        "acceptance_rule": "all six frozen criteria pass",
        "reviewers": 1,
        "pairs_analyzed": len(pairs),
        "rows": rows,
        "notes": {
            "records_with_note": len(with_note),
            "by_arm_after_unblinding": notes_by_arm,
            "interpretation": (
                "Counts only. Notes are qualitative reviewer evidence and are "
                "not classified, scored, or summarised."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(
    manifest: dict[str, Any],
    validation: ValidationReport,
    summary: dict[str, Any] | None,
    pairs: list[dict[str, Any]],
    automatic: dict[str, Any],
    hashes: dict[str, str],
    experiment_dir: Path,
    analysis_dir: Path,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("QUERY REPRESENTATION A/B - BLIND REVIEW ANALYSIS")
    add("=" * 72)
    add("")
    add("1. EXPERIMENT IDENTITY")
    add(f"   experiment_id           : {manifest.get('experiment_id')}")
    add(f"   version                 : {manifest.get('version')}")
    add(f"   experiment git SHA      : {manifest.get('code_revision')}")
    add(f"   model                   : {manifest.get('model', {}).get('name')}")
    add(
        f"   criteria source         : "
        f"{manifest.get('review', {}).get('criteria_source')}"
    )
    add(f"   analyzer version        : {ANALYZER_VERSION}")
    add(
        f"   review records          : {validation.valid} / {validation.expected_items}"
    )
    add("")

    add("2. REVIEW COMPLETENESS")
    add(f"   expected reviews        : {validation.expected_items}")
    add(f"   received                : {validation.received}")
    add(f"   valid                   : {validation.valid}")
    add(f"   invalid                 : {validation.invalid}")
    add(f"   missing                 : {validation.missing}")
    add(f"   duplicated              : {validation.duplicated}")
    add(
        f"   status                  : "
        f"{'complete' if validation.complete else 'REVIEW INCOMPLETE'}"
    )
    if validation.missing_review_ids:
        shown = validation.missing_review_ids[:20]
        suffix = " ..." if len(shown) < len(validation.missing_review_ids) else ""
        add(f"   missing review_ids      : {', '.join(shown)}{suffix}")
    if validation.duplicate_review_ids:
        add(
            f"   duplicate review_ids    : {', '.join(validation.duplicate_review_ids)}"
        )
    if validation.unknown_review_ids:
        add(f"   unknown review_ids      : {', '.join(validation.unknown_review_ids)}")
    for error in validation.errors[:20]:
        add(f"   error                   : {error}")
    if len(validation.errors) > 20:
        add(f"   ... {len(validation.errors) - 20} further error(s)")
    add("")

    if summary is None:
        add("No paired analysis is reported: the review is not complete.")
        add("This tool does not impute, drop, or partially analyze reviews.")
        add("")
    else:
        overall = summary["rows"][0]
        add("3. OVERALL ACCEPTANCE")
        add(f"   rule                    : {summary['acceptance_rule']}")
        add(
            f"   A accepted              : {overall['a_pass']}/{overall['a_total']}"
            f"  ({overall['a_rate'] * 100:.1f}%)"
        )
        add(
            f"   B accepted              : {overall['b_pass']}/{overall['b_total']}"
            f"  ({overall['b_rate'] * 100:.1f}%)"
        )
        add(
            f"   A minus B               : "
            f"{overall['a_minus_b_percentage_points']:+.1f} percentage points"
        )
        add(f"   both accepted           : {overall['both']}")
        add(f"   A accepted / B rejected : {overall['a_only']}")
        add(f"   B accepted / A rejected : {overall['b_only']}")
        add(f"   both rejected           : {overall['neither']}")
        add(f"   discordant pairs        : {overall['discordant_pairs']}")
        add(f"   exact McNemar p         : {overall['exact_mcnemar_p']}")
        add("")

        add("4. CRITERION RESULTS")
        add(
            f"   {'criterion':<32}{'A':>9}{'B':>9}{'A-B pp':>9}"
            f"{'both':>6}{'A':>4}{'B':>4}{'none':>6}{'p':>9}"
        )
        add("   " + "-" * 84)
        for row in summary["rows"]:
            a_cell = f"{row['a_pass']}/{row['a_total']}"
            b_cell = f"{row['b_pass']}/{row['b_total']}"
            add(
                f"   {row['outcome']:<32}"
                f"{a_cell:>9}{b_cell:>9}"
                f"{row['a_minus_b_percentage_points']:>+9.1f}"
                f"{row['both']:>6}{row['a_only']:>4}{row['b_only']:>4}"
                f"{row['neither']:>6}{row['exact_mcnemar_p']:>9.4f}"
            )
        add("")
        add("   Columns: A accepted/total, B accepted/total, A minus B in")
        add("   percentage points, both, A only, B only, neither, and the exact")
        add("   two-sided McNemar p-value. Criteria are independent; no combined")
        add("   quality score is computed.")
        add("")

    add("5. REVIEW NOTES")
    if summary is None:
        add("   not analyzed: review incomplete")
    else:
        notes = summary["notes"]
        add(f"   records with a note     : {notes['records_with_note']}")
        for arm in ARMS:
            add(
                f"   after unblinding, {arm}     : "
                f"{notes['by_arm_after_unblinding'][arm]}"
            )
        add(f"   {notes['interpretation']}")
    add("")

    add("6. GENERATION CONTEXT (reported separately from human review)")
    overall_gen = automatic.get("overall", {})
    for arm in ARMS:
        stats = overall_gen.get(arm, {})
        add(
            f"   {f'{arm} generation success':<24}: "
            f"{stats.get('success')}/{stats.get('units')}"
        )
    add(f"   {'pairs generated':<24}: {overall_gen.get('paired', {}).get('pairs')}")
    add(f"   {'pairs reviewed by human':<24}: {len(pairs)} (both arms successful)")
    add("   Generation validity and human-reviewed query quality are reported")
    add("   separately. No combined or weighted metric is produced, and the")
    add("   generation success rate is not multiplied by review acceptance.")
    add("")

    add("7. PROTECTED STATE")
    add(
        f"   frozen inputs hashed    : "
        f"{len([k for k in hashes if k.startswith('frozen/')])}"
    )
    add(
        f"   benchmark artifacts     : "
        f"{len([k for k in hashes if k.startswith('protected/')])}"
    )
    add("   All frozen inputs and benchmark-protection artifacts are unchanged")
    add("   across this analysis. Analysis outputs are written only under the")
    add("   analysis directory.")
    add("")

    add("8. ARTIFACT LOCATIONS")
    add(f"   experiment directory     : {experiment_dir.as_posix()}")
    add(f"   analysis directory      : {analysis_dir.as_posix()}")
    for name in ANALYSIS_OUTPUTS:
        add(f"   analysis output         : {name}")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def clear_analysis_outputs(analysis_dir: Path) -> list[str]:
    """Remove final analysis outputs left by an earlier run.

    A stale ``blind_review_analysis.json`` sitting next to a failed validation
    is exactly the misleading output this tool must never produce.
    """
    removed: list[str] = []
    for name in ANALYSIS_OUTPUTS:
        if name == "review_validation.json":
            continue
        path = analysis_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def run(args: argparse.Namespace) -> int:
    experiment_dir: Path = args.experiment_dir
    analysis_dir: Path = args.analysis_dir
    review_file: Path = args.review_file

    for name in FROZEN_INPUTS:
        if not (experiment_dir / name).exists():
            print(
                f"missing frozen experiment input: {experiment_dir / name}",
                file=sys.stderr,
            )
            return 2

    before = snapshot_hashes(experiment_dir)
    blind_input = read_json(args.input)
    key = read_json(experiment_dir / "blind_review_key.json")
    manifest = read_json(experiment_dir / "manifest.json")
    automatic = read_json(experiment_dir / "automatic_summary.json")

    expected_ids = {item.get("review_id") for item in blind_input.get("items", [])}
    if not review_file.exists():
        removed = clear_analysis_outputs(analysis_dir)
        report = ValidationReport(expected_items=len(expected_ids))
        report.missing = len(expected_ids)
        report.missing_review_ids = sorted(expected_ids)
        report.errors.append(
            f"review results file not found: {review_file}; expected "
            f"{len(expected_ids)} review records"
        )
        write_json(analysis_dir / "review_validation.json", report.to_dict())
        print("REVIEW INCOMPLETE")
        print(f"  expected reviews : {len(expected_ids)}")
        print("  received         : 0")
        print(f"  missing          : {len(expected_ids)}")
        if removed:
            print(f"  removed stale    : {', '.join(removed)}")
        print(f"  wrote            : {analysis_dir / 'review_validation.json'}")
        print("No paired analysis written.")
        return 1

    try:
        payload = read_json(review_file)
    except json.JSONDecodeError as exc:
        print(f"review file is not valid JSON: {exc}", file=sys.stderr)
        return 2

    validation, cleaned = validate_reviews(blind_input, key, manifest, payload)
    write_json(analysis_dir / "review_validation.json", validation.to_dict())

    if not validation.complete:
        removed = clear_analysis_outputs(analysis_dir)
        print("REVIEW INCOMPLETE")
        print(f"  expected reviews : {validation.expected_items}")
        print(f"  received         : {validation.received}")
        print(f"  valid            : {validation.valid}")
        print(f"  missing          : {validation.missing}")
        print(f"  duplicated       : {validation.duplicated}")
        print(f"  invalid          : {validation.invalid}")
        if validation.missing_review_ids:
            shown = validation.missing_review_ids[:20]
            suffix = " ..." if len(shown) < len(validation.missing_review_ids) else ""
            print(f"  missing ids      : {', '.join(shown)}{suffix}")
        for error in validation.errors[:20]:
            print(f"  error            : {error}")
        if len(validation.errors) > 20:
            print(f"  ... {len(validation.errors) - 20} further error(s)")
        if removed:
            print(f"  removed stale    : {', '.join(removed)}")
        print(f"  wrote            : {analysis_dir / 'review_validation.json'}")
        print("No paired analysis written: the review is not complete.")
        return 1

    if args.validate_only:
        print("review complete and valid")
        print(f"  reviews : {validation.valid}")
        print(f"  pairs   : {validation.expected_pairs}")
        print(f"  wrote   : {analysis_dir / 'review_validation.json'}")
        print("validation-only: no analysis written")
        return 0

    generations = {
        (record["code_unit_id"], record["arm"]): record
        for record in read_jsonl(experiment_dir / "generations.jsonl")
    }
    pairs = unblind(key, cleaned, generations)
    summary = build_summary(pairs, cleaned, key)

    write_json(
        analysis_dir / "blind_review_analysis.json",
        {
            "analyzer_version": ANALYZER_VERSION,
            "experiment_id": manifest.get("experiment_id"),
            "version": manifest.get("version"),
            "experiment_code_revision": manifest.get("code_revision"),
            "acceptance_rule": summary["acceptance_rule"],
            "criteria": list(REVIEW_CRITERIA),
            "pairs": pairs,
        },
    )
    write_json(
        analysis_dir / "review_summary.json",
        {
            "analyzer_version": ANALYZER_VERSION,
            "experiment_id": manifest.get("experiment_id"),
            "version": manifest.get("version"),
            "experiment_code_revision": manifest.get("code_revision"),
            "validation": validation.to_dict(),
            "reviewers": summary["reviewers"],
            "review": summary,
            "generation_context": {
                "note": (
                    "Generation validity is reported separately from human "
                    "review; the two are never combined."
                ),
                "overall": automatic.get("overall"),
            },
            "frozen_input_sha256": before,
        },
    )
    write_text(
        analysis_dir / "review_report.txt",
        render_report(
            manifest,
            validation,
            summary,
            pairs,
            automatic,
            before,
            experiment_dir,
            analysis_dir,
        ),
    )

    changed = changed_hashes(before, snapshot_hashes(experiment_dir))
    if changed:
        raise SystemExit(f"frozen inputs changed during analysis: {changed}")

    overall = summary["rows"][0]
    print("review complete and analyzed")
    print(f"  pairs               : {len(pairs)}")
    print(f"  A accepted          : {overall['a_pass']}/{overall['a_total']}")
    print(f"  B accepted          : {overall['b_pass']}/{overall['b_total']}")
    print(
        f"  both / A / B / none : {overall['both']} / {overall['a_only']} / "
        f"{overall['b_only']} / {overall['neither']}"
    )
    print(f"  exact McNemar p     : {overall['exact_mcnemar_p']}")
    print(f"  outputs             : {analysis_dir.as_posix()}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_dir = (
        REPO_ROOT / "artifacts" / "experiments" / "query_representation_ab" / "v1"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Analyze completed blind human-review decisions for the frozen "
            "query-representation A/B experiment. Validates the review, then "
            "unblinds and reports neutral paired counts with exact McNemar "
            "p-values. Computes no score, ranking, or recommendation."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=default_dir,
        help="frozen experiment directory (default: %(default)s)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="review input the review set was drawn from; the expected review "
        "count is derived from it (default: "
        "<experiment-dir>/blind_review_input.json)",
    )
    parser.add_argument(
        "--review-file",
        type=Path,
        default=None,
        help="completed review file (default: "
        "<experiment-dir>/blind_review_results.json)",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="analysis output directory (default: <experiment-dir>/analysis)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate completeness and stop without writing any analysis",
    )
    args = parser.parse_args(argv)
    if args.input is None:
        args.input = args.experiment_dir / "blind_review_input.json"
    if args.review_file is None:
        args.review_file = args.experiment_dir / "blind_review_results.json"
    if args.analysis_dir is None:
        args.analysis_dir = args.experiment_dir / "analysis"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
