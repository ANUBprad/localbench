"""Development-only interactive CLI for the blind query A/B human review.

The frozen query-representation experiment produced a blinded review set.  This
tool is the reviewer-facing half: it shows the queries in the exact order the
experiment randomized them, collects the six frozen criterion decisions, and
saves each completed item atomically so an interrupted session loses at most the
item currently in hand.

It is deliberately incapable of unblinding.  The only artifact it reads is
``blind_review_input.json``; the review key, the generations, the automatic
summary, the dataset, and the source code are never opened.  No analysis, no
paired statistics, no p-values, no model calls: those belong exclusively to
``scripts/analyze_query_representation_ab_review.py``.

Acceptance is *derived* from the six criteria using the frozen rule ("A query is
accepted only if all criteria pass") rather than asked, so the reviewer cannot
create a decision that contradicts the criteria.  The derived value is written
explicitly and the analyzer re-checks it.

Schema compatibility is not a second opinion: the criteria, the acceptance rule,
and the results-file reader are imported from the analyzer, so the file this
tool writes is by construction the file the analyzer expects.

Usage:
    python scripts/run_blind_query_ab_review.py --help
    python scripts/run_blind_query_ab_review.py
    python scripts/run_blind_query_ab_review.py --restart
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_DIR = (
    REPO_ROOT / "artifacts" / "experiments" / "query_representation_ab" / "v1"
)

#: Artifacts this tool must never open, whatever the arguments say.  Kept as a
#: constant so the test can assert the source never references them.
FORBIDDEN_ARTIFACTS: tuple[str, ...] = (
    "blind_review_key.json",
    "generations.jsonl",
    "automatic_summary.json",
    "sample.jsonl",
    "manifest.json",
)


@lru_cache(maxsize=1)
def _analyzer() -> Any:
    """Import the analysis script for its schema constants and helpers.

    Sharing the criteria list, the acceptance rule, and the results reader is
    what guarantees this tool and the analyzer cannot drift into two formats.
    Importing the module has no side effects: it defines functions and reads no
    artifacts at import time.
    """
    path = REPO_ROOT / "scripts" / "analyze_query_representation_ab_review.py"
    if not path.exists():
        raise SystemExit(f"cannot find the review analyzer at {path}")
    spec = importlib.util.spec_from_file_location(
        "analyze_query_representation_ab_review", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REVIEW_CRITERIA: tuple[str, ...] = _analyzer().REVIEW_CRITERIA
EXPECTED_REVIEW_ITEMS: int = _analyzer().EXPECTED_REVIEW_ITEMS
EXPECTED_PAIRS: int = _analyzer().EXPECTED_PAIRS
ALLOWED_REVIEW_KEYS = _analyzer().ALLOWED_REVIEW_KEYS
BLIND_FORBIDDEN_KEYS = _analyzer().BLIND_FORBIDDEN_KEYS
derive_accepted = _analyzer().derive_accepted
_extract_records = _analyzer()._extract_records


class ReviewDataError(RuntimeError):
    """The blind input or an existing results file is malformed."""


# ---------------------------------------------------------------------------
# Blind input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlindInput:
    """The validated, reviewer-facing view of the blind review set."""

    experiment_id: str
    version: str
    criteria: tuple[tuple[str, str], ...]
    items: tuple[dict[str, Any], ...]

    @property
    def review_ids(self) -> list[str]:
        return [item["review_id"] for item in self.items]

    def item(self, review_id: str) -> dict[str, Any]:
        for candidate in self.items:
            if candidate["review_id"] == review_id:
                return candidate
        raise KeyError(review_id)


def load_blind_input(path: Path) -> BlindInput:
    """Load and structurally validate the blind review input.

    Fails closed.  The check is structural only: item identity, pairing, and
    criteria.  Nothing here consults the arm mapping, because this tool never
    reads it.
    """
    if not path.exists():
        raise ReviewDataError(f"blind review input not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewDataError(f"blind review input is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ReviewDataError("blind review input must be a JSON object")

    criteria = data.get("criteria")
    if not isinstance(criteria, list):
        raise ReviewDataError("blind review input has no 'criteria' array")
    names = tuple(
        entry.get("name") if isinstance(entry, dict) else None for entry in criteria
    )
    if names != REVIEW_CRITERIA:
        raise ReviewDataError(
            f"blind review input criteria do not match the frozen six: {list(names)}"
        )
    for entry in criteria:
        if not isinstance(entry.get("definition"), str):
            raise ReviewDataError(f"criterion {entry.get('name')!r} has no definition")

    items = data.get("items")
    if not isinstance(items, list):
        raise ReviewDataError("blind review input has no 'items' array")
    if len(items) != EXPECTED_REVIEW_ITEMS:
        raise ReviewDataError(
            f"expected {EXPECTED_REVIEW_ITEMS} review items, found {len(items)}"
        )

    seen: set[str] = set()
    per_pair: dict[str, int] = {}
    for index, item in enumerate(items):
        where = f"item {index}"
        if not isinstance(item, dict):
            raise ReviewDataError(f"{where}: not an object")
        extra = sorted(set(item) - {"review_id", "pair_id", "query"})
        if extra:
            # A reviewer item must carry only its own text; anything else
            # could reveal the arm.
            raise ReviewDataError(f"{where}: unexpected field(s) {extra}")
        review_id = item.get("review_id")
        if not isinstance(review_id, str) or not review_id:
            raise ReviewDataError(f"{where}: review_id must be a non-empty string")
        if review_id in seen:
            raise ReviewDataError(f"duplicate review_id in input: {review_id}")
        seen.add(review_id)
        pair_id = item.get("pair_id")
        if (
            not isinstance(pair_id, str)
            or not pair_id.startswith("p")
            or not pair_id[1:].isdigit()
        ):
            raise ReviewDataError(f"{review_id}: invalid pair_id {pair_id!r}")
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ReviewDataError(f"{review_id}: query must be a non-empty string")
        per_pair[pair_id] = per_pair.get(pair_id, 0) + 1

    if len(per_pair) != EXPECTED_PAIRS:
        raise ReviewDataError(f"expected {EXPECTED_PAIRS} pairs, found {len(per_pair)}")
    unpaired = sorted(p for p, count in per_pair.items() if count != 2)
    if unpaired:
        raise ReviewDataError(f"pairs without exactly two items: {unpaired}")

    return BlindInput(
        experiment_id=str(data.get("experiment_id", "")),
        version=str(data.get("version", "")),
        criteria=tuple((c["name"], c["definition"]) for c in criteria),
        items=tuple(dict(item) for item in items),
    )


# ---------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------


def validate_record(record: Any, blind: BlindInput) -> dict[str, Any]:
    """Validate one completed result record against the frozen input.

    Deliberately silent about repair: a malformed record is an error, not
    something to quietly fix, because rewriting a reviewer's decision without
    telling them is worse than stopping.
    """
    if not isinstance(record, dict):
        raise ReviewDataError(f"result record is not an object: {record!r}")
    review_id = record.get("review_id")
    if not isinstance(review_id, str):
        raise ReviewDataError(f"result record has no review_id: {record!r}")
    try:
        item = blind.item(review_id)
    except KeyError:
        raise ReviewDataError(
            f"{review_id}: not present in the blind review input"
        ) from None

    forbidden = sorted(set(record) & BLIND_FORBIDDEN_KEYS)
    if forbidden:
        raise ReviewDataError(
            f"{review_id}: result carries unblinding field(s) {forbidden}"
        )
    unexpected = sorted(set(record) - ALLOWED_REVIEW_KEYS)
    if unexpected:
        raise ReviewDataError(f"{review_id}: unexpected field(s) {unexpected}")
    if record.get("pair_id") != item["pair_id"]:
        raise ReviewDataError(
            f"{review_id}: pair_id {record.get('pair_id')!r} does not match the "
            f"blind input ({item['pair_id']!r})"
        )

    criteria = record.get("criteria")
    if not isinstance(criteria, dict):
        raise ReviewDataError(f"{review_id}: 'criteria' must be an object")
    missing = [name for name in REVIEW_CRITERIA if name not in criteria]
    unknown = sorted(set(criteria) - set(REVIEW_CRITERIA))
    if missing:
        raise ReviewDataError(f"{review_id}: missing criterion {missing}")
    if unknown:
        raise ReviewDataError(f"{review_id}: unknown criterion {unknown}")
    for name in REVIEW_CRITERIA:
        # bool is a subclass of int, so the type must be exact: 1 and "true"
        # are not decisions.
        if not isinstance(criteria[name], bool):
            raise ReviewDataError(
                f"{review_id}: criterion {name!r} must be a boolean, got "
                f"{type(criteria[name]).__name__}"
            )

    accepted = record.get("accepted")
    if accepted is not None:
        if not isinstance(accepted, bool):
            raise ReviewDataError(f"{review_id}: 'accepted' must be a boolean")
        if accepted != derive_accepted(criteria):
            raise ReviewDataError(
                f"{review_id}: accepted={accepted} contradicts the frozen rule "
                f"(all criteria pass), which gives {derive_accepted(criteria)}"
            )

    note = record.get("note")
    if note is not None and not isinstance(note, str):
        raise ReviewDataError(f"{review_id}: 'note' must be a string")
    return record


def load_results(path: Path, blind: BlindInput) -> list[dict[str, Any]]:
    """Load completed reviews, failing closed on anything malformed.

    A missing file is simply an unstarted review, not an error.
    """
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewDataError(
            f"results file is not valid JSON ({path}); refusing to continue "
            f"rather than overwriting reviewer work: {exc}"
        ) from exc
    try:
        records = _extract_records(payload)
    except ValueError as exc:
        raise ReviewDataError(f"results file is malformed: {exc}") from exc

    completed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        validated = validate_record(record, blind)
        review_id = validated["review_id"]
        if review_id in seen:
            raise ReviewDataError(f"duplicate review_id in results: {review_id}")
        seen.add(review_id)
        completed.append(validated)
    return completed


def atomic_write_results(
    path: Path,
    blind: BlindInput,
    records: Sequence[dict[str, Any]],
) -> None:
    """Write the results file via temp file + atomic replace.

    Same shape as the existing ``_atomic_write_json`` in
    ``scripts/generate_query_candidates.py``: a torn write can never leave a
    half-written results file that would discard the reviewer's work.  Records
    keep the blind input's order so the file stays diff-stable across resumes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "experiment_id": blind.experiment_id,
        "version": blind.version,
        "reviews": list(records),
    }
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def build_record(
    item: dict[str, Any], answers: dict[str, bool], note: str
) -> dict[str, Any]:
    """One completed item, in exactly the shape the analyzer expects."""
    record: dict[str, Any] = {
        "review_id": item["review_id"],
        "pair_id": item["pair_id"],
        "criteria": {name: answers[name] for name in REVIEW_CRITERIA},
    }
    # Derived, never asked: the frozen rule is the only source of truth.
    record["accepted"] = derive_accepted(record["criteria"])
    if note.strip():
        record["note"] = note
    return record


def first_missing_index(blind: BlindInput, completed: Sequence[dict[str, Any]]) -> int:
    """Index of the first unanswered item, in the frozen input order."""
    done = {record["review_id"] for record in completed}
    for index, review_id in enumerate(blind.review_ids):
        if review_id not in done:
            return index
    return len(blind.items)


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

Ask = Callable[[str], str]


def _show_item(item: dict[str, Any], position: int, total: int) -> None:
    print()
    print("=" * 42)
    print(f"Blind Query Review {position} / {total}")
    print("=" * 42)
    print()
    print(f"Review ID: {item['review_id']}")
    print(f"Pair ID: {item['pair_id']}")
    print()
    print("Query:")
    for line in item["query"].splitlines() or [""]:
        print(f"  {line}")


def _collect(
    item: dict[str, Any],
    criteria: tuple[tuple[str, str], ...],
    ask: Ask,
    position: int,
    total: int,
) -> tuple[dict[str, bool], str] | None:
    """Ask the six criteria plus an optional note for one item.

    Returns ``None`` when the reviewer quits, discarding the unfinished item.
    ``b`` restarts the item so the answers can be corrected before saving.
    """
    while True:
        _show_item(item, position, total)
        answers: dict[str, bool] = {}
        restart = False
        for index, (name, definition) in enumerate(criteria, start=1):
            print()
            print(f"  {index}. {name}")
            print(f"     {definition}")
            while True:
                raw = ask("     [y/n] (q quits): ").strip().lower()
                if raw == "q":
                    return None
                if raw in ("y", "n"):
                    answers[name] = raw == "y"
                    break
                if raw == "b":
                    restart = True
                    break
                print("     Please answer y or n.")
            if restart:
                break

        if restart:
            print()
            print("  Restarting this item.")
            continue

        print()
        note = ask("  Optional note (press Enter to skip): ").strip()
        print()
        action = ask("  [Enter] save   [b] change answers   [q] quit: ").strip().lower()
        if action == "b":
            print()
            print("  Changing the answers for this item.")
            continue
        if action == "q":
            return None
        return answers, note


def _print_progress(done: int, total: int) -> None:
    print(f"Completed: {done} / {total}")
    print(f"Remaining: {total - done}")


def review_session(
    blind: BlindInput,
    results_path: Path,
    ask: Ask,
    *,
    restart: bool = False,
) -> int:
    """Drive the review from the first missing item to the end.

    Prior work is preserved unless ``restart`` is set, and even then only after
    an explicit typed confirmation.
    """
    total = len(blind.items)
    records = load_results(results_path, blind)

    if restart and records:
        answer = (
            ask(
                f"This discards {len(records)} completed review(s). Type 'yes' to "
                f"confirm: "
            )
            .strip()
            .lower()
        )
        if answer != "yes":
            print("Restart cancelled. Existing reviews kept.")
            return 1
        records = []
        atomic_write_results(results_path, blind, records)

    if not records:
        print("No completed reviews found. Starting from the first item.")

    start = first_missing_index(blind, records)
    _print_progress(len(records), total)
    if start >= total:
        print()
        print("All reviews are already complete.")
        return 0

    position = start
    while position < total:
        item = blind.items[position]
        collected = _collect(item, blind.criteria, ask, position + 1, total)
        if collected is None:
            print()
            print("Quit. Completed reviews are saved.")
            break
        answers, note = collected
        record = build_record(item, answers, note)
        validate_record(record, blind)
        records.append(record)
        # Save immediately: an interrupted terminal loses at most this item.
        atomic_write_results(results_path, blind, records)
        position += 1
        print("  Saved.")
        _print_progress(len(records), total)

    # Final fail-closed check on what is actually on disk.
    final = load_results(results_path, blind)
    print()
    if len(final) == total:
        print(f"All {total} reviews complete. Results written to {results_path}")
        return 0
    print(f"Progress saved to {results_path}")
    _print_progress(len(final), total)
    print("Run this command again to resume where you left off.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive reviewer for the frozen query-representation A/B blind "
            "review. Shows the 120 blinded queries in the frozen order, records "
            "the six criterion decisions, and saves after every item. Reads only "
            "blind_review_input.json: it never unblinds, never analyses, and "
            "never calls a model."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "blind_review_input.json",
        help="blinded review input (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "blind_review_results.json",
        help="results file, resumed if it exists (default: %(default)s)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard existing results (requires typing 'yes' to confirm)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the input and any existing results, then exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, ask: Ask | None = None) -> int:
    args = parse_args(argv)
    ask = ask or (lambda prompt: input(prompt))
    try:
        blind = load_blind_input(args.input)
        records = load_results(args.output, blind)
    except ReviewDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Experiment: {blind.experiment_id} {blind.version}")
    print(f"Items: {len(blind.items)}  Criteria: {len(blind.criteria)}")
    print("Answer y if the query satisfies the criterion, n if it does not.")
    if args.validate_only:
        _print_progress(len(records), len(blind.items))
        print("Validation only. Nothing written.")
        return 0
    return review_session(blind, args.output, ask, restart=args.restart)


if __name__ == "__main__":
    raise SystemExit(main())
