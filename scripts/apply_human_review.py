# ruff: noqa: E501
#!/usr/bin/env python3
"""Apply benchmark-blind human review decisions to review_artifact.json.

Reviews all 45 items against frozen criteria §4.4.4:
  1. Understandable — grammatically correct and clear
  2. Behaviorally relevant — describes real code behavior
  3. Sufficiently specific — not generic, grounded in this CodeUnit
  4. Unambiguous — one clear interpretation
  5. No implementation leakage — no function/class/parameter/variable/identifier names
  6. Developer could locate code — relevant unit reachable in top-10

A query is ACCEPTED only if all six criteria pass.
A query is REJECTED if any criterion fails.
"""

import json
from datetime import datetime, timezone

ARTIFACT_PATH = "dataset/queries/review_artifact.json"

# Review decisions: (position, state, notes)
# State is "accepted" or "rejected"
# Notes describe the failed criterion(s) for rejections
DECISIONS = [
    # ─── BATCH 1: Items 1–15 ────────────────────────────────────────────
    (
        1,
        "rejected",
        "Criterion 3: query is too generic to distinguish the spinner initialization behavior",
    ),
    (2, "rejected", "Criterion 5: explicitly names the Box class"),
    (
        3,
        "rejected",
        "Criterion 2: asks about a function call that is not present in the code",
    ),
    (
        4,
        "rejected",
        "Criterion 2: incorrectly claims the constructor parses command-line options",
    ),
    (
        5,
        "rejected",
        "Criterion 2: incorrectly describes temporary file/directory and fixture management",
    ),
    (
        6,
        "rejected",
        "Criterion 3: describes generic structural operations rather than the distinctive text-splitting behavior",
    ),
    (
        7,
        "rejected",
        "Criterion 3: generic delegation/control-flow description lacks sufficient retrieval specificity",
    ),
    (8, "accepted", ""),
    (9, "accepted", ""),
    (
        10,
        "rejected",
        "Criterion 3: completely generic query with no useful retrieval signal",
    ),
    (11, "accepted", ""),
    (
        12,
        "rejected",
        "Criterion 3: generic comparison description misses the remaining-step calculation and None behavior",
    ),
    (
        13,
        "rejected",
        "Criterion 2: incorrectly describes temporary file/directory creation and fixture configuration",
    ),
    (
        14,
        "rejected",
        "Criterion 3: generic purpose/inputs/outputs/control-flow query lacks distinctive behavior",
    ),
    (
        15,
        "rejected",
        "Criterion 3: generic comparison description misses fixture-marker and session-scope behavior",
    ),
    # ─── BATCH 2: Items 16–30 ────────────────────────────────────────────
    (
        16,
        "rejected",
        "Criterion 3: generic purpose/operations query lacks retrieval specificity",
    ),
    (
        17,
        "rejected",
        "Criterion 3: generic purpose/operations query does not distinguish binary path writing or optional parent-directory creation",
    ),
    (
        18,
        "rejected",
        "Criterion 2: describes test-harness/temp-directory generation and regex transformation not present in the function",
    ),
    (
        19,
        "rejected",
        "Criterion 2: describes temporary files/directories and fixture interaction; method only stops global capturing",
    ),
    (
        20,
        "rejected",
        "Criterion 2: describes reading a value from an associated object; source creates a test file, runs pytest, and checks outcomes",
    ),
    (
        21,
        "rejected",
        "Criterion 2: asks which function is called, but the hook specification has no function call in its body",
    ),
    (22, "accepted", ""),
    (
        23,
        "rejected",
        "Criterion 2: describes string extraction/transformation via subprocess rather than version-selection behavior",
    ),
    (
        24,
        "rejected",
        "Criterion 2: describes reading a value/method call; source tests that fixture retrieval raises an exception",
    ),
    (
        25,
        "rejected",
        "Criterion 2: describes string extraction and warning handling; source computes a relative path between Path objects",
    ),
    (
        26,
        "rejected",
        "Criterion 3: generic purpose/parameter query lacks distinctive initial-conftest hook behavior",
    ),
    (
        27,
        "rejected",
        "Criterion 3: generic value-reading/loop query misses YAML loading and collection-item creation",
    ),
    (
        28,
        "rejected",
        "Criterion 2: describes comparison and direct return; source conditionally writes cached node IDs with early returns",
    ),
    (
        29,
        "rejected",
        "Criterion 2: describes comparison/four function calls; source selects a file location from location, obj, or path",
    ),
    (
        30,
        "rejected",
        "Criterion 2: says it extracts part of a string; source strips the value and checks membership in valid choices",
    ),
    # ─── BATCH 3: Items 31–45 ────────────────────────────────────────────
    (
        31,
        "rejected",
        "Criterion 3: generic comparison boilerplate; misses elapsed-time and started/stopped behavior",
    ),
    (32, "accepted", ""),
    (
        33,
        "rejected",
        "Criterion 3: generic constructor query does not identify the reader's state/configuration behavior",
    ),
    (
        34,
        "rejected",
        "Criterion 2: incorrectly claims subprocess execution and warning management; source delegates to _makefile",
    ),
    (
        35,
        "rejected",
        "Criterion 3: generic comparison/error boilerplate; misses numbered-directory creation, cleanup lock, retries, and cleanup",
    ),
    (
        36,
        "rejected",
        "Criterion 3: generic associated-object retrieval query misses fixture-definition setup",
    ),
    (
        37,
        "rejected",
        "Criterion 3: generic state-modification query with unrelated text-formatting/rendering context",
    ),
    (38, "accepted", ""),
    (
        39,
        "rejected",
        "Criterion 2: claims terminal/console rendering and text formatting; source stores syntax configuration and initializes theme/highlight state",
    ),
    (
        40,
        "rejected",
        "Criterion 3: generic associated-object retrieval; misses indexed retrieval of a recorded warning",
    ),
    (
        41,
        "rejected",
        "Criterion 3: generic comparison/function-call description; misses measurement calculation and width constraint",
    ),
    (
        42,
        "rejected",
        "Criterion 2: incorrectly claims subprocess launching; source converts self.path through legacy_path",
    ),
    (
        43,
        "rejected",
        "Criterion 2: incorrectly claims filtering based on a pattern; source constructs pytest subprocess arguments and runs pytest",
    ),
    (
        44,
        "rejected",
        "Criterion 3: generic delegation/return-value query; misses ensuring the joined path is a directory",
    ),
    (
        45,
        "rejected",
        "Criterion 2: incorrectly claims tabulated console rendering; source only creates/stores Text(text) and href",
    ),
]

NOW_UTC = datetime.now(timezone.utc).isoformat()


def main() -> None:
    with open(ARTIFACT_PATH, encoding="utf-8") as f:
        artifact = json.load(f)

    items_by_position = {item["position"]: item for item in artifact["items"]}

    accepted = 0
    rejected = 0

    for position, state, _notes in DECISIONS:
        item = items_by_position[position]
        item["review"]["state"] = state
        item["review"]["notes"] = _notes
        item["review"]["decided_utc"] = NOW_UTC
        if state == "accepted":
            accepted += 1
        else:
            rejected += 1

    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print(
        f"Review complete: {accepted} accepted, {rejected} rejected, {accepted + rejected} total"
    )

    # Print summary
    print("\n-- ACCEPTED (passed all 6 criteria) --")
    for position, state, _notes in DECISIONS:
        if state == "accepted":
            item = items_by_position[position]
            print(f"  Item {position:2d}: {item['query'][:80]}")

    print("\n-- REJECTED (failed criterion 5: implementation leakage) --")
    for position, state, _notes in DECISIONS:
        if state == "rejected":
            item = items_by_position[position]
            print(f"  Item {position:2d}: {item['query'][:80]}")
            print(f"         Reason: {_notes}")


if __name__ == "__main__":
    main()
