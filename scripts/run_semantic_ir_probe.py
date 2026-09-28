"""Read-only comparison of StructuredBehaviorFacts against the Semantic IR.

Answers one question with evidence: for real code units in this repository's
own dataset, does the Tree-sitter Semantic IR carry more retrievable
behavioral signal than the AST-based facts the pipeline uses today?

Development-only and read-only.  It opens the split JSONL files, builds both
representations, prints a report, and writes nothing.  It never touches the
candidate pool, the query generator, or any review artifact.

The splits carry no module-level ``imports`` list (that lives with the file,
not the extracted unit), so both sides are called with ``imports=None``.
That is symmetric, which is what matters for a comparison, but it does mean
the import-grounded domain signals are absent on both sides.

Usage:
    python scripts/run_semantic_ir_probe.py --sample 60
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localbench.workloads.code_retrieval.behavior_extraction import (  # noqa: E402
    extract_behavior_facts,
)
from localbench.workloads.code_retrieval.semantic_ir_experiment import (  # noqa: E402
    assert_identifier_free,
    build_semantic_ir,
    informativeness,
    rename_identifiers,
)

SPLITS = ("train", "validation")
#: placeholders both extractors emit when the source did not parse
_EMPTY_MARKERS = ("unknown", "parses as invalid code", "no ")
#: how many side-by-side examples to print in full
_SHOW = 3


def load_sample(split: str, limit: int) -> list[dict]:
    """Every ``limit``-th unit of a split, in id order, so runs are stable."""
    path = REPO_ROOT / "dataset" / "splits" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: row["id"])
    if not rows:
        return []
    stride = max(1, len(rows) // limit)
    return rows[::stride][:limit]


def facts_signal(facts) -> int:
    """How many of the old facts carry something other than a placeholder."""
    values = [
        facts.primary_purpose, facts.input_summary, facts.output_summary,
        facts.domain_concepts, facts.observable_effects, facts.side_effects,
        facts.error_handling, facts.control_flow, facts.key_operations,
        facts.raises,
    ]
    count = 0
    for value in values:
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            text = str(item).lower()
            if text and not any(marker in text for marker in _EMPTY_MARKERS):
                count += 1
    return count


def describe(unit: dict) -> dict:
    """Build both representations for one unit.  Pure read."""
    source = unit["source_code"]
    language = unit.get("language", "python")
    old = extract_behavior_facts(source, imports=None)
    new = build_semantic_ir(source, language, unit_kind=unit.get("symbol_type"))
    return {
        "unit": unit,
        "old": old,
        "new": new,
        "old_signal": facts_signal(old),
        "anchors": informativeness(new),
    }


def render_facts(facts) -> str:
    lines = [
        f"primary_purpose    {facts.primary_purpose}",
        f"input_summary      {facts.input_summary}",
        f"output_summary     {facts.output_summary}",
        f"key_operations     {list(facts.key_operations)}",
        f"domain_concepts    {list(facts.domain_concepts)}",
        f"control_flow       {facts.control_flow}",
        f"side_effects       {list(facts.side_effects)}",
        f"observable_effects {list(facts.observable_effects)}",
        f"error_handling     {facts.error_handling}",
        f"raises             {list(facts.raises)}",
    ]
    return "\n".join("  " + line for line in lines)


def render_ir(ir) -> str:
    fields = [
        ("input_roles", list(ir.input_roles)),
        ("output_behavior", ir.output_behavior),
        ("domain_signals", list(ir.domain_signals)),
        ("transformations", list(ir.transformations)),
        ("relations", list(ir.relations)),
        ("conditions", list(ir.conditions)),
        ("state_changes", list(ir.state_changes)),
        ("side_effects", list(ir.side_effects)),
        ("observable_effects", list(ir.observable_effects)),
        ("error_behavior", ir.error_behavior),
    ]
    return "\n".join(f"  {name:<20} {value}" for name, value in fields)


def report(records: list[dict], label: str) -> dict:
    totals = {
        "units": len(records),
        "parsed": sum(1 for r in records if r["new"].parse_ok),
        "old_mean_signal": statistics.fmean(r["old_signal"] for r in records),
        "anchor_mean": statistics.fmean(r["anchors"].anchor_count for r in records),
        "with_relations": sum(1 for r in records if r["new"].relations),
        "old_with_relations": 0,
    }
    for record in records:
        old = record["old"]
        linked = any(
            old.input_summary not in ("unknown", "") and old.output_summary
            not in ("unknown", "")
            for _ in (0,)
        )
        totals["old_with_relations"] += int(linked)

    print(f"\n=== {label} " + "=" * (60 - len(label)))
    print(f"  units sampled                {totals['units']}")
    print(f"  Semantic IR parsed cleanly   {totals['parsed']}")
    print(f"  units with an input->output  {totals['with_relations']}  "
          f"({_pct(totals['with_relations'], totals['units'])})")
    print(f"  old facts with both summaries {totals['old_with_relations']}  "
          f"({_pct(totals['old_with_relations'], totals['units'])})")
    print(f"  mean old non-placeholder fact count  "
          f"{totals['old_mean_signal']:.2f}")
    print(f"  mean Semantic IR anchor count        "
          f"{totals['anchor_mean']:.2f}")

    breakdown_totals: dict[str, int] = {}
    for record in records:
        for key, value in record["anchors"].breakdown.items():
            breakdown_totals[key] = breakdown_totals.get(key, 0) + value
    print("  anchors by field (total across units):")
    for key, value in sorted(breakdown_totals.items(), key=lambda kv: -kv[1]):
        print(f"    {key:<20} {value}")
    return totals


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole else "n/a"


def show_examples(records: list[dict], count: int) -> None:
    ranked = sorted(records, key=lambda r: -r["anchors"].anchor_count)
    for record in ranked[:count]:
        unit = record["unit"]
        print("\n" + "-" * 72)
        print(f"  {unit['repository']}  {unit['file_path']}  "
              f"({unit['symbol_type']})")
        print(f"  id: {unit['id']}")
        print(f"  source is {len(unit['source_code'])} chars; "
              f"the symbol's own name never reaches either representation")
        print("\n  StructuredBehaviorFacts (production today):")
        print(render_facts(record["old"]))
        print("\n  Semantic IR (pilot):")
        print(render_ir(record["new"]))
        print(f"\n  anchors: {record['anchors'].anchor_count}  "
              f"old non-placeholder facts: {record['old_signal']}")


def describe_difference(record: dict) -> str:
    """Field-by-field before/after for one unit that failed the guard.

    Naming the field and both values is the difference between a count and a
    diagnosis: the offending classifier is usually obvious once you can see
    which field moved and which way.
    """
    unit = record["unit"]
    language = unit.get("language", "python")
    original = record["new"]
    renamed = build_semantic_ir(
        rename_identifiers(unit["source_code"], language), language,
        original.unit_kind,
    )
    before, after = original.to_dict(), renamed.to_dict()
    lines = [
        f"code_unit_id : {unit['id']}",
        f"language     : {language}",
        f"repository   : {unit['repository']}  {unit['file_path']}",
    ]
    for field in before:
        if before[field] == after[field]:
            continue
        lines.append(f"  field: {field}")
        lines.append(f"    original: {before[field]}")
        lines.append(f"    renamed : {after[field]}")
    return "\n".join(lines)


def verify_identifiers(records: list[dict]) -> tuple[int, list[str]]:
    """Run the provenance guard over real units, not just fixtures."""
    checked = 0
    failures: list[str] = []
    for record in records:
        unit = record["unit"]
        try:
            assert_identifier_free(
                record["new"], unit["source_code"], unit.get("language", "python")
            )
        except AssertionError:
            failures.append(describe_difference(record))
        else:
            checked += 1
    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=int, default=60,
        help="units per split to examine (default: 60)",
    )
    parser.add_argument(
        "--show", type=int, default=_SHOW,
        help="full side-by-side examples to print (default: 3)",
    )
    args = parser.parse_args()

    all_records: list[dict] = []
    for split in SPLITS:
        records = [describe(unit) for unit in load_sample(split, args.sample)]
        report(records, split)
        all_records.extend(records)

    show_examples(all_records, args.show)

    checked, failures = verify_identifiers(all_records)
    print("\n=== provenance guard " + "=" * 45)
    print(f"  units where the IR is byte-identical under a full rename: "
          f"{checked}/{len(all_records)}")
    if failures:
        # Reported, not fatal: ~1% residual is the documented steady state, and
        # a probe that always exits non-zero teaches people to ignore it.
        print(f"  {len(failures)} unit(s) still change under rename. Each is a "
              "classifier heuristic reading source text instead of tree")
        print("  structure; see the ponytail: note on "
              "semantic_ir_experiment.assert_identifier_free.")
        for failure in failures:
            print("\n  " + failure.replace("\n", "\n  "))
    print("\n  no dataset files, review artifacts, or production modules were "
          "written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
