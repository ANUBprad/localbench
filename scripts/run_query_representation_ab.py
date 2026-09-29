"""Development-only A/B: StructuredBehaviorFacts vs Semantic IR for query generation.

Compares exactly one variable — the code-unit representation handed to the
query generator — while holding everything else fixed: the same sampled
CodeUnits, the same shared system instruction and task wording, the same model
and sampling parameters, the same bounded retry policy, and the same validation
chain.  Arm A uses the production ``extract_behavior_facts`` output; arm B uses
the development Semantic IR after the existing ``assert_identifier_free`` guard.

Scope limits, all deliberate:

* Development-only.  This script is never imported by the production query
  pipeline and is not wired into any dataset, benchmark, or selection step.
* It writes only under an ignored experiment root.  It never opens the
  candidate pool, the review artifact, or the 45-selection for writing.
* It contains no automatic winner logic.  It reports per-arm and paired
  counts, and stops there.  Judgement is left to the frozen blind review.

One implementation note that is a real deviation worth knowing about: the
provider-agnostic ``GenerationRequest`` has no system-message field, and the
Ollama adapter posts only ``prompt``.  The production pipeline therefore never
transmits the shared system instruction.  To keep the arms identical *and*
reuse that instruction, the dev prompt folds ``get_query_system_prompt()`` into
the single prompt string for both arms.  Production behaviour is unchanged.

Usage:
    python scripts/run_query_representation_ab.py --sample
    python scripts/run_query_representation_ab.py --train-sample 5 \\
        --validation-sample 5
    python scripts/run_query_representation_ab.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localbench.runtime.generation.executor import (  # noqa: E402
    RetryResult,
    run_with_retry,
)
from localbench.runtime.generation.policy import RetryPolicy  # noqa: E402
from localbench.runtime.model import (  # noqa: E402
    GenerationRequest,
    LocalModel,
)
from localbench.runtime.ollama.adapter import OllamaAdapter  # noqa: E402
from localbench.workloads.code_retrieval.behavior_extraction import (  # noqa: E402
    extract_behavior_facts,
)
from localbench.workloads.code_retrieval.extraction import (  # noqa: E402
    ExtractedCodeUnit,
)
from localbench.workloads.code_retrieval.query_generator import (  # noqa: E402
    check_meta_query,
    check_query_leakage,
    check_query_provenance,
)
from localbench.workloads.code_retrieval.query_prompt import (  # noqa: E402
    get_query_system_prompt,
)
from localbench.workloads.code_retrieval.schemas import (  # noqa: E402
    CandidateQuery,
    CodeUnitContext,
)
from localbench.workloads.code_retrieval.semantic_ir_experiment import (  # noqa: E402
    assert_identifier_free,
    build_semantic_ir,
    informativeness,
)

# ---------------------------------------------------------------------------
# Frozen experiment constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "query_representation_ab"
EXPERIMENT_VERSION = "v1"
PROMPT_TEMPLATE_VERSION = "ab-dev-1.0.0"

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_SEED = 42
DEFAULT_TRAIN_SAMPLE = 100
DEFAULT_VALIDATION_SAMPLE = 100

#: Mirrors the production generator's hard-coded sampling parameters
#: (``QueryGenerator._make_generate_fn``) so the arms match the pipeline.
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 128

#: The two arms, in fixed order.  Order matters for the seeded blind review.
ARMS: tuple[str, str] = ("A", "B")

#: Frozen review criteria, copied verbatim from DATASET_SPECIFICATION.md
#: section 4.4.4.  Kept here because the package exposes no criteria constant
#: and the review script that does is not part of the importable package.
REVIEW_CRITERIA: tuple[str, ...] = (
    "Understandable",
    "Behaviorally relevant",
    "Sufficiently specific",
    "Unambiguous",
    "No implementation leakage",
    "Developer could locate the code",
)

#: Number of pairs handed to the blind reviewer (two queries per pair).
BLIND_REVIEW_PAIRS = 60

SPLITS: tuple[str, str] = ("train", "validation")

#: Every artifact this experiment writes, under the ignored experiment root.
ARTIFACT_NAMES: tuple[str, ...] = (
    "manifest.json",
    "sample.jsonl",
    "generations.jsonl",
    "automatic_summary.json",
    "blind_review_input.json",
    "blind_review_key.json",
)

#: Paths that must never be written by this experiment.  Asserted in tests.
PROTECTED_ARTIFACTS: tuple[str, ...] = (
    "dataset/queries/final_45_selection.json",
    "dataset/queries/review_artifact.json",
    "dataset/queries/quarantine_overbudget.json",
    "dataset/queries/candidates_v3.jsonl",
    "dataset/queries/candidate_failures_v3.jsonl",
    "dataset/queries/candidates.jsonl",
    "dataset/queries/candidate_failures.jsonl",
)

#: Keys that must never appear in the blinded review input.  A review item
#: carries only its own text, so any of these would reveal the arm.
BLIND_FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "arm",
    "arm_label",
    "representation",
    "variant",
    "group",
    "code_unit_id",
    "split",
    "symbol",
    "file_path",
    "repository",
    "symbol_type",
    "is_public",
    "query_style",
    "query_intent",
})

#: Whether either extractor receives the file's ``context.imports``.
#:
#: False, and not by accident.  ``assert_identifier_free`` rebuilds the renamed
#: IR internally *without* imports, so an IR built with imports cannot be
#: compared against its own rename and the guard fails spuriously on any
#: import-grounded domain signal.  Since arm B must pass that guard, B has to
#: be built without imports; leaving imports on for A alone would then hand A
#: extra grounded domain concepts that B cannot match, which is exactly the
#: kind of one-sided head start this experiment must not create.  Both arms
#: therefore drop imports, and the cost -- absent import-grounded domain
#: signals on both sides -- is recorded in the manifest.
USE_IMPORTS = False

#: The dev prompt forbids these outright.  A prompt that contains any of them
#: would leak implementation detail the retrieval task is meant to avoid.
#: Payload field labels that must never appear in a rendered prompt.  Matched
#: in ``label:`` form so the check targets the payload, not the task wording.
_FORBIDDEN_PROMPT_TOKENS: tuple[str, ...] = (
    "source_code:",
    "docstring:",
    "file_path:",
    "repository:",
    "symbol:",
    "class_name:",
    "parent_methods:",
    "module_docstring:",
    "code_unit_id:",
    "source_url:",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str:
    """Current commit, or ``unknown`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Loading and sampling
# ---------------------------------------------------------------------------


def load_split(split: str) -> list[dict]:
    """Read one split JSONL, sorted by id so downstream order is stable."""
    path = REPO_ROOT / "dataset" / "splits" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: row["id"])
    return rows


def to_code_unit(row: dict) -> ExtractedCodeUnit:
    """Adapt a split row to the type the production checks expect."""
    context = row.get("context") or {}
    return ExtractedCodeUnit(
        repository=row["repository"],
        language=row["language"],
        file_path=row["file_path"],
        symbol=row["symbol"],
        symbol_type=row["symbol_type"],
        source_code=row["source_code"],
        context=CodeUnitContext(
            class_name=context.get("class_name"),
            module_docstring=context.get("module_docstring"),
            imports=list(context.get("imports") or []),
            parent_methods=list(context.get("parent_methods") or []),
        ),
        source_url=row["source_url"],
        is_public=row["is_public"],
        docstring=row.get("docstring", ""),
        source_file_lines=row["source_file_lines"],
        content_hash=row["content_hash"],
        extracted_at=row["extracted_at"],
    )


#: Sampling stratum: the CodeUnit shape a sample must preserve.
Stratum = tuple[str, bool]


def _stratum(row: dict) -> Stratum:
    return (row["symbol_type"], bool(row["is_public"]))


def _allocate(counts: dict[Stratum, int], total: int) -> dict[Stratum, int]:
    """Largest-remainder proportional allocation, summing exactly to *total*."""
    population = sum(counts.values())
    if population == 0 or total <= 0:
        return {key: 0 for key in counts}
    if total >= population:
        return dict(counts)

    exact = {key: total * size / population for key, size in counts.items()}
    allocated = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(allocated.values())
    # Hand out the leftover seats to the largest fractional parts, breaking
    # ties on the stratum key so the result never depends on dict order.
    order = sorted(counts, key=lambda key: (-(exact[key] - allocated[key]), key))
    for key in order[:remainder]:
        allocated[key] += 1
    return allocated


def select_sample(
    rows: Sequence[dict], total: int, seed: int, split: str
) -> list[dict]:
    """Deterministic, stratified sample of *total* rows.

    Stratifies on ``(symbol_type, is_public)`` and allocates proportionally, so
    the sample keeps the split's shape instead of over-sampling whichever kind
    of unit happens to sort first.  Each stratum draws from its own seeded RNG
    keyed by ``seed:split:stratum``, which makes the draw independent of
    stratum ordering and of any earlier draw.
    """
    groups: dict[tuple[str, bool], list[dict]] = {}
    for row in sorted(rows, key=lambda item: item["id"]):
        groups.setdefault(_stratum(row), []).append(row)

    allocated = _allocate({key: len(value) for key, value in groups.items()}, total)
    picked: list[dict] = []
    for key in sorted(groups):
        want = allocated[key]
        if want <= 0:
            continue
        pool = groups[key]
        if want >= len(pool):
            picked.extend(pool)
            continue
        rng = random.Random(f"{seed}:{split}:{key[0]}:{key[1]}")
        picked.extend(rng.sample(pool, want))
    return sorted(picked, key=lambda item: item["id"])


def build_sample(
    train_n: int, validation_n: int, seed: int
) -> list[dict]:
    """The shared CodeUnit sample both arms run on."""
    sample: list[dict] = []
    for split, count in (("train", train_n), ("validation", validation_n)):
        for row in select_sample(load_split(split), count, seed, split):
            sample.append(
                {
                    "code_unit_id": row["id"],
                    "split": row["split"],
                    "symbol_type": row["symbol_type"],
                    "is_public": bool(row["is_public"]),
                    "repository": row["repository"],
                    "file_path": row["file_path"],
                    "language": row["language"],
                    "symbol": row["symbol"],
                    "source_code_sha256": sha256_text(row["source_code"]),
                    "source_lines": row["source_file_lines"],
                }
            )
    return sample


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------


def _arm_imports(code_unit: ExtractedCodeUnit) -> list[str] | None:
    """Module paths for the extractors, or ``None`` when imports are dropped."""
    if not USE_IMPORTS:
        return None
    return list(code_unit.context.imports) or None


def representation_a(code_unit: ExtractedCodeUnit) -> dict[str, Any]:
    """Arm A payload: the production ``StructuredBehaviorFacts``, unchanged.

    The extractor itself is untouched; only the arguments are held symmetric
    with arm B.  See :data:`USE_IMPORTS`.
    """
    facts = extract_behavior_facts(
        code_unit.source_code,
        imports=_arm_imports(code_unit),
    )
    return {
        "primary_purpose": facts.primary_purpose,
        "input_summary": facts.input_summary,
        "output_summary": facts.output_summary,
        "side_effects": list(facts.side_effects),
        "key_operations": list(facts.key_operations),
        "error_handling": facts.error_handling,
        "control_flow": facts.control_flow,
        "raises": list(facts.raises),
        "domain_concepts": list(facts.domain_concepts),
        "observable_effects": list(facts.observable_effects),
    }


def representation_b(code_unit: ExtractedCodeUnit) -> dict[str, Any]:
    """Arm B payload: the Semantic IR, after the existing provenance guard.

    Raises ``AssertionError`` if the guard fails; the caller records that as a
    B-side failure rather than substituting a weaker check.
    """
    ir = build_semantic_ir(
        code_unit.source_code,
        code_unit.language,
        unit_kind=code_unit.symbol_type,
        imports=_arm_imports(code_unit),
    )
    assert_identifier_free(ir, code_unit.source_code, code_unit.language)
    return {
        "input_roles": list(ir.input_roles),
        "output_behavior": ir.output_behavior,
        "domain_signals": list(ir.domain_signals),
        "transformations": list(ir.transformations),
        "relations": list(ir.relations),
        "conditions": list(ir.conditions),
        "state_changes": list(ir.state_changes),
        "side_effects": list(ir.side_effects),
        "error_behavior": ir.error_behavior,
        "observable_effects": list(ir.observable_effects),
        # Diagnostics only; never part of the prompt payload.
        "parse_ok": ir.parse_ok,
        "anchor_count": informativeness(ir).anchor_count,
    }


def _render(payload: dict[str, Any], skip: Iterable[str] = ()) -> str:
    """Render a payload as ``label: value`` lines, lists as ``- item``."""
    skip = set(skip)
    lines: list[str] = []
    for key, value in payload.items():
        if key in skip:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            lines.append(f"{key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_TASK_BLOCK = (
    "You are given behaviour-level facts about exactly one code unit. The facts "
    "were derived from its source and deliberately contain no function, class, "
    "parameter, variable, file, repository, module, or library names, and no "
    "documentation text.\n"
    "\n"
    "Write ONE developer-style search query that a developer could type into "
    "code search to retrieve this code unit.\n"
    "\n"
    "Rules:\n"
    "- Ground the query only in the facts given. Do not invent behaviour.\n"
    "- Never name an identifier, file, path, repository, library, or API, even "
    "if the facts hint at one.\n"
    "- Describe what the unit does and what a caller would observe, not how it "
    "is implemented.\n"
    "- Make the query specific enough to distinguish this unit from similar "
    "ones, without copying implementation vocabulary.\n"
    "\n"
    "Return strict JSON with exactly these keys:\n"
    '{"query": "<the search query>",'
    ' "query_style": "natural|technical|verbose|concise",'
    ' "query_intent": "<one sentence: what the developer is looking for>"}\n'
    "\n"
    "JSON only. No prose, no code fences."
)


def build_prompt(arm: str, payload: dict[str, Any]) -> str:
    """Render the dev prompt for one arm.

    Deliberately takes the *rendered payload text* rather than the CodeUnit, so
    the prompt structurally cannot carry source, docstring, or identifiers.  The
    shared system instruction, the task wording, and the output contract are
    byte-identical across arms; only the facts block differs.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    skip = {"parse_ok", "anchor_count"} if arm == "B" else set()
    return (
        f"{get_query_system_prompt()}\n"
        f"\n"
        f"{_TASK_BLOCK}\n"
        f"\n"
        f"BEHAVIOUR FACTS:\n{_render(payload, skip)}\n"
    )


def prompt_shell() -> str:
    """The arm-independent part of every prompt, for the equivalence test."""
    return (
        f"{get_query_system_prompt()}\n\n{_TASK_BLOCK}\n\nBEHAVIOUR FACTS:\n"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def make_generate_fn(
    model: LocalModel,
    seed: int,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
) -> Callable[[str], str]:
    """Generate function matching the production sampling parameters."""

    def _generate(prompt: str) -> str:
        result = model.generate(
            GenerationRequest(
                prompt=prompt,
                model=model.name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
            )
        )
        if result.error:
            raise RuntimeError(result.error)
        return result.text

    return _generate


@dataclass
class AttemptLog:
    attempt_number: int
    status: str
    raw_text: str
    will_retry: bool
    generation_ms: float | None
    errors: list[str] = field(default_factory=list)


def _attempts(retry_result: RetryResult) -> list[AttemptLog]:
    return [
        AttemptLog(
            attempt_number=attempt.attempt_number,
            status=attempt.status,
            raw_text=attempt.raw_text,
            will_retry=attempt.will_retry,
            generation_ms=attempt.generation_ms,
            errors=[str(error) for error in attempt.errors],
        )
        for attempt in retry_result.attempts
    ]


def run_one(
    arm: str,
    entry: dict,
    row: dict,
    model: LocalModel,
    policy: RetryPolicy,
    seed: int,
) -> dict[str, Any]:
    """Run one arm for one CodeUnit and return a fully auditable record."""
    code_unit = to_code_unit(row)
    started = time.perf_counter()

    record: dict[str, Any] = {
        "code_unit_id": entry["code_unit_id"],
        "split": entry["split"],
        "symbol_type": entry["symbol_type"],
        "is_public": entry["is_public"],
        "arm": arm,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "model": model.name,
        "seed": seed,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "max_attempts": policy.max_attempts,
        "success": False,
        "stage": "pending",
        "failure_reason": None,
        "query": None,
        "query_style": None,
        "query_intent": None,
        "schema_valid": False,
        "leakage_passed": None,
        "provenance_passed": None,
        "meta_passed": None,
        "attempts": [],
        "payload_sha256": None,
        "payload_chars": 0,
        "ir_guard_passed": None,
        "ir_anchor_count": None,
        "total_generation_ms": 0.0,
        "total_validation_ms": 0.0,
        "wall_ms": 0.0,
    }

    try:
        payload = (
            representation_a(code_unit) if arm == "A" else representation_b(code_unit)
        )
    except AssertionError as exc:
        record["stage"] = "representation_guard_failed"
        record["failure_reason"] = str(exc)
        record["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return record

    if arm == "B":
        record["ir_guard_passed"] = True
        record["ir_anchor_count"] = payload.get("anchor_count")

    prompt = build_prompt(arm, payload)
    record["payload_sha256"] = sha256_text(prompt)
    record["payload_chars"] = len(prompt)

    retry_result = run_with_retry(
        prompt=prompt,
        schema=CandidateQuery,
        generate_fn=make_generate_fn(model, seed),
        policy=policy,
    )
    record["attempts"] = [asdict(attempt) for attempt in _attempts(retry_result)]
    record["total_generation_ms"] = retry_result.total_generation_ms
    record["total_validation_ms"] = retry_result.total_validation_ms

    if not retry_result.success or retry_result.result is None:
        record["stage"] = "generation_exhausted"
        last = record["attempts"][-1]["errors"] if record["attempts"] else []
        record["failure_reason"] = "; ".join(last) or "no attempt recorded"
        record["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return record

    candidate = retry_result.result.data
    record["schema_valid"] = True
    record["query"] = candidate.query
    record["query_style"] = candidate.query_style
    record["query_intent"] = candidate.query_intent

    # Identical validation chain to the production generator, in the same order.
    leakage = check_query_leakage(candidate.query, code_unit)
    record["leakage_passed"] = leakage.passed
    if not leakage.passed:
        record["stage"] = "leakage"
        record["failure_reason"] = "; ".join(leakage.violations)
        record["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return record

    provenance = check_query_provenance(candidate.query, code_unit.docstring)
    record["provenance_passed"] = provenance.passed
    if not provenance.passed:
        record["stage"] = "provenance"
        record["failure_reason"] = "; ".join(provenance.violations)
        record["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return record

    record["meta_passed"] = not check_meta_query(candidate.query)
    if not record["meta_passed"]:
        record["stage"] = "meta_query"
        record["failure_reason"] = "meta-task query"
        record["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return record

    record["success"] = True
    record["stage"] = "complete"
    record["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return record


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _ngrams(text: str, n: int) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def distinct_n(queries: Sequence[str], n: int) -> float:
    """Corpus-level distinct-n: unique n-grams over the arm's valid queries."""
    grams: set[str] = set()
    for query in queries:
        grams |= _ngrams(query, n)
    return len(grams)


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def arm_summary(records: Sequence[dict]) -> dict[str, Any]:
    total = len(records)
    valid = [r for r in records if r["success"]]
    queries = [r["query"] for r in valid]
    attempts = sum(len(r["attempts"]) for r in records)
    retried = sum(1 for r in records if len(r["attempts"]) > 1)
    guard_failed = sum(
        1 for r in records if r["stage"] == "representation_guard_failed"
    )
    return {
        "units": total,
        "success": len(valid),
        "success_rate": _rate(len(valid), total),
        "schema_valid": sum(1 for r in records if r["schema_valid"]),
        "leakage_passed": sum(1 for r in records if r["leakage_passed"] is True),
        "provenance_passed": sum(
            1 for r in records if r["provenance_passed"] is True
        ),
        "meta_passed": sum(1 for r in records if r["meta_passed"] is True),
        "non_empty_query": sum(1 for q in queries if q and q.strip()),
        "guard_failures": guard_failed,
        "retried_units": retried,
        "retry_rate": _rate(retried, total),
        "generation_attempts": attempts,
        "mean_attempts": round(attempts / total, 4) if total else 0.0,
        "mean_query_words": (
            round(sum(len(q.split()) for q in queries) / len(queries), 2)
            if queries
            else 0.0
        ),
        "distinct_1": distinct_n(queries, 1),
        "distinct_2": distinct_n(queries, 2),
        "distinct_3": distinct_n(queries, 3),
        "unique_queries": len(set(queries)),
        "mean_generation_ms": (
            round(sum(r["total_generation_ms"] for r in records) / total, 2)
            if total
            else 0.0
        ),
        "failure_stages": dict(
            Counter(r["stage"] for r in records if not r["success"])
        ),
    }


def paired_summary(records: Sequence[dict]) -> dict[str, Any]:
    by_unit: dict[str, dict[str, dict]] = {}
    for record in records:
        by_unit.setdefault(record["code_unit_id"], {})[record["arm"]] = record
    both = [pair for pair in by_unit.values() if len(pair) == len(ARMS)]
    both_ok = sum(1 for pair in both if all(r["success"] for r in pair.values()))
    a_only = sum(
        1
        for pair in both
        if pair["A"]["success"] and not pair["B"]["success"]
    )
    b_only = sum(
        1
        for pair in both
        if pair["B"]["success"] and not pair["A"]["success"]
    )
    neither = sum(
        1
        for pair in both
        if not any(r["success"] for r in pair.values())
    )
    return {
        "pairs": len(both),
        "both_succeeded": both_ok,
        "neither_succeeded": neither,
        "a_only_succeeded": a_only,
        "b_only_succeeded": b_only,
        "same_validity": both_ok + neither,
        "different_validity": a_only + b_only,
    }


def build_summary(records: Sequence[dict]) -> dict[str, Any]:
    """Neutral, symmetric aggregate.  No ranking, no winner, no score."""
    per_split: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        subset = [r for r in records if r["split"] == split]
        per_split[split] = {
            "A": arm_summary([r for r in subset if r["arm"] == "A"]),
            "B": arm_summary([r for r in subset if r["arm"] == "B"]),
            "paired": paired_summary(subset),
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": EXPERIMENT_VERSION,
        "generated_at": _stamp(),
        "interpretation": (
            "Counts only. No winner is computed; A and B are reported side by "
            "side under identical conditions."
        ),
        "overall": {
            "A": arm_summary([r for r in records if r["arm"] == "A"]),
            "B": arm_summary([r for r in records if r["arm"] == "B"]),
            "paired": paired_summary(records),
        },
        "by_split": per_split,
    }


def build_blind_review(records: Sequence[dict], seed: int) -> dict[str, Any]:
    """Select comparable pairs and lay them out in a fixed blinded order.

    Only pairs where *both* arms produced a valid query are eligible: a pair
    with a missing side cannot be compared.  The input file carries no arm
    label, and the key is written separately so the mapping survives.
    """
    by_unit: dict[str, dict[str, dict]] = {}
    for record in records:
        by_unit.setdefault(record["code_unit_id"], {})[record["arm"]] = record

    eligible = [
        (unit_id, pair)
        for unit_id, pair in sorted(by_unit.items())
        if len(pair) == len(ARMS) and all(r["success"] for r in pair.values())
    ]
    rng = random.Random(f"{seed}:blind-review")
    order = list(range(len(eligible)))
    rng.shuffle(order)

    chosen = [eligible[i] for i in order[: min(BLIND_REVIEW_PAIRS, len(order))]]

    # Lay out both arms in one shuffled sequence.  Presenting all of A then all
    # of B would hand the reviewer the label by position alone, so the arm is
    # randomised independently of the pair order and review ids are assigned
    # only after that shuffle.
    entries: list[tuple[int, str, str]] = [
        (position, arm, pair[arm]["query"])
        for position, (_unit_id, pair) in enumerate(chosen)
        for arm in ARMS
    ]
    random.Random(f"{seed}:blind-review-items").shuffle(entries)

    review_input = {
        "experiment_id": EXPERIMENT_ID,
        "version": EXPERIMENT_VERSION,
        "generated_at": _stamp(),
        "instructions": (
            "Judge each query against every criterion independently. A query "
            "is accepted only if all criteria pass. You do not know which "
            "arm produced which query, and must not try to infer it."
        ),
        "criteria": [
            {"name": name, "definition": _criterion_definition(name)}
            for name in REVIEW_CRITERIA
        ],
        "items": [
            {
                "review_id": f"r{slot:03d}",
                "pair_id": f"p{position:03d}",
                "query": query,
            }
            for slot, (position, _arm, query) in enumerate(entries, start=1)
        ],
    }
    key = {
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "pairs": len(chosen),
        "eligible_pairs": len(eligible),
        "mapping": {
            f"r{slot:03d}": {
                "pair_id": f"p{position:03d}",
                "code_unit_id": chosen[position - 1][0],
                "arm": arm,
            }
            for slot, (position, arm, _query) in enumerate(entries, start=1)
        },
    }
    return {"input": review_input, "key": key}


def _criterion_definition(name: str) -> str:
    return {
        "Understandable": "grammatically correct and clear",
        "Behaviorally relevant": "describes real code behavior",
        "Sufficiently specific": "not generic; grounded in this code unit",
        "Unambiguous": "one clear interpretation",
        "No implementation leakage": (
            "no function, class, parameter, variable, file, or library names"
        ),
        "Developer could locate the code": (
            "the relevant unit is reachable from this query in a top-10 search"
        ),
    }[name]


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def protected_hashes() -> dict[str, str]:
    """SHA-256 of every protected artifact, for the before/after check."""
    out: dict[str, str] = {}
    for relative in PROTECTED_ARTIFACTS:
        path = REPO_ROOT / relative
        out[relative] = sha256_file(path) if path.exists() else "missing"
    return out


def build_manifest(
    train_n: int, validation_n: int, seed: int, model_name: str, policy: RetryPolicy
) -> dict[str, Any]:
    split_files = {
        split: {
            "path": f"dataset/splits/{split}.jsonl",
            "sha256": sha256_file(REPO_ROOT / "dataset" / "splits" / f"{split}.jsonl"),
        }
        for split in SPLITS
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": EXPERIMENT_VERSION,
        "created_at": _stamp(),
        "code_revision": git_revision(),
        "python": platform.python_version(),
        "model": {
            "name": model_name,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "seed": seed,
            "max_attempts": policy.max_attempts,
        },
        "prompt": {
            "template_version": PROMPT_TEMPLATE_VERSION,
            "system_instruction_sha256": sha256_text(get_query_system_prompt()),
            "task_block_sha256": sha256_text(_TASK_BLOCK),
            "shared_shell_sha256": sha256_text(prompt_shell()),
            "note": (
                "GenerationRequest has no system field, so the shared system "
                "instruction is folded into the single prompt for both arms."
            ),
        },
        "sampling": {
            "seed": seed,
            "train_requested": train_n,
            "validation_requested": validation_n,
            "stratified_on": ["symbol_type", "is_public"],
            "allocation": "largest-remainder proportional per stratum",
        },
        "arms": {
            "A": "StructuredBehaviorFacts via extract_behavior_facts (production)",
            "B": "Semantic IR via build_semantic_ir, guarded by assert_identifier_free",
        },
        "deviations_from_production": {
            "imports_dropped": {
                "value": not USE_IMPORTS,
                "reason": (
                    "assert_identifier_free rebuilds the renamed IR without "
                    "imports, so an import-grounded IR cannot be compared "
                    "against its own rename. Arm B must pass the guard, and "
                    "keeping imports for A alone would give A import-grounded "
                    "domain signals B cannot match."
                ),
                "cost": (
                    "import-grounded domain signals are absent on both arms"
                ),
            },
            "prompt_context_omitted": {
                "value": True,
                "reason": (
                    "The production prompt passes class_name, "
                    "parent_methods, module_docstring and imports. Identifiers "
                    "and docstrings are forbidden here, so both arms get an "
                    "identifier-free payload only."
                ),
            },
            "system_instruction_folded_into_prompt": {
                "value": True,
                "reason": (
                    "GenerationRequest carries no system field, so the shared "
                    "system instruction is prepended to the prompt for both "
                    "arms."
                ),
            },
        },
        "validation_chain": [
            "CandidateQuery schema",
            "check_query_leakage",
            "check_query_provenance",
            "check_meta_query",
        ],
        "review": {
            "criteria": list(REVIEW_CRITERIA),
            "criteria_source": "DATASET_SPECIFICATION.md section 4.4.4",
            "blind_pairs": BLIND_REVIEW_PAIRS,
            "blinded": True,
        },
        "inputs": split_files,
        "protected_artifacts": protected_hashes(),
        "artifacts": list(ARTIFACT_NAMES),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_experiment(
    out_dir: Path,
    train_n: int,
    validation_n: int,
    seed: int,
    model: LocalModel,
    policy: RetryPolicy,
) -> dict[str, Any]:
    """Sample, generate both arms, and write every artifact."""
    sample = build_sample(train_n, validation_n, seed)
    manifest = build_manifest(train_n, validation_n, seed, model.name, policy)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "sample.jsonl", sample)

    rows_by_id = {
        row["id"]: row
        for split in SPLITS
        for row in load_split(split)
    }
    records: list[dict] = []
    for entry in sample:
        row = rows_by_id[entry["code_unit_id"]]
        for arm in ARMS:
            records.append(run_one(arm, entry, row, model, policy, seed))
            write_jsonl(out_dir / "generations.jsonl", records)

    write_json(out_dir / "automatic_summary.json", build_summary(records))
    review = build_blind_review(records, seed)
    write_json(out_dir / "blind_review_input.json", review["input"])
    write_json(out_dir / "blind_review_key.json", review["key"])

    after = protected_hashes()
    if after != manifest["protected_artifacts"]:
        changed = [
            name
            for name in after
            if after[name] != manifest["protected_artifacts"].get(name)
        ]
        raise RuntimeError(f"protected artifacts changed during the run: {changed}")
    return {"manifest": manifest, "records": records, "review": review}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only A/B comparing StructuredBehaviorFacts against "
            "the Semantic IR for query generation. Reports counts only; it "
            "computes no winner."
        )
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="write the deterministic sample and manifest, then exit",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-sample", type=int, default=DEFAULT_TRAIN_SAMPLE)
    parser.add_argument(
        "--validation-sample", type=int, default=DEFAULT_VALIDATION_SAMPLE
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "experiments"
        / EXPERIMENT_ID
        / EXPERIMENT_VERSION,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    policy = RetryPolicy(max_attempts=args.max_attempts)
    out_dir: Path = args.out_dir

    if args.sample:
        sample = build_sample(args.train_sample, args.validation_sample, args.seed)
        manifest = build_manifest(
            args.train_sample, args.validation_sample, args.seed, args.model, policy
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "manifest.json", manifest)
        write_jsonl(out_dir / "sample.jsonl", sample)
        print(f"sample written: {len(sample)} units -> {out_dir}")
        return 0

    model = OllamaAdapter(model_name=args.model)
    try:
        if not model.health_check():
            print(f"model host unreachable for {args.model}", file=sys.stderr)
            return 2
        outcome = run_experiment(
            out_dir=out_dir,
            train_n=args.train_sample,
            validation_n=args.validation_sample,
            seed=args.seed,
            model=model,
            policy=policy,
        )
    finally:
        model.close()

    summary = build_summary(outcome["records"])
    overall = summary["overall"]
    print(f"units sampled: {len(outcome['records']) // len(ARMS)}")
    for arm in ARMS:
        stats = overall[arm]
        print(
            f"  {arm}: success {stats['success']}/{stats['units']}, "
            f"retries {stats['retried_units']}, "
            f"mean words {stats['mean_query_words']}"
        )
    paired = overall["paired"]
    print(
        f"  pairs: {paired['pairs']} "
        f"(both {paired['both_succeeded']}, "
        f"A only {paired['a_only_succeeded']}, "
        f"B only {paired['b_only_succeeded']}, "
        f"neither {paired['neither_succeeded']})"
    )
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
