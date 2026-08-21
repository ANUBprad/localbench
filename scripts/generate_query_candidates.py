"""Phase 4F-I-B: candidate retrieval-query generation (test split).

Loads the canonical test split from ``dataset/splits/test.jsonl`` and
generates exactly ONE candidate retrieval query per test CodeUnit using
the dedicated query-generation model (``qwen2.5-coder:7b``, never a
benchmark model). Reuses the existing Phase 3 retry infrastructure and
the Phase 4F QueryGenerator unchanged.

The canonical CodeUnit dataset is NOT modified. Only query-generation
artifacts are written:
  dataset/queries/candidates.jsonl          successful candidate pool
  dataset/queries/candidate_failures.jsonl  failed candidates (audit)
  dataset/queries/generation_metadata.json  reproducibility + statistics
  dataset/meta/version.json                 query_generation block only,
                                            per DATASET_SPECIFICATION.md
                                            §4.4.1 (opt-in via
                                            --update-meta)

There is NO resume mechanism: results are written once, after all units
complete. An interrupted run leaves the previous artifacts untouched;
rerun from scratch (delete outputs first) to avoid duplicates.

Usage:
    python scripts/generate_query_candidates.py
    python scripts/generate_query_candidates.py --limit 1 \
        --output-dir <temp dir>          # smoke test, touches nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localbench.runtime.generation.attempt import AttemptRecord, AttemptStatus
from localbench.runtime.generation.policy import DEFAULT_MAX_ATTEMPTS, RetryPolicy
from localbench.runtime.ollama.adapter import OllamaAdapter
from localbench.workloads.code_retrieval.extraction import (
    ExtractedCodeUnit,
    _build_code_unit_id,
)
from localbench.workloads.code_retrieval.query_generator import QueryGenerator
from localbench.workloads.code_retrieval.query_prompt import (
    QUERY_PROMPT_TEMPLATE_VERSION,
)
from localbench.workloads.code_retrieval.schemas import (
    CodeUnitContext,
    QueryGenerationInput,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen configuration (Phase 4F-I-B)
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
TEST_SPLIT_PATH = DATASET_ROOT / "splits" / "test.jsonl"

MODEL_NAME = "qwen2.5-coder:7b"
MODEL_VERSION = "7b"
SEED = 42
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 128
MAX_ATTEMPTS = DEFAULT_MAX_ATTEMPTS

PROGRESS_LOG_INTERVAL = 25
# Abort the run after this many consecutive provider-level failures;
# isolated transient failures are simply recorded as failed candidates.
CONSECUTIVE_PROVIDER_FAILURE_LIMIT = 3

_PROVIDER_FAILURE_CATEGORIES = frozenset(
    {"timeout", "provider_unavailable", "model_error", "provider_or_model_error"}
)

_PERMITTED_INPUT_FIELDS = frozenset(
    {
        "source_code",
        "docstring",
        "symbol_type",
        "class_name",
        "module_docstring",
        "imports",
        "parent_methods",
    }
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class CandidateAuditRecord:
    """Full audit trail for one CodeUnit's candidate generation.

    Field names mirror the existing candidate artifact schema; the
    trailing fields extend it additively (attempt history, timing,
    classification).
    """

    code_unit_id: str
    candidate_id: str
    query: str
    query_style: str
    query_intent: str
    model: str
    model_version: str
    prompt_version: str
    seed: int
    temperature: float
    top_p: float
    max_tokens: int
    attempt_count: int
    attempts: list[dict] = field(default_factory=list)
    generation_ms: float = 0.0
    validation_ms: float = 0.0
    validation_passed: bool = False
    leakage_passed: bool = False
    leakage_violations: list[str] = field(default_factory=list)
    success: bool = False
    failure_category: str | None = None
    failure_reason: str | None = None
    completed_utc: str = ""


def _serialize_attempt(attempt: AttemptRecord) -> dict:
    """JSON-serializable form of one AttemptRecord."""
    return {
        "attempt_number": attempt.attempt_number,
        "status": attempt.status,
        "will_retry": attempt.will_retry,
        "generation_ms": attempt.generation_ms,
        "errors": [str(error) for error in attempt.errors],
        "raw_text": attempt.raw_text,
    }


def _build_record(unit_id: str, result, completed_utc: str) -> CandidateAuditRecord:
    """Build the audit record for a QueryGenerationResult."""
    success = result.success and result.candidate is not None
    category = None
    reason = None
    if not success:
        category, reason = classify_failure(result)

    return CandidateAuditRecord(
        code_unit_id=unit_id,
        candidate_id=f"candidate_{unit_id}",
        query=result.candidate.query if success else "",
        query_style=result.candidate.query_style if success else "",
        query_intent=result.candidate.query_intent if success else "",
        model=MODEL_NAME,
        model_version=MODEL_VERSION,
        prompt_version=QUERY_PROMPT_TEMPLATE_VERSION,
        seed=SEED,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        attempt_count=len(result.attempts),
        attempts=[_serialize_attempt(a) for a in result.attempts],
        generation_ms=result.total_generation_ms,
        validation_ms=result.total_validation_ms,
        validation_passed=bool(result.attempts)
        and result.attempts[-1].status == AttemptStatus.SUCCESS,
        leakage_passed=bool(result.leakage and result.leakage.passed),
        leakage_violations=result.leakage.violations if result.leakage else [],
        success=success,
        failure_category=category,
        failure_reason=reason,
        completed_utc=completed_utc,
    )


_VALIDATION_ERROR_CATEGORIES = {
    "MalformedJSONError": "malformed_json",
    "MissingFieldError": "missing_field",
    "TypeMismatchError": "type_mismatch",
    "ConstraintViolationError": "constraint_violation",
}


def classify_failure(result) -> tuple[str, str]:
    """Classify why a QueryGenerationResult failed.

    Returns (category, reason). Categories distinguish validation
    failures, leakage, and provider-level failures.
    """
    if result.leakage is not None and not result.leakage.passed:
        violations = "; ".join(result.leakage.violations)
        return "leakage", violations

    if not result.attempts:
        return "unknown", "No attempts were recorded."

    last = result.attempts[-1]
    if last.status == AttemptStatus.SUCCESS:
        # Validation passed but the candidate was rejected downstream;
        # only the leakage check rejects at that point.
        return "leakage", "Candidate rejected after successful validation."

    if last.errors:
        error = last.errors[0]
        category = _VALIDATION_ERROR_CATEGORIES.get(type(error).__name__)
        if category:
            return category, str(error)

        message = str(error)
        if message.startswith("OllamaUnavailableError:") and (
            "timed out" in message or "timed out" in message.lower()
        ):
            return "timeout", message
        if message.startswith("OllamaUnavailableError:"):
            return "provider_unavailable", message
        if message.startswith(
            ("ModelNotFoundError:", "GenerationError:", "RuntimeError:")
        ):
            return "model_error", message
        return "provider_or_model_error", message

    return "unknown", "Last attempt failed without a recorded error."


# ---------------------------------------------------------------------------
# Input loading and blindness verification
# ---------------------------------------------------------------------------


def load_test_units(path: Path) -> list[ExtractedCodeUnit]:
    """Rebuild ExtractedCodeUnits from the canonical test split."""
    units: list[ExtractedCodeUnit] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            units.append(
                ExtractedCodeUnit(
                    repository=row["repository"],
                    language=row["language"],
                    file_path=row["file_path"],
                    symbol=row["symbol"],
                    symbol_type=row["symbol_type"],
                    source_code=row["source_code"],
                    context=CodeUnitContext(**row["context"]),
                    source_url=row["source_url"],
                    is_public=row["is_public"],
                    docstring=row["docstring"],
                    source_file_lines=row["source_file_lines"],
                    content_hash=row["content_hash"],
                    extracted_at=row["extracted_at"],
                )
            )
    return units


def verify_input_blindness() -> None:
    """Structurally verify the generator can only see permitted fields.

    The prompt builder consumes ONLY ``QueryGenerationInput``; asserting
    its exact field set proves no repository/file/symbol/label identity
    can reach the model input.
    """
    actual_fields = frozenset(QueryGenerationInput.__dataclass_fields__)
    if actual_fields != _PERMITTED_INPUT_FIELDS:
        unexpected = sorted(actual_fields - _PERMITTED_INPUT_FIELDS)
        missing = sorted(_PERMITTED_INPUT_FIELDS - actual_fields)
        raise SystemExit(
            f"BLINDNESS VIOLATION: QueryGenerationInput fields changed. "
            f"Unexpected: {unexpected}, missing: {missing}"
        )
    logger.info(
        "Input blindness verified: %d permitted fields only", len(actual_fields)
    )


def _unit_id(unit: ExtractedCodeUnit) -> str:
    return _build_code_unit_id(
        unit.repository, unit.file_path, unit.symbol, unit.content_hash
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_CANDIDATES_FILENAME = "candidates.jsonl"
_FAILURES_FILENAME = "candidate_failures.jsonl"
_METADATA_FILENAME = "generation_metadata.json"


def write_artifacts(
    output_dir: Path,
    successful: list[CandidateAuditRecord],
    failed: list[CandidateAuditRecord],
    metadata: dict,
) -> None:
    """Overwrite the three query-generation artifacts atomically-at-end."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / _CANDIDATES_FILENAME, "w", encoding="utf-8") as f:
        for record in successful:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with open(output_dir / _FAILURES_FILENAME, "w", encoding="utf-8") as f:
        for record in failed:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with open(output_dir / _METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def update_dataset_metadata() -> None:
    """Record query-generation provenance in ``meta/version.json``.

    Writes ONLY the ``query_generation`` block required by
    DATASET_SPECIFICATION.md §4.4.1; all other metadata is preserved.
    ``total_queries`` stays untouched: it counts FINAL evaluation
    queries (§5.1), not this candidate pool.
    """
    version_path = DATASET_ROOT / "meta" / "version.json"
    with open(version_path, encoding="utf-8") as f:
        meta = json.load(f)

    meta["query_generation"] = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "prompt_template_version": QUERY_PROMPT_TEMPLATE_VERSION,
        "seed": SEED,
        "generation_params": {"temperature": TEMPERATURE, "top_p": TOP_P},
    }

    with open(version_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Updated query_generation block in %s", version_path)


# ---------------------------------------------------------------------------
# Generation run
# ---------------------------------------------------------------------------


def discover_model_metadata(adapter: OllamaAdapter) -> dict:
    """Best available runtime metadata for the generation model.

    Ollama exposes no semantic version string per model; size and
    modification timestamp are recorded instead of being fabricated.
    """
    try:
        for info in adapter.discover_models():
            if info.name == MODEL_NAME:
                return {
                    "size_bytes": info.size_bytes,
                    "modified_at": info.modified_at,
                }
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort
        logger.warning("Could not discover model metadata: %s", exc)
    return {}


def generate_all(
    test_units: list[ExtractedCodeUnit],
    output_dir: Path,
    provider_failure_limit: int,
    update_meta: bool,
) -> int:
    """Run candidate generation for every test unit. Returns exit code."""
    started = time.perf_counter()

    adapter = OllamaAdapter(model_name=MODEL_NAME)
    if not adapter.health_check():
        logger.error("Ollama is not reachable at localhost:11434")
        return 2
    model_runtime_metadata = discover_model_metadata(adapter)
    logger.info("Model runtime metadata: %s", model_runtime_metadata)

    policy = RetryPolicy(max_attempts=MAX_ATTEMPTS)
    generator = QueryGenerator(model=adapter, policy=policy, top_p=TOP_P, seed=SEED)

    total = len(test_units)
    successful: list[CandidateAuditRecord] = []
    failed: list[CandidateAuditRecord] = []
    consecutive_provider_failures = 0

    for index, unit in enumerate(test_units, 1):
        unit_id = _unit_id(unit)
        result = generator.generate(unit)
        record = _build_record(unit_id, result, datetime.now(timezone.utc).isoformat())

        if record.success:
            successful.append(record)
            consecutive_provider_failures = 0
        else:
            failed.append(record)
            logger.warning(
                "[%d/%d] FAILED %s (%s): %s",
                index,
                total,
                unit_id,
                record.failure_category,
                record.failure_reason,
            )
            if record.failure_category in _PROVIDER_FAILURE_CATEGORIES:
                consecutive_provider_failures += 1
            else:
                consecutive_provider_failures = 0

        if consecutive_provider_failures >= provider_failure_limit:
            logger.error(
                "Aborting: %d consecutive provider failures suggest "
                "Ollama is down. No artifacts written; rerun from scratch.",
                consecutive_provider_failures,
            )
            adapter.close()
            return 3

        if index % PROGRESS_LOG_INTERVAL == 0 or index == total:
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed > 0 else 0.0
            eta_minutes = ((total - index) / rate / 60) if rate > 0 else 0.0
            logger.info(
                "[%d/%d] ok=%d failed=%d | %.1f units/min | ETA %.0f min",
                index,
                total,
                len(successful),
                len(failed),
                rate * 60,
                eta_minutes,
            )

    adapter.close()

    wall_clock = time.perf_counter() - started
    metadata = build_metadata(
        successful, failed, total, wall_clock, model_runtime_metadata
    )
    write_artifacts(output_dir, successful, failed, metadata)
    logger.info("Artifacts written to %s", output_dir)

    if update_meta:
        update_dataset_metadata()

    logger.info("=" * 60)
    logger.info("PHASE 4F-I-B CANDIDATE GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Test CodeUnits:        %d", total)
    logger.info("Successful candidates: %d", len(successful))
    logger.info("Failed candidates:     %d", len(failed))
    all_count = sum(r.attempt_count for r in successful + failed)
    logger.info("Total attempts:        %d", all_count)
    logger.info("Wall clock:            %.1f s", wall_clock)
    return 0


def build_metadata(
    successful: list[CandidateAuditRecord],
    failed: list[CandidateAuditRecord],
    total: int,
    wall_clock_seconds: float,
    model_runtime_metadata: dict,
) -> dict:
    """Aggregate reproducibility and statistics metadata."""
    all_records = successful + failed
    total_attempts = sum(r.attempt_count for r in all_records)
    categories = Counter(r.failure_category for r in failed)
    styles = Counter(r.query_style for r in successful)
    validation_failures = sum(
        1 for r in failed if r.failure_category in _VALIDATION_ERROR_CATEGORIES.values()
    )
    provider_failures = sum(
        1 for r in failed if r.failure_category in _PROVIDER_FAILURE_CATEGORIES
    )
    retry_exhausted = sum(
        1
        for r in failed
        if r.attempt_count >= MAX_ATTEMPTS and r.failure_category != "leakage"
    )
    avg_gen_ms = (
        sum(r.generation_ms for r in successful) / len(successful)
        if successful
        else 0.0
    )

    return {
        "phase": "4F-I-B",
        "purpose": "candidate_pool_not_final_evaluation_set",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(wall_clock_seconds, 1),
        "input": {
            "source": "dataset/splits/test.jsonl",
            "total_test_code_units": total,
            "attempted": len(all_records),
            "successful_candidates": len(successful),
            "failed_candidates": len(failed),
        },
        "model": {
            "name": MODEL_NAME,
            "version_label": MODEL_VERSION,
            "runtime_metadata": model_runtime_metadata,
            "is_benchmark_model": False,
        },
        "prompt_template_version": QUERY_PROMPT_TEMPLATE_VERSION,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "max_attempts_per_unit": MAX_ATTEMPTS,
        "stats": {
            "success_rate": round(len(successful) / total, 4) if total else 0.0,
            "total_attempts": total_attempts,
            "avg_attempts_per_code_unit": (
                round(total_attempts / len(all_records), 3) if all_records else 0.0
            ),
            "leakage_failures": categories.get("leakage", 0),
            "validation_failures": validation_failures,
            "provider_failures": provider_failures,
            "retry_exhausted": retry_exhausted,
            "failure_breakdown": dict(categories),
            "query_style_distribution": dict(styles),
            "avg_generation_ms_per_successful_unit": round(avg_gen_ms, 1),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate for only the first N test units (smoke testing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_ROOT / "queries",
        help="Artifact output directory (override for smoke tests).",
    )
    parser.add_argument(
        "--update-meta",
        action="store_true",
        help="Also write the query_generation block to meta/version.json.",
    )
    parser.add_argument(
        "--provider-failure-limit",
        type=int,
        default=CONSECUTIVE_PROVIDER_FAILURE_LIMIT,
        help="Abort after N consecutive provider-level failures.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logger.info("=" * 60)
    logger.info("Phase 4F-I-B: Test Query Candidate Generation")
    logger.info("=" * 60)

    verify_input_blindness()

    test_units = load_test_units(TEST_SPLIT_PATH)
    if args.limit is not None:
        test_units = test_units[: args.limit]
    if not test_units:
        logger.error("No test CodeUnits loaded from %s", TEST_SPLIT_PATH)
        return 1
    logger.info("Loaded %d test CodeUnits from %s", len(test_units), TEST_SPLIT_PATH)

    return generate_all(
        test_units=test_units,
        output_dir=args.output_dir,
        provider_failure_limit=args.provider_failure_limit,
        update_meta=args.update_meta,
    )


if __name__ == "__main__":
    sys.exit(main())
