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

Persistence is INCREMENTAL: every completed CodeUnit (success or
failure) is appended to its JSONL artifact and fsynced immediately.
The two JSONL files double as the checkpoint — a restarted run skips
already-completed CodeUnit IDs, so an interrupted run can be resumed
without duplicates. A torn trailing line from a mid-write crash is
truncated at load time; corrupt or duplicated records abort the run.

Usage:
    python scripts/generate_query_candidates.py
    python scripts/generate_query_candidates.py --limit 1 \
        --output-dir <temp dir>          # smoke test, touches nothing
    python scripts/generate_query_candidates.py --fresh  # discard
        existing artifacts and regenerate from scratch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localbench.runtime.generation.attempt import AttemptRecord, AttemptStatus
from localbench.runtime.generation.policy import DEFAULT_MAX_ATTEMPTS, RetryPolicy
from localbench.runtime.ollama.adapter import OllamaAdapter
from localbench.workloads.code_retrieval.candidate_store import CandidateStore
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


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON via a temp file + atomic replace.

    An interrupted write can never leave a half-written artifact at the
    target path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def filter_pending(
    test_units: list[ExtractedCodeUnit],
    completed_ids: set[str],
) -> list[ExtractedCodeUnit]:
    """Drop CodeUnits already recorded in the checkpoint (either outcome)."""
    return [u for u in test_units if _unit_id(u) not in completed_ids]


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

    _atomic_write_json(version_path, meta)
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


def _generate_one(
    generator: QueryGenerator, unit: ExtractedCodeUnit
) -> CandidateAuditRecord:
    """Generate one candidate record (thread-safe: no shared mutation)."""
    unit_id = _unit_id(unit)
    result = generator.generate(unit)
    return _build_record(
        unit_id, result, datetime.now(timezone.utc).isoformat()
    )


def generate_all(
    test_units: list[ExtractedCodeUnit],
    output_dir: Path,
    provider_failure_limit: int,
    update_meta: bool,
    workers: int = 1,
) -> int:
    """Run candidate generation for every pending test unit.

    Completed units (success or failure) are appended to the JSONL
    artifacts immediately and skipped on restart. With ``workers > 1``
    inference runs on a thread pool while persistence stays serialized
    on this thread; record content per CodeUnit is identical for any
    worker count (only append order varies).
    """
    started = time.perf_counter()

    adapter = OllamaAdapter(model_name=MODEL_NAME)
    if not adapter.health_check():
        logger.error("Ollama is not reachable at localhost:11434")
        return 2
    model_runtime_metadata = discover_model_metadata(adapter)
    logger.info("Model runtime metadata: %s", model_runtime_metadata)

    store = CandidateStore(
        output_dir / _CANDIDATES_FILENAME,
        output_dir / _FAILURES_FILENAME,
    )
    resumed_success, resumed_failed = store.load()
    completed = store.completed_ids
    pending = filter_pending(test_units, completed)
    total = len(test_units)
    logger.info(
        "Checkpoint: %d completed (%d success / %d failed), "
        "%d remaining of %d",
        len(completed),
        len(resumed_success),
        len(resumed_failed),
        len(pending),
        total,
    )

    policy = RetryPolicy(max_attempts=MAX_ATTEMPTS)
    generator = QueryGenerator(model=adapter, policy=policy, top_p=TOP_P, seed=SEED)

    fresh_success: list[CandidateAuditRecord] = []
    fresh_failed: list[CandidateAuditRecord] = []
    consecutive_provider_failures = 0
    exit_code = 0
    processed = 0

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_generate_one, generator, unit): unit
                for unit in pending
            }
            outstanding = set(futures)
            aborted = False

            while outstanding:
                done, outstanding = wait(
                    outstanding, return_when=FIRST_COMPLETED
                )
                for future in done:
                    record = future.result()
                    record_dict = asdict(record)
                    processed += 1

                    if record.success:
                        store.append_success(record_dict)
                        fresh_success.append(record)
                        consecutive_provider_failures = 0
                    else:
                        store.append_failure(record_dict)
                        fresh_failed.append(record)
                        logger.warning(
                            "[%d/%d] FAILED %s (%s): %s",
                            processed,
                            len(pending),
                            record.code_unit_id,
                            record.failure_category,
                            record.failure_reason,
                        )
                        if (
                            record.failure_category
                            in _PROVIDER_FAILURE_CATEGORIES
                        ):
                            consecutive_provider_failures += 1
                        else:
                            consecutive_provider_failures = 0

                    if (
                        not aborted
                        and consecutive_provider_failures
                        >= provider_failure_limit
                    ):
                        logger.error(
                            "Aborting after %d consecutive provider "
                            "failures; Ollama appears to be down. All "
                            "completed records are already persisted — "
                            "rerun to resume.",
                            consecutive_provider_failures,
                        )
                        aborted = True
                        exit_code = 3
                        for queued in outstanding:
                            queued.cancel()

                    if (
                        index := processed
                    ) % PROGRESS_LOG_INTERVAL == 0 or processed == len(
                        pending
                    ):
                        elapsed = time.perf_counter() - started
                        rate = index / elapsed if elapsed > 0 else 0.0
                        eta_minutes = (
                            ((len(pending) - index) / rate / 60)
                            if rate > 0
                            else 0.0
                        )
                        logger.info(
                            "[%d/%d] completed=%d ok=%d failed=%d "
                            "remaining=%d | %.1f units/min | ETA %.0f min",
                            index,
                            len(pending),
                            len(completed) + index,
                            len(resumed_success) + len(fresh_success),
                            len(resumed_failed) + len(fresh_failed),
                            len(pending) - index,
                            rate * 60,
                            eta_minutes,
                        )
    finally:
        store.close()
        adapter.close()

    successful = resumed_success + [asdict(r) for r in fresh_success]
    failed = resumed_failed + [asdict(r) for r in fresh_failed]
    wall_clock = time.perf_counter() - started

    metadata = build_metadata(
        successful,
        failed,
        total,
        wall_clock,
        model_runtime_metadata,
        resumed={
            "successful": len(resumed_success),
            "failed": len(resumed_failed),
        },
    )
    _atomic_write_json(output_dir / _METADATA_FILENAME, metadata)
    logger.info("Generation metadata written to %s", output_dir)

    if update_meta and exit_code == 0:
        update_dataset_metadata()

    all_records = successful + failed
    logger.info("=" * 60)
    logger.info("PHASE 4F-I-B CANDIDATE GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Test CodeUnits:        %d", total)
    logger.info("Attempted (this pass): %d", len(fresh_success) + len(fresh_failed))
    logger.info("Successful candidates: %d", len(successful))
    logger.info("Failed candidates:     %d", len(failed))
    logger.info("Total attempts:        %d",
                sum(r["attempt_count"] for r in all_records))
    logger.info("Wall clock (this pass): %.1f s", wall_clock)
    return exit_code


def build_metadata(
    successful: list[dict],
    failed: list[dict],
    total: int,
    wall_clock_seconds: float,
    model_runtime_metadata: dict,
    resumed: dict[str, int],
) -> dict:
    """Aggregate reproducibility and statistics metadata.

    ``successful``/``failed`` include records restored from the
    checkpoint so totals always describe the full candidate pool.
    """
    all_records = successful + failed
    total_attempts = sum(r["attempt_count"] for r in all_records)
    categories = Counter(r["failure_category"] for r in failed)
    styles = Counter(r["query_style"] for r in successful)
    validation_failures = sum(
        1
        for r in failed
        if r["failure_category"] in _VALIDATION_ERROR_CATEGORIES.values()
    )
    provider_failures = sum(
        1
        for r in failed
        if r["failure_category"] in _PROVIDER_FAILURE_CATEGORIES
    )
    retry_exhausted = sum(
        1
        for r in failed
        if r["attempt_count"] >= MAX_ATTEMPTS
        and r["failure_category"] != "leakage"
    )
    avg_gen_ms = (
        sum(r["generation_ms"] for r in successful) / len(successful)
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
            "resumed_from_checkpoint": resumed,
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
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing candidate artifacts in the output "
        "directory before generating (otherwise resume).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent inference requests (persistence stays "
        "serialized; record content is worker-count independent).",
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

    if args.fresh:
        for filename in (
            _CANDIDATES_FILENAME,
            _FAILURES_FILENAME,
            _METADATA_FILENAME,
        ):
            artifact = args.output_dir / filename
            if artifact.exists():
                artifact.unlink()
                logger.warning("Deleted existing artifact %s", artifact)

    return generate_all(
        test_units=test_units,
        output_dir=args.output_dir,
        provider_failure_limit=args.provider_failure_limit,
        update_meta=args.update_meta,
        workers=max(1, args.workers),
    )


if __name__ == "__main__":
    sys.exit(main())
