"""Phase 4F-I-C8 Remediation: regenerate human-rejected queries.

Regenerates candidates for two CodeUnits rejected during human review.
Uses the frozen regeneration infrastructure (QueryGenerator,
CandidateStore, leakage detection, bounded retry).

Writes new candidates to candidates_v2.jsonl and failures to
candidate_failures_v2.jsonl.

DO NOT modify: candidates.jsonl, candidate_failures.jsonl, test.jsonl
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

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
from localbench.workloads.code_retrieval.run_lock import (
    GenerationLockError,
    generation_run_lock,
)
from localbench.workloads.code_retrieval.schemas import CodeUnitContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
TEST_SPLIT_PATH = DATASET_ROOT / "splits" / "test.jsonl"
CANDIDATES_PATH = DATASET_ROOT / "queries" / "candidates.jsonl"
FAILURES_PATH = DATASET_ROOT / "queries" / "candidate_failures.jsonl"
V2_CANDIDATES_PATH = DATASET_ROOT / "queries" / "candidates_v2.jsonl"
V2_FAILURES_PATH = DATASET_ROOT / "queries" / "candidate_failures_v2.jsonl"

MODEL_NAME = "qwen2.5-coder:7b"
MODEL_VERSION = "7b"
SEED = 42
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 128
MAX_ATTEMPTS = DEFAULT_MAX_ATTEMPTS
V2_CANDIDATE_PREFIX = "candidate_v2_"

# The two rejected code unit IDs
# Item 9 is EXHAUSTED (3 prior attempts, 0 remaining) — must be excluded from pool.
# Item 35 has 1 remaining attempt — regenerate.
REJECTED_IDS = [
    "repo006_py_testing_python_metafunc_py__TestMetafuncFunctional_test_two_functions_a584c3632477",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _unit_id(unit: ExtractedCodeUnit) -> str:
    return _build_code_unit_id(
        unit.repository, unit.file_path, unit.symbol, unit.content_hash
    )


def _load_test_units() -> dict[str, ExtractedCodeUnit]:
    units = {}
    with open(TEST_SPLIT_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            eu = ExtractedCodeUnit(
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
            units[_unit_id(eu)] = eu
    return units


def _count_prior_attempts(
    code_unit_id: str,
    candidates: list[dict],
    failures: list[dict],
) -> int:
    count = 0
    for rec in candidates:
        if rec["code_unit_id"] == code_unit_id:
            count += rec.get("attempt_count", 1)
    for rec in failures:
        if rec["code_unit_id"] == code_unit_id:
            count += rec.get("attempt_count", 1)
    return count


def _serialize_attempt(attempt: AttemptRecord) -> dict:
    return {
        "attempt_number": attempt.attempt_number,
        "status": attempt.status,
        "will_retry": attempt.will_retry,
        "generation_ms": attempt.generation_ms,
        "errors": [str(error) for error in attempt.errors],
        "raw_text": attempt.raw_text,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def remediate() -> int:
    logger.info("=" * 60)
    logger.info("Phase 4F-I-C8 Remediation: Two Rejected Queries")
    logger.info("=" * 60)

    test_units = _load_test_units()
    candidates = _load_jsonl(CANDIDATES_PATH)
    failures = _load_jsonl(FAILURES_PATH)
    v2_candidates = _load_jsonl(V2_CANDIDATES_PATH)
    v2_failures = _load_jsonl(V2_FAILURES_PATH)
    all_candidates = candidates + v2_candidates
    all_failures = failures + v2_failures

    # Budget check
    budget_info = {}
    for uid in REJECTED_IDS:
        prior = _count_prior_attempts(uid, all_candidates, all_failures)
        remaining = MAX_ATTEMPTS - prior
        budget_info[uid] = {"prior": prior, "remaining": remaining}
        logger.info(
            "  %s: prior=%d, remaining=%d", uid, prior, remaining
        )

    regenerable = [uid for uid, info in budget_info.items() if info["remaining"] > 0]
    exhausted = [uid for uid, info in budget_info.items() if info["remaining"] <= 0]

    if exhausted:
        for uid in exhausted:
            logger.error(
                "  EXHAUSTED: %s (prior=%d, remaining=0)",
                uid, budget_info[uid]["prior"],
            )
        logger.error("Cannot proceed: some items exhausted.")
        return 1

    logger.info("All %d items regenerable.", len(regenerable))

    adapter = OllamaAdapter(model_name=MODEL_NAME)
    if not adapter.health_check():
        logger.error("Ollama is not reachable at localhost:11434")
        return 2

    store = CandidateStore(V2_CANDIDATES_PATH, V2_FAILURES_PATH)
    store.load()

    new_success = 0
    new_failed = 0

    try:
        for uid in regenerable:
            eu = test_units.get(uid)
            if eu is None:
                logger.warning("SKIP: %s not found in test split", uid)
                continue

            remaining = budget_info[uid]["remaining"]
            logger.info("Regenerating %s (remaining=%d)...", uid, remaining)

            policy = RetryPolicy(max_attempts=remaining)
            generator = QueryGenerator(
                model=adapter, policy=policy, top_p=TOP_P, seed=SEED,
            )
            result = generator.generate(eu)
            success = result.success and result.candidate is not None

            if success:
                record = {
                    "code_unit_id": uid,
                    "candidate_id": f"{V2_CANDIDATE_PREFIX}{uid}",
                    "query": result.candidate.query,
                    "query_style": result.candidate.query_style,
                    "query_intent": result.candidate.query_intent,
                    "model": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "prompt_version": QUERY_PROMPT_TEMPLATE_VERSION,
                    "seed": SEED,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_tokens": MAX_TOKENS,
                    "attempt_count": len(result.attempts),
                    "attempts": [_serialize_attempt(a) for a in result.attempts],
                    "generation_ms": result.total_generation_ms,
                    "validation_ms": result.total_validation_ms,
                    "validation_passed": bool(result.attempts)
                    and result.attempts[-1].status == AttemptStatus.SUCCESS,
                    "leakage_passed": bool(result.leakage and result.leakage.passed),
                    "leakage_violations": (
                        result.leakage.violations if result.leakage else []
                    ),
                    "success": True,
                    "failure_category": None,
                    "failure_reason": None,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
                store.append_success(record)
                new_success += 1
                logger.info(
                    "  OK: %s (query=%s)",
                    uid, result.candidate.query,
                )
            else:
                category = "unknown"
                reason = "Generation failed"
                if result.leakage is not None and not result.leakage.passed:
                    category = "leakage"
                    reason = "; ".join(result.leakage.violations)
                elif result.attempts:
                    last = result.attempts[-1]
                    if last.errors:
                        reason = str(last.errors[0])

                record = {
                    "code_unit_id": uid,
                    "candidate_id": f"{V2_CANDIDATE_PREFIX}{uid}",
                    "query": "",
                    "query_style": "",
                    "query_intent": "",
                    "model": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "prompt_version": QUERY_PROMPT_TEMPLATE_VERSION,
                    "seed": SEED,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_tokens": MAX_TOKENS,
                    "attempt_count": len(result.attempts),
                    "attempts": [_serialize_attempt(a) for a in result.attempts],
                    "generation_ms": result.total_generation_ms,
                    "validation_ms": result.total_validation_ms,
                    "validation_passed": False,
                    "leakage_passed": False,
                    "leakage_violations": (
                        result.leakage.violations if result.leakage else []
                    ),
                    "success": False,
                    "failure_category": category,
                    "failure_reason": reason,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
                store.append_failure(record)
                new_failed += 1
                logger.warning("  FAILED: %s (%s): %s", uid, category, reason)

    finally:
        store.close()
        adapter.close()

    logger.info("=" * 60)
    logger.info("REMEDIATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Regenerable attempted: %d", len(regenerable))
    logger.info("New successful: %d", new_success)
    logger.info("New failed: %d", new_failed)

    return 0


if __name__ == "__main__":
    try:
        with generation_run_lock(DATASET_ROOT / "queries"):
            sys.exit(remediate())
    except GenerationLockError as exc:
        logger.error("%s", exc)
        sys.exit(4)
