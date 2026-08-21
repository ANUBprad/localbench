"""Phase 4F-H: Dataset ingestion and candidate query generation.

This script:
1. Clones 6 Python repositories at pinned commits
2. Extracts code units via AST parsing
3. Deduplicates by SHA-256 content hash
4. Creates repository-disjoint splits (train/val/test, seed=42)
5. Persists dataset artifacts to disk
6. Generates candidate queries for all test-split CodeUnits

Output:
  dataset/
    meta/version.json
    meta/stats.json
    repositories/{repo_id}/functions.jsonl
    repositories/{repo_id}/metadata.json
    splits/train.jsonl
    splits/validation.jsonl
    splits/test.jsonl
    queries/candidates.jsonl        # Candidate pool (not final 45)
    queries/candidate_failures.jsonl # Failed candidates (audit trail)
    queries/generation_metadata.json

Usage:
    python scripts/ingest_and_generate.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localbench.runtime.generation.policy import RetryPolicy
from localbench.runtime.ollama.adapter import OllamaAdapter
from localbench.workloads.code_retrieval.extraction import (
    ExtractedCodeUnit,
    ExtractionResult,
    extract_code_units,
)
from localbench.workloads.code_retrieval.query_generator import (
    QueryGenerator,
    check_query_leakage,
)
from localbench.workloads.code_retrieval.schemas import (
    DatasetMetadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
CLONE_ROOT = Path(__file__).resolve().parent.parent / ".clone_cache"
SPLIT_SEED = 42

# Repository selection: 6 well-known Python libraries, permissive license
REPOSITORIES = [
    {
        "id": "repo001",
        "name": "requests",
        "url": "https://github.com/psf/requests.git",
        "commit": "v2.31.0",
        "base_url": "https://github.com/psf/requests",
        "license": "Apache-2.0",
        "description": "HTTP library for Python",
        "split": "train",
    },
    {
        "id": "repo002",
        "name": "click",
        "url": "https://github.com/pallets/click.git",
        "commit": "8.1.7",
        "base_url": "https://github.com/pallets/click",
        "license": "BSD-3-Clause",
        "description": "CLI toolkit for Python",
        "split": "train",
    },
    {
        "id": "repo003",
        "name": "rich",
        "url": "https://github.com/Textualize/rich.git",
        "commit": "v13.7.0",
        "base_url": "https://github.com/Textualize/rich",
        "license": "MIT",
        "description": "Terminal formatting for Python",
        "split": "test",
    },
    {
        "id": "repo004",
        "name": "flask",
        "url": "https://github.com/pallets/flask.git",
        "commit": "3.0.3",
        "base_url": "https://github.com/pallets/flask",
        "license": "BSD-3-Clause",
        "description": "Web framework for Python",
        "split": "train",
    },
    {
        "id": "repo005",
        "name": "pydantic",
        "url": "https://github.com/pydantic/pydantic.git",
        "commit": "v2.6.4",
        "base_url": "https://github.com/pydantic/pydantic",
        "license": "MIT",
        "description": "Data validation for Python",
        "split": "validation",
    },
    {
        "id": "repo006",
        "name": "pytest",
        "url": "https://github.com/pytest-dev/pytest.git",
        "commit": "8.1.1",
        "base_url": "https://github.com/pytest-dev/pytest",
        "license": "MIT",
        "description": "Testing framework for Python",
        "split": "test",
    },
]

# Generation config (frozen)
MODEL_NAME = "qwen2.5-coder:7b"
MODEL_VERSION = "7b"
SEED = 42
TEMPERATURE = 0.7
TOP_P = 0.9
PROMPT_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Repository cloning
# ---------------------------------------------------------------------------


def clone_repository(repo: dict) -> Path:
    """Clone a repository at a specific commit. Returns local path."""
    local_path = CLONE_ROOT / repo["id"]

    if local_path.exists():
        logger.info("Repository %s already cloned at %s", repo["id"], local_path)
        return local_path

    logger.info("Cloning %s → %s", repo["name"], local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            repo["url"],
            str(local_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # Checkout specific commit/tag
    subprocess.run(
        ["git", "-C", str(local_path), "fetch", "--depth=1", "origin", repo["commit"]],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(local_path), "checkout", "FETCH_HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )

    # Record actual commit hash
    result = subprocess.run(
        ["git", "-C", str(local_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    actual_commit = result.stdout.strip()
    logger.info("  Commit: %s", actual_commit)

    return local_path


# ---------------------------------------------------------------------------
# Dataset persistence
# ---------------------------------------------------------------------------


def _unit_to_dict(unit: ExtractedCodeUnit) -> dict:
    """Convert ExtractedCodeUnit to a JSON-serializable dict."""
    return {
        "repository": unit.repository,
        "language": unit.language,
        "file_path": unit.file_path,
        "symbol": unit.symbol,
        "symbol_type": unit.symbol_type,
        "source_code": unit.source_code,
        "context": {
            "class_name": unit.context.class_name,
            "module_docstring": unit.context.module_docstring,
            "imports": unit.context.imports,
            "parent_methods": unit.context.parent_methods,
        },
        "source_url": unit.source_url,
        "is_public": unit.is_public,
        "docstring": unit.docstring,
        "source_file_lines": unit.source_file_lines,
        "content_hash": unit.content_hash,
        "extracted_at": unit.extracted_at,
    }


def persist_dataset(
    all_units: list[ExtractedCodeUnit],
    splits: dict[str, list[ExtractedCodeUnit]],
    repo_metadata: list[dict],
) -> None:
    """Write dataset artifacts to disk."""
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "meta").mkdir(exist_ok=True)
    (DATASET_ROOT / "repositories").mkdir(exist_ok=True)
    (DATASET_ROOT / "splits").mkdir(exist_ok=True)
    (DATASET_ROOT / "queries").mkdir(exist_ok=True)

    # Write repository metadata and code units
    for repo in repo_metadata:
        repo_dir = DATASET_ROOT / "repositories" / repo["id"]
        repo_dir.mkdir(exist_ok=True)

        # metadata.json
        with open(repo_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(repo, f, indent=2)

        # functions.jsonl — all units from this repo
        repo_units = [u for u in all_units if u.repository == repo["id"]]
        with open(repo_dir / "functions.jsonl", "w", encoding="utf-8") as f:
            for unit in repo_units:
                f.write(json.dumps(_unit_to_dict(unit), ensure_ascii=False) + "\n")

    # Write splits
    for split_name, units in splits.items():
        split_path = DATASET_ROOT / "splits" / f"{split_name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for unit in units:
                record = _unit_to_dict(unit)
                record["split"] = split_name
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write dataset metadata
    repos = sorted(set(u.repository for u in all_units))
    repo_commits = {r["id"]: r["commit"] for r in repo_metadata}

    metadata = DatasetMetadata(
        version="1.0.0",
        schema_version="1.0.0",
        release_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        repositories=repos,
        repository_commits=repo_commits,
        total_code_units=len(all_units),
        train_cases=len(splits["train"]),
        validation_cases=len(splits["validation"]),
        test_cases=len(splits["test"]),
        total_queries=0,  # Updated after generation
        split_seed=SPLIT_SEED,
        parser="python_ast",
        frozen=False,
    )
    with open(DATASET_ROOT / "meta" / "version.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(metadata.model_dump_json()), f, indent=2)

    # Write stats
    stats = {
        "code_units": {
            "total": len(all_units),
            "by_repository": dict(Counter(u.repository for u in all_units)),
            "by_split": {k: len(v) for k, v in splits.items()},
            "symbol_types": dict(Counter(u.symbol_type for u in all_units)),
        }
    }
    with open(DATASET_ROOT / "meta" / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Dataset persisted to %s", DATASET_ROOT)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def create_splits(
    units: list[ExtractedCodeUnit],
) -> dict[str, list[ExtractedCodeUnit]]:
    """Assign every CodeUnit to its repository's fixed split.

    Assignment is repository-level and deterministic: all units from a
    repository belong to that repository's single assigned split. The
    authoritative assignment is the ``split`` field in REPOSITORIES.
    Units must already be globally deduplicated so no content hash can
    land in two splits.
    """
    repo_split = {repo["id"]: repo["split"] for repo in REPOSITORIES}

    splits: dict[str, list[ExtractedCodeUnit]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for unit in units:
        splits[repo_split[unit.repository]].append(unit)

    for split_name, split_units in splits.items():
        logger.info("  %s: %d units", split_name, len(split_units))

    return splits


# ---------------------------------------------------------------------------
# Candidate query generation
# ---------------------------------------------------------------------------


@dataclass
class CandidateRecord:
    """Full audit trail for a generated candidate query."""

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
    attempt_number: int
    generation_ms: float
    validation_passed: bool
    leakage_passed: bool
    leakage_violations: list[str]
    success: bool
    failure_reason: str | None = None


def generate_candidates(
    test_units: list[ExtractedCodeUnit],
) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    """Generate candidate queries for all test CodeUnits.

    Returns (successful_candidates, failed_candidates).
    """
    adapter = OllamaAdapter(model_name=MODEL_NAME)
    policy = RetryPolicy(max_attempts=3)
    generator = QueryGenerator(model=adapter, policy=policy)

    successful: list[CandidateRecord] = []
    failed: list[CandidateRecord] = []

    total = len(test_units)
    logger.info("Generating candidates for %d test CodeUnits...", total)

    for i, unit in enumerate(test_units, 1):
        logger.info(
            "[%d/%d] %s",
            i,
            total,
            unit.symbol,
        )

        result = generator.generate(unit)

        # Build attempt metadata
        attempt_num = len(result.attempts)
        gen_ms = result.total_generation_ms

        if result.success and result.candidate is not None:
            # Run leakage check (already done in generator, but record it)
            leakage = check_query_leakage(result.candidate.query, unit)

            record = CandidateRecord(
                code_unit_id=_build_id(unit),
                candidate_id=f"candidate_{_build_id(unit)}",
                query=result.candidate.query,
                query_style=result.candidate.query_style,
                query_intent=result.candidate.query_intent,
                model=MODEL_NAME,
                model_version=MODEL_VERSION,
                prompt_version=PROMPT_VERSION,
                seed=SEED,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                attempt_number=attempt_num,
                generation_ms=gen_ms,
                validation_passed=True,
                leakage_passed=leakage.passed,
                leakage_violations=leakage.violations,
                success=True,
            )
            successful.append(record)

            if not leakage.passed:
                logger.warning(
                    "  Leakage detected: %s",
                    "; ".join(leakage.violations),
                )
        else:
            # Determine failure reason
            failure_reason = "unknown"
            if result.attempts:
                last_attempt = result.attempts[-1]
                if last_attempt.errors:
                    failure_reason = str(last_attempt.errors[0])
                elif not last_attempt.will_retry:
                    failure_reason = "validation_failed_no_retry"

            record = CandidateRecord(
                code_unit_id=_build_id(unit),
                candidate_id=f"candidate_{_build_id(unit)}",
                query="",
                query_style="",
                query_intent="",
                model=MODEL_NAME,
                model_version=MODEL_VERSION,
                prompt_version=PROMPT_VERSION,
                seed=SEED,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                attempt_number=attempt_num,
                generation_ms=gen_ms,
                validation_passed=False,
                leakage_passed=False,
                leakage_violations=[],
                success=False,
                failure_reason=failure_reason,
            )
            failed.append(record)

    adapter.close()
    return successful, failed


def _build_id(unit: ExtractedCodeUnit) -> str:
    """Build deterministic code unit ID."""
    normalized = unit.symbol.replace(".", "_")
    return f"{unit.repository}_py_{normalized}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=" * 60)
    logger.info("Phase 4F-H: Dataset Ingestion & Candidate Generation")
    logger.info("=" * 60)

    # Step 1: Clone repositories
    logger.info("\n--- Step 1: Clone repositories ---")
    repo_paths: dict[str, Path] = {}
    actual_commits: dict[str, str] = {}

    for repo in REPOSITORIES:
        local_path = clone_repository(repo)
        repo_paths[repo["id"]] = local_path

        # Get actual commit
        result = subprocess.run(
            ["git", "-C", str(local_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        actual_commits[repo["id"]] = result.stdout.strip()

    # Step 2: Extract code units
    logger.info("\n--- Step 2: Extract code units ---")
    all_units: list[ExtractedCodeUnit] = []
    extraction_results: dict[str, ExtractionResult] = {}

    for repo in REPOSITORIES:
        local_path = repo_paths[repo["id"]]
        logger.info("Extracting from %s (%s)...", repo["name"], local_path)

        result = extract_code_units(
            local_path=local_path,
            repository_id=repo["id"],
            commit=actual_commits[repo["id"]],
            base_url=repo["base_url"],
        )
        extraction_results[repo["id"]] = result
        all_units.extend(result.code_units)

        logger.info(
            "  Extracted: %d units, %d skipped, %d empty, %d errors",
            len(result.code_units),
            len(result.skipped_files),
            len(result.empty_files),
            len(result.parse_errors),
        )

    logger.info("\nTotal extracted: %d code units", len(all_units))

    # Step 3: Deduplicate
    logger.info("\n--- Step 3: Deduplicate ---")
    seen_hashes: set[str] = set()
    unique_units: list[ExtractedCodeUnit] = []
    duplicates = 0

    for unit in all_units:
        if unit.content_hash not in seen_hashes:
            seen_hashes.add(unit.content_hash)
            unique_units.append(unit)
        else:
            duplicates += 1

    logger.info(
        "Deduplicated: %d unique, %d duplicates removed",
        len(unique_units),
        duplicates,
    )

    # Step 4: Create splits
    logger.info("\n--- Step 4: Create splits ---")
    splits = create_splits(unique_units)

    logger.info(
        "Split totals: train=%d, validation=%d, test=%d",
        len(splits["train"]),
        len(splits["validation"]),
        len(splits["test"]),
    )

    # Step 5: Persist dataset
    logger.info("\n--- Step 5: Persist dataset ---")
    repo_metadata = []
    for repo in REPOSITORIES:
        repo_metadata.append(
            {
                "id": repo["id"],
                "name": repo["name"],
                "url": repo["url"],
                "commit": actual_commits[repo["id"]],
                "base_url": repo["base_url"],
                "license": repo["license"],
                "description": repo["description"],
            }
        )
    persist_dataset(unique_units, splits, repo_metadata)

    # Step 6: Generate candidate queries
    logger.info("\n--- Step 6: Generate candidate queries ---")
    test_units = splits["test"]
    logger.info("Test CodeUnits: %d", len(test_units))

    successful, failed = generate_candidates(test_units)

    # Step 7: Persist generation results
    logger.info("\n--- Step 7: Persist generation results ---")
    queries_dir = DATASET_ROOT / "queries"
    queries_dir.mkdir(exist_ok=True)

    # Write successful candidates
    with open(queries_dir / "candidates.jsonl", "w", encoding="utf-8") as f:
        for rec in successful:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    # Write failed candidates (audit trail)
    with open(queries_dir / "candidate_failures.jsonl", "w", encoding="utf-8") as f:
        for rec in failed:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    # Write generation metadata
    gen_metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "prompt_template_version": PROMPT_VERSION,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_code_units": len(test_units),
        "successful_candidates": len(successful),
        "failed_candidates": len(failed),
        "leakage_failures": sum(
            1 for r in successful if not r.leakage_passed
        ),
    }
    with open(queries_dir / "generation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(gen_metadata, f, indent=2)

    # Update dataset metadata with query count
    version_path = DATASET_ROOT / "meta" / "version.json"
    with open(version_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["total_queries"] = len(successful)
    with open(version_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4F-H COMPLETE")
    logger.info("=" * 60)
    logger.info("Test CodeUnits:      %d", len(test_units))
    logger.info("Successful candidates: %d", len(successful))
    logger.info("Failed candidates:   %d", len(failed))
    logger.info(
        "Leakage failures:    %d",
        sum(1 for r in successful if not r.leakage_passed),
    )
    total_attempts = sum(r.attempt_number for r in successful + failed)
    logger.info("Total attempts:      %d", total_attempts)
    logger.info("Dataset:             %s", DATASET_ROOT)


if __name__ == "__main__":
    main()
