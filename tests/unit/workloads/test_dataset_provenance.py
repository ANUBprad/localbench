"""Verify dataset provenance metadata (Phase 4F-I-A3)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from localbench.workloads.code_retrieval.schemas import DatasetMetadata

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "ingest_and_generate.py"
)
_script_spec = importlib.util.spec_from_file_location(
    "localbench_ingest_and_generate_provenance_script", _SCRIPT_PATH
)
ingest = importlib.util.module_from_spec(_script_spec)
sys.modules[_script_spec.name] = ingest
_script_spec.loader.exec_module(ingest)

FROZEN_COMMITS = {
    "repo001": "v2.31.0",
    "repo002": "8.1.7",
    "repo003": "v13.7.0",
    "repo004": "3.0.3",
    "repo005": "v2.6.4",
    "repo006": "8.1.1",
}

FROZEN_SPLITS = {
    "repo001": "train",
    "repo002": "train",
    "repo003": "test",
    "repo004": "train",
    "repo005": "validation",
    "repo006": "test",
}


def make_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        version="1.0.0",
        schema_version="1.0.0",
        repositories=sorted(FROZEN_SPLITS),
        repository_commits=dict(FROZEN_COMMITS),
        repository_splits=dict(FROZEN_SPLITS),
        manifest_hash="a" * 64,
        total_code_units=450,
        extracted_code_units=500,
        duplicate_code_units=50,
        train_cases=225,
        validation_cases=112,
        test_cases=113,
        extraction_version="1.0.0",
        deduplication_method="sha256_of_stripped_source_utf8",
        eligibility_rules={"min_source_lines": 3, "max_source_lines": 100},
    )


class TestRequiredFields:
    def test_provenance_fields_present(self):
        meta = make_metadata()
        assert meta.extraction_version == "1.0.0"
        assert meta.deduplication_method == "sha256_of_stripped_source_utf8"
        assert meta.eligibility_rules == {
            "min_source_lines": 3,
            "max_source_lines": 100,
        }
        assert meta.manifest_hash == "a" * 64
        assert meta.parser == "python_ast"

    def test_defaults_keep_legacy_metadata_valid(self):
        meta = DatasetMetadata(version="1.0.0", schema_version="1.0.0")
        assert meta.extraction_version == ""
        assert meta.deduplication_method == ""
        assert meta.eligibility_rules == {}
        assert meta.manifest_hash == ""
        assert meta.extracted_code_units == 0
        assert meta.duplicate_code_units == 0
        assert meta.repository_splits == {}


class TestSeedPreserved:
    def test_split_seed_defaults_to_42(self):
        assert DatasetMetadata(version="1.0.0", schema_version="1.0.0").split_seed == 42
        assert make_metadata().split_seed == 42


class TestRepositoryProvenance:
    def test_revisions_preserved(self):
        meta = make_metadata()
        assert meta.repository_commits == FROZEN_COMMITS
        assert meta.repositories == sorted(FROZEN_SPLITS)

    def test_manifest_matches_frozen_revisions(self):
        actual = {r["id"]: r["commit"] for r in ingest.REPOSITORIES}
        assert actual == FROZEN_COMMITS

    def test_split_assignments_preserved(self):
        assert make_metadata().repository_splits == FROZEN_SPLITS

    def test_invalid_split_value_rejected(self):
        with pytest.raises(ValidationError):
            DatasetMetadata(
                version="1.0.0",
                schema_version="1.0.0",
                repository_splits={"repo001": "dev"},
            )


class TestCountConsistency:
    def test_consistent_counts_accepted(self):
        meta = make_metadata()
        assert meta.total_code_units == 450
        assert meta.train_cases + meta.validation_cases + meta.test_cases == 450

    def test_duplicate_relationship_enforced(self):
        with pytest.raises(ValidationError, match="duplicate_code_units"):
            DatasetMetadata(
                version="1.0.0",
                schema_version="1.0.0",
                total_code_units=450,
                extracted_code_units=500,
                duplicate_code_units=10,
                train_cases=225,
                validation_cases=112,
                test_cases=113,
            )

    def test_unrecorded_extraction_counts_skip_check(self):
        meta = DatasetMetadata(
            version="1.0.0",
            schema_version="1.0.0",
            total_code_units=450,
            train_cases=225,
            validation_cases=112,
            test_cases=113,
        )
        assert meta.total_code_units == 450


class TestSerializationDeterminism:
    def test_serialization_is_deterministic(self):
        first = make_metadata().model_dump_json()
        second = make_metadata().model_dump_json()
        assert first == second

    def test_roundtrip_preserves_provenance(self):
        meta = make_metadata()
        restored = DatasetMetadata.model_validate_json(meta.model_dump_json())
        assert restored == meta


class TestQueryGenerationUntouched:
    def test_query_generation_not_populated_at_build(self):
        assert make_metadata().query_generation is None


class TestManifestIdentity:
    def test_manifest_hash_is_deterministic(self):
        assert ingest.compute_manifest_hash() == ingest.compute_manifest_hash()

    def test_manifest_hash_matches_canonical_recipe(self):
        canonical = json.dumps(ingest.REPOSITORIES, sort_keys=True)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert ingest.compute_manifest_hash() == expected

    def test_manifest_hash_covers_split_assignments(self):
        import copy

        mutated = copy.deepcopy(ingest.REPOSITORIES)
        mutated[0]["split"] = "test" if mutated[0]["split"] != "test" else "train"
        canonical = json.dumps(mutated, sort_keys=True)
        mutated_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert mutated_hash != ingest.compute_manifest_hash()
