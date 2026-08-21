"""Verify repository-disjoint split assignment (Phase 4F-I-A2)."""

import importlib.util
import sys
from pathlib import Path

from localbench.workloads.code_retrieval.extraction import ExtractedCodeUnit
from localbench.workloads.code_retrieval.schemas import CodeUnitContext

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "ingest_and_generate.py"
)
_script_spec = importlib.util.spec_from_file_location(
    "localbench_ingest_and_generate_script", _SCRIPT_PATH
)
ingest = importlib.util.module_from_spec(_script_spec)
sys.modules[_script_spec.name] = ingest
_script_spec.loader.exec_module(ingest)

FROZEN_ASSIGNMENT = {
    "repo001": "train",
    "repo002": "train",
    "repo003": "test",
    "repo004": "train",
    "repo005": "validation",
    "repo006": "test",
}


def make_unit(repository: str, symbol: str, content_hash: str) -> ExtractedCodeUnit:
    return ExtractedCodeUnit(
        repository=repository,
        language="python",
        file_path=f"src/{symbol}.py",
        symbol=symbol,
        symbol_type="function",
        source_code="def f():\n    return 1\n",
        context=CodeUnitContext(),
        source_url=f"https://github.com/example/repo/blob/abc/src/{symbol}.py",
        is_public=True,
        docstring="",
        source_file_lines=2,
        content_hash=content_hash,
        extracted_at="2026-01-01T00:00:00+00:00",
    )


class TestManifestSplitField:
    def test_every_repository_has_exactly_one_split(self):
        assert len(ingest.REPOSITORIES) == 6
        for repo in ingest.REPOSITORIES:
            assert repo["split"] in {"train", "validation", "test"}

    def test_frozen_assignment_matches_audit(self):
        actual = {repo["id"]: repo["split"] for repo in ingest.REPOSITORIES}
        assert actual == FROZEN_ASSIGNMENT


class TestRepositoryAssignment:
    def test_each_repository_lands_in_its_assigned_split(self):
        units = [
            make_unit(repo_id, f"sym_{repo_id}", f"hash-{repo_id}")
            for repo_id in FROZEN_ASSIGNMENT
        ]
        splits = ingest.create_splits(units)
        for split_name, split_units in splits.items():
            for unit in split_units:
                assert FROZEN_ASSIGNMENT[unit.repository] == split_name


class TestDisjointness:
    def test_repository_sets_are_pairwise_disjoint(self):
        repos_by_split = {}
        for repo in ingest.REPOSITORIES:
            repos_by_split.setdefault(repo["split"], set()).add(repo["id"])

        split_names = ["train", "validation", "test"]
        for i, first in enumerate(split_names):
            for second in split_names[i + 1 :]:
                assert repos_by_split[first] & repos_by_split[second] == set()


class TestCodeUnitInheritance:
    def test_all_units_of_a_repository_share_one_split(self):
        units = []
        counter = 0
        for repo_id in FROZEN_ASSIGNMENT:
            for _ in range(3):
                counter += 1
                units.append(make_unit(repo_id, f"sym{counter}", f"hash{counter:04d}"))

        splits = ingest.create_splits(units)
        splits_by_repo = {}
        for split_name, split_units in splits.items():
            for unit in split_units:
                splits_by_repo.setdefault(unit.repository, set()).add(split_name)

        for repo_id, assigned in splits_by_repo.items():
            assert assigned == {FROZEN_ASSIGNMENT[repo_id]}, repo_id


class TestNoRatioSplitting:
    def test_repository_with_n_units_keeps_all_n_in_one_split(self):
        units = [
            make_unit("repo005", f"sym{i}", f"hash{i:04d}") for i in range(10)
        ]
        splits = ingest.create_splits(units)
        assert len(splits["validation"]) == 10
        assert sum(len(splits[name]) for name in ("train", "test")) == 0


class TestDuplicateSafety:
    def test_globally_deduplicated_hash_cannot_span_two_splits(self):
        duplicated = make_unit("repo001", "shared", "hash-dup")
        same_content_elsewhere = make_unit("repo003", "shared", "hash-dup")

        deduped = list(
            {u.content_hash: u for u in [duplicated, same_content_elsewhere]}.values()
        )
        splits = ingest.create_splits(deduped)

        splits_by_hash = {}
        for split_name, split_units in splits.items():
            for unit in split_units:
                splits_by_hash.setdefault(unit.content_hash, set()).add(split_name)

        for content_hash, assigned in splits_by_hash.items():
            assert len(assigned) == 1, content_hash


class TestDeterminism:
    def test_same_input_produces_identical_assignment(self):
        units = [
            make_unit(repo_id, f"sym{i}", f"hash-{repo_id}-{i}")
            for repo_id in FROZEN_ASSIGNMENT
            for i in range(2)
        ]
        first = ingest.create_splits(list(units))
        second = ingest.create_splits(list(units))
        assert [
            [u.content_hash for u in first[name]]
            for name in ("train", "validation", "test")
        ] == [
            [u.content_hash for u in second[name]]
            for name in ("train", "validation", "test")
        ]


class TestSeedPreserved:
    def test_canonical_seed_remains_42(self):
        assert ingest.SPLIT_SEED == 42
