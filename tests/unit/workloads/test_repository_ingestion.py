"""Tests for repository ingestion — uses local temporary Git repos."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from localbench.workloads.code_retrieval.errors import (
    CloneFailedError,
    RevisionNotFoundError,
    WorkspaceError,
)
from localbench.workloads.code_retrieval.repository import (
    GitRepository,
    RepositorySnapshot,
    RepositorySource,
    _compute_content_hash,
    _run_git,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def local_git_repo(tmp_path: Path) -> Path:
    """Create a local bare git repository with two commits and a tag."""
    bare = tmp_path / "upstream.git"
    subprocess.run(
        ["git", "init", "--bare", str(bare)],
        check=True,
        capture_output=True,
    )

    # Create a working clone, add content, push
    work = tmp_path / "work"
    subprocess.run(
        ["git", "clone", str(bare), str(work)],
        check=True,
        capture_output=True,
    )

    # Commit 1
    (work / "hello.py").write_text("def greet():\n    return 'hello'\n")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=test@test.com", "-c", "user.name=Test",
         "commit", "-m", "initial"],
        cwd=work, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=work, check=True, capture_output=True,
    )

    # Tag v1.0
    subprocess.run(
        ["git", "tag", "v1.0"],
        cwd=work, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "push", "origin", "v1.0"],
        cwd=work, check=True, capture_output=True,
    )

    # Commit 2
    (work / "hello.py").write_text("def greet():\n    return 'hello world'\n")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=test@test.com", "-c", "user.name=Test",
         "commit", "-m", "update greeting"],
        cwd=work, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=work, check=True, capture_output=True,
    )

    return bare


def _commit_sha(bare: Path, ref: str) -> str:
    """Get the full SHA for a ref in a bare repo."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", ref],
        cwd=bare, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


# ------------------------------------------------------------------
# RepositorySource
# ------------------------------------------------------------------


class TestRepositorySource:
    def test_valid(self):
        src = RepositorySource(
            id="repo001",
            url="https://github.com/example/repo.git",
            revision="v1.0",
            license="MIT",
        )
        assert src.id == "repo001"
        assert src.revision == "v1.0"

    def test_default_revision(self):
        src = RepositorySource(id="r", url="u")
        assert src.revision == "main"

    def test_missing_id(self):
        with pytest.raises(ValidationError):
            RepositorySource(url="u")

    def test_missing_url(self):
        with pytest.raises(ValidationError):
            RepositorySource(id="r")


# ------------------------------------------------------------------
# GitRepository — acquire
# ------------------------------------------------------------------


class TestGitRepositoryAcquire:
    def test_acquire_returns_snapshot(
        self, local_git_repo: Path, tmp_path: Path
    ):
        workspace = tmp_path / "ws"
        src = RepositorySource(
            id="repo001",
            url=str(local_git_repo),
            revision="main",
        )
        repo = GitRepository(src, workspace=workspace)
        snap = repo.acquire()

        assert isinstance(snap, RepositorySnapshot)
        assert snap.repository_id == "repo001"
        assert snap.commit == _commit_sha(local_git_repo, "main")
        assert snap.reference == "main"
        assert Path(snap.local_path).is_dir()

    def test_acquire_with_tag(
        self, local_git_repo: Path, tmp_path: Path
    ):
        workspace = tmp_path / "ws"
        src = RepositorySource(
            id="repo002",
            url=str(local_git_repo),
            revision="v1.0",
        )
        repo = GitRepository(src, workspace=workspace)
        snap = repo.acquire()

        assert snap.commit == _commit_sha(local_git_repo, "v1.0")
        assert snap.reference == "v1.0"

    def test_acquire_with_explicit_sha(
        self, local_git_repo: Path, tmp_path: Path
    ):
        workspace = tmp_path / "ws"
        sha = _commit_sha(local_git_repo, "v1.0")
        src = RepositorySource(
            id="repo003",
            url=str(local_git_repo),
            revision=sha,
        )
        repo = GitRepository(src, workspace=workspace)
        snap = repo.acquire()
        assert snap.commit == sha

    def test_acquire_creates_workspace(
        self, local_git_repo: Path, tmp_path: Path
    ):
        workspace = tmp_path / "deep" / "nested" / "ws"
        src = RepositorySource(id="r", url=str(local_git_repo))
        repo = GitRepository(src, workspace=workspace)
        repo.acquire()
        assert workspace.is_dir()

    def test_content_hash_is_deterministic(
        self, local_git_repo: Path, tmp_path: Path
    ):
        workspace = tmp_path / "ws"
        src = RepositorySource(id="r", url=str(local_git_repo))
        repo = GitRepository(src, workspace=workspace)
        snap1 = repo.acquire()
        # Re-acquire updates existing clone
        snap2 = repo.acquire()
        assert snap1.content_hash == snap2.content_hash

    def test_existing_clone_fetches(
        self, local_git_repo: Path, tmp_path: Path
    ):
        workspace = tmp_path / "ws"
        src = RepositorySource(id="r", url=str(local_git_repo))
        repo = GitRepository(src, workspace=workspace)
        snap1 = repo.acquire()
        # Second acquire should fetch, not re-clone
        snap2 = repo.acquire()
        assert snap1.commit == snap2.commit


# ------------------------------------------------------------------
# GitRepository — failures
# ------------------------------------------------------------------


class TestGitRepositoryFailures:
    def test_nonexistent_repository(self, tmp_path: Path):
        src = RepositorySource(
            id="bad",
            url="/nonexistent/path.git",
            revision="main",
        )
        repo = GitRepository(src, workspace=tmp_path / "ws")
        with pytest.raises(CloneFailedError):
            repo.acquire()

    def test_invalid_revision(
        self, local_git_repo: Path, tmp_path: Path
    ):
        src = RepositorySource(
            id="r",
            url=str(local_git_repo),
            revision="nonexistent-tag",
        )
        repo = GitRepository(src, workspace=tmp_path / "ws")
        with pytest.raises(RevisionNotFoundError) as exc_info:
            repo.acquire()
        assert exc_info.value.revision == "nonexistent-tag"

    def test_workspace_creation_failure(
        self, local_git_repo: Path, tmp_path: Path
    ):
        # Create a file where the workspace directory should be
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory")
        src = RepositorySource(id="r", url=str(local_git_repo))
        repo = GitRepository(src, workspace=blocked / "sub")
        with pytest.raises(WorkspaceError):
            repo.acquire()


# ------------------------------------------------------------------
# _compute_content_hash
# ------------------------------------------------------------------


class TestContentHash:
    def test_hash_deterministic(self, tmp_path: Path):
        d = tmp_path / "repo"
        d.mkdir()
        (d / "a.py").write_text("x = 1")
        h1 = _compute_content_hash(d)
        h2 = _compute_content_hash(d)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_differs_on_content_change(self, tmp_path: Path):
        d = tmp_path / "repo"
        d.mkdir()
        (d / "a.py").write_text("x = 1")
        h1 = _compute_content_hash(d)
        (d / "a.py").write_text("x = 2")
        h2 = _compute_content_hash(d)
        assert h1 != h2

    def test_hash_empty_dir(self, tmp_path: Path):
        d = tmp_path / "empty"
        d.mkdir()
        h = _compute_content_hash(d)
        assert len(h) == 64


# ------------------------------------------------------------------
# _run_git helper
# ------------------------------------------------------------------


class TestRunGit:
    def test_invalid_command(self):
        with pytest.raises(subprocess.CalledProcessError):
            _run_git(["--version-pypoetry-pypi-pkg-naming-error"])

    def test_cwd_parameter(self, local_git_repo: Path):
        result = _run_git(["rev-parse", "--verify", "HEAD"], cwd=local_git_repo)
        assert len(result.stdout.strip()) == 40
