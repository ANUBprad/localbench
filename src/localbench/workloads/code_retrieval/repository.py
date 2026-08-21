"""Repository source configuration and Git-based acquisition.

Implements deterministic repository ingestion for Phase 4C.
Uses subprocess (no GitPython dependency).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

from localbench.workloads.code_retrieval.errors import (
    CheckoutFailedError,
    CloneFailedError,
    RevisionNotFoundError,
    WorkspaceError,
)

logger = logging.getLogger(__name__)

_DEFAULT_WORKSPACE = Path.home() / ".localbench" / "datasets"


class RepositorySource(BaseModel):
    """Configuration for a repository to ingest.

    Defines the stable identity and source details for reproducible
    acquisition.  The ``id`` is the dataset-level identifier
    (e.g. ``repo001``) that appears in CodeUnit and metadata records.
    """

    id: str
    url: str
    revision: str = "main"
    license: str = ""


class RepositorySnapshot(BaseModel):
    """Record of an acquired repository checkout.

    Produced by :class:`GitRepository` after successful acquisition.
    Contains everything needed to trace provenance and locate source
    files for future AST extraction.
    """

    repository_id: str
    url: str
    local_path: str
    commit: str
    reference: str
    content_hash: str = ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _run_git(
    args: list[str],
    cwd: str | Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a git command safely via subprocess argument list."""
    cmd = ["git"] + args
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
        timeout=120,
    )


def _compute_content_hash(directory: Path) -> str:
    """SHA-256 hash of sorted file paths + contents in *directory*.

    Excludes ``.git`` directories.  Reads file contents to detect
    changes even when file sizes remain the same.
    """
    hasher = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            rel = path.relative_to(directory)
            hasher.update(str(rel).encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


# ------------------------------------------------------------------
# GitRepository
# ------------------------------------------------------------------


class GitRepository:
    """Deterministic Git repository acquisition.

    Parameters
    ----------
    source:
        The repository source configuration.
    workspace:
        Root directory for cloned repositories.  Each repository is
        cloned into ``<workspace>/<source.id>``.
    """

    def __init__(
        self,
        source: RepositorySource,
        workspace: Path | str | None = None,
    ) -> None:
        self.source = source
        self.workspace = Path(workspace) if workspace else _DEFAULT_WORKSPACE

    @property
    def local_dir(self) -> Path:
        """Path where the repository is (or will be) cloned."""
        return self.workspace / self.source.id

    # -- public API ---------------------------------------------------

    def acquire(self) -> RepositorySnapshot:
        """Full acquisition pipeline: clone → resolve → checkout.

        Returns a :class:`RepositorySnapshot` with provenance metadata.
        """
        self._ensure_workspace()
        self._clone_or_update()
        commit = self._resolve_revision(self.source.revision)
        self._checkout(commit)
        content_hash = _compute_content_hash(self.local_dir)
        return RepositorySnapshot(
            repository_id=self.source.id,
            url=self.source.url,
            local_path=str(self.local_dir),
            commit=commit,
            reference=self.source.revision,
            content_hash=content_hash,
        )

    # -- internal steps -----------------------------------------------

    def _ensure_workspace(self) -> None:
        """Create the workspace root if it does not exist."""
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WorkspaceError(
                f"Cannot create workspace {self.workspace}: {exc}"
            ) from exc

    def _clone_or_update(self) -> None:
        """Clone if absent, otherwise fetch + reset."""
        if (self.local_dir / ".git").is_dir():
            self._fetch()
        else:
            self._clone()

    def _clone(self) -> None:
        """Clone the repository into local_dir."""
        if self.local_dir.exists():
            shutil.rmtree(self.local_dir)
        try:
            _run_git(["clone", self.source.url, str(self.local_dir)])
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise CloneFailedError(
                url=self.source.url,
                message=f"Clone failed: {exc}",
            ) from exc

    def _fetch(self) -> None:
        """Fetch latest refs in an existing clone."""
        try:
            _run_git(["fetch", "--all"], cwd=self.local_dir)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise CloneFailedError(
                url=self.source.url,
                message=f"Fetch failed: {exc}",
            ) from exc

    def _resolve_revision(self, reference: str) -> str:
        """Resolve *reference* to a full 40-char commit SHA.

        ``^{commit}`` peels annotated tags to their commit so the
        snapshot always records a checkout-able commit SHA.
        """
        try:
            result = _run_git(
                ["rev-parse", "--verify", f"{reference}^{{commit}}"],
                cwd=self.local_dir,
            )
            sha = result.stdout.strip()
            if len(sha) < 40:
                raise RevisionNotFoundError(reference, self.source.id)
            return sha
        except subprocess.CalledProcessError as exc:
            raise RevisionNotFoundError(
                reference, self.source.id
            ) from exc

    def _checkout(self, commit: str) -> None:
        """Checkout a specific commit (detached HEAD)."""
        try:
            _run_git(["checkout", commit], cwd=self.local_dir)
        except subprocess.CalledProcessError as exc:
            raise CheckoutFailedError(commit, self.source.id) from exc


# ------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------


def acquire_repository(
    source: RepositorySource,
    workspace: Path | str | None = None,
) -> RepositorySnapshot:
    """One-call convenience for repository acquisition."""
    repo = GitRepository(source, workspace=workspace)
    return repo.acquire()
