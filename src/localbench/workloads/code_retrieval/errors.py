"""Ingestion-specific error types for repository acquisition.

Extends the base LocalBenchError hierarchy for Phase 4C.
"""

from __future__ import annotations


class RepositoryError(Exception):
    """Base exception for repository ingestion failures."""


class CloneFailedError(RepositoryError):
    """Repository clone or fetch failed."""

    def __init__(self, url: str, message: str | None = None) -> None:
        self.url = url
        super().__init__(
            message or f"Failed to clone repository: {url}"
        )


class RevisionNotFoundError(RepositoryError):
    """Requested revision (tag, branch, SHA) does not exist."""

    def __init__(self, revision: str, repository: str) -> None:
        self.revision = revision
        self.repository = repository
        super().__init__(
            f"Revision '{revision}' not found in repository '{repository}'"
        )


class CheckoutFailedError(RepositoryError):
    """Checkout of resolved revision failed."""

    def __init__(self, commit: str, repository: str) -> None:
        self.commit = commit
        self.repository = repository
        super().__init__(
            f"Failed to checkout revision '{commit}' "
            f"in repository '{repository}'"
        )


class WorkspaceError(RepositoryError):
    """Local workspace/directory operation failed."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Workspace operation failed")
