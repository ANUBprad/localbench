"""Tests for repository ingestion error types."""

from localbench.workloads.code_retrieval.errors import (
    CheckoutFailedError,
    CloneFailedError,
    RepositoryError,
    RevisionNotFoundError,
    WorkspaceError,
)


class TestRepositoryErrorHierarchy:
    def test_clone_is_repository_error(self):
        assert issubclass(CloneFailedError, RepositoryError)

    def test_revision_not_found_is_repository_error(self):
        assert issubclass(RevisionNotFoundError, RepositoryError)

    def test_checkout_failed_is_repository_error(self):
        assert issubclass(CheckoutFailedError, RepositoryError)

    def test_workspace_is_repository_error(self):
        assert issubclass(WorkspaceError, RepositoryError)

    def test_all_are_exception(self):
        for cls in (
            RepositoryError,
            CloneFailedError,
            RevisionNotFoundError,
            CheckoutFailedError,
            WorkspaceError,
        ):
            assert issubclass(cls, Exception)


class TestCloneFailedError:
    def test_message_default(self):
        err = CloneFailedError(url="https://github.com/example/repo.git")
        assert "https://github.com/example/repo.git" in str(err)
        assert err.url == "https://github.com/example/repo.git"

    def test_message_custom(self):
        err = CloneFailedError(
            url="https://example.com/repo.git",
            message="Network timeout",
        )
        assert str(err) == "Network timeout"
        assert err.url == "https://example.com/repo.git"


class TestRevisionNotFoundError:
    def test_attributes(self):
        err = RevisionNotFoundError(revision="v1.0", repository="repo001")
        assert err.revision == "v1.0"
        assert err.repository == "repo001"
        assert "v1.0" in str(err)
        assert "repo001" in str(err)


class TestCheckoutFailedError:
    def test_attributes(self):
        err = CheckoutFailedError(commit="abc1234", repository="repo001")
        assert err.commit == "abc1234"
        assert err.repository == "repo001"
        assert "abc1234" in str(err)
        assert "repo001" in str(err)


class TestWorkspaceError:
    def test_message_default(self):
        err = WorkspaceError()
        assert "Workspace" in str(err)

    def test_message_custom(self):
        err = WorkspaceError(message="Disk full")
        assert str(err) == "Disk full"
