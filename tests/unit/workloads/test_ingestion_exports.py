"""Verify public API exports from the code_retrieval package."""

from localbench.workloads.code_retrieval import (
    CheckoutFailedError,
    CloneFailedError,
    CodeUnit,
    CodeUnitContext,
    DatasetMetadata,
    GitRepository,
    QueryCase,
    QueryGenerationMetadata,
    QueryRelevance,
    RepositoryError,
    RepositorySnapshot,
    RepositorySource,
    RevisionNotFoundError,
    SemanticLabel,
    SourceRepositorySnapshot,
    WorkspaceError,
    acquire_repository,
)


class TestPackageExports:
    def test_error_types_exported(self):
        assert issubclass(CloneFailedError, RepositoryError)
        assert issubclass(RevisionNotFoundError, RepositoryError)
        assert issubclass(CheckoutFailedError, RepositoryError)
        assert issubclass(WorkspaceError, RepositoryError)

    def test_repository_classes_exported(self):
        assert GitRepository is not None
        assert RepositorySource is not None
        assert RepositorySnapshot is not None
        assert callable(acquire_repository)

    def test_schema_classes_exported(self):
        assert CodeUnit is not None
        assert CodeUnitContext is not None
        assert DatasetMetadata is not None
        assert QueryCase is not None
        assert QueryGenerationMetadata is not None
        assert QueryRelevance is not None
        assert SemanticLabel is not None
        assert SourceRepositorySnapshot is not None

    def test_all_matches_actual_exports(self):
        import localbench.workloads.code_retrieval as pkg

        for name in pkg.__all__:
            assert hasattr(pkg, name), f"__all__ lists '{name}' but not exported"
