"""Code semantic retrieval workload — dataset contracts and schemas."""

from localbench.workloads.code_retrieval.errors import (
    CheckoutFailedError,
    CloneFailedError,
    RepositoryError,
    RevisionNotFoundError,
    WorkspaceError,
)
from localbench.workloads.code_retrieval.repository import (
    GitRepository,
    RepositorySnapshot,
    RepositorySource,
    acquire_repository,
)
from localbench.workloads.code_retrieval.schemas import (
    CodeUnit,
    CodeUnitContext,
    DatasetMetadata,
    QueryCase,
    QueryGenerationMetadata,
    QueryRelevance,
    SemanticLabel,
    SourceRepositorySnapshot,
)

__all__ = [
    # Errors
    "CheckoutFailedError",
    "CloneFailedError",
    "RepositoryError",
    "RevisionNotFoundError",
    "WorkspaceError",
    # Repository ingestion
    "GitRepository",
    "RepositorySnapshot",
    "RepositorySource",
    "acquire_repository",
    # Schemas
    "CodeUnit",
    "CodeUnitContext",
    "DatasetMetadata",
    "QueryCase",
    "QueryGenerationMetadata",
    "QueryRelevance",
    "SemanticLabel",
    "SourceRepositorySnapshot",
]
