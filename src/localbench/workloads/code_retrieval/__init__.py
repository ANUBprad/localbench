"""Code semantic retrieval workload — dataset contracts and schemas."""

from localbench.workloads.code_retrieval.errors import (
    CheckoutFailedError,
    CloneFailedError,
    RepositoryError,
    RevisionNotFoundError,
    WorkspaceError,
)
from localbench.workloads.code_retrieval.extraction import (
    ExtractedCodeUnit,
    ExtractionResult,
    ParseError,
    SkippedFile,
    discover_python_files,
    extract_code_units,
)
from localbench.workloads.code_retrieval.query_generator import (
    LeakageCheckResult,
    QueryGenerationResult,
    QueryGenerator,
    check_query_leakage,
    generate_query,
)
from localbench.workloads.code_retrieval.query_prompt import (
    QUERY_PROMPT_TEMPLATE_VERSION,
    build_query_generation_prompt,
    get_query_system_prompt,
)
from localbench.workloads.code_retrieval.repository import (
    GitRepository,
    RepositorySnapshot,
    RepositorySource,
    acquire_repository,
)
from localbench.workloads.code_retrieval.schemas import (
    CandidateQuery,
    CodeUnit,
    CodeUnitContext,
    DatasetMetadata,
    QueryCase,
    QueryGenerationInput,
    QueryGenerationMetadata,
    QueryRelevance,
    SemanticLabel,
    SourceRepositorySnapshot,
)
from localbench.workloads.code_retrieval.semantic_generator import (
    LABEL_VERSION,
    SemanticLabelGenerator,
    SemanticLabelResult,
    generate_semantic_label,
)
from localbench.workloads.code_retrieval.semantic_prompt import (
    PROMPT_TEMPLATE_VERSION,
    build_semantic_label_prompt,
    get_system_prompt,
)

__all__ = [
    # Errors
    "CheckoutFailedError",
    "CloneFailedError",
    "RepositoryError",
    "RevisionNotFoundError",
    "WorkspaceError",
    # Extraction
    "ExtractedCodeUnit",
    "ExtractionResult",
    "ParseError",
    "SkippedFile",
    "discover_python_files",
    "extract_code_units",
    # Repository ingestion
    "GitRepository",
    "RepositorySnapshot",
    "RepositorySource",
    "acquire_repository",
    # Schemas
    "CandidateQuery",
    "CodeUnit",
    "CodeUnitContext",
    "DatasetMetadata",
    "QueryCase",
    "QueryGenerationInput",
    "QueryGenerationMetadata",
    "QueryRelevance",
    "SemanticLabel",
    "SourceRepositorySnapshot",
    # Semantic label generation
    "LABEL_VERSION",
    "PROMPT_TEMPLATE_VERSION",
    "SemanticLabelGenerator",
    "SemanticLabelResult",
    "build_semantic_label_prompt",
    "generate_semantic_label",
    "get_system_prompt",
    # Query generation
    "QUERY_PROMPT_TEMPLATE_VERSION",
    "LeakageCheckResult",
    "QueryGenerationResult",
    "QueryGenerator",
    "build_query_generation_prompt",
    "check_query_leakage",
    "generate_query",
    "get_query_system_prompt",
]
