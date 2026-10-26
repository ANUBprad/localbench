"""Verify public API exports from the code_retrieval package."""

from localbench.workloads.code_retrieval import (
    LABEL_VERSION,
    PROMPT_TEMPLATE_VERSION,
    QUERY_PROMPT_TEMPLATE_VERSION,
    CandidateQuery,
    CheckoutFailedError,
    CloneFailedError,
    CodeUnit,
    CodeUnitContext,
    DatasetMetadata,
    ExtractedCodeUnit,
    ExtractionResult,
    GitRepository,
    LeakageCheckResult,
    ParseError,
    QueryCase,
    QueryGenerationInput,
    QueryGenerationMetadata,
    QueryGenerationResult,
    QueryGenerator,
    QueryRelevance,
    RepositoryError,
    RepositorySnapshot,
    RepositorySource,
    RevisionNotFoundError,
    SemanticLabel,
    SemanticLabelGenerator,
    SemanticLabelResult,
    SkippedFile,
    SourceRepositorySnapshot,
    WorkspaceError,
    acquire_repository,
    build_query_generation_prompt,
    build_semantic_label_prompt,
    check_query_leakage,
    discover_python_files,
    extract_code_units,
    generate_query,
    generate_semantic_label,
    get_query_system_prompt,
    get_system_prompt,
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

    def test_extraction_classes_exported(self):
        assert ExtractedCodeUnit is not None
        assert ExtractionResult is not None
        assert ParseError is not None
        assert SkippedFile is not None
        assert callable(extract_code_units)
        assert callable(discover_python_files)

    def test_semantic_generation_exported(self):
        assert SemanticLabelGenerator is not None
        assert SemanticLabelResult is not None
        assert callable(generate_semantic_label)
        assert callable(build_semantic_label_prompt)
        assert callable(get_system_prompt)
        assert LABEL_VERSION == "1.0.0"
        assert PROMPT_TEMPLATE_VERSION == "1.0.0"

    def test_query_generation_exported(self):
        assert CandidateQuery is not None
        assert QueryGenerationInput is not None
        assert QueryGenerationResult is not None
        assert QueryGenerator is not None
        assert LeakageCheckResult is not None
        assert callable(build_query_generation_prompt)
        assert callable(get_query_system_prompt)
        assert callable(check_query_leakage)
        assert callable(generate_query)
        assert QUERY_PROMPT_TEMPLATE_VERSION == "3.2.0"

    def test_all_matches_actual_exports(self):
        import localbench.workloads.code_retrieval as pkg

        for name in pkg.__all__:
            assert hasattr(pkg, name), f"__all__ lists '{name}' but not exported"
