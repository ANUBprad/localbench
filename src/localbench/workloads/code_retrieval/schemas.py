"""Canonical dataset contracts for the code semantic retrieval workload.

All models are derived directly from DATASET_SPECIFICATION.md §3 and §5.
No speculative fields are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Type aliases (frozen categories per spec)
# ---------------------------------------------------------------------------

SplitType = Literal["train", "validation", "test"]
"""Dataset split assignment (§3.1)."""

SymbolType = Literal["function", "method"]
"""Code-unit symbol kind (§3.1)."""

Language = Literal["python"]
"""Supported language for v1."""

RelevanceLabel = Literal[
    "direct_match", "highly_relevant", "related", "not_relevant"
]
"""Ground-truth relevance categories (§3.4)."""

CreatedBy = Literal["human", "model_generated", "hybrid"]
"""Semantic-label creator (§3.2)."""

QueryStyle = Literal["natural", "technical", "verbose", "concise"]
"""Query presentation style (§3.3)."""

Difficulty = Literal["easy", "medium", "hard"]
"""Query difficulty level (§3.3)."""


# ---------------------------------------------------------------------------
# Query generation input contract (§4.4 — source-only, no SemanticLabels)
# ---------------------------------------------------------------------------


@dataclass
class QueryGenerationInput:
    """Source-only input to the query prompt builder.

    Contains only the information the query generator is permitted to see.
    Deliberately excludes: repository ID, file path, symbol path, source URL,
    content hash, and SemanticLabel fields.

    The prompt builder consumes ONLY this type, making accidental
    SemanticLabel leakage structurally impossible.
    """

    source_code: str
    docstring: str = ""
    symbol_type: Literal["function", "method"] = "function"
    class_name: str | None = None
    module_docstring: str | None = None
    imports: list[str] = dc_field(default_factory=list)
    parent_methods: list[str] = dc_field(default_factory=list)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIN_SOURCE_LINES = 3
_MAX_SOURCE_LINES = 100
_MIN_DESCRIPTION_WORDS = 20
_MAX_DESCRIPTION_WORDS = 256
_MIN_CONCEPTS = 2


def _count_source_lines(source: str) -> int:
    """Count non-empty lines in source code (v1 approximation)."""
    return len([line for line in source.splitlines() if line.strip()])


def _word_count(text: str) -> int:
    """Count whitespace-delimited words."""
    return len(text.split())


# ---------------------------------------------------------------------------
# §3.1  Code Unit
# ---------------------------------------------------------------------------


class CodeUnitContext(BaseModel):
    """Surrounding class/module context for a code unit (§3.1)."""

    class_name: str | None = None
    module_docstring: str | None = None
    imports: list[str] = Field(default_factory=list)
    parent_methods: list[str] = Field(default_factory=list)


class CodeUnit(BaseModel):
    """A single function or method with context — the benchmarkable unit.

    Corresponds to DATASET_SPECIFICATION.md §3.1.
    """

    id: str
    repository: str
    language: Language
    file_path: str
    symbol: str
    symbol_type: SymbolType
    source_code: str
    context: CodeUnitContext = Field(default_factory=CodeUnitContext)
    source_url: str
    split: SplitType
    is_public: bool
    docstring: str = ""
    source_file_lines: int
    extracted_at: str

    @model_validator(mode="after")
    def _validate_source_lines(self) -> CodeUnit:
        """Source code must be 3–100 non-empty lines (§4.2)."""
        lines = _count_source_lines(self.source_code)
        if lines < _MIN_SOURCE_LINES:
            raise ValueError(
                f"Source code has {lines} non-empty lines, "
                f"minimum is {_MIN_SOURCE_LINES}"
            )
        if lines > _MAX_SOURCE_LINES:
            raise ValueError(
                f"Source code has {lines} non-empty lines, "
                f"maximum is {_MAX_SOURCE_LINES}"
            )
        return self


# ---------------------------------------------------------------------------
# §3.2  Semantic Label
# ---------------------------------------------------------------------------


class SemanticLabel(BaseModel):
    """Human-assigned or AI-generated metadata for a code unit.

    Corresponds to DATASET_SPECIFICATION.md §3.2.
    """

    code_unit_id: str
    description: str
    summary: str
    concepts: list[str]
    input_types: list[str] = Field(default_factory=list)
    output_type: str = ""
    side_effects: list[str] = Field(default_factory=list)
    created_by: CreatedBy
    label_version: str

    @model_validator(mode="after")
    def _validate_description(self) -> SemanticLabel:
        """Description must be 20–256 words (§4.3)."""
        wc = _word_count(self.description)
        if wc < _MIN_DESCRIPTION_WORDS:
            raise ValueError(
                f"Description has {wc} words, minimum is {_MIN_DESCRIPTION_WORDS}"
            )
        if wc > _MAX_DESCRIPTION_WORDS:
            raise ValueError(
                f"Description has {wc} words, maximum is {_MAX_DESCRIPTION_WORDS}"
            )
        return self

    @model_validator(mode="after")
    def _validate_concepts(self) -> SemanticLabel:
        """At least 2 concepts required (§4.3)."""
        if len(self.concepts) < _MIN_CONCEPTS:
            raise ValueError(
                f"Concepts has {len(self.concepts)} items, "
                f"minimum is {_MIN_CONCEPTS}"
            )
        return self


# ---------------------------------------------------------------------------
# §3.3  Query Case
# ---------------------------------------------------------------------------


class QueryCase(BaseModel):
    """A developer-style retrieval query — test split only.

    Corresponds to DATASET_SPECIFICATION.md §3.3.
    """

    id: str
    query: str
    query_style: QueryStyle
    query_intent: str
    relevant_code_units: list[str]
    related_concepts: list[str] = Field(default_factory=list)
    split: Literal["test"]
    difficulty: Difficulty
    created_at: str

    @model_validator(mode="after")
    def _validate_relevant_code_units(self) -> QueryCase:
        """At least one relevant code unit required (§3.3)."""
        if len(self.relevant_code_units) < 1:
            raise ValueError("At least one relevant code unit is required")
        return self


# ---------------------------------------------------------------------------
# §3.4  Query Relevance
# ---------------------------------------------------------------------------


class QueryRelevance(BaseModel):
    """Ground-truth mapping from query to code unit.

    Corresponds to DATASET_SPECIFICATION.md §3.4.
    """

    query_id: str
    code_unit_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    relevance_label: RelevanceLabel
    explanation: str


# ---------------------------------------------------------------------------
# Candidate query (model output from query generation, §4.4)
# ---------------------------------------------------------------------------


class CandidateQuery(BaseModel):
    """Structured output from the query-generation model.

    Contains only the fields the model produces.  Pipeline metadata
    (id, relevant_code_units, split, difficulty, created_at) is added
    later by the dataset assembly layer.
    """

    query: str
    query_style: QueryStyle
    query_intent: str


# ---------------------------------------------------------------------------
# §3.1  Source Repository Snapshot
# ---------------------------------------------------------------------------


class SourceRepositorySnapshot(BaseModel):
    """Exact source snapshot record for a repository.

    Preserves provenance: repository identity + exact commit/tag.
    """

    repository: str
    commit: str
    content_hash: str = ""


# ---------------------------------------------------------------------------
# §5.1  Query Generation Metadata
# ---------------------------------------------------------------------------


class QueryGenerationMetadata(BaseModel):
    """Reproducibility record for query generation (§4.4).

    One dedicated model separate from all benchmark models.
    """

    model_name: str
    model_version: str
    prompt_template_version: str
    seed: int


# ---------------------------------------------------------------------------
# §5.1  Dataset Metadata
# ---------------------------------------------------------------------------


class DatasetMetadata(BaseModel):
    """Top-level dataset versioning and provenance (§5.1).

    Records everything needed to reproduce or identify a dataset release.
    """

    version: str
    schema_version: str
    release_date: str = ""
    repositories: list[str] = Field(default_factory=list)
    repository_commits: dict[str, str] = Field(default_factory=dict)
    repository_splits: dict[str, SplitType] = Field(default_factory=dict)
    manifest_hash: str = ""
    total_code_units: int = 0
    extracted_code_units: int = 0
    duplicate_code_units: int = 0
    train_cases: int = 0
    validation_cases: int = 0
    test_cases: int = 0
    total_queries: int = 0
    split_seed: int = 42
    query_generation: QueryGenerationMetadata | None = None
    parser: str = "python_ast"
    extraction_version: str = ""
    deduplication_method: str = ""
    eligibility_rules: dict[str, int] = Field(default_factory=dict)
    frozen: bool = False

    @model_validator(mode="after")
    def _validate_counts(self) -> DatasetMetadata:
        """Code-unit counts must sum correctly."""
        expected = self.train_cases + self.validation_cases + self.test_cases
        if self.total_code_units != expected:
            raise ValueError(
                f"total_code_units ({self.total_code_units}) != "
                f"train ({self.train_cases}) + validation "
                f"({self.validation_cases}) + test ({self.test_cases}) "
                f"= {expected}"
            )
        if self.extracted_code_units and (
            self.extracted_code_units - self.duplicate_code_units
            != self.total_code_units
        ):
            raise ValueError(
                f"extracted_code_units ({self.extracted_code_units}) - "
                f"duplicate_code_units ({self.duplicate_code_units}) != "
                f"total_code_units ({self.total_code_units})"
            )
        return self
