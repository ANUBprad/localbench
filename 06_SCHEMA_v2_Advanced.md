# LocalBench --- Data Schema v2.0

**Status:** v2.0 (Advanced)\
**Revision Date:** August 2026\
**Schema Version:** 1.0.0

---

## Philosophy

Schemas are contracts between components. They should be:
- **Typed** (Pydantic models for runtime, JSON Schema for persistence).
- **Versioned** (breaking changes require new version).
- **Validated at boundaries** (strict mode).
- **Serializable** (JSON roundtrip).
- **Self-documenting** (clear field names, docstrings).

**Immutability Rule:** Once a schema version is released, it's frozen. New versions only add fields, never remove.

---

## 1. Core Application Schemas

### 1.1 GenerationRequest (Runtime Contract)

```python
from pydantic import BaseModel, Field
from typing import Optional

class GenerationRequest(BaseModel):
    """Request to generate text from a model.
    
    Used for both simple text generation and structured prompts.
    """
    model_config = ConfigDict(validate_assignment=True)
    
    model: str = Field(
        ...,
        description="Model identifier (e.g., 'qwen:7b')",
    )
    prompt: str = Field(
        ...,
        description="User prompt",
        min_length=1,
        max_length=10000,
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Optional system message",
    )
    temperature: float = Field(
        0.0,
        description="Sampling temperature (0.0-2.0)",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        512,
        description="Maximum tokens to generate",
        gt=0,
        le=4096,
    )
    stream: bool = Field(
        False,
        description="Stream output (if supported)",
    )
    timeout_seconds: int = Field(
        60,
        description="Generation timeout in seconds",
        gt=0,
    )
```

### 1.2 GenerationResult (Runtime Contract)

```python
class GenerationResult(BaseModel):
    """Result of text generation."""
    
    model: str = Field(description="Model that generated")
    text: str = Field(description="Generated text")
    duration_ms: float = Field(description="Total generation time")
    time_to_first_token_ms: Optional[float] = Field(
        None,
        description="Time until first token (streaming)",
    )
    input_tokens: Optional[int] = Field(None)
    output_tokens: Optional[int] = Field(None)
    tokens_per_second: Optional[float] = Field(None)
    finish_reason: str = Field(
        description="Why generation stopped: 'stop', 'length', 'error'",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 1.3 StructuredGenerationRequest

```python
class StructuredGenerationRequest(BaseModel):
    """Request for structured (JSON) output."""
    
    model: str
    prompt: str
    system_prompt: Optional[str] = None
    schema: type[BaseModel]  # The Pydantic schema to validate against
    temperature: float = 0.0  # Should be deterministic
    max_tokens: int = 1024
    retry_config: Optional['RetryConfig'] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RetryConfig(BaseModel):
    """Configuration for structured generation retries."""
    
    max_attempts: int = Field(
        3,
        description="Maximum retry attempts",
        ge=1,
        le=10,
    )
    backoff_multiplier: float = Field(
        1.0,
        description="Multiplier for backoff (not used in v1, reserved)",
    )
    include_diagnostics: bool = Field(
        True,
        description="Include error feedback in retry prompt",
    )
```

### 1.4 AttemptRecord (Structured Generation History)

```python
class AttemptRecord(BaseModel):
    """Record of a single generation attempt."""
    
    attempt_number: int
    raw_output: str = Field(description="Exact model output")
    duration_ms: float
    error_category: Optional[str] = Field(
        None,
        description="If failed: 'json_parse', 'schema_validation', 'timeout', etc",
    )
    validation_errors: list[str] = Field(
        default_factory=list,
        description="Validation error messages",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 1.5 StructuredGenerationResult

```python
class StructuredGenerationResult(BaseModel):
    """Result of structured generation with retry history."""
    
    valid: bool = Field(description="Did we get valid output?")
    attempts: int = Field(description="Number of attempts made")
    schema_name: str = Field(description="Name of schema expected")
    parsed: Optional[BaseModel] = Field(
        None,
        description="The actual Pydantic object (if valid)",
    )
    validation_errors: list[str] = Field(
        default_factory=list,
        description="Final validation errors (if invalid)",
    )
    last_raw_output: str = Field(
        description="Last raw output from model (always preserved)",
    )
    attempt_log: list[AttemptRecord] = Field(
        description="Complete history of all attempts",
    )
    total_duration_ms: float = Field(
        description="Total time for all attempts",
    )
```

---

## 2. Runtime & Model Schemas

### 2.1 ModelMetadata

```python
class ModelMetadata(BaseModel):
    """Metadata about a local model."""
    
    name: str = Field(description="Model identifier (e.g., 'qwen:7b')")
    runtime: str = Field(description="Runtime: 'ollama', 'etc'")
    identifier: str = Field(
        description="Exact identifier for the runtime (e.g., model hash)",
    )
    parameter_count: Optional[str] = Field(
        None,
        description="Parameter scale (e.g., '7B', '13B')",
    )
    quantization: Optional[str] = Field(
        None,
        description="Quantization level (e.g., 'Q4_K_M')",
    )
    disk_size_bytes: Optional[int] = Field(None)
    pull_command: Optional[str] = Field(
        None,
        description="How to pull this model (e.g., 'ollama pull qwen:7b')",
    )
    metadata_timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 2.2 SystemMetadata

```python
class CPUInfo(BaseModel):
    model: str
    logical_cores: int
    physical_cores: int
    frequency_ghz: Optional[float]

class MemoryInfo(BaseModel):
    total_bytes: int
    available_bytes: int

class GPUInfo(BaseModel):
    name: str
    vram_bytes: int

class SystemMetadata(BaseModel):
    """System state at benchmark time."""
    
    os: str  # "Windows", "macOS", "Linux"
    architecture: str  # "x86_64", "arm64"
    python_version: str
    
    cpu: CPUInfo
    memory: MemoryInfo
    gpu: Optional[GPUInfo] = None
    
    ollama_version: Optional[str] = None
    ollama_home: Optional[str] = None
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

---

## 3. Benchmark Schemas

### 3.1 EvaluationConfig

```python
class EvaluationConfig(BaseModel):
    """How to score a benchmark case."""
    
    strategy: str = Field(
        description="Evaluation strategy: 'keyword', 'exact', 'numeric', 'judge'",
    )
    
    # Strategy-specific fields (optional, validated per strategy)
    required_keywords: Optional[list[str]] = Field(
        None,
        description="For 'keyword' strategy: required keywords",
    )
    tolerance: Optional[float] = Field(
        None,
        description="For 'numeric' strategy: tolerance around reference number",
    )
    judge_model: Optional[str] = Field(
        None,
        description="For 'judge' strategy: which model evaluates",
    )
    case_sensitive: Optional[bool] = Field(
        None,
        description="For 'exact' strategy: case-sensitive matching",
    )
    
    metadata: dict = Field(
        default_factory=dict,
        description="Strategy-specific metadata",
    )
```

### 3.2 BenchmarkCase

```python
class BenchmarkCase(BaseModel):
    """Single benchmark test case."""
    
    id: str = Field(
        description="Unique case ID (e.g., 'conceptual-001')",
        regex=r'^[a-z0-9\-]+$',
    )
    category: str = Field(
        description="Category (e.g., 'conceptual', 'code', 'math')",
    )
    question: str = Field(
        description="The question/prompt",
        min_length=1,
    )
    reference_answer: str = Field(
        description="Reference answer (for evaluation)",
    )
    evaluation_config: EvaluationConfig
    
    # Metadata
    metadata: dict = Field(
        default_factory=dict,
        description="Case-specific metadata",
    )
    created_timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 3.3 CaseResult (Persisted)

```python
class EvaluationResult(BaseModel):
    """Evaluation of a model's response."""
    
    strategy: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool = Field(description="Score >= threshold?")
    details: dict = Field(
        default_factory=dict,
        description="Strategy-specific details",
    )

class ResourceSample(BaseModel):
    """Single resource measurement."""
    
    timestamp: datetime
    rss_bytes: int
    vss_bytes: Optional[int] = None
    cpu_percent: float
    vram_bytes: Optional[int] = None

class CaseResult(BaseModel):
    """Result of running one case on one model."""
    
    case_id: str
    model_name: str
    status: str = Field(description="'success', 'error', 'timeout'")
    
    # Generation details
    response: str
    latency_ms: float
    structured_valid: Optional[bool] = None
    structured_attempts: Optional[int] = None
    structured_error: Optional[str] = None
    
    # Resource measurements
    peak_rss_bytes: Optional[int] = None
    avg_cpu_percent: Optional[float] = None
    peak_vram_bytes: Optional[int] = None
    resource_samples: list[ResourceSample] = Field(
        default_factory=list,
    )
    
    # Evaluation
    evaluation: EvaluationResult
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
```

### 3.4 ModelSummary (Aggregated)

```python
class ModelSummary(BaseModel):
    """Aggregated metrics for one model across all cases."""
    
    model_name: str
    
    # Quality metrics
    quality_mean: float
    quality_std: Optional[float] = None
    quality_min: float
    quality_max: float
    
    # Latency metrics (ms)
    latency_mean: float
    latency_std: Optional[float] = None
    latency_min: float
    latency_max: float
    
    # Resource metrics
    peak_rss_mean_bytes: Optional[float] = None
    peak_rss_max_bytes: Optional[int] = None
    avg_cpu_percent: Optional[float] = None
    peak_vram_bytes: Optional[int] = None
    
    # Structured generation reliability
    structured_success_rate: Optional[float] = None  # 0.0-1.0
    retry_attempts_mean: Optional[float] = None
    
    # Summary
    total_cases: int
    successful_cases: int
    failed_cases: int
```

### 3.5 BenchmarkRun (Complete Result)

```python
class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""
    
    benchmark_version: str = Field(description="e.g., '0.1.0'")
    dataset_version: str = Field(description="e.g., '1.0.0'")
    model_names: list[str]
    generation_params: GenerationRequest = Field(
        description="Template generation config",
    )
    resource_monitoring_enabled: bool = True

class BenchmarkRun(BaseModel):
    """Complete benchmark run."""
    
    # Identifiers
    run_id: str = Field(description="ISO timestamp, e.g., '2026-08-14T143000Z'")
    benchmark_version: str
    dataset_version: str
    software_version: str = Field(description="Package version at run time")
    
    # Context
    system_metadata: SystemMetadata
    configuration: BenchmarkConfig
    
    # Data
    models: list[str]
    cases: list[BenchmarkCase]
    case_results: list[CaseResult]
    
    # Aggregation
    summary: dict[str, ModelSummary]  # keyed by model name
    
    # Artifacts
    report_markdown: str
    csv_export: str
    
    # Metadata
    started_timestamp: datetime
    completed_timestamp: datetime
```

---

## 4. Recommendation Schemas

### 4.1 RecommendationConstraints

```python
class RecommendationConstraints(BaseModel):
    """User constraints for model selection."""
    
    min_quality: float = Field(
        0.75,
        description="Minimum quality score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    max_latency_ms: int = Field(
        3000,
        description="Maximum acceptable latency",
        gt=0,
    )
    max_ram_gb: int = Field(
        8,
        description="Maximum RAM usage",
        gt=0,
    )
    min_structured_success: float = Field(
        0.95,
        description="Minimum structured output success rate",
        ge=0.0,
        le=1.0,
    )
    
    metadata: dict = Field(default_factory=dict)
```

### 4.2 RecommendationResult

```python
class RejectedModel(BaseModel):
    """Why a model was rejected."""
    
    model: str
    reasons: list[str] = Field(
        description="Which constraints were violated",
    )

class RecommendationResult(BaseModel):
    """Recommendation with explanation."""
    
    recommended_model: str
    reason: str = Field(
        description="Why this model was selected",
    )
    
    passed_constraints: list[str] = Field(
        description="Which constraints the model passed",
    )
    rejected_models: list[RejectedModel] = Field(
        description="Models that didn't qualify and why",
    )
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

---

## 5. Study Assistant Schemas

### 5.1 StudyContext

```python
class StudyContext(BaseModel):
    """Loaded document context."""
    
    source_path: str
    document_type: str  # "pdf", "text"
    text_content: str = Field(description="Full extracted text")
    
    # Metadata
    size_bytes: int
    num_pages: Optional[int] = None
    loaded_timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)
```

### 5.2 Quiz Schemas

```python
class QuizQuestion(BaseModel):
    """Single quiz question."""
    
    question: str
    options: list[str] = Field(min_items=2, max_items=5)
    correct_answer_index: int = Field(
        description="Index of correct option (0-based)",
    )
    explanation: Optional[str] = Field(
        description="Why this answer is correct",
    )

class Quiz(BaseModel):
    """Generated quiz."""
    
    title: str
    questions: list[QuizQuestion] = Field(min_items=1)
    metadata: dict = Field(default_factory=dict)
    generated_timestamp: datetime = Field(default_factory=datetime.utcnow)
```

---

## 6. Persistence Format

### 6.1 Artifact Layout

```
results/
├── <run-id>/
│   ├── metadata.json              # BenchmarkRun (run info)
│   ├── config.json                # BenchmarkConfig
│   ├── system.json                # SystemMetadata
│   ├── raw_outputs.jsonl          # GenerationResult (one per line)
│   ├── case_results.jsonl         # CaseResult (one per line)
│   ├── summary.json               # Aggregated ModelSummary
│   ├── report.md                  # Human-readable Markdown
│   └── results.csv                # CSV export
```

### 6.2 JSONL Format (raw_outputs.jsonl)

Each line is a complete `GenerationResult`:

```json
{"model": "qwen:7b", "text": "Paging is...", "duration_ms": 2100, "output_tokens": 128, "finish_reason": "stop", "timestamp": "2026-08-14T14:30:00Z"}
{"model": "mistral:7b", "text": "Paging allows...", "duration_ms": 1500, "output_tokens": 95, "finish_reason": "stop", "timestamp": "2026-08-14T14:30:05Z"}
```

### 6.3 JSONL Format (case_results.jsonl)

Each line is a complete `CaseResult`:

```json
{"case_id": "conceptual-001", "model_name": "qwen:7b", "status": "success", "response": "...", "latency_ms": 2100, "evaluation": {"strategy": "keyword", "score": 0.92, "passed": true}, "timestamp": "2026-08-14T14:30:00Z"}
```

---

## 7. Versioning & Breaking Changes

### Version Numbering

- **MAJOR:** Breaking changes (removed fields, type changes).
- **MINOR:** Additive changes (new optional fields).
- **PATCH:** Non-schema changes (docs, tooling).

### Migration Strategy

**Rule:** Never remove or change type of fields in released schema versions.

**Adding a field:**
```python
# v1.0.0
class CaseResult(BaseModel):
    case_id: str
    response: str

# v1.1.0 (additive, backward compatible)
class CaseResult(BaseModel):
    case_id: str
    response: str
    new_field: Optional[str] = None  # ← New field, optional
```

**Breaking change (requires v2.0.0):**
```python
# v2.0.0 (NOT backward compatible)
class CaseResult(BaseModel):
    case_id: str
    response: str
    new_field: str  # ← Now required (change from Optional)
```

---

## 8. Validation Rules (Pydantic Config)

All schemas use:

```python
class Config:
    # Validate on assignment (runtime checks)
    validate_assignment = True
    
    # Use enum values (not names)
    use_enum_values = True
    
    # Allow serialization of datetime
    arbitrary_types_allowed = True
```

---

## 9. JSON Schema (For External Clients)

Pydantic models can generate JSON Schema for documentation:

```python
import json
from localbench.benchmark import CaseResult

schema = CaseResult.model_json_schema()
print(json.dumps(schema, indent=2))
```

This produces OpenAPI-compatible JSON Schema.

---

## 10. Serialization Examples

### To JSON
```python
result = CaseResult(...)
json_str = result.model_dump_json()  # Pydantic v2
```

### From JSON
```python
json_str = '{"case_id": "...", ...}'
result = CaseResult.model_validate_json(json_str)  # Pydantic v2
```

### Round-trip Test
```python
def test_round_trip():
    original = CaseResult(...)
    json_str = original.model_dump_json()
    restored = CaseResult.model_validate_json(json_str)
    assert original == restored
```

---

## 11. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-08-13 | Initial schema |
| 2.0 | 2026-08-14 | Advanced version: full Pydantic examples, validation rules, versioning strategy, JSON-LD structure |

