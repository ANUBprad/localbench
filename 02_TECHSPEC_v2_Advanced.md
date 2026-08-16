# LocalBench --- Technical Specification v2.0

**Status:** v2.0 (Advanced)\
**Revision Date:** August 2026

---

## 1. Architecture Overview

LocalBench is a layered, modular system with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                      CLI Layer                          │
│              (Typer, Rich Terminal UI)                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│            Application / Workloads Layer                │
│         (Study Assistant, Benchmark Runner)             │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│       Generation + Reliability Layer                    │
│    (Structured validation, Retry engine, Parsing)       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│      Model Runtime Abstraction Layer                    │
│         (LocalModel protocol, metadata)                 │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│            Ollama Adapter Layer                         │
│         (HTTP client, model discovery)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                ┌────▼────┐
                │  Ollama  │
                │ Runtime  │
                └────┬─────┘
                     │
                ┌────▼──────────────────────────┐
                │   Local LLM Models             │
                │  (Qwen, Mistral, Llama, etc)  │
                └───────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│         Benchmark Engine (Parallel to above)            │
│ ┌────────────┬──────────┬──────────┬──────────────────┐ │
│ │ Dataset    │ Runner   │ Evaluator│ Resource Profiler│ │
│ ├────────────┼──────────┼──────────┼──────────────────┤ │
│ │ Metrics    │ Reporter │ Artifacts│ Error Handling   │ │
│ └────────────┴──────────┴──────────┴──────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

**Key Principle:** Benchmark engine and application workloads share the same generation/runtime contracts. No duplication.

---

## 2. Technology Stack (Locked)

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.10+ | Modern async, Pydantic v2 support |
| Local Runtime | Ollama | Mature, well-documented, widely available |
| Validation | Pydantic v2 | Strong typing, serialization, validation |
| CLI | Typer | Clean, type-hinted, Sphinx-friendly |
| Terminal UI | Rich | Readable tables, no decoration bloat |
| Hardware Metrics | psutil | Cross-platform, low overhead |
| PDF Handling | PyMuPDF (fitz) | Fast, minimal dependencies |
| Data Analysis | pandas | Standard, well-integrated |
| Visualization | matplotlib | No external dashboard needed |
| Testing | pytest | Industry standard, fixture support |
| Persistence | JSON/JSONL/CSV | No database; data is immutable |
| Packaging | pyproject.toml | PEP 517/518 compliant |

**Explicitly NOT used:**
- LangChain (adds abstraction we don't need).
- Vector databases (keyword search is sufficient).
- Async/await at the benchmark level (benchmarks are sequential).
- Cloud APIs (offline requirement).
- Web frameworks (CLI-first).

---

## 3. Module Architecture

### 3.1 `localbench.runtime`

**Responsibility:** Communication with Ollama and local model runtimes.

**Must Provide:**
- Model discovery and enumeration.
- Health check (Ollama running?).
- Model metadata normalization.
- Text generation (streaming and non-streaming).

**Must NOT Know About:**
- Benchmark scoring.
- Study assistant logic.
- Evaluation strategies.
- Retry policy (that's generation layer).

**Key Classes/Functions:**

```python
class LocalModel(Protocol):
    """Abstract model contract."""
    name: str
    identifier: str  # Exact Ollama model ID
    
    def generate(
        self, 
        request: GenerationRequest
    ) -> GenerationResult:
        """Synchronous generation."""
        ...
    
    def stream(
        self, 
        request: GenerationRequest
    ) -> Iterator[str]:
        """Streaming generation (optional)."""
        ...
    
    def metadata(self) -> ModelMetadata:
        """Return model info."""
        ...

class OllamaModel(LocalModel):
    """Concrete Ollama implementation."""
    _client: httpx.Client
    _model_name: str
    
    def __init__(self, client: httpx.Client, model_name: str):
        ...

class OllamaRuntime:
    """Manages Ollama connection and model discovery."""
    
    def health_check(self) -> bool:
        """Is Ollama running and accessible?"""
        ...
    
    def list_models(self) -> list[ModelMetadata]:
        """Discover available models."""
        ...
    
    def get_model(self, name: str) -> LocalModel:
        """Load a specific model."""
        ...

class ModelRegistry:
    """Centralized model registry."""
    _models: dict[str, LocalModel]
    
    def register(self, model: LocalModel) -> None:
        ...
    
    def get(self, name: str) -> LocalModel:
        ...
    
    def all(self) -> list[LocalModel]:
        ...
```

**Error Handling:**

| Error | Handling | User Message |
|-------|----------|--------------|
| Ollama unavailable | RuntimeError | "Ollama is not running. Start it with `ollama serve`" |
| Model not found | ModelNotFound | "Model '{name}' not found. Available: {list}" |
| Generation timeout | GenerationTimeout | "Model did not respond within {timeout}s" |
| Network error | RuntimeError | "Network error communicating with Ollama" |

---

### 3.2 `localbench.generation`

**Responsibility:** Request execution, structured output parsing, validation, and retry orchestration.

**Must Provide:**
- Prompt construction (simple + structured).
- JSON extraction from model output.
- Pydantic validation.
- Retry logic with failure classification.
- Attempt tracking and diagnostics.

**Must NOT Know About:**
- Benchmark dataset details.
- Evaluation criteria.
- Application workload logic.

**Key Classes/Functions:**

```python
class GenerationRequest(BaseModel):
    """Request contract."""
    model: str
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    stream: bool = False
    timeout_seconds: int = 60

class GenerationResult(BaseModel):
    """Result contract."""
    model: str
    text: str
    duration_ms: float
    time_to_first_token_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    finish_reason: str  # "stop", "length", "error"

class StructuredGenerationRequest(BaseModel):
    """Structured request contract."""
    model: str
    prompt: str
    system_prompt: Optional[str] = None
    schema: type[BaseModel]  # e.g., Quiz, Answer
    temperature: float = 0.0
    max_tokens: int = 1024
    retry_config: Optional[RetryConfig] = None

class StructuredGenerationResult(BaseModel):
    """Structured result contract."""
    valid: bool
    attempts: int
    schema_name: str
    parsed: Optional[BaseModel] = None  # The actual Pydantic object
    validation_errors: list[ValidationError]
    last_raw_output: str  # Always preserve raw output
    attempt_log: list[AttemptRecord]

class AttemptRecord(BaseModel):
    """Single attempt in retry sequence."""
    attempt_number: int
    raw_output: str
    duration_ms: float
    validation_errors: list[str]
    error_category: str  # "json_parse", "schema_validation", "timeout"

class RetryConfig(BaseModel):
    """Retry policy."""
    max_attempts: int = 3
    backoff_multiplier: float = 1.0
    include_diagnostics: bool = True  # Include error feedback in retry prompt

class StructuredGenerator:
    """Orchestrates structured generation with retries."""
    
    def generate(
        self,
        request: StructuredGenerationRequest,
        model: LocalModel,
    ) -> StructuredGenerationResult:
        """Generate with validation and retry."""
        ...
    
    def _classify_failure(
        self, 
        error: Exception
    ) -> str:
        """Categorize failure type."""
        ...
    
    def _build_retry_prompt(
        self,
        original_prompt: str,
        validation_errors: list[str],
        previous_output: str,
    ) -> str:
        """Construct diagnostic retry prompt."""
        ...
```

**Failure Classification:**

| Category | When | Retry? | Example |
|----------|------|--------|---------|
| json_parse_error | Output is not valid JSON | Yes | "Missing closing brace" |
| schema_validation_error | JSON valid but doesn't match schema | Yes | "Field 'score' is required" |
| timeout | Generation exceeded time limit | No | Model hung |
| model_unavailable | Model crashed or disappeared | No | OOM situation |
| upstream_error | Ollama is down | No | Network failure |
| max_retries_exceeded | All retries exhausted | No | Genuinely broken model |

---

### 3.3 `localbench.benchmark`

**Responsibility:** Dataset management, benchmark execution, evaluation, metrics, and reporting.

**Must Provide:**
- Dataset loading and validation.
- Benchmark configuration.
- Case execution loop.
- Evaluation strategy dispatch.
- Metric calculation.
- Result persistence.
- Report generation.

**Must NOT Know About:**
- Ollama-specific details (use runtime abstraction).
- Application workload logic.

**Key Classes/Functions:**

```python
class BenchmarkCase(BaseModel):
    """Single benchmark case."""
    id: str  # e.g., "conceptual-001"
    category: str  # e.g., "conceptual", "code", "math"
    question: str
    reference_answer: str  # For evaluation
    evaluation_config: EvaluationConfig
    metadata: dict = {}

class EvaluationConfig(BaseModel):
    """How to score this case."""
    strategy: str  # "keyword", "exact", "numeric", "judge"
    # Strategy-specific fields:
    required_keywords: Optional[list[str]] = None  # For keyword strategy
    tolerance: Optional[float] = None  # For numeric
    judge_model: Optional[str] = None  # For judge strategy

class BenchmarkCase(BaseModel):
    """A single test case with evaluation rules."""
    id: str
    category: str
    question: str
    reference_answer: str
    evaluation_config: EvaluationConfig

class CaseResult(BaseModel):
    """Result of executing one case on one model."""
    case_id: str
    model_name: str
    status: str  # "success", "error", "timeout"
    response: str
    latency_ms: float
    
    # Structured generation details (if applicable)
    structured_valid: bool
    structured_attempts: int
    
    # Resource measurements
    peak_rss_bytes: Optional[int]
    cpu_percent: Optional[float]
    vram_bytes: Optional[int]
    
    # Evaluation result
    evaluation_score: float  # 0.0 to 1.0
    evaluation_strategy: str
    evaluation_details: dict  # Strategy-specific details
    
    # Timestamps and metadata
    timestamp: datetime
    error_message: Optional[str] = None

class BenchmarkRun(BaseModel):
    """Complete benchmark run across all models."""
    run_id: str  # ISO timestamp, e.g., "2026-08-14T143000Z"
    benchmark_version: str  # e.g., "0.1.0"
    dataset_version: str  # e.g., "1.0.0"
    software_version: str  # Package version
    
    system_metadata: SystemMetadata
    configuration: BenchmarkConfig
    
    models: list[str]
    cases: list[BenchmarkCase]
    results: list[CaseResult]
    
    # Aggregated metrics per model
    summary: dict[str, ModelSummary]
    
    # Report artifacts
    report_markdown: str
    csv_export: str

class BenchmarkRunner:
    """Orchestrates benchmark execution."""
    
    def run(
        self,
        config: BenchmarkConfig,
        models: list[LocalModel],
        dataset: BenchmarkDataset,
    ) -> BenchmarkRun:
        """Execute complete benchmark."""
        ...
    
    def _warm_up(self, model: LocalModel) -> None:
        """Warm up model with dummy requests."""
        ...
    
    def _execute_case(
        self,
        model: LocalModel,
        case: BenchmarkCase,
    ) -> CaseResult:
        """Execute single case with timing and resource monitoring."""
        ...
    
    def _evaluate(
        self,
        case: BenchmarkCase,
        result: CaseResult,
    ) -> EvaluationResult:
        """Score the result."""
        ...
```

**Evaluation Strategies:**

| Strategy | Logic | Use Case |
|----------|-------|----------|
| keyword | Count required keywords in response | Conceptual Q&A |
| exact | Exact string match (case-insensitive) | Factual answers |
| numeric | Number within tolerance of reference | Math problems |
| structured | Validate JSON schema matches | Structured generation |
| judge | Use a local LLM to score | Open-ended answers |

---

### 3.4 `localbench.profiling`

**Responsibility:** Hardware detection and resource measurement.

**Must Provide:**
- System metadata (CPU, RAM, GPU).
- Process-level metrics (RSS, CPU utilization).
- Temperature and power (if available).
- Measurement methodology documentation.

**Key Classes/Functions:**

```python
class SystemMetadata(BaseModel):
    """System profile at benchmark time."""
    os: str  # "Windows", "macOS", "Linux"
    architecture: str  # "x86_64", "arm64"
    python_version: str
    
    cpu: CPUInfo
    memory: MemoryInfo
    gpu: Optional[GPUInfo]
    
    timestamp: datetime

class CPUInfo(BaseModel):
    """CPU details."""
    model: str  # e.g., "Intel Core i7-10700K"
    logical_cores: int
    physical_cores: int
    frequency_ghz: float

class MemoryInfo(BaseModel):
    """RAM details."""
    total_bytes: int
    available_bytes: int

class GPUInfo(BaseModel):
    """GPU details (if available)."""
    name: str  # e.g., "NVIDIA RTX 3070"
    vram_bytes: int

class ResourceSample(BaseModel):
    """Snapshot of resource usage during inference."""
    timestamp: datetime
    rss_bytes: int  # Process resident set size
    vss_bytes: Optional[int]  # Virtual memory
    cpu_percent: float  # 0-100
    cpu_cores_used: Optional[float]
    vram_bytes: Optional[int]

class ResourceMonitor:
    """Measures resource usage during generation."""
    
    def sample(self, process_id: int) -> ResourceSample:
        """Take one resource sample."""
        ...
    
    def monitor_during_generation(
        self,
        process_id: int,
        duration_seconds: float,
        sample_interval_ms: int = 100,
    ) -> list[ResourceSample]:
        """Collect samples during inference."""
        ...
    
    def peak_stats(
        self,
        samples: list[ResourceSample],
    ) -> dict[str, float]:
        """Summarize peak usage from samples."""
        ...
```

---

### 3.5 `localbench.workloads`

**Responsibility:** Reusable benchmark and application workload definitions.

**Contract:**

```python
class Workload(Protocol):
    """Abstract workload interface."""
    name: str
    version: str
    
    def cases(self) -> Iterable[BenchmarkCase]:
        """Yield all benchmark cases."""
        ...
    
    def evaluate(
        self,
        case: BenchmarkCase,
        result: GenerationResult,
    ) -> EvaluationResult:
        """Score a result for this workload."""
        ...

class BenchmarkWorkload(Workload):
    """Base class for benchmark workloads."""
    
    cases_data: list[BenchmarkCase]
    
    def cases(self) -> Iterable[BenchmarkCase]:
        return iter(self.cases_data)

class EducationWorkload(Workload):
    """Semester notes Q&A workload."""
    name = "education"
    version = "0.1.0"
    
    def cases(self) -> Iterable[BenchmarkCase]:
        """20 OS/CS/Data Structures questions."""
        ...
    
    def evaluate(
        self,
        case: BenchmarkCase,
        result: GenerationResult,
    ) -> EvaluationResult:
        """Grade using keywords + semantic similarity."""
        ...
```

---

### 3.6 `localbench.assistant`

**Responsibility:** Study assistant application (Q&A, quiz generation).

**Must Provide:**
- Document ingestion (PDF, text).
- Simple retrieval (keyword-based, no vectors).
- Question answering against local context.
- Quiz generation (structured, validated).

**Must NOT Use:**
- External APIs.
- Vector databases (keyword search is fine).
- Complex RAG (simple context window is okay).

**Key Classes/Functions:**

```python
class StudyContext(BaseModel):
    """Loaded document context."""
    source_path: str
    document_type: str  # "pdf", "text"
    text_content: str
    metadata: dict

class QuizQuestion(BaseModel):
    """Single quiz question."""
    question: str
    options: list[str]
    correct_answer_index: int
    explanation: Optional[str]

class Quiz(BaseModel):
    """Generated quiz."""
    title: str
    questions: list[QuizQuestion]
    metadata: dict

class StudyAssistant:
    """Local study assistant."""
    context: Optional[StudyContext]
    model: LocalModel
    generator: StructuredGenerator
    
    def load_document(self, path: str) -> StudyContext:
        """Load PDF or text file."""
        ...
    
    def answer_question(self, question: str) -> str:
        """Answer based on loaded context."""
        ...
    
    def generate_quiz(
        self,
        num_questions: int = 5,
    ) -> Quiz:
        """Generate validated quiz."""
        ...
```

---

### 3.7 `localbench.cli`

**Responsibility:** Command-line interface only. Business logic lives elsewhere.

**Must Provide:**
- Command routing.
- Argument parsing (via Typer).
- Output formatting (via Rich).
- Error presentation to user.

**Must NOT Contain:**
- Business logic.
- Direct model inference.
- Benchmark execution (delegate to benchmark module).

**Commands:**

| Command | Purpose |
|---------|---------|
| `localbench models` | List available models |
| `localbench ask <prompt>` | Ask a model (ad hoc) |
| `localbench benchmark` | Run benchmark suite |
| `localbench compare` | Compare latest benchmark results |
| `localbench recommend` | Get model recommendation based on constraints |
| `localbench study <file>` | Start study assistant with document |
| `localbench system` | Show system information |

---

## 4. Core Data Contracts (Pydantic)

All contracts must be:
- Versioned (breaking changes require new version).
- Documented (docstrings on all fields).
- Validated (strict mode in Pydantic).
- Serializable (JSON roundtrip).

```python
# Example: Strongly typed everything

class GenerationRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    model: str  # Model identifier
    prompt: str  # User prompt
    system_prompt: Optional[str] = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(512, gt=0, le=4096)
    stream: bool = False
    timeout_seconds: int = Field(60, gt=0)
```

---

## 5. Error Handling Strategy

**Principle:** Errors are first-class. Classify, log, and handle explicitly.

**Typed Exceptions:**

```python
class LocalBenchError(Exception):
    """Base exception."""
    code: str
    message: str
    context: dict

class RuntimeUnavailable(LocalBenchError):
    """Ollama not running."""
    code = "RUNTIME_UNAVAILABLE"

class ModelNotFound(LocalBenchError):
    """Requested model doesn't exist."""
    code = "MODEL_NOT_FOUND"

class GenerationTimeout(LocalBenchError):
    """Model inference took too long."""
    code = "GENERATION_TIMEOUT"

class ValidationError(LocalBenchError):
    """Structured output invalid."""
    code = "VALIDATION_ERROR"

class BenchmarkConfigurationError(LocalBenchError):
    """Config is invalid."""
    code = "BENCHMARK_CONFIG_ERROR"
```

**Handling Strategy:**

| Layer | Strategy |
|-------|----------|
| Runtime adapter | Raise typed exceptions. Don't retry. |
| Generation | Classify, retry if recoverable, else raise. |
| Benchmark | Catch, log, record error in result, continue. |
| CLI | Catch all, format error message, exit with code. |

---

## 6. Testing Strategy

### Unit Tests

```
tests/
  test_runtime/
    test_ollama_discovery.py
    test_model_metadata.py
    test_health_check.py
  test_generation/
    test_retry_policy.py
    test_pydantic_validation.py
    test_json_parsing.py
  test_benchmark/
    test_evaluation_strategies.py
    test_metric_calculation.py
    test_recommendation_engine.py
  test_profiling/
    test_system_metadata.py
    test_resource_measurement.py
```

### Integration Tests

```
tests/
  test_integration/
    test_end_to_end_generation.py
    test_end_to_end_benchmark.py
    test_study_assistant.py
    test_cli_commands.py
```

### Failure Tests (Critical)

```
tests/
  test_failures/
    test_ollama_unavailable.py
    test_malformed_json.py
    test_schema_validation_failure.py
    test_timeout.py
    test_no_models_qualify_recommendation.py
```

---

## 7. Performance Guidelines

- **Do NOT:** Load/unload a model per case (keep it in memory).
- **Do NOT:** Saturate CPU with resource sampling (100ms intervals).
- **Do NOT:** Persist incrementally (batch writes).
- **Do:** Warm up models before measurements.
- **Do:** Separate cold and warm metrics.
- **Do:** Use connection pooling for Ollama.

---

## 8. Security & Privacy

- **No external logging** of prompts or documents.
- **No telemetry** without explicit opt-in.
- **No cloud fallback** (fail locally, not remotely).
- **No credentials in logs** (strip before logging).
- **Benchmark artifacts** are local user data (treat as sensitive).

---

## 9. Extensibility

Future adapters without changing core:

```python
class LocalModel(Protocol):  # Stable
    ...

# Future implementations:
class LLaMAModel(LocalModel):
    ...

class VLLMModel(LocalModel):
    ...

class TensorRTModel(LocalModel):
    ...
```

**Do not implement future adapters now.**

---

## 10. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-08-13 | Initial tech spec |
| 2.0 | 2026-08-14 | Advanced version: detailed module specs, contracts, error handling, testing strategy |

