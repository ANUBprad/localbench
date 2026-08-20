# LocalBench --- Architecture & System Design v3

**Status:** Research-oriented, experimental platform architecture  
**Last Updated:** 2026-08-19  
**Role:** Defines component responsibilities, data flow, integration boundaries, and design decisions.

---

## 1. System Overview

LocalBench is a modular, layered research platform for evaluating whether small, locally deployable language models can be specialized for narrowly scoped real-world workloads and achieve competitive downstream performance while reducing computational resources.

The architecture separates concerns into **five primary layers**:

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI & User Interface (Typer, Rich)                             │
│  Command routing, argument parsing, error presentation          │
└──────────────┬────────────────────────────────────┬─────────────┘
               │                                    │
┌──────────────v──────────────┐  ┌─────────────────v────────────────┐
│  Application Layer          │  │  Workload Layer                   │
│  - Config management        │  │  - Task definition                │
│  - Benchmark orchestration  │  │  - Dataset contracts              │
│  - Report generation        │  │  - Evaluation logic               │
└──────────────┬──────────────┘  └─────────────────┬────────────────┘
               │                                   │
┌──────────────v─────────────────────────────────v────────────────┐
│  Runtime Layer (Provider-agnostic inference)                    │
│  - LocalModel protocol                                          │
│  - Generation request/response handling                         │
│  - Structured output validation & retry                        │
│  - Hardware profiling & resource tracking                      │
└──────────────┬─────────────────────────────────────────────────┘
               │
┌──────────────v────────────────────────────────────────────────────┐
│  Inference Backend (Ollama)                                       │
│  - Model lifecycle management                                     │
│  - Generation execution                                           │
│  - Hardware-specific optimization                                │
└────────────────────────────────────────────────────────────────────┘
```

**Key principle:** Task-specific evaluation stays in the workload layer. Runtime remains inference-provider agnostic.

---

## 2. Module Responsibilities

### 2.1 CLI Module (`cli.py`)

**Role:** User-facing command interface and routing.

**Responsibilities:**
- Parse command-line arguments (Typer)
- Route to application handlers
- Format and present results (Rich tables, markdown, JSON)
- Handle user-facing errors with actionable messages
- Provide help and command documentation

**Key Commands:**
```
localbench system          # Display hardware/environment info
localbench models          # List available local models
localbench ask <prompt>    # Single-shot inference verification
localbench benchmark       # Run full benchmark workload
localbench compare         # Compare benchmark results
localbench recommend       # Get model recommendations
localbench workload        # Workload management
```

**Constraints:**
- CLI must remain thin; core logic lives elsewhere.
- All decisions deferred to Application layer.
- Error handling is presentational, not logical.

---

### 2.2 Application Layer

#### 2.2.1 Configuration Module (`config/`)

**Role:** Configuration loading, validation, and environment setup.

**Files:**
- `config.py` — Pydantic models for configuration
- `loader.py` — YAML/JSON loading and validation
- `defaults.py` — System defaults and constants

**Responsibilities:**
- Load configuration from files, environment, CLI overrides
- Validate against schema
- Resolve hardware capabilities
- Provide runtime-safe configuration objects
- Track configuration versions for reproducibility

**Key schema:**
```python
class BenchmarkConfig:
    workload: str
    models: List[str]
    dataset_version: str
    generation_config: GenerationConfig
    hardware_profile: HardwareProfile
    seed: int
    output_dir: Path
```

---

#### 2.2.2 Application Orchestration Module (`application.py`)

**Role:** Benchmark workflow orchestration, state management, report coordination.

**Responsibilities:**
- Coordinate benchmark phases (warmup → execution → evaluation)
- Manage experiment state (run-id, versioning, artifact tracking)
- Coordinate across Runtime and Workload layers
- Trigger profiling and measurement collection
- Persist run metadata and results
- Invoke workload-specific evaluation

**Key methods:**
- `run_benchmark(config)` — Execute full benchmark pipeline
- `profile_system()` — Capture hardware snapshot
- `verify_model(model_id)` — Health check before execution
- `record_artifacts(run_id, results)` — Persist raw outputs

---

#### 2.2.3 Profiling Module (`profiling/`)

**Role:** System and process-level resource measurement.

**Files:**
- `hardware.py` — Machine/OS/RAM/CPU detection
- `monitor.py` — Real-time process monitoring (RSS, CPU, VRAM)
- `metrics.py` — Aggregation and reporting

**Responsibilities:**
- Detect available RAM, CPU cores, GPU
- Capture baseline system state
- Profile process resource consumption during inference
- Record disk footprint of models
- Support cold/warm measurement distinction
- Export metrics in schema-compliant format

---

#### 2.2.4 Reporting Module (`reporting/`)

**Role:** Result aggregation, comparison, and output formatting.

**Files:**
- `comparison.py` — Model/result comparison logic
- `export.py` — CSV, JSON, markdown export
- `visualization.py` — matplotlib-based charts
- `formatter.py` — Terminal and report formatting

**Responsibilities:**
- Aggregate metrics from raw artifacts
- Compare baseline vs specialized models
- Generate markdown reports with methodology/results
- Export data for external analysis
- Visualize quality/resource trade-offs

---

### 2.3 Runtime Layer

**Role:** Provider-agnostic inference abstraction and structured output handling.

#### 2.3.1 LocalModel Protocol (`runtime/model.py`)

**Role:** Define the inference contract.

```python
class LocalModel(Protocol):
    """Contract for local inference providers."""
    
    @property
    def name(self) -> str:
        """Unique model identifier."""
    
    @property
    def metadata(self) -> ModelMetadata:
        """Static metadata: parameters, quantization, footprint."""
    
    async def generate(
        self, 
        request: GenerationRequest
    ) -> GenerationResult:
        """Execute generation with timing/token info."""
```

**Responsibilities:**
- Define clean abstraction for inference providers
- Specify required metadata and return formats
- Enable provider-agnostic application code

---

#### 2.3.2 Ollama Adapter (`runtime/ollama/`)

**Role:** Implement LocalModel protocol for Ollama.

**Files:**
- `adapter.py` — LocalModel implementation
- `client.py` — Ollama HTTP client wrapper
- `health.py` — Health check and model discovery

**Responsibilities:**
- Connect to Ollama server (local/remote)
- Translate GenerationRequest → Ollama API
- Parse Ollama responses and extract timing
- Discover available models and normalize metadata
- Handle connection failures with retries
- Record actual token counts if available

**Constraints:**
- All Ollama-specific logic isolated here.
- Health checks are non-fatal.
- Failures trigger actionable error messages.

---

#### 2.3.3 Generation & Structured Output (`runtime/generation/`)

**Role:** Generation request handling, validation, and retry logic.

**Files:**
- `request.py` — GenerationRequest schema
- `result.py` — GenerationResult schema
- `validator.py` — Pydantic validation logic
- `retry.py` — Bounded retry engine
- `failures.py` — Failure taxonomy and classification

**Responsibilities:**
- Validate generation parameters before request
- Parse raw model output
- Validate structured output against Pydantic schema
- Classify failures (malformed JSON, validation error, etc.)
- Execute bounded retries with exponential backoff
- Record all attempts (attempt count, error, recovery status)
- Distinguish generation recovery from benchmark repetition

**Failure Taxonomy:**
```
GenerationFailure
├── MalformedJSON
├── ValidationError
│   ├── MissingField
│   ├── TypeMismatch
│   └── ConstraintViolation
├── SemanticFailure
│   └── GrindingFailure (repetitive/nonsense output)
└── ProviderFailure
    ├── Timeout
    ├── ResourceExhausted
    └── Unavailable
```

**Key constraints:**
- Retries are bounded (default: 3 attempts).
- Retries do not count toward benchmark repetitions.
- All attempt records are preserved.
- Retry decisions are reason-based (not blind retries).

---

### 2.4 Workload Layer

**Role:** Task-specific implementation—dataset, evaluation, downstream metrics.

#### 2.4.1 Workload Base Contract (`workloads/base.py`)

**Role:** Define the workload abstraction.

```python
class Workload(ABC):
    """Contract for benchmark workloads."""
    
    @property
    def name(self) -> str:
        """Unique workload identifier."""
    
    @property
    def version(self) -> str:
        """Versioned workload schema."""
    
    @abstractmethod
    def dataset(self) -> Dataset:
        """Load benchmark dataset."""
    
    @abstractmethod
    def evaluate_generation(
        self, 
        case: BenchmarkCase, 
        result: GenerationResult
    ) -> GenerationEvaluation:
        """Evaluate generated artifact quality."""
    
    @abstractmethod
    def evaluate_downstream(
        self,
        artifacts: SemanticArtifactStore,
        evaluator: DownstreamEvaluator
    ) -> DownstreamMetrics:
        """Evaluate downstream task performance."""
```

---

#### 2.4.2 Code Semantic Retrieval Workload (`workloads/code_retrieval/`)

**Role:** Flagship workload implementation.

**Files:**
- `workload.py` — Workload orchestration
- `dataset.py` — Dataset loading, versioning, splitting
- `extraction.py` — Source code parsing and method extraction
- `queries.py` — Query generation and relevance labeling
- `retrieval.py` — Local retrieval index and query execution
- `evaluation.py` — Hit@K, MRR computation

**Responsibilities:**

**Dataset Management:**
- Load source repositories
- Extract methods/functions with context
- Define semantic labels (description, concepts)
- Generate developer-style queries
- Create relevance relationships
- Perform repository-disjoint train/val/test split
- Version dataset schema and contents

**Generation Evaluation:**
- Validate semantic artifact schema
- Check field completeness
- Assess semantic grounding
- Record validation errors

**Downstream Evaluation:**
- Build local embedding index from semantic artifacts
- Execute queries against index
- Compute ranked retrieval results
- Calculate Hit@1, Hit@3, Hit@5, Hit@10, MRR
- Compare against ground-truth relevance

**Key invariants:**
- Dataset versioning is immutable.
- Test set is frozen before final evaluation.
- Ground-truth labels are fixed.
- Evaluation protocol is deterministic.

---

## 3. Data Flow

### 3.1 Benchmark Execution Flow

```
User: localbench benchmark
         ↓
    CLI (parse args)
         ↓
    Application.run_benchmark()
         ↓
    Load config & workload
         ↓
    Profiling.profile_system()
         ↓
    Runtime.verify_model() — Ollama health check
         ↓
    Workload.dataset() — Load benchmark cases
         ↓
    [For each case]
         ├─ Runtime.generate(request)
         │      ↓
         │  Ollama adapter → Ollama → result + timing
         │      ↓
         │  Validation & retry if needed
         │      ↓
         │  Record: GenerationResult + attempts
         │
         ├─ Workload.evaluate_generation()
         │      ↓
         │  Validate artifact structure
         │      ↓
         │  Store SemanticArtifact
         │
         └─ [End loop]
         ↓
    Profiling.record_resources()
         ↓
    Workload.evaluate_downstream()
         ├─ Build retrieval index
         ├─ Execute queries
         └─ Compute Hit@K, MRR
         ↓
    Application.record_artifacts()
         ├─ results/<run-id>/raw_outputs.jsonl
         ├─ results/<run-id>/generation_results.jsonl
         ├─ results/<run-id>/retrieval_results.jsonl
         ├─ results/<run-id>/metrics.json
         └─ results/<run-id>/report.md
         ↓
    Reporting.format_summary()
         ↓
    CLI.present_results()
```

---

### 3.2 Generation with Validation & Retry

```
Workload: generate semantic artifact for code unit
         ↓
    GenerationRequest {
        prompt: "...",
        temperature: 0.3,
        max_tokens: 256,
        format: "json"
    }
         ↓
    Runtime.generate(request)
         ├─ Ollama adapter translates to Ollama API
         ├─ Record start time, hardware state
         └─ Execute: Ollama → raw model output
         ↓
    Attempt 1:
    ├─ Parse JSON from raw output
    ├─ Validate against SemanticArtifact schema
    ├─ If valid → return with attempt count
    └─ If invalid → classify error
         ↓
    Invalid (e.g., MalformedJSON)
         ↓
    Attempt 2 (bounded retry):
    ├─ Adjust prompt (e.g., "output must be valid JSON")
    ├─ Execute: Ollama → raw output
    ├─ Validate
    └─ If valid → return with attempt count=2
         ↓
    Still invalid after max_attempts
         ↓
    Record GenerationFailure
    ├─ attempt_count: 3
    ├─ final_error: ValidationError
    └─ status: FAILED
```

---

## 4. Module Interactions

### 4.1 Dependency Graph

```
CLI
 └─ Application (orchestration)
     ├─ Config (configuration loading)
     ├─ Profiling (resource measurement)
     ├─ Runtime (inference)
     │   ├─ Ollama adapter (provider-specific)
     │   └─ Generation/validation (provider-agnostic)
     ├─ Workload (task-specific)
     │   ├─ Dataset (workload-specific)
     │   ├─ Extraction (workload-specific)
     │   └─ Evaluation (workload-specific)
     └─ Reporting (result aggregation)
```

**Key principle:** No circular dependencies. Runtime does not depend on Workload. Workload does not depend on specific model implementations.

---

### 4.2 Integration Contracts

**Application → Runtime:**
```python
request: GenerationRequest
result = runtime.generate(request)
# result.status ∈ {SUCCESS, FAILED}
# if FAILED: result.attempts, result.error_category accessible
```

**Application → Workload:**
```python
dataset = workload.dataset()
artifacts = workload.evaluate_generation(cases, results)
metrics = workload.evaluate_downstream(artifacts)
```

**Runtime → Ollama Adapter:**
```python
# LocalModel protocol implementation
adapter.generate(request) → GenerationResult with timing
```

---

## 5. Data Structures & Serialization

All major objects are Pydantic v2 models with JSON serialization.

**Core objects:**
- `GenerationRequest` — Input to inference
- `GenerationResult` — Output from inference
- `SemanticArtifact` — Validated, generated content
- `BenchmarkCase` — Individual benchmark unit
- `BenchmarkRun` — Configuration + metadata for a run
- `DownstreamMetrics` — Retrieval quality measurements

**Persistence:**
- JSONL for streaming (raw_outputs, generation_results, retrieval_results)
- JSON for structured data (config, metrics, summary)
- Markdown for human-readable reports

---

## 6. Error Handling & Resilience

### 6.1 Levels of Failure

**Fatal (stops execution):**
- Configuration validation failure
- Ollama unavailable before warm-up
- Dataset load failure

**Recoverable (continues with status):**
- Single generation timeout → retry
- Single benchmark case failure → record and continue
- Profiling metric unavailable → use default

**Acceptable (recorded, analyzed):**
- Model generates invalid JSON → record failure, analyze patterns
- Query returns no results → record in statistics

### 6.2 Failure Recording

All failures are recorded with:
- Error category (taxonomy)
- Timestamp
- Relevant context (model, case ID, attempt count)
- Recovery attempt (if applicable)
- Final outcome (success/failure)

---

## 7. Extensibility

### 7.1 Adding a New Workload

1. Create `workloads/new_workload/` directory
2. Implement `Workload` abstract base class
3. Define dataset schema and loading
4. Implement `evaluate_generation()` for artifact validation
5. Implement `evaluate_downstream()` for task-specific metrics
6. Add tests in `tests/workloads/test_new_workload.py`
7. Update workload registry in `config/workloads.py`

### 7.2 Adding a New Runtime Provider

1. Create `runtime/new_provider/` directory
2. Implement `LocalModel` protocol
3. Create adapter translating GenerationRequest to provider API
4. Implement health check and model discovery
5. Add tests in `tests/runtime/test_new_provider.py`
6. Update runtime registry in `config/providers.py`

---

## 8. Testing Strategy

**Unit tests:**
- Config loading/validation
- Failure classification
- Metrics calculation
- Schema validation

**Integration tests:**
- Full benchmark flow (with mock Ollama)
- Generation + validation + retry
- Downstream evaluation

**System tests:**
- End-to-end benchmark with real Ollama
- Artifact persistence and retrieval
- Report generation

**Test structure:**
```
tests/
├── unit/
│   ├── config/
│   ├── runtime/
│   ├── workloads/
│   └── reporting/
├── integration/
│   ├── test_benchmark_flow.py
│   └── test_workload_eval.py
└── fixtures/
    ├── mock_ollama.py
    ├── sample_dataset.py
    └── expected_outputs/
```

---

## 9. Future Extensibility Considerations

- **Multiple workloads:** Runtime isolated to support future workloads without modification
- **Distributed training:** Training artifacts separate from core platform
- **Quantization:** Model footprint/latency tracked; quantized variants supported
- **Web UI:** Application layer provides API surface; web tier can be added independently
- **Cloud fallback:** Ollama adapter can be swapped; no application-level cloud coupling

---

## 10. Key Design Principles

1. **Separation of concerns** — Runtime, workload, and application logic remain independent
2. **Reproducibility** — All decisions versioned and recorded
3. **Transparency** — Raw data remains accessible; no black-box aggregation
4. **Failure visibility** — Errors are classified, recorded, and analyzable
5. **Local-first** — Default to local execution; no cloud coupling
6. **Extensibility** — Workloads and providers pluggable without core changes
7. **Testing** — Failure paths are first-class test cases
8. **Documentation** — Architecture is self-documenting via contracts and schemas

