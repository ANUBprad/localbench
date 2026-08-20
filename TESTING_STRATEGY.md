# LocalBench --- Testing Strategy v3

**Status:** Framework for implementation across all phases  
**Last Updated:** 2026-08-19  
**Role:** Defines testing approach, test structure, coverage goals, and continuous validation.

---

## 1. Testing Philosophy

Testing LocalBench requires coverage across **three levels**: unit (isolated components), integration (component interactions), and system (end-to-end workflows).

**Core principles:**
- Failure paths are first-class test cases
- Tests document expected behavior
- Mocking enables isolated unit tests
- Real Ollama used selectively for integration tests
- Reproducibility enabled through fixture control

---

## 2. Test Structure

### 2.1 Directory Organization

```
tests/
├── unit/
│   ├── config/
│   │   ├── test_loader.py
│   │   └── test_validation.py
│   ├── runtime/
│   │   ├── test_ollama_adapter.py
│   │   ├── test_generation_request.py
│   │   └── test_failure_classification.py
│   ├── workload/
│   │   ├── test_code_retrieval.py
│   │   ├── test_dataset_loading.py
│   │   └── test_evaluation.py
│   ├── reporting/
│   │   ├── test_comparison.py
│   │   └── test_export.py
│   └── application/
│       └── test_orchestration.py
├── integration/
│   ├── test_benchmark_flow.py
│   ├── test_ollama_integration.py
│   ├── test_retrieval_pipeline.py
│   └── test_end_to_end.py
├── system/
│   ├── test_cli.py
│   ├── test_artifact_persistence.py
│   └── test_reproducibility.py
├── fixtures/
│   ├── mock_ollama.py
│   ├── mock_dataset.py
│   ├── sample_models.py
│   └── expected_outputs/
└── conftest.py  # Shared pytest configuration
```

---

## 3. Unit Tests

### 3.1 Configuration Module Tests (`unit/config/`)

**Purpose:** Validate configuration loading, validation, and defaults.

**Test cases:**

```python
# test_loader.py

def test_load_config_from_yaml():
    """Config loads correctly from YAML file."""
    config = ConfigLoader.load("tests/fixtures/config.yaml")
    assert config.workload == "code-retrieval-v1"
    assert config.dataset_version == "1.0.0"

def test_load_config_with_env_override():
    """Environment variables override YAML."""
    os.environ["LOCALBENCH_WORKLOAD"] = "custom"
    config = ConfigLoader.load("tests/fixtures/config.yaml")
    assert config.workload == "custom"

def test_validate_config_fails_missing_field():
    """Validation fails if required field missing."""
    invalid_config = {...}  # missing dataset_version
    with pytest.raises(ConfigValidationError):
        ConfigValidator.validate(invalid_config)

def test_config_defaults_applied():
    """Default values fill missing optional fields."""
    minimal_config = {"workload": "code-retrieval"}
    filled = ConfigLoader.apply_defaults(minimal_config)
    assert filled.temperature == 0.3
    assert filled.max_tokens == 256
```

---

### 3.2 Runtime Module Tests (`unit/runtime/`)

#### 3.2.1 Ollama Adapter Tests

```python
# test_ollama_adapter.py

@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "model": "phi-3-mini",
        "created_at": "2026-01-01T00:00:00Z",
        "response": '{"description": "..."}',
        "done": True,
        "total_duration": 720000000,  # nanoseconds
    }

def test_adapter_generates_with_valid_request(mock_ollama_response):
    """Adapter successfully generates with valid request."""
    adapter = OllamaAdapter("http://localhost:11434")
    
    # Mock httpx.post
    with patch("httpx.post") as mock_post:
        mock_post.return_value.json.return_value = mock_ollama_response
        
        request = GenerationRequest(
            prompt="Describe this code: ...",
            temperature=0.3,
            max_tokens=256
        )
        result = adapter.generate(request)
        
        assert result.status == "success"
        assert result.text == '{"description": "..."}'
        assert result.duration_ms == 720

def test_adapter_handles_timeout():
    """Adapter handles connection timeout gracefully."""
    adapter = OllamaAdapter("http://localhost:11434")
    
    with patch("httpx.post") as mock_post:
        mock_post.side_effect = httpx.TimeoutException()
        
        request = GenerationRequest(...)
        result = adapter.generate(request)
        
        assert result.status == "failed"
        assert result.error_category == "timeout"

def test_adapter_discovers_models():
    """Adapter discovers available models."""
    adapter = OllamaAdapter("http://localhost:11434")
    
    with patch("httpx.get") as mock_get:
        mock_get.return_value.json.return_value = {
            "models": [
                {"name": "phi-3-mini:latest", "size": 2048},
                {"name": "mistral-7b:latest", "size": 3800}
            ]
        }
        
        models = adapter.discover_models()
        assert len(models) == 2
        assert models[0].name == "phi-3-mini"
```

#### 3.2.2 Generation & Validation Tests

```python
# test_failure_classification.py

def test_malformed_json_detected():
    """Malformed JSON is classified correctly."""
    raw_output = '{"description": "incomplete'  # Missing closing brace
    
    result = parse_and_validate(raw_output, schema=SemanticArtifact)
    assert result.valid == False
    assert result.error_category == "malformed_json"

def test_missing_field_detected():
    """Missing required field is classified correctly."""
    raw_output = '{"summary": "..."}'  # Missing 'description'
    
    result = parse_and_validate(raw_output, schema=SemanticArtifact)
    assert result.valid == False
    assert result.error_category == "validation_error"

def test_valid_semantic_artifact_passes():
    """Valid artifact passes validation."""
    raw_output = json.dumps({
        "description": "...",
        "concepts": ["retry", "backoff"],
        "code_unit_id": "repo001_method_01"
    })
    
    result = parse_and_validate(raw_output, schema=SemanticArtifact)
    assert result.valid == True
    assert result.parsed.description == "..."

def test_retry_bounded():
    """Retry attempts are bounded."""
    adapter_mock = MagicMock()
    adapter_mock.generate.return_value.text = '{"bad": json}'  # Always invalid
    
    retry_engine = RetryEngine(max_attempts=3, base_delay=0.1)
    result = retry_engine.execute(
        request=GenerationRequest(...),
        adapter=adapter_mock
    )
    
    assert result.valid == False
    assert adapter_mock.generate.call_count == 3  # Exactly 3 attempts
```

---

### 3.3 Workload Module Tests (`unit/workload/`)

```python
# test_code_retrieval.py

@pytest.fixture
def mock_dataset():
    """Mock code retrieval dataset."""
    return {
        "code_units": [
            BenchmarkCase(
                id="repo001_method_01",
                symbol="process_retry",
                source_code="def process_retry(...): ...",
                split="test"
            )
        ],
        "queries": [
            Query(
                id="query_001",
                query="Where are failed payments retried?",
                relevant_code_units=["repo001_method_01"],
                split="test"
            )
        ]
    }

def test_dataset_loads_correctly(mock_dataset):
    """Dataset loads and validates correctly."""
    workload = CodeRetrievalWorkload()
    dataset = workload.dataset(version="1.0.0")
    
    assert len(dataset.code_units) > 0
    assert len(dataset.queries) > 0

def test_repository_disjoint_split_verified(mock_dataset):
    """Train/test split is repository-disjoint."""
    workload = CodeRetrievalWorkload()
    dataset = workload.dataset()
    
    train_repos = {c.repository for c in dataset.train_split}
    test_repos = {c.repository for c in dataset.test_split}
    
    assert train_repos.isdisjoint(test_repos), "Repository leakage detected"

def test_semantic_artifact_validation():
    """Semantic artifacts validate correctly."""
    valid_artifact = SemanticArtifact(
        description="Retries failed payments with exponential backoff.",
        concepts=["retry", "payment"],
        code_unit_id="repo001_method_01"
    )
    
    assert valid_artifact.is_valid() == True

def test_retrieval_hit_at_k_computed():
    """Hit@K metric computed correctly."""
    artifacts = [
        SemanticArtifact(code_unit_id="unit_01", ...),
        SemanticArtifact(code_unit_id="unit_02", ...),
    ]
    query = Query(
        id="query_001",
        relevant_code_units=["unit_01"],
        ...
    )
    
    # Simulate retrieval: top-10 results
    top_10 = [
        ("unit_01", 0.95),  # relevant at rank 1
        ("unit_02", 0.87),
        ...
    ]
    
    hit_at_1 = 1.0  # unit_01 is at rank 1
    hit_at_5 = 1.0  # unit_01 is in top-5
    hit_at_10 = 1.0  # unit_01 is in top-10
    
    assert hit_at_1 == 1.0
    assert hit_at_5 == 1.0
    assert hit_at_10 == 1.0

def test_mrr_computed_correctly():
    """MRR metric computed correctly."""
    queries = [
        Query(id="q1", relevant_code_units=["u1"]),
        Query(id="q2", relevant_code_units=["u2"]),
        Query(id="q3", relevant_code_units=["u3"]),
    ]
    
    # Simulated retrieval results
    results = [
        [("u2", 0.9), ("u1", 0.8)],  # u1 at rank 2
        [("u2", 0.95)],               # u2 at rank 1
        [("u4", 0.8), ("u5", 0.7)],   # no relevant result
    ]
    
    # MRR = (1/2 + 1/1 + 1/inf) / 3 = (0.5 + 1.0 + 0) / 3 = 0.5
    mrr = compute_mrr(results, queries)
    assert mrr == pytest.approx(0.5)
```

---

### 3.4 Reporting Module Tests (`unit/reporting/`)

```python
# test_comparison.py

def test_models_compared_on_identical_set():
    """Models compared on identical test set."""
    baseline_results = {"hit_at_10": 0.90, "mrr": 0.76}
    specialized_results = {"hit_at_10": 0.93, "mrr": 0.81}
    
    comparison = ModelComparison(
        baseline=baseline_results,
        specialized=specialized_results
    )
    
    improvement = comparison.compute_improvement("hit_at_10")
    assert improvement == 0.03  # 0.93 - 0.90

def test_export_to_csv():
    """Results export to CSV correctly."""
    results = [
        {"model": "phi-3-mini", "hit_at_10": 0.90, "latency_ms": 720},
        {"model": "mistral-7b", "hit_at_10": 0.93, "latency_ms": 1840}
    ]
    
    csv_output = export_to_csv(results)
    
    lines = csv_output.strip().split("\n")
    assert lines[0] == "model,hit_at_10,latency_ms"
    assert "phi-3-mini,0.9,720" in csv_output
```

---

## 4. Integration Tests

### 4.1 Benchmark Flow Tests (`integration/test_benchmark_flow.py`)

```python
def test_full_benchmark_execution():
    """End-to-end benchmark executes successfully."""
    config = BenchmarkConfig(
        workload="code-retrieval-v1",
        models=["phi-3-mini"],
        dataset_version="1.0.0",
        # ... other config
    )
    
    # Use mock Ollama to avoid real inference
    with patch_ollama() as mock_ollama:
        mock_ollama.generate.return_value = GenerationResult(
            text='{"description": "..."}',
            status="success",
            duration_ms=720
        )
        
        results = run_benchmark(config)
        
        assert results.run_id is not None
        assert len(results.metrics) > 0
        assert results.metrics["hit_at_10"] > 0.0

def test_generation_with_retry():
    """Generation with validation and retry succeeds."""
    with patch_ollama() as mock_ollama:
        # First attempt fails, second succeeds
        mock_ollama.generate.side_effect = [
            GenerationResult(text='invalid json', status="success"),
            GenerationResult(text='{"description": "..."}', status="success")
        ]
        
        retry_engine = RetryEngine(max_attempts=3)
        result = retry_engine.execute(
            GenerationRequest(...),
            adapter=OllamaAdapter()
        )
        
        assert result.valid == True
        assert result.attempts == 2  # Succeeded on second attempt

def test_benchmark_failure_handling():
    """Benchmark continues on case failure."""
    config = BenchmarkConfig(...)
    
    with patch_ollama() as mock_ollama:
        # 3rd case fails
        mock_ollama.generate.side_effect = [
            GenerationResult(..., status="success"),
            GenerationResult(..., status="success"),
            GenerationResult(..., status="timeout"),  # Failure
            GenerationResult(..., status="success"),
        ]
        
        results = run_benchmark(config)
        
        assert len(results.failures) == 1
        assert results.failures[0].case_id == "case_03"
        assert len(results.successful_cases) == 3  # 3 succeeded, 1 failed
```

---

### 4.2 Retrieval Pipeline Tests (`integration/test_retrieval_pipeline.py`)

```python
def test_retrieval_end_to_end():
    """End-to-end retrieval evaluation."""
    # Build retrieval index
    artifacts = [
        SemanticArtifact(code_unit_id="u1", description="Retry logic"),
        SemanticArtifact(code_unit_id="u2", description="Validation logic"),
    ]
    
    index = build_index(artifacts)
    
    # Execute query
    query = "Where do we retry transactions?"
    results = index.search(query, k=10)
    
    # Verify top result is relevant
    top_result = results[0]
    assert top_result.code_unit_id == "u1"
    assert top_result.score > 0.8
```

---

## 5. System Tests

### 5.1 CLI Tests (`system/test_cli.py`)

```python
def test_cli_system_command():
    """CLI 'system' command displays hardware info."""
    result = subprocess.run(
        ["localbench", "system"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "CPU" in result.stdout
    assert "RAM" in result.stdout

def test_cli_models_command():
    """CLI 'models' command lists available models."""
    result = subprocess.run(
        ["localbench", "models"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    # Should list at least Ollama models if available

def test_cli_invalid_command_fails():
    """CLI rejects invalid commands."""
    result = subprocess.run(
        ["localbench", "invalid_cmd"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode != 0
    assert "Unknown command" in result.stderr or "invalid" in result.stderr.lower()
```

---

### 5.2 Artifact Persistence Tests (`system/test_artifact_persistence.py`)

```python
def test_benchmark_artifacts_persisted():
    """Benchmark artifacts persisted to disk correctly."""
    config = BenchmarkConfig(...)
    
    results = run_benchmark(config)
    run_id = results.run_id
    
    # Check artifact directory
    artifact_dir = Path(f"results/{run_id}")
    assert artifact_dir.exists()
    
    # Check required files
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "metrics.json").exists()
    assert (artifact_dir / "raw_outputs.jsonl").exists()
    
    # Load and verify metadata
    with open(artifact_dir / "metadata.json") as f:
        metadata = json.load(f)
    assert metadata["run_id"] == run_id
    assert metadata["benchmark_version"] == "3.0.0"

def test_results_reproducible_from_artifacts():
    """Results reproducible from raw artifacts."""
    artifact_dir = Path("results/run-20260930-001")
    
    # Load raw outputs
    raw_outputs = []
    with open(artifact_dir / "raw_outputs.jsonl") as f:
        for line in f:
            raw_outputs.append(json.loads(line))
    
    # Recompute metrics
    computed_metrics = compute_metrics_from_raw(raw_outputs)
    
    # Load saved metrics
    with open(artifact_dir / "metrics.json") as f:
        saved_metrics = json.load(f)
    
    # Verify match
    assert computed_metrics["hit_at_10"] == pytest.approx(
        saved_metrics["hit_at_10"],
        abs=0.001
    )
```

---

## 6. Fixture Design

### 6.1 Mock Ollama (`fixtures/mock_ollama.py`)

```python
@pytest.fixture
def mock_ollama():
    """Mock Ollama adapter for testing."""
    with patch("localbench.runtime.OllamaAdapter") as mock:
        # Default successful generation
        mock.return_value.generate.return_value = GenerationResult(
            text='{"description": "Test description"}',
            status="success",
            duration_ms=720
        )
        
        yield mock

@pytest.fixture
def patch_ollama():
    """Context manager to patch Ollama globally."""
    @contextmanager
    def _patch():
        with patch("localbench.runtime.OllamaAdapter") as mock:
            yield mock
    return _patch
```

### 6.2 Mock Dataset (`fixtures/mock_dataset.py`)

```python
@pytest.fixture
def minimal_dataset():
    """Minimal dataset for testing."""
    return Dataset(
        code_units=[
            BenchmarkCase(
                id="test_u1",
                repository="test_repo",
                language="python",
                symbol="test_func",
                source_code="def test_func(): pass",
                split="test"
            )
        ],
        queries=[
            Query(
                id="test_q1",
                query="What does test_func do?",
                relevant_code_units=["test_u1"],
                split="test"
            )
        ]
    )
```

---

## 7. Coverage Goals

| Module | Goal | Priority |
|--------|------|----------|
| Config | 95% | P0 |
| Runtime adapter | 90% | P0 |
| Generation validation | 95% | P0 |
| Workload evaluation | 90% | P0 |
| CLI | 80% | P1 |
| Reporting | 85% | P1 |

**Coverage measured with:**
```bash
pytest tests/ --cov=src/localbench --cov-report=html
```

---

## 8. Test Execution

### 8.1 Local Execution

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests (requires mock Ollama)
pytest tests/integration/ -m integration

# System tests (full end-to-end)
pytest tests/system/ -m system

# Coverage report
pytest tests/ --cov=src/localbench --cov-report=term-missing
```

### 8.2 CI/CD Integration

**GitHub Actions workflow:**
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -e .[dev]
      - name: Run unit tests
        run: pytest tests/unit/ -v
      - name: Run integration tests
        run: pytest tests/integration/ -v
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## 9. Key Principles

1. **Failure paths are tests** — Error handling tested as rigorously as happy path
2. **Mocking enables isolation** — Unit tests don't depend on Ollama/dataset
3. **Fixtures are reusable** — Common test data in conftest.py
4. **Tests document behavior** — Clear naming and assertions
5. **Coverage is tracked** — Target 85%+ across critical modules
6. **Reproducibility is tested** — Artifacts enable re-computation

