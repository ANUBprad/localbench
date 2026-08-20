# LocalBench --- Evaluation Protocol v3

**Status:** Frozen (subject to phase verification)  
**Last Updated:** 2026-08-19  
**Role:** Defines benchmark methodology, metric definitions, fair comparison constraints, and statistical rigor.

---

## 1. Evaluation Philosophy

LocalBench evaluates models through **downstream task performance**, not surface-level text quality. A generated artifact is useful only if it improves retrieval, not merely because it looks plausible.

**Core principles:**
- Measure what matters: downstream utility
- One protocol, applied uniformly
- Fairness through controlled variation
- Failure visibility and analysis
- Resource constraints in context

---

## 2. Benchmark Protocol: Frozen Specification

### 2.1 Pre-Benchmark Phase

**Before any model execution:**

1. **Configuration freeze** → Workload, dataset, queries immutable
2. **Hardware profiling** → Capture baseline system state
3. **Model verification** → Ollama health check, model size validation
4. **Dataset validation** → All splits present, ground truth loaded

**Recorded metadata:**
```json
{
  "benchmark_version": "3.0.0",
  "workload": "code-retrieval-v1",
  "workload_version": "1.0.0",
  "dataset_version": "1.0.0",
  "protocol_version": "1.0.0",
  "timestamp": "2026-09-30T14:00:00Z",
  "git_commit": "abc123def...",
  "hardware": {
    "machine_id": "...",
    "os": "Ubuntu 22.04",
    "cpu_model": "Intel Core i7-11700K",
    "cpu_cores": 8,
    "ram_total_gb": 32,
    "gpu": "NVIDIA RTX 3090",
    "vram_gb": 24
  }
}
```

---

### 2.2 Warm-up Phase

**Purpose:** Stabilize system state before measurement.

**Procedure:**
1. Execute first 5 benchmark cases (not counted toward results)
2. Discard warm-up outputs
3. Allow hardware to reach steady state (CPU frequency, cache, RAM usage)
4. Record system state after warm-up

**Rationale:**
- First inference is often slower (model loading, GPU initialization)
- Subsequent runs are more representative of deployment performance
- Warm-up outputs excluded from benchmark metrics

---

### 2.3 Benchmark Execution Phase

**For each benchmark case:**

```
Case i ∈ dataset.test_split
  ↓
Record start time & system state
  ↓
Generate semantic artifact with exact config
  ↓
Validate & retry if needed (record attempts)
  ↓
Record generation time, tokens (if available)
  ↓
Record end system state (RSS, CPU, VRAM)
  ↓
Store: (case_id, model, generation_result, system_metrics)
```

**Execution constraints:**
- **Temperature:** Fixed at 0.3 (deterministic, low variance)
- **Max tokens:** 256 (sufficient for descriptions)
- **Timeout:** 30 seconds per generation (Ollama default)
- **Retries:** Bounded at 3 attempts per case
- **Repetitions:** Single pass (not repeated runs; sufficient for Phase 5)

**Recorded per case:**
```json
{
  "case_id": "repo001_py_class_PaymentProcessor_method_process_retry",
  "model": "phi-3-mini",
  "attempt": 1,
  "generation_result": {
    "text": "...",
    "duration_ms": 720,
    "finish_reason": "stop",
    "status": "success"
  },
  "validation": {
    "valid": true,
    "attempts": 1,
    "schema": "SemanticArtifact"
  },
  "system_metrics": {
    "start_rss_mb": 1024,
    "end_rss_mb": 1128,
    "cpu_percent": 45.2,
    "start_timestamp": "2026-09-30T14:00:05Z",
    "end_timestamp": "2026-09-30T14:00:06Z"
  }
}
```

---

### 2.4 Retrieval Evaluation Phase

**After all generations complete:**

1. **Load all semantic artifacts** (from successful generations)
2. **Build retrieval index** → Embed artifacts locally
3. **Execute queries** → For each test query, retrieve top-K
4. **Compare results** → Against ground-truth relevance
5. **Compute metrics** → Hit@K, MRR aggregated

**Index construction:**
- **Embedding method:** Local model (same as generation model) or fixed baseline (e.g., all-MiniLM-L6-v2)
- **Embedding consistency:** Use identical embedding model across all benchmarks
- **Index format:** FAISS or similar (deterministic, reproducible)

**Query execution:**
```python
for query_id in test_queries:
    query = queries[query_id]
    top_k_results = index.search(query.text, k=10)
    
    ground_truth = query.relevant_code_units
    retrieved_ids = [result.code_unit_id for result in top_k_results]
    
    # Compute metrics
    hits = {1: ..., 3: ..., 5: ..., 10: ...}
    mrr = mean_reciprocal_rank(retrieved_ids, ground_truth)
```

---

## 3. Metrics: Definitions & Rationale

### 3.1 Primary Metrics (Downstream Task)

**Hit@K** — Proportion of queries with at least one relevant result in top-K

```
Hit@K = (# queries with ≥1 relevant result in top K) / (# total queries)
Range: [0.0, 1.0]
Higher is better
```

**Key values tracked:**
- **Hit@1** — Top result is relevant (strict)
- **Hit@3** — Relevant result in top-3 (practical for developer workflows)
- **Hit@5** — Lenient threshold
- **Hit@10** — Comprehensive retrieval

**Interpretation:**
- Hit@10 ≥ 0.9 → Strong performance (developer finds code in one search)
- Hit@10 ∈ [0.7, 0.9) → Acceptable with caveats
- Hit@10 < 0.7 → Inadequate for deployment

---

**MRR (Mean Reciprocal Rank)** — Average position of first relevant result

```
MRR = (1/|Q|) × Σ (1 / rank_of_first_relevant)
Range: (0.0, 1.0]
Higher is better
```

**Example:**
- Query 1: relevant at position 2 → 1/2 = 0.5
- Query 2: relevant at position 1 → 1/1 = 1.0
- Query 3: no relevant result → 1/∞ = 0.0
- MRR = (0.5 + 1.0 + 0.0) / 3 = 0.5

**Interpretation:**
- MRR > 0.8 → Excellent ranking (first/second result usually relevant)
- MRR ∈ [0.6, 0.8) → Good ranking
- MRR < 0.6 → Ranking needs improvement

---

### 3.2 Secondary Metrics (Resource & Reliability)

**Latency** — Time from input to first output token

```
latency_ms = (generation_end_time - generation_start_time)
Tracked: min, max, median, mean
Relevant for: Interactive use cases
```

**Throughput** — Tokens per second

```
throughput_tps = output_tokens / (duration_seconds)
When available: extracted from Ollama
Fallback: estimated from token count / wall time
```

**Memory** — Process-level RAM usage

```
peak_rss_mb = max(process.rss) during benchmark
Tracked: min, max, mean across cases
Relevant for: Resource-constrained deployment
```

**Disk Footprint** — Model size on disk

```
model_size_mb = (checkpoint_bytes / 1_000_000)
Directly from model metadata
Relevant for: Storage and download constraints
```

**Reliability** — Success rate without retry

```
reliability = (# successful generations without retry) / (# total cases)
Range: [0.0, 1.0]
Tracked separately from "success after retry"
```

---

### 3.3 Artifact Quality Metrics (Supporting Evidence)

These are **not** primary success metrics but support analysis.

**Semantic Artifact Validity** — Passes JSON/schema validation

```
validity = (# valid artifacts) / (# attempted generations)
Range: [0.0, 1.0]
```

**Grounding** — Artifact references actual code concepts

Manual assessment:
- Does the description mention specific variables, control flow, or logic from the code?
- Example: ✓ "Exponential backoff with max_attempts parameter"
- Example: ✗ "Does something with retries" (too generic)

---

## 4. Fair Comparison Framework

### 4.1 Identical Conditions for Comparable Models

When comparing models, all must be evaluated under **identical conditions**:

| Parameter | Identical? | Notes |
|-----------|-----------|-------|
| Dataset & split | ✓ YES | Same test set, same ground truth |
| Workload version | ✓ YES | Same version, same protocol |
| Prompt template | ✓ YES | Identical input construction |
| Temperature | ✓ YES | Fixed at 0.3 |
| Max tokens | ✓ YES | Fixed at 256 |
| Timeout | ✓ YES | Fixed at 30s |
| Hardware | ✓ YES | Same machine (or documented equivalence) |
| Embedding model (for retrieval) | ✓ YES | Same embedder across all runs |
| Seed (if applicable) | ✓ YES | Deterministic inference |

---

### 4.2 Intentional Differences & Documentation

Some differences are expected. **All must be recorded:**

| Difference | Reason | Documentation |
|-----------|--------|---|
| Model size | Specialization vs baseline | Parameter count, quantization |
| Training data | Ablation study | Dataset version, examples count |
| Quantization | Resource investigation | Q4_K_M vs Q5_K_M, etc. |
| Hardware (cross-machine) | Scaling study | CPU/GPU differences noted |

**Protocol violations** (never allowed without explicit decision):
- Tuning prompt for specific model
- Changing max_tokens per model
- Using different embedding model
- Removing failed cases from metrics

---

## 5. Failure Handling

### 5.1 Failure Taxonomy

```
GenerationFailure
├── ProviderFailure
│   ├── Timeout (> 30s)
│   ├── OOM (out of memory)
│   └── ConnectionError
├── StructureFailure
│   ├── MalformedJSON
│   ├── MissingRequiredField
│   └── TypeMismatch
├── SemanticFailure
│   ├── Incoherent (nonsense output)
│   └── Incomplete (truncated/repetitive)
└── Retrieval Failure
    ├── NoRelevantResult (query impossible to answer)
    └── IndexError (lookup failed)
```

---

### 5.2 Failure Recording & Analysis

**Every failure is recorded:**

```json
{
  "case_id": "repo001_method_01",
  "model": "phi-3-mini",
  "failure_type": "MalformedJSON",
  "attempt": 2,
  "raw_output": "...",
  "error_message": "JSON decode error at line 3",
  "recovery_attempted": true,
  "final_status": "FAILED"
}
```

**Failure analysis:**
1. Count failures by type
2. Identify patterns (specific models, cases, concepts)
3. Assess impact on metrics (included or excluded?)
4. Decide on remediation (retry threshold, prompt adjustment, etc.)

**Inclusion in metrics:**
- Failed cases **excluded** from Hit@K, MRR
- Reliability = (successful cases / total cases)
- Report separately: "4/100 cases failed; metrics based on 96 successful cases"

---

## 6. Statistical Rigor

### 6.1 When Applicable

**Statistical testing is applied when:**
- Comparing two models on same test set
- Sample size > 30 (sufficient for normality assumption)
- Metric has natural variance (e.g., Hit@K, MRR)

**Not applicable when:**
- Single model, single run (no comparison)
- Deterministic metrics (e.g., disk size)
- Aggregated results (summary statistics)

---

### 6.2 Comparison Methodology

**Scenario: Comparing Baseline vs Specialized model**

```python
# Data: Hit@10 for each test query
baseline_hits = [query_hit@10 for query in test_queries]
specialized_hits = [query_hit@10 for query in test_queries]

# Test for difference
effect_size = mean(specialized_hits) - mean(baseline_hits)

# Significance (paired t-test)
from scipy import stats
t_stat, p_value = stats.ttest_rel(specialized_hits, baseline_hits)

# Report
print(f"Hit@10: Baseline={mean(baseline_hits):.3f}, Specialized={mean(specialized_hits):.3f}")
print(f"Improvement: {effect_size:.3f} ({effect_size/mean(baseline_hits)*100:.1f}%)")
if p_value < 0.05:
    print(f"Significant at α=0.05 (p={p_value:.4f})")
else:
    print(f"Not significant (p={p_value:.4f})")
```

**Constraints:**
- Do **not** p-hack (no selective testing)
- Report sample size and assumptions
- Include confidence intervals
- Accept null results (specialization may not help)

---

## 7. Benchmark Report Template

### 7.1 Structure

```markdown
# LocalBench: Code Retrieval Benchmark Report

## Experiment Metadata
- Run ID: run-20260930-phi3-baseline
- Timestamp: 2026-09-30 14:00 UTC
- Hardware: Intel i7-11700K, 32GB RAM, RTX 3090
- Models: phi-3-mini, mistral-7b, gemma-2
- Protocol version: 3.0.0

## Primary Results

### Hit@K Metrics
| Model | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|-------|-------|-------|-------|--------|-----|
| phi-3-mini (baseline) | 0.62 | 0.76 | 0.84 | 0.90 | 0.76 |
| mistral-7b (baseline) | 0.68 | 0.82 | 0.88 | 0.93 | 0.81 |
| gemma-2 (baseline) | 0.71 | 0.84 | 0.89 | 0.94 | 0.82 |

### Resource Metrics
| Model | Latency (ms) | Throughput (tps) | Peak RAM (MB) | Disk (MB) |
|-------|--------------|------------------|---------------|-----------|
| phi-3-mini | 720 | 45.2 | 1128 | 2048 |
| mistral-7b | 1840 | 38.1 | 2456 | 3800 |
| gemma-2 | 2210 | 31.5 | 3104 | 5400 |

## Analysis
[Interpretation of results, trade-offs, failure patterns]

## Failures
- phi-3-mini: 2 timeouts, 1 malformed JSON (3/100 = 3% failure rate)
- mistral-7b: 1 timeout (1/100 = 1% failure rate)
- gemma-2: 0 failures (0/100 = 0% failure rate)

## Limitations & Caveats
- Single hardware configuration; results may vary on different machines
- Test set from specific repository types; may not generalize
- Temperature fixed at 0.3; higher randomness at higher temps

## Artifacts
- Raw results: results/run-20260930-phi3-baseline/
- Metrics: results/run-20260930-phi3-baseline/metrics.json
- Detailed logs: results/run-20260930-phi3-baseline/raw_outputs.jsonl
```

---

## 8. Ablation Studies

### 8.1 Training Data Size Ablation

**Hypothesis:** Larger training sets improve specialization.

**Protocol:**
1. Freeze baseline benchmark
2. Train specialized model on:
   - 50 examples (small subset)
   - 112 examples (medium subset)
   - 225 examples (full train set)
3. Benchmark each specialized variant on **identical test set**
4. Compare Hit@10, MRR, latency

**Analysis:**
- Plot Hit@10 vs training examples
- Assess scaling curve (linear, logarithmic, plateau)
- Identify diminishing returns

---

### 8.2 Model Size Ablation

**Hypothesis:** Small models can approach medium/large performance with specialization.

**Protocol:**
1. Baseline: phi-3-mini (0.5B), mistral-7b (7B), gemma-2 (9B)
2. Specialized: phi-3-mini fine-tuned
3. Compare:
   - Hit@10 across sizes
   - Latency vs quality (trade-offs)
   - Resource efficiency (Hit@10 per GB RAM)

---

## 9. Reproducibility Checklist

- [ ] Exact dataset version recorded
- [ ] Protocol version recorded
- [ ] Workload version recorded
- [ ] Model identifier & quantization recorded
- [ ] Hardware specification recorded
- [ ] Software versions recorded (Python, Ollama, libraries)
- [ ] Random seed (if applicable) recorded
- [ ] All hyperparameters (temperature, max_tokens, timeout) recorded
- [ ] Raw outputs persisted
- [ ] Metrics reproducible from raw outputs
- [ ] Report includes all recorded metadata
- [ ] Git commit hash for reproducibility

---

## 10. Key Principles

1. **Downstream task is truth** — Retrieval metrics matter most
2. **One protocol, uniform application** — All models evaluated identically
3. **Fairness through constraints** — No tuning for specific models
4. **Failure is visible** — All failures recorded and analyzed
5. **Raw data preserved** — Derived metrics reproducible
6. **Metadata is complete** — Full reproducibility possible
7. **Claims are scoped** — Tied to workload, dataset, protocol, hardware
8. **Statistics are rigorous** — No p-hacking, proper sample sizes

