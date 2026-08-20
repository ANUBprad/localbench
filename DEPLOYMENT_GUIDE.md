# LocalBench --- Deployment & Usage Guide v3

**Status:** Operational guide for Phase 1+ implementation  
**Last Updated:** 2026-08-19  
**Role:** Provides deployment instructions, configuration examples, and usage workflows.

---

## 1. Prerequisites

### 1.1 System Requirements

**Minimum:**
- CPU: 4 cores, 2.5+ GHz
- RAM: 8 GB
- Disk: 20 GB free
- GPU: Optional (speeds up inference 5–10x)
- OS: Linux/macOS/Windows (with WSL2)

**Recommended:**
- CPU: 8+ cores
- RAM: 16+ GB
- Disk: 50 GB free
- GPU: NVIDIA (CUDA 11.8+) or equivalent
- OS: Linux (Ubuntu 20.04+)

### 1.2 Software Dependencies

```bash
# Required
Python 3.10+
pip / conda
Ollama 0.29+

# Optional (for GPU acceleration)
CUDA 11.8+ (NVIDIA)
cuDNN (NVIDIA)
```

---

## 2. Installation

### 2.1 Clone Repository

```bash
git clone https://github.com/your-org/localbench.git
cd localbench
```

### 2.2 Install LocalBench

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .

# Install development dependencies (optional)
pip install -e .[dev]
```

### 2.3 Install Ollama

```bash
# Visit https://ollama.ai and download
# Or on macOS/Linux:
curl https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve
```

**Verify Ollama is running:**
```bash
ollama list  # Should show available models
```

---

## 3. Configuration

### 3.1 Default Configuration File

Create `config.yaml` in project root:

```yaml
# LocalBench Configuration v3

workload: "code-retrieval-v1"
workload_version: "1.0.0"
dataset_version: "1.0.0"
protocol_version: "1.0.0"

# Model selection
models:
  - "phi-3-mini"     # Target for specialization
  - "mistral-7b"     # Baseline medium
  - "gemma-2"        # Baseline larger

# Generation parameters
generation:
  temperature: 0.3   # Deterministic, low variance
  max_tokens: 256
  timeout_seconds: 30
  retry_max_attempts: 3

# Benchmark parameters
benchmark:
  seed: 42
  warmup_cases: 5    # Number of cases to discard before measurement
  repetitions: 1     # Single pass for Phase 5
  output_dir: "./results"

# Hardware profiling
profiling:
  capture_rss: true
  capture_cpu: true
  capture_vram: true
  capture_disk: true

# Ollama configuration
ollama:
  base_url: "http://localhost:11434"
  timeout_seconds: 60
  retry_max_attempts: 3

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_dir: "./logs"
  console_output: true
```

### 3.2 Environment Variable Overrides

```bash
# Override configuration via environment
export LOCALBENCH_WORKLOAD="code-retrieval-v1"
export LOCALBENCH_MODELS="phi-3-mini,mistral-7b"
export LOCALBENCH_TEMPERATURE="0.3"
export LOCALBENCH_SEED="42"

# Run benchmark
localbench benchmark
```

### 3.3 Command-Line Overrides

```bash
# Inline overrides
localbench benchmark \
  --workload code-retrieval-v1 \
  --models phi-3-mini,mistral-7b \
  --temperature 0.3 \
  --seed 42 \
  --output-dir ./results
```

---

## 4. Quick Start

### 4.1 Verify Installation

```bash
# Check LocalBench version
localbench --version

# Display system information
localbench system

# List available models
localbench models
```

**Expected output:**
```
LocalBench v3.0.0

System Information
──────────────────────────────────────────
OS: Ubuntu 22.04
CPU: Intel(R) Core(TM) i7-11700K CPU @ 3.60GHz
Cores: 8
RAM: 31.8 GB
GPU: NVIDIA GeForce RTX 3090

Available Models
──────────────────────────────────────────
phi-3-mini (0.5B)  → 2.0 GB
mistral-7b (7B)    → 3.8 GB
gemma-2 (9B)       → 5.4 GB
```

### 4.2 Test Single Model Inference

```bash
# Quick inference test (not part of benchmark)
localbench ask "Analyze this Python function and describe its purpose"

# Model output + timing displayed
```

---

## 5. Running Benchmarks

### 5.1 Baseline Benchmark (Phase 5)

```bash
# Run full benchmark with default config
localbench benchmark

# Output:
# ✓ Loaded workload: code-retrieval-v1
# ✓ Verified dataset: 113 test cases
# ✓ Profiling system...
# ✓ Checking Ollama models...
# ✓ Starting warm-up (5 cases)...
# ✓ Executing benchmark (113 cases)
#   [████████████████████] 100% (2m 45s)
# ✓ Running retrieval evaluation...
# ✓ Computing metrics...
# 
# Benchmark Complete
# ─────────────────────────────────────────
# Run ID: run-20260930-phi3-baseline
# Workload: code-retrieval-v1
# Models: 3
# Cases: 113 (0 failures)
# 
# Results:
# ─────────────────────────────────────────
# results/run-20260930-phi3-baseline/
```

**View detailed results:**
```bash
# Summary report
cat results/run-20260930-phi3-baseline/report.md

# Metrics JSON
cat results/run-20260930-phi3-baseline/metrics.json | jq

# Raw outputs (for re-analysis)
head -20 results/run-20260930-phi3-baseline/raw_outputs.jsonl
```

---

### 5.2 Specialized Model Benchmark (Phase 7)

```bash
# Benchmark the fine-tuned model
localbench benchmark \
  --models phi-3-mini-lora \
  --lora-checkpoint experiments/phi3_lora_full/final
```

---

### 5.3 Comparison & Recommendation

```bash
# Compare two benchmark runs
localbench compare \
  run-20260930-phi3-baseline \
  run-20261001-phi3-specialized

# Output:
# ─────────────────────────────────────────
# Comparison: phi3-baseline vs phi3-specialized
# ─────────────────────────────────────────
# 
# Hit@10:     0.900 → 0.927 (+3.0%, p=0.023)
# MRR:        0.763 → 0.802 (+5.1%, p=0.011)
# Latency:    720ms → 728ms (+1.1%)
# Peak RAM:   1128MB → 1140MB (+1.1%)
#
# Specialization improves retrieval quality with minimal overhead.
```

```bash
# Get model recommendations
localbench recommend \
  --min-hit-at-10 0.85 \
  --max-latency-ms 1000 \
  --max-ram-gb 4

# Output:
# ─────────────────────────────────────────
# Model Recommendation
# ─────────────────────────────────────────
# Recommended: phi-3-mini-specialized
# 
# Why: Highest retrieval quality among models satisfying all constraints.
# 
# Constraints Applied:
#   ✓ Hit@10 ≥ 0.85
#   ✓ Latency ≤ 1000ms
#   ✓ RAM ≤ 4GB
# 
# Candidate Models:
#   ✓ phi-3-mini-specialized: Hit@10=0.927, Latency=728ms, RAM=1.14GB
#   ✗ mistral-7b: Hit@10=0.933, RAM=2.46GB (satisfies)
#   ✗ gemma-2: Hit@10=0.945, RAM=3.10GB (satisfies)
#
# Rejected:
#   - phi-3-mini (baseline): Hit@10=0.900 (below minimum)
#   - gemma-9b: RAM=5.40GB (exceeds maximum)
```

---

## 6. Workload Management

### 6.1 List Available Workloads

```bash
localbench workload list

# Output:
# Available Workloads
# ────────────────────────────────────────
# code-retrieval-v1.0.0 (✓ available)
#   Flagship: Benchmark code semantic retrieval
#   Dataset: 450 functions from 3 repositories
#   Queries: 45 test queries
#   Status: Complete & Frozen
#
# future-workload-v1.0.0 (○ planned)
#   Status: Not yet implemented
```

### 6.2 Download Workload/Dataset

```bash
# Datasets are packaged with LocalBench v3.0.0
# But can be refreshed:
localbench workload download code-retrieval-v1

# Verify dataset
localbench workload validate code-retrieval-v1

# Output:
# Dataset Validation: code-retrieval-v1
# ────────────────────────────────────────
# ✓ Schema valid
# ✓ Referential integrity
# ✓ Repository-disjoint splits
# ✓ 450 code units, 45 queries
# ✓ All checks passed
```

---

## 7. Advanced Usage

### 7.1 Custom Configuration Profile

Create `profiles/research.yaml`:

```yaml
# Research-focused profile (more verbosity, extra metrics)
workload: "code-retrieval-v1"
models: ["phi-3-mini"]

benchmark:
  repetitions: 3        # Multiple runs for statistics
  seed: 42
  output_dir: "./research_results"

logging:
  level: "DEBUG"        # Verbose logging
  log_dir: "./logs/research"

# Additional metrics collection
extra_metrics:
  - "token_distribution"
  - "semantic_similarity"
  - "concept_coverage"
```

Use profile:
```bash
localbench benchmark --profile research
```

---

### 7.2 Debugging a Specific Case

```bash
# Verbose output for a single case
localbench debug-case repo001_method_0001 \
  --model phi-3-mini \
  --verbose

# Output:
# Debugging: repo001_method_0001
# ────────────────────────────────────────
# Case ID: repo001_method_0001
# 
# Code Unit:
#   Symbol: PaymentProcessor.process_retry
#   Lines: 42–67
#   Language: python
# 
# Generation:
#   Prompt: [full prompt shown]
#   Model: phi-3-mini
#   Temperature: 0.3
#   
#   Raw output:
#   {"description": "...", ...}
#   
#   Validation: ✓ PASS
#   Duration: 720ms
# 
# Retrieved artifacts:
#   1. repo001_method_0001 (score: 0.95) [RELEVANT]
#   2. repo002_method_0042 (score: 0.82)
#   3. repo001_method_0003 (score: 0.78)
```

---

### 7.3 Export Results for External Analysis

```bash
# Export to CSV
localbench export \
  results/run-20260930-baseline \
  --format csv \
  --output results/baseline_metrics.csv

# Export to JSON (full details)
localbench export \
  results/run-20260930-baseline \
  --format json \
  --output results/baseline_full.json

# Generate comparison report (Markdown)
localbench export \
  results/run-20260930-baseline \
  results/run-20261001-specialized \
  --format markdown \
  --output results/COMPARISON.md
```

---

## 8. Troubleshooting

### 8.1 Ollama Connection Issues

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If connection refused:
# 1. Start Ollama
ollama serve

# 2. Override base URL if Ollama on different host
export LOCALBENCH_OLLAMA_BASE_URL="http://192.168.1.100:11434"
localbench system

# 3. Check Ollama logs
tail -f ~/.ollama/logs/server.log
```

### 8.2 Model Not Found

```bash
# Pull model from Ollama
ollama pull phi-3-mini

# Verify model is available
ollama list

# LocalBench will auto-discover
localbench models
```

### 8.3 Out of Memory

```bash
# Reduce batch size or max_tokens
localbench benchmark \
  --max-tokens 128 \
  --temperature 0.3

# Or use smaller model
localbench benchmark --models phi-3-mini
```

### 8.4 Slow Performance

```bash
# Check GPU availability
localbench system

# If GPU present but not used, check CUDA:
python -c "import torch; print(torch.cuda.is_available())"

# Ensure Ollama is using GPU (check Ollama logs)
tail ~/.ollama/logs/server.log | grep GPU
```

---

## 9. Performance Tuning

### 9.1 Inference Optimization

| Setting | Impact | Trade-off |
|---------|--------|-----------|
| Lower max_tokens | Faster | Truncated descriptions |
| Higher temperature | More variance | Less reproducible |
| Batch inference | Faster | Not supported yet |
| Model quantization | Faster, lower RAM | Slight quality loss |

**Recommended for speed:**
```yaml
generation:
  temperature: 0.3    # Fixed
  max_tokens: 128     # Reduce from 256
  timeout_seconds: 20 # Tighten from 30
```

### 9.2 Disk Space Optimization

```bash
# Archive old results
tar -czf results/run-20260930-archived.tar.gz results/run-20260930-*/

# Remove raw outputs if metrics saved
rm results/run-20260930-*/raw_outputs.jsonl  # Careful: metrics must be computed first!

# Check disk usage
du -sh results/
```

---

## 10. Data Privacy

### 10.1 Privacy by Default

- **No telemetry** — All data remains local
- **No cloud sync** — Benchmarks never leave your machine
- **No model uploads** — Models run locally via Ollama
- **Artifacts are local** — Results stored in `./results/`

### 10.2 Sensitive Data Handling

If benchmarking proprietary code:
```bash
# Disable source code logging
localbench benchmark --no-source-logging

# This removes source code from artifacts (only metadata preserved)
# Useful for compliance, security reviews
```

---

## 11. Reproducibility

### 11.1 Document Your Environment

```bash
# Create environment report
localbench report-environment > ENVIRONMENT.txt

# This captures:
# - LocalBench version & commit
# - Python version
# - OS & kernel
# - Ollama version
# - Available models
# - GPU info
# - Configuration
```

### 11.2 Archive Results

```bash
# Create reproducible archive
mkdir -p archives/
cp -r results/run-20260930-baseline/ archives/
tar -czf archives/baseline-v1.tar.gz archives/run-20260930-baseline/

# Later: reproduce
tar -xzf archives/baseline-v1.tar.gz
localbench analyze archives/run-20260930-baseline/
```

---

## 12. Monitoring & Logging

### 12.1 View Logs

```bash
# Real-time logging during benchmark
localbench benchmark --log-level DEBUG

# View saved logs
tail -f logs/localbench.log

# Filter by level
grep ERROR logs/localbench.log
grep WARNING logs/localbench.log
```

### 12.2 Health Monitoring

```bash
# Monitor system resources during benchmark
localbench benchmark &
watch -n 1 'ps aux | grep localbench'  # Check CPU/memory
```

---

## 13. Summary: Typical Workflow

```bash
# 1. Install & verify
git clone https://github.com/your-org/localbench.git
cd localbench
pip install -e .
ollama pull phi-3-mini mistral-7b gemma-2
localbench system

# 2. Run baseline benchmark
localbench benchmark > baseline.log 2>&1

# 3. Review results
cat results/run-20260930-*/report.md
cat results/run-20260930-*/metrics.json | jq .

# 4. (Later) Train specialized model
# [Training steps...]

# 5. Benchmark specialized model
localbench benchmark --models phi-3-mini-lora --lora-checkpoint ...

# 6. Compare & recommend
localbench compare run-baseline run-specialized
localbench recommend --min-hit-at-10 0.85 --max-ram-gb 4

# 7. Export & analyze
localbench export run-baseline --format csv --output results.csv
python analyze_results.py results.csv
```

---

## 14. Getting Help

```bash
# Command-line help
localbench --help
localbench benchmark --help

# Report issues
# → GitHub Issues: https://github.com/your-org/localbench/issues

# Documentation
# → Architecture: ARCHITECTURE.md
# → Evaluation Protocol: EVALUATION_PROTOCOL.md
```

