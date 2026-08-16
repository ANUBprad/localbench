# LocalBench --- Application Flow v2.0

**Status:** v2.0 (Advanced)\
**Revision Date:** August 2026

---

## 1. Initialization & Health Check Flow

### Startup Sequence

```
┌─────────────────┐
│  User launches  │
│   CLI command   │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Load configuration  │
│  (hardcoded defaults)│
└────────┬─────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Detect system metadata      │
│  (CPU, RAM, GPU, OS)         │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Health check: Ollama    │
│  running?                │
└────┬─────────────┬───────┘
     │ YES         │ NO
     ▼             ▼
   [Continue]   [Error]
                  │
                  ▼
              ┌──────────────────────────┐
              │ Output actionable error: │
              │ "Ollama is not running"  │
              │ "Start with: ollama      │
              │ serve"                   │
              └──────────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  Discover models via Ollama    │
│  API (list available)          │
└────────┬─────────────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Cache model metadata      │
│  (parameter count,         │
│   quantization, etc)       │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  CLI ready                 │
│  (show available commands) │
└────────────────────────────┘
```

**Exit Criteria:**
- System metadata captured.
- Ollama status known (available or unavailable).
- Model list cached.

**Error Handling:**
- Ollama unavailable → clear error with recovery steps.
- Network timeout → same error (treat as unavailable).
- Malformed model metadata → skip model, log warning.

---

## 2. Model Discovery & Listing Flow

### `localbench models`

```
┌─────────────────────────────────┐
│  User: localbench models        │
└────────┬────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  CLI: Handle 'models' command    │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  App: Query ModelRegistry        │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Registry: Return cached models  │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  App: Normalize metadata         │
│  (ensure all fields present)     │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  CLI: Render table               │
│  ┌─────────────────────────────┐ │
│  │ Model        │ Params │ RAM │ │
│  │ qwen:7b      │ 7B     │ 4GB │ │
│  │ mistral:7b   │ 7B     │ 5GB │ │
│  └─────────────────────────────┘ │
└──────────────────────────────────┘
```

**Exit Criteria:**
- List is non-empty OR empty list with explanation.
- Each model shows: name, parameter count, estimated RAM.

**Error Cases:**
- No models available: "No models found. Run `ollama pull qwen:7b` to download one."
- Ollama unavailable: "Ollama is not running. Start with `ollama serve`."

---

## 3. Simple Text Generation Flow

### `localbench ask "Explain paging"`

```
┌───────────────────────────────────┐
│  User: localbench ask <prompt>    │
└────────┬────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  CLI: Parse arguments              │
│  - prompt (positional)             │
│  - model (flag, default: first)    │
│  - temperature (flag, default: 0.0)│
│  - max_tokens (flag, default: 512) │
└────────┬─────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Validate args                     │
│  - prompt non-empty? ✓             │
│  - model exists? ✓                 │
└────────┬─────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  App: Get model from registry      │
└────────┬─────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Create GenerationRequest          │
│  - model: "qwen:7b"               │
│  - prompt: "Explain paging"       │
│  - temperature: 0.0               │
│  - max_tokens: 512                │
└────────┬─────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Model adapter: Send to Ollama     │
│  via HTTP POST                     │
└────────┬─────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Ollama: Generate                  │
│  (inference happening)             │
└────────┬─────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Model adapter: Receive response   │
│  - raw text                        │
│  - token counts                    │
│  - timing                          │
└────────┬─────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Create GenerationResult           │
│  - text: "Paging is..."           │
│  - duration_ms: 2100              │
│  - tokens_per_second: 24.3        │
└────────┬─────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  CLI: Format and display output    │
│                                    │
│  >> Paging is a memory management │
│  technique that...                 │
│                                    │
│  [Timing: 2.1s, ~24 tokens/sec]  │
└────────────────────────────────────┘
```

**Exit Criteria:**
- Output is displayed.
- Timing information shown.

**Error Cases:**
- Model not found: "Model 'unknown' not found. Available: qwen:7b, mistral:7b"
- Ollama unavailable: "Ollama is not running."
- Timeout: "Model did not respond within 60s."
- Empty output: "Model returned empty response. Try different prompt."

---

## 4. Structured Generation Flow (with Retry)

### Quiz Generation (Internal)

```
┌──────────────────────────────────────┐
│  App: Request Quiz generation       │
│  - context: "OS notes"              │
│  - num_questions: 5                 │
└────────┬───────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Build Quiz generation prompt       │
│  "Generate a 5-question quiz in     │
│   JSON format: {...schema...}"      │
└────────┬───────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  StructuredGenerator: Attempt 1      │
│  - Model generates text             │
│  - Extract JSON from output         │
│  - Parse JSON → dict                │
└────────┬────────┬───────────────────┘
         │        │
      valid?      invalid
         │        (JSON parse error)
         ▼        │
       [Proceed]  ▼
                ┌──────────────────────┐
                │ Classify: json_parse │
                └────────┬─────────────┘
                         │
                         ▼
                ┌───────────────────────┐
                │ Recoverable? YES      │
                └────────┬──────────────┘
                         │
                         ▼
                ┌───────────────────────┐
                │ Construct retry       │
                │ prompt with           │
                │ diagnostics:          │
                │ "Your output was      │
                │  invalid JSON: {err}" │
                │ "Please retry: {...}" │
                └────────┬──────────────┘
                         │
                         ▼
                ┌───────────────────────┐
                │ Attempt 2: Generate   │
                │ with diagnostic       │
                └────────┬──────────────┘
                         │
                         ▼
                      valid?
                    ╱        ╲
                  yes         no
                  │            │
                  ▼            ▼
                [Proceed]  [Attempt 3]
                              │
                              ▼
                           valid?
                         ╱        ╲
                       yes         no
                       │            │
                       ▼            ▼
                    [Proceed]    [Max retries reached]
                                      │
                                      ▼
                                 Raise error:
                                 "Quiz generation
                                  failed after 3
                                  attempts"
         │
         ▼
┌──────────────────────────────────────┐
│  Validate JSON against Quiz schema   │
│  using Pydantic                      │
└────────┬───────────────────────────────┘
         │
      valid?
    ╱      ╲
  yes       no
  │         │
  ▼         ▼
[Done]  [Classify: schema_validation]
            │
            ▼
        [Attempt 2 with diagnostics]
            │
            ▼
           ...
         │
         ▼
┌──────────────────────────────────────┐
│  Return StructuredGenerationResult   │
│  - valid: true                       │
│  - attempts: 2                       │
│  - parsed: Quiz object               │
│  - attempt_log: [details...]         │
└──────────────────────────────────────┘
```

**Result Structure:**
```json
{
  "valid": true,
  "attempts": 2,
  "schema_name": "Quiz",
  "parsed": {
    "title": "OS Midterm",
    "questions": [...]
  },
  "validation_errors": [],
  "attempt_log": [
    {
      "attempt_number": 1,
      "raw_output": "...{...broken JSON...",
      "error_category": "json_parse",
      "validation_errors": ["Expecting value: line 1 column 5"]
    },
    {
      "attempt_number": 2,
      "raw_output": "{valid JSON}",
      "error_category": null,
      "validation_errors": []
    }
  ]
}
```

**Exit Criteria:**
- `valid: true` after retries, OR
- `valid: false` after max attempts with error log.

**Error Handling:**
- Non-recoverable errors (timeout): No retry.
- Recoverable errors (JSON parse, schema validation): Retry up to 3 times.

---

## 5. Benchmark Execution Flow

### `localbench benchmark`

```
┌─────────────────────────────┐
│  User: localbench benchmark │
└────────┬────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Load BenchmarkConfig                │
│  - models: [qwen:7b, mistral:7b]    │
│  - dataset: v1.0.0                   │
│  - output_dir: ./results/<run-id>/   │
└────────┬───────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Create run ID (ISO timestamp)       │
│  e.g., 2026-08-14T143000Z            │
└────────┬───────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Capture system metadata             │
│  (CPU, RAM, GPU, Ollama version)     │
│  Persist to system.json              │
└────────┬───────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Load dataset (20–30 cases)          │
│  Validate schema                     │
└────────┬───────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  For each model:                     │
│                                      │
│  ┌──────────────────────────────┐   │
│  │ 1. Verify model exists       │   │
│  │ 2. Warm-up (3–5 dummy req)  │   │
│  │ 3. For each case:            │   │
│  │    - Execute case            │   │
│  │    - Measure timing          │   │
│  │    - Evaluate result         │   │
│  │    - Measure resources       │   │
│  │    - Persist case_result     │   │
│  │ 4. Aggregate model metrics   │   │
│  │ 5. Persist model summary     │   │
│  └──────────────────────────────┘   │
└────────┬───────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Aggregate across all models         │
│  - Per-category metrics              │
│  - Overall comparison                │
└────────┬───────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Generate report                     │
│  - Markdown table                    │
│  - Charts (matplotlib)               │
│  - Methodology section               │
│  - Limitations section               │
└────────┬───────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Persist artifacts:                  │
│  - metadata.json (run info)          │
│  - config.json (benchmark config)    │
│  - raw_outputs.jsonl (all text)      │
│  - case_results.jsonl (scored)       │
│  - summary.json (aggregated)         │
│  - report.md (human-readable)        │
└────────┬───────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  CLI: Display summary                │
│  - Results location                  │
│  - Key metrics                       │
│  - Next steps (recommend command)    │
└──────────────────────────────────────┘
```

**Case Execution Detail:**

```
For each (model, case) pair:

┌─────────────────────────────────┐
│  Start timer                    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Build prompt from case         │
│  template                       │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Start resource monitoring      │
│  (sample CPU, RSS every 100ms)  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Generate response via model    │
│  adapter                        │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Stop resource monitoring       │
│  Calculate peak RSS, avg CPU    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Stop timer                     │
│  Latency = end - start          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Evaluate result:               │
│  - Apply evaluation strategy    │
│  - Calculate quality score      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Create CaseResult object       │
│  - status: success/error        │
│  - response: raw text           │
│  - latency_ms: measured         │
│  - peak_rss: from monitoring    │
│  - evaluation_score: calculated │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Persist to case_results.jsonl  │
│  (append mode)                  │
└─────────────────────────────────┘
```

**Error Handling:**
- Single case fails: Record error, continue benchmark.
- Model becomes unavailable: Record error, move to next model.
- Ollama crashes mid-benchmark: Error, benchmark stops.
- Resource monitoring fails: Continue without resource metrics.

---

## 6. Evaluation Flow

### Quality Scoring

```
For each case result:

┌──────────────────────────────────┐
│  Get evaluation config            │
│  (strategy, parameters)           │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Dispatch to evaluator            │
│  (keyword, exact, numeric, judge) │
└────────┬─────────────────────────┘
         │
    ┌────┴────┬────────┬─────────┐
    │          │        │         │
    ▼          ▼        ▼         ▼
keyword    exact    numeric    judge
    │          │        │         │
    │          │        │         └─→ [LLM scoring]
    │          │        │
    └─→ Count   │        └─→ Numeric
        keywords│            tolerance
                │
                └─→ String
                    equality

All evaluators:

    │
    ▼
┌──────────────────────────────────┐
│  Calculate score (0.0–1.0)       │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Return EvaluationResult         │
│  - score: 0.92                   │
│  - strategy: "keyword"           │
│  - details: {...}                │
└──────────────────────────────────┘
```

---

## 7. Recommendation Flow

### `localbench recommend`

```
┌──────────────────────────────────────┐
│  User: localbench recommend          │
│  [+ optional flags: --min-quality, --max-ram] │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  CLI: Parse constraints              │
│  - min_quality: 0.85 (default)      │
│  - max_ram_gb: 8 (default)          │
│  - max_latency_ms: 3000 (default)   │
│  - min_structured_success: 0.95     │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Load latest benchmark results       │
│  from ./results/latest/              │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  For each model, check constraints   │
│                                      │
│  ┌──────────────────────────────┐   │
│  │ Qwen 7B:                     │   │
│  │  ✓ Quality 87% >= 85%        │   │
│  │  ✓ RAM 6.2GB <= 8GB          │   │
│  │  ✓ Latency 1.2s <= 3s        │   │
│  │  ✓ Struct success 96% >= 95% │   │
│  │  → PASS all constraints      │   │
│  │                              │   │
│  │ Mistral 7B:                  │   │
│  │  ✗ Quality 72% < 85%         │   │
│  │  ✓ RAM 5.8GB <= 8GB          │   │
│  │  ✓ Latency 0.9s <= 3s        │   │
│  │  → FAIL (quality threshold)  │   │
│  │                              │   │
│  │ Llama 13B:                   │   │
│  │  ✓ Quality 89% >= 85%        │   │
│  │  ✗ RAM 10.4GB > 8GB          │   │
│  │  ✓ Latency 2.1s <= 3s        │   │
│  │  → FAIL (RAM limit)          │   │
│  └──────────────────────────────┘   │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Remaining qualifying models:        │
│  - Qwen 7B (only one)               │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Single model? Use it.               │
│  Multiple models? Rank by quality    │
│  (highest quality wins)              │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Build recommendation explanation:   │
│                                      │
│  "Qwen 7B is recommended because:    │
│   - Highest quality (87%)            │
│   - Meets all constraints:           │
│     ✓ Quality >= 85%                 │
│     ✓ RAM <= 8GB (6.2GB actual)     │
│     ✓ Latency <= 3s (1.2s actual)   │
│   - Rejected alternatives:           │
│     Mistral: Quality 72% < 85%      │
│     Llama: RAM 10.4GB > 8GB"       │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  CLI: Display recommendation         │
│  (with explanation)                  │
└──────────────────────────────────────┘
```

**Edge Cases:**
- No models qualify: "No models meet constraints. Relax limits or add models."
- Single model: "Only one model qualifies: {model}."
- All models qualify: Recommend highest quality.

---

## 8. Study Assistant Flow

### `localbench study notes.pdf`

```
┌──────────────────────────────────────┐
│  User: localbench study notes.pdf    │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  CLI: Parse file path                │
│  - File exists? ✓                    │
│  - Is PDF or text? ✓                 │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  StudyAssistant: Load document       │
│  - If PDF: extract text via PyMuPDF  │
│  - If text: read as-is               │
│  - Normalize (remove headers, etc)   │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Create StudyContext                 │
│  - document_type: "pdf"              │
│  - text_content: full text           │
│  - metadata: path, size, etc         │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Interactive menu:                   │
│                                      │
│  [1] Ask question                    │
│  [2] Generate quiz                   │
│  [3] Show summary (TODO)             │
│  [0] Exit                            │
└────────┬─────────────────────────────┘
         │
    ┌────┴─────┬─────────┐
    │           │         │
    ▼           ▼         ▼
   [1]         [2]       [0]
    │           │         │
    │           ▼         └─→ [Exit]
    │      ┌──────────────────┐
    │      │ Gen quiz (struc. │
    │      │ gen w/ retry)    │
    │      │ Display quiz in  │
    │      │ terminal (iter.) │
    │      └──────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ Ask question:                        │
│ - User enters question               │
│ - Extract relevant context from doc  │
│   (keyword search, not vector)       │
│ - Build prompt:                      │
│   "Context: {relevant text}"         │
│   "Question: {user question}"        │
│   "Answer:"                          │
│ - Generate answer via model          │
│ - Display answer                     │
│ - Loop back to menu                  │
└──────────────────────────────────────┘
```

**Quiz Interaction:**
```
Quiz: OS Fundamentals

Question 1 of 5:
What is virtual memory?

a) Extended RAM using disk
b) Memory accessed via network
c) Encrypted memory storage
d) Memory used only for processes

Your answer: a

✓ Correct!
Explanation: Virtual memory extends physical RAM...

Next → (or quit)
```

---

## 9. Comparison Flow

### `localbench compare`

```
┌──────────────────────────────────┐
│  User: localbench compare        │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Load latest benchmark results   │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Aggregate metrics per model     │
│  - Quality (avg score)           │
│  - Latency (avg, min, max)       │
│  - Throughput (tokens/sec)       │
│  - Peak RAM (max)                │
│  - Structured success rate       │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  CLI: Render comparison table    │
│                                  │
│  ┌──────────────────────────────┐│
│  │ Model  │ Quality │ Latency  ││
│  │ Qwen   │  87%    │  1.2s    ││
│  │ Mistral│  72%    │  0.9s    ││
│  │ Llama  │  89%    │  2.1s    ││
│  └──────────────────────────────┘│
│                                  │
│  Note: Results specific to this  │
│  machine and benchmark version.  │
└──────────────────────────────────┘
```

---

## 10. Failure Handling Patterns

### Ollama Unavailable

```
User command
    │
    ▼
Health check → FAIL
    │
    ▼
Error: RuntimeUnavailable
    │
    ▼
CLI formats message:

"Ollama is not running.

Start Ollama with:
  ollama serve

Then run your command again."
    │
    ▼
Exit with code 1
```

### Model Missing

```
User: localbench ask "..." --model llama:100b

    │
    ▼
Model lookup → NOT FOUND
    │
    ▼
Error: ModelNotFound("llama:100b")
    │
    ▼
CLI formats message:

"Model 'llama:100b' not found.

Available models:
  - qwen:7b
  - mistral:7b

Download a model with:
  ollama pull <model-name>"
    │
    ▼
Exit with code 1
```

### Structured Generation Fails (All Retries Exhausted)

```
Quiz generation attempt 1: malformed JSON
    │
    ▼
Retry with diagnostic...
    │
    ▼
Attempt 2: schema validation error
    │
    ▼
Retry with diagnostic...
    │
    ▼
Attempt 3: still invalid
    │
    ▼
Max retries reached
    │
    ▼
Error: ValidationError with full attempt log
    │
    ▼
CLI formats message:

"Failed to generate valid quiz after 3 attempts.

Last attempt raw output:
[raw output from attempt 3]

Errors:
  - Field 'questions' is required
  - Field 'title' is missing

Try:
  1. Different model
  2. Simpler prompt
  3. Adjust generation parameters"
    │
    ▼
Exit with code 1
```

---

## 11. State Machines (Per Flow)

### Benchmark Run State Machine

```
[IDLE] → (user runs benchmark)
  │
  ▼
[LOADING_CONFIG]
  │
  ├─→ (config invalid) → [ERROR]
  │
  ▼
[LOADING_DATASET]
  │
  ├─→ (dataset missing) → [ERROR]
  │
  ▼
[SYSTEM_PROFILING]
  │
  ▼
[RUNNING_BENCHMARK] ← iterating models
  │
  ├─→ (model unavailable) → [RECORDING_ERROR]
  │                             │
  │                             ▼
  │                         (continue with next model)
  │
  ├─→ (case fails) → [RECORDING_CASE_ERROR]
  │                     │
  │                     ▼
  │                 (continue with next case)
  │
  ▼
[GENERATING_REPORT]
  │
  ▼
[PERSISTING_ARTIFACTS]
  │
  ▼
[COMPLETE] → CLI displays results
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-08-13 | Initial flows |
| 2.0 | 2026-08-14 | Advanced version: detailed state machines, error paths, JSON examples |

