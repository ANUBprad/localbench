# LocalBench --- Implementation Plan v2.0

**Status:** v2.0 (Advanced)\
**Revision Date:** August 2026\
**Total Estimated Duration:** 15–21 weeks\
**Target Completion:** December 2026

---

## Executive Summary

This plan breaks the project into 10 phases across 4 months. Each phase has:
- **Clear deliverables** (what ships).
- **Dependencies** (what must exist first).
- **Exit criteria** (how you know it's done).
- **Risk flags** (watch out for these).
- **Time estimate** (weeks, with contingency).

**The critical path is Phase 0–3 (foundation to reliability).** Everything downstream depends on these working.

---

## Phase 0: Repository Foundation (1–2 weeks)

### Objectives
Create a clean Python package with all tooling. Make `localbench --help` work.

### Deliverables
- GitHub repository initialized.
- `pyproject.toml` with all dependencies.
- Package structure: `src/localbench/`, `tests/`, `docs/`.
- CLI skeleton with Typer.
- pytest configuration.
- `.gitignore`, LICENSE, basic README.
- GitHub Actions CI (optional but recommended).

### Dependencies
- None (starting point).

### Tasks (Ordered by Priority)

| # | Task | Time | Owner | Notes |
|---|------|------|-------|-------|
| 0.1 | Create repo structure | 1h | You | `src/localbench/__init__.py`, `tests/__init__.py` |
| 0.2 | Write `pyproject.toml` | 2h | You | Lock all versions; see tech spec for stack |
| 0.3 | CLI skeleton (Typer) | 2h | You | Just `--help`, `--version` |
| 0.4 | pytest + conftest | 2h | You | Fixture for mock Ollama client |
| 0.5 | Add GitHub Actions | 1h | You | Run tests on push (optional) |
| 0.6 | First commit | 30m | You | Clean, focused message |

### Exit Criteria
- [ ] `localbench --help` shows output.
- [ ] `pytest` runs (no tests yet, but framework works).
- [ ] Package is installable (`pip install -e .`).
- [ ] Git history is clean (small, focused commits).

### Risk Flags
- **Dependency conflicts:** Test installation early on your actual dev machine.
- **Python version mismatch:** Verify Python 3.10+ before starting.

### Time Estimate
**1–2 weeks** (includes debugging environment issues).

---

## Phase 1: Hardware + Ollama Runtime (2–3 weeks)

### Objectives
Establish reliable inference. Make Ollama health check and model discovery work. Build the runtime abstraction.

### Deliverables
- `localbench.runtime` module.
- `LocalModel` protocol (abstract interface).
- `OllamaModel` concrete implementation.
- `OllamaRuntime` and `ModelRegistry` classes.
- `models` CLI command.
- Basic `ask` CLI command.
- Unit tests for runtime layer.

### Dependencies
- Phase 0 complete.

### Tasks (Ordered by Priority)

| # | Task | Time | Owner | Notes |
|---|------|------|-------|-------|
| 1.1 | SystemMetadata class | 1h | You | CPU, RAM, GPU detection via psutil |
| 1.2 | LocalModel protocol | 1h | You | Define abstract interface |
| 1.3 | OllamaModel class | 3h | You | HTTP client to Ollama, caching |
| 1.4 | OllamaRuntime class | 2h | You | Health check, model discovery |
| 1.5 | ModelRegistry | 1h | You | Central model store |
| 1.6 | `models` command | 1h | You | List models with metadata table |
| 1.7 | `ask` command (basic) | 2h | You | Simple text generation |
| 1.8 | Unit tests | 3h | You | Test health check, discovery, metadata |
| 1.9 | Integration test (mock Ollama) | 2h | You | Mock HTTP responses |

### Exit Criteria
- [ ] `localbench models` lists available models.
- [ ] `localbench ask "What is paging?"` generates text (if Ollama is running).
- [ ] Health check correctly identifies Ollama down/up.
- [ ] All unit tests pass.
- [ ] Runtime module has no direct Ollama references outside adapter.

### Risk Flags
- **Ollama connectivity issues:** Test on your actual machine. Network proxies can break httpx.
- **Model metadata inconsistency:** Ollama may not always return all fields. Handle missing fields gracefully.

### Time Estimate
**2–3 weeks** (includes Ollama integration debugging).

---

## Phase 2: Structured Generation (1–2 weeks)

### Objectives
Add Pydantic validation and JSON parsing. Set up contracts for structured output.

### Deliverables
- `localbench.generation` module.
- `GenerationRequest`, `GenerationResult` schemas.
- `StructuredGenerationRequest`, `StructuredGenerationResult` schemas.
- JSON extraction and Pydantic parsing.
- Structured output tests (happy path).

### Dependencies
- Phase 1 complete (need `LocalModel` to use).

### Tasks (Ordered by Priority)

| # | Task | Time | Owner | Notes |
|---|------|------|-------|-------|
| 2.1 | GenerationRequest/Result schemas | 1h | You | Basic Pydantic models |
| 2.2 | StructuredGenerationRequest schema | 1h | You | Include schema and retry config |
| 2.3 | JSON extraction from text | 2h | You | Find `{...}` blocks in output |
| 2.4 | Pydantic validation | 1h | You | Parse JSON → BaseModel |
| 2.5 | StructuredGenerationResult schema | 1h | You | Include validation_errors, attempts |
| 2.6 | StructuredGenerator class (no retry yet) | 2h | You | Just validate once |
| 2.7 | Unit tests (happy path) | 2h | You | Valid output → parsed object |
| 2.8 | Unit tests (failure cases) | 2h | You | Malformed JSON, missing fields |

### Exit Criteria
- [ ] Valid JSON → parsed Pydantic object.
- [ ] Invalid JSON → caught and reported with error details.
- [ ] Missing required fields → clear validation error.
- [ ] 80%+ test coverage for generation module.

### Risk Flags
- **JSON extraction fragility:** Models may output JSON with extra text before/after. Be liberal in extraction.
- **Pydantic strict mode:** Decide early if you'll coerce types or reject strictly.

### Time Estimate
**1–2 weeks**.

---

## Phase 3: Reliability / Retry Engine (1 week)

### Objectives
Make structured generation fault tolerant. Implement retries with diagnostics.

### Deliverables
- `RetryConfig` schema.
- Bounded retry loop with max attempts.
- Failure classification (recoverable vs. not).
- Diagnostic prompt construction.
- Attempt logging.
- Retry tests.

### Dependencies
- Phase 2 complete (need StructuredGenerator).

### Tasks (Ordered by Priority)

| # | Task | Time | Owner | Notes |
|---|------|------|-------|-------|
| 3.1 | RetryConfig schema | 1h | You | max_attempts, backoff, diagnostics flag |
| 3.2 | Failure classification logic | 1h | You | json_parse, schema_validation, timeout, etc. |
| 3.3 | Diagnostic prompt builder | 1.5h | You | "Your output was invalid: {errors}. Try again." |
| 3.4 | Retry loop in StructuredGenerator | 2h | You | Bounded, observable, logged |
| 3.5 | AttemptRecord and attempt logging | 1h | You | Track each attempt |
| 3.6 | Retry tests (all failure types) | 3h | You | Test each recoverable failure |
| 3.7 | Integration test (model repeatedly fails) | 1h | You | Max retries exhausted |

### Exit Criteria
- [ ] Malformed JSON on attempt 1 → retry attempt 2 with diagnostics.
- [ ] All attempts are logged (raw output, duration, error).
- [ ] Non-recoverable errors (timeout, model down) do NOT retry.
- [ ] Max retries reached → clear error message to user.
- [ ] `StructuredGenerationResult` shows full attempt history.

### Risk Flags
- **Infinite retry loops:** Use absolute max_attempts=3 as safety.
- **Retry prompt confusion:** Test that diagnostic prompts don't make model worse.

### Time Estimate
**1 week**.

---

## Phase 4: Benchmark Dataset (2–3 weeks)

### Objectives
Build a fair, versioned benchmark dataset. Write reference answers and evaluation rules.

### Deliverables
- 20–30 benchmark cases (versioned as v1.0.0).
- Evaluation strategy defined per case.
- Reference answers written.
- `BenchmarkCase`, `EvaluationConfig` schemas.
- Dataset validation and schema tests.

### Dependencies
- Phase 1 complete (need model to test cases).

### Tasks (Ordered by Priority)

| # | Task | Time | Owner | Notes |
|---|------|------|-------|-------|
| 4.1 | Define benchmark categories | 2h | You | e.g., conceptual, code, math, Q&A, structured |
| 4.2 | Write 5–8 high-confidence cases per category | 6h | You | Write question + reference answer |
| 4.3 | Define evaluation strategy per case | 3h | You | keyword, exact, numeric, or judge |
| 4.4 | BenchmarkCase schema | 1h | You | Pydantic model |
| 4.5 | EvaluationConfig schema | 1h | You | Strategy-specific fields |
| 4.6 | Dataset JSONL file (versioned) | 1h | You | Serialize all cases |
| 4.7 | Dataset validation tests | 2h | You | Load, validate schema, spot-check |
| 4.8 | README section: Dataset Methodology | 1h | You | Explain categories, evaluation strategy |

### Exit Criteria
- [ ] 20–30 cases written with reference answers.
- [ ] Evaluation strategy defined for every case.
- [ ] Dataset loads without errors.
- [ ] All cases have proper schema.
- [ ] Dataset JSONL is checked into repo as `data/benchmark_v1.0.0.jsonl`.

### Risk Flags
- **Weak evaluation strategy:** Test your evaluation on 3–5 cases first. Does it match ground truth?
- **Biased dataset:** Avoid cases that favor certain model architectures (e.g., only LLaMA-specific).
- **Too ambitious:** 40 cases is hard. Start with 20 solid cases.

### Time Estimate
**2–3 weeks** (writing cases is slow; evaluation design takes thought).

---

## Phase 5: Benchmark Runner (2 weeks)

### Objectives
Execute identical benchmark on multiple models. Produce raw outputs and case results.

### Deliverables
- `BenchmarkRunner` class.
- `BenchmarkConfig` schema.
- Warm-up sequence.
- Case execution loop (for each model, for each case).
- Raw output persistence to JSONL.
- Case result persistence to JSONL.
- Integration tests end-to-end.

### Dependencies
- Phase 1 (runtime), Phase 3 (reliability), Phase 4 (dataset).

### Tasks (Ordered by Priority)

| # | Task | Time | Owner | Notes |
|---|------|------|-------|-------|
| 5.1 | BenchmarkConfig schema | 1h | You | Models, dataset, generation params |
| 5.2 | BenchmarkRunner class skeleton | 1h | You | High-level run() method |
| 5.3 | Warm-up sequence | 1h | You | 3–5 dummy requests per model |
| 5.4 | Case execution loop | 2h | You | For each model, for each case |
| 5.5 | Timing measurement | 1h | You | Record latency |
| 5.6 | Output capture and persistence | 1.5h | You | Write raw_outputs.jsonl |
| 5.7 | Case result schema + persistence | 1.5h | You | Write case_results.jsonl |
| 5.8 | Error handling (single case fails) | 1h | You | Record error, continue benchmark |
| 5.9 | Integration tests | 2h | You | Mock models, verify outputs persist |

### Exit Criteria
- [ ] `BenchmarkRunner.run()` executes and completes.
- [ ] Two models benchmarked on same 10 cases.
- [ ] `raw_outputs.jsonl` contains all generated text.
- [ ] `case_results.jsonl` contains timing and status.
- [ ] Single case failure doesn't crash entire benchmark.
- [ ] All metrics recorded (latency minimum).

### Risk Flags
- **Benchmark too slow:** If 10 cases take 10 minutes per model, consider reducing case count.
- **Model crashes mid-benchmark:** Ollama may OOM. Record error and continue.

### Time Estimate
**2 weeks**.

---

## Phase 6: Evaluation + Resource Metrics (1–2 weeks)

### Objectives
Score case results. Measure CPU/RAM/VRAM usage.

### Deliverables
- Evaluation strategy implementations (keyword, exact, numeric, judge).
- Keyword evaluator (count required keywords).
- Exact match evaluator.
- Numeric tolerance evaluator.
- Resource profiling (peak RSS, CPU %).
- Metrics schema and calculation tests.

### Dependencies
- Phase 4 (dataset with evaluation configs), Phase 5 (case results).

### Tasks (Ordered by Priority)

| # | Task | Time | Owner | Notes |
|---|------|------|-------|-------|
| 6.1 | EvaluationResult schema | 1h | You | score, strategy, details |
| 6.2 | Keyword evaluator | 1.5h | You | Count matches in response |
| 6.3 | Exact match evaluator | 1h | You | String equality (case-insensitive) |
| 6.4 | Numeric evaluator | 1.5h | You | Parse numbers, check tolerance |
| 6.5 | ResourceMonitor class | 2h | You | psutil sampling during inference |
| 6.6 | Peak RSS calculation | 1h | You | Max memory during inference |
| 6.7 | CPU utilization sampling | 1h | You | Average CPU% during inference |
| 6.8 | Evaluation scoring tests | 2h | You | Test each strategy |
| 6.9 | Metric calculation tests | 1h | You | Peak stats from samples |

### Exit Criteria
- [ ] Each case gets a quality score (0.0–1.0).
- [ ] Peak RSS recorded for each case.
- [ ] CPU% recorded for each case.
- [ ] Evaluation strategy correctly applied per evaluation config.
- [ ] 80%+ test coverage for evaluation and metrics.

### Risk Flags
- **Resource measurement noise:** Background processes affect readings. Document methodology.
- **Evaluator bias (judge):** If using LLM-as-judge, clearly label. Don't present as objective truth.

### Time Estimate
**1–2 weeks**.

---

## Phase 7: Comparison + Recommendation (1–2 weeks)

### Objectives
Turn metrics into recommendations. Implement constraint filtering and ranking.

### Deliverables
- `RecommendationConstraints` schema.
- Constraint filtering (hard constraints).
- Ranking algorithm (soft objectives).
- Explanation generator (why this model?).
- `compare` CLI command (show table).
- `recommend` CLI command (show recommendation).
- Edge case tests (no models qualify).

### Dependencies
- Phase 6 (metrics).

### Tasks (Ordered by Priority)

| # | Task | Time | Owner | Notes |
|---|------|------|-------|-------|
| 7.1 | RecommendationConstraints schema | 1h | You | min_quality, max_ram_gb, max_latency_ms |
| 7.2 | Constraint validator | 1h | You | Ensure constraints are sensible |
| 7.3 | Constraint filtering logic | 1.5h | You | Remove violating models |
| 7.4 | Ranking algorithm | 1.5h | You | Score remaining models |
| 7.5 | Explanation generator | 2h | You | "Selected X because...", "Rejected Y because..." |
| 7.6 | RecommendationResult schema | 1h | You | recommended_model, reason, rejected_models |
| 7.7 | RecommendationEngine class | 1.5h | You | High-level generate() method |
| 7.8 | `compare` command | 1h | You | Show results table |
| 7.9 | `recommend` command | 1h | You | Show recommendation with explanation |
| 7.10 | Edge case tests | 2h | You | No models qualify, single model, all pass |

### Exit Criteria
- [ ] Recommendation is explainable (user sees why model was chosen).
- [ ] Rejected models show violated constraints.
- [ ] Edge case: no models qualify → clear message.
- [ ] Edge case: single model → recommended with caveat.
- [ ] Compare table shows key metrics.

### Risk Flags
- **Ranking complexity:** Start simple (e.g., highest quality). Don't over-engineer scoring.
- **Constraint edge cases:** What if user sets impossible constraints? Explain clearly.

### Time Estimate
**1–2 weeks**.

---

## Phase 8: Education Workload (2 weeks)

### Objectives
Demonstrate real-world utility. Implement Q&A and quiz generation.

### Deliverables
- `StudyAssistant` class.
- Document ingestion (PDF via PyMuPDF, text).
- Simple keyword-based retrieval.
- Q&A against context.
- `Quiz` schema and `QuizQuestion`.
- Quiz generation (structured, validated).
- `study` CLI command.

### Dependencies
- Phase 1 (runtime), Phase 3 (reliability for quiz generation).

### Tasks (Ordered by Priority)

| # | Task | Time | Owner | Notes |
|---|------|------|-------|-------|
| 8.1 | StudyContext schema | 1h | You | document_type, text_content, metadata |
| 8.2 | PDF ingestion (PyMuPDF) | 1.5h | You | Load PDF, extract text, handle errors |
| 8.3 | Text file ingestion | 1h | You | Read UTF-8 text |
| 8.4 | Document normalization | 1h | You | Remove headers, footers, page breaks |
| 8.5 | Simple retrieval | 1h | You | Keyword search (no vectors) |
| 8.6 | Q&A against context | 1.5h | You | Load context, answer question |
| 8.7 | QuizQuestion schema | 1h | You | Question, options, correct_answer_index, explanation |
| 8.8 | Quiz schema | 1h | You | Title, questions list |
| 8.9 | Quiz generation (structured) | 1.5h | You | Use StructuredGenerator to create Quiz |
| 8.10 | Quiz CLI | 1h | You | Interactive quiz in terminal |
| 8.11 | Study command | 1h | You | Load doc, offer Q&A and quiz options |
| 8.12 | Study workload tests | 2h | You | Test ingestion, Q&A, quiz generation |

### Exit Criteria
- [ ] Load PDF (e.g., OS textbook chapter).
- [ ] Ask question about content.
- [ ] Get reasonable answer (no cloud API).
- [ ] Generate quiz with 5 questions.
- [ ] Quiz validates (Pydantic).
- [ ] `localbench study notes.pdf` works end-to-end.

### Risk Flags
- **Simple retrieval limitations:** Keyword search won't handle semantic paraphrasing. That's okay for MVP; document limitation.
- **Quiz generation hallucination:** Model may generate questions not in text. Acceptable as long as they're not factually wrong.
- **Scope creep:** Resist adding summarization, flashcards, etc. Focus on Q&A + quiz.

### Time Estimate
**2 weeks**.

---

## Phase 9: Reporting (1 week)

### Objectives
Make benchmark results publishable and understandable.

### Deliverables
- Markdown report generator.
- CSV export of results.
- Summary metrics table.
- Latency/resource charts (matplotlib).
- Methodology documentation.
- Limitations section.
- README with actual results.

### Dependencies
- Phase 6 (metrics), Phase 7 (recommendation).

### Tasks (Ordered by Priority)

| # | Task | Time | Owner | Notes |
|---|------|------|-------|-------|
| 9.1 | ReportGenerator class | 1h | You | High-level generate_markdown() |
| 9.2 | Markdown summary table | 1h | You | Model, quality, latency, RAM, etc. |
| 9.3 | Matplotlib latency chart | 1h | You | Bar chart: latency per model |
| 9.4 | Matplotlib resource chart | 1h | You | Bar chart: peak RSS per model |
| 9.5 | Methodology section | 1.5h | You | Explain how benchmark was run |
| 9.6 | Limitations section | 1h | You | Sample size, hardware, model versions |
| 9.7 | CSV export | 1h | You | All case results to CSV |
| 9.8 | Report metadata (run ID, timestamp) | 1h | You | Ensure reproducibility info is included |
| 9.9 | README results section | 1h | You | Show actual benchmark output |

### Exit Criteria
- [ ] `BenchmarkRun` produces Markdown report.
- [ ] Report includes summary table.
- [ ] Report includes charts.
- [ ] Methodology clearly documented.
- [ ] Limitations section present.
- [ ] README can be copy-pasted from actual run.

### Risk Flags
- **Chart readability:** Ensure axes are labeled, legends are clear.
- **Report length:** Keep concise; link to detailed artifacts in repo.

### Time Estimate
**1 week**.

---

## Phase 10: Hardening + Release (1 week)

### Objectives
Make the project production-ready. Expand testing, audit for privacy, verify reproducibility.

### Deliverables
- Full test suite (70–80%+ coverage).
- Failure path testing.
- Privacy audit (no prompt logging).
- Reproducibility audit (same run twice → identical results).
- First release tag.

### Dependencies
- Phases 0–9 complete.

### Tasks (Ordered by Priority)

| # | Task | Time | Owner | Notes |
|---|------|------|-------|-------|
| 10.1 | Expand unit test coverage | 3h | You | Target 70%+ of core modules |
| 10.2 | Failure path audit | 2h | You | Ollama down, model missing, timeout, etc. |
| 10.3 | Privacy audit | 1h | You | No prompts in logs, documents stay local |
| 10.4 | Reproducibility audit | 1h | You | Run same benchmark twice, diff results |
| 10.5 | Full end-to-end test | 1h | You | models → benchmark → recommend |
| 10.6 | Error message review | 1h | You | All error messages are actionable |
| 10.7 | Documentation polish | 1.5h | You | README, architecture, API docs |
| 10.8 | Version bump (0.1.0) | 30m | You | Update pyproject, tag release |
| 10.9 | Release checklist | 1h | You | Final verification before tagging |

### Exit Criteria
- [ ] All unit tests pass.
- [ ] All integration tests pass.
- [ ] Failure paths handled gracefully (no crashes).
- [ ] No unexpected prompts/documents in logs.
- [ ] Reproducible results (same input → same output within tolerance).
- [ ] README has actual benchmark results.
- [ ] Repository tagged as v0.1.0.

### Risk Flags
- **Coverage gaps:** Don't aim for 100%; focus on critical paths.
- **Flaky tests:** Tests depending on model timing are flaky. Mock where possible.

### Time Estimate
**1 week**.

---

## Critical Path Summary

```
Phase 0 (1-2w) → Phase 1 (2-3w) → Phase 3 (1w) → Phase 4 (2-3w)
                                                       ↓
                                                Phase 5 (2w)
                                                       ↓
                                                Phase 6 (1-2w)
                                                       ↓
                                                Phase 7 (1-2w)
                                                       ↓
Phase 8 (2w) ⊕ Phase 9 (1w) ⊕ Phase 10 (1w)
```

**Critical phases (must not slip):** 0, 1, 3, 4, 5, 6.
**Compressible phases (can be cut if needed):** 8 (workload), 9 (reporting).

---

## Timeline at a Glance

| Phase | Duration | Cumulative | Must Be Done |
|-------|----------|-----------|--------------|
| 0 | 1–2w | 1–2w | Foundation |
| 1 | 2–3w | 3–5w | Runtime working |
| 2 | 1–2w | 4–7w | Structured gen ready |
| 3 | 1w | 5–8w | Reliability done |
| 4 | 2–3w | 7–11w | Dataset versioned |
| 5 | 2w | 9–13w | Benchmark runs |
| 6 | 1–2w | 10–15w | Metrics calculated |
| 7 | 1–2w | 11–17w | Recommendation works |
| 8 | 2w | 13–19w | Workload demo |
| 9 | 1w | 14–20w | Reports generated |
| 10 | 1w | 15–21w | Release ready |

**Realistic estimate: 15–21 weeks (~4–5 months).**
**You have 26 weeks before placement season. Buffer: 5–11 weeks.**

---

## Common Slip Scenarios

### Scenario 1: Phase 0 Environment Issues (adds 1–2 weeks)

**Action:**
- Don't try to debug complex environment setup alone. Use standard Python venv.
- If you hit dependency conflicts, cut non-critical dependencies temporarily.
- Move forward with Phase 1; resolve dependencies in parallel.

### Scenario 2: Phase 4 Dataset Takes Longer (adds 2–3 weeks)

**Action:**
- Reduce target from 30 to 20 cases. Quality > quantity.
- Focus on one category deeply (e.g., 10 OS questions).
- Expand dataset post-release as P1 work.

### Scenario 3: Phase 5 Benchmark Too Slow (adds 1 week)

**Action:**
- Reduce case count to 10 for development.
- Run full 20–30 case benchmark only for final release.
- Parallelize case execution if latency is critical.

### Scenario 4: Phase 8 Workload Scope Creep (adds 2+ weeks)

**Action:**
- Implement Q&A only. Skip quiz generation initially.
- Move quiz generation to P1.
- This is not critical for portfolio; benchmark engine is.

---

## Success Definition by Phase

| Phase | Success | How to Verify |
|-------|---------|--------------|
| 0 | `pytest` works | `pytest --version` returns output |
| 1 | `localbench models` lists models | Run command, see table |
| 2 | JSON → Pydantic object | Write unit test, it passes |
| 3 | Retry on validation failure | Unit test: malformed JSON → retry → success |
| 4 | Dataset loads without errors | `pytest tests/test_benchmark/` passes |
| 5 | Benchmark produces JSONL | Run benchmark, check `case_results.jsonl` exists |
| 6 | Metrics calculated correctly | Verify latency/RAM calculations in unit tests |
| 7 | Recommendation explains itself | `localbench recommend` shows reasoning |
| 8 | Q&A works on local PDF | Load PDF, ask question, get answer |
| 9 | Report is readable Markdown | Run benchmark, check `report.md` |
| 10 | Everything still works | Full end-to-end: models → benchmark → recommend |

---

## Pre-Phase Checklist

### Before Starting Phase 0
- [ ] GitHub account created (or use GitLab/etc).
- [ ] Python 3.10+ installed on dev machine.
- [ ] Ollama installed and tested (can run `ollama pull qwen:7b`).
- [ ] Text editor/IDE ready (VSCode, PyCharm, etc).
- [ ] 1–2 hours of uninterrupted time.

### Before Starting Phase 1
- [ ] Phase 0 complete and committed.
- [ ] `pip install -e .` works without errors.
- [ ] Ollama is running on your machine.
- [ ] You have at least one model downloaded (e.g., `qwen:7b`).

### Before Starting Phase 4
- [ ] Phase 3 complete.
- [ ] You've manually tested 1–2 models to understand their behavior.
- [ ] You have reference answers ready (write these by hand first).

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-08-13 | Initial plan |
| 2.0 | 2026-08-14 | Advanced version: concrete tasks, dependencies, exit criteria, risk flags |

