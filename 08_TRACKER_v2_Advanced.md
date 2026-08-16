# LocalBench --- Project Tracker v2.0

**Status:** v2.0 (Advanced)\
**Revision Date:** August 2026\
**Current Phase:** Planning (Phase 0 pending start)\
**Target Completion:** December 2026

---

## Status Legend

| Status | Meaning |
|--------|---------|
| `TODO` | Not started |
| `IN_PROGRESS` | Actively being implemented |
| `BLOCKED` | Waiting on dependency or external factor |
| `REVIEW` | Implementation complete, needs verification |
| `DONE` | Completed and verified |
| `DEFERRED` | Explicitly postponed to P1+ |

---

## Phase Overview

```
Phase 0  Phase 1  Phase 2  Phase 3  Phase 4  Phase 5  Phase 6  Phase 7  Phase 8  Phase 9  Phase 10
  │────────────────────────────────────────────────────────────────────────────────────────────────│
  │ Critical Path (Must Not Slip) ────────────────────────────────────────────│
  │                                                                            │
  └─────────────────────── Compressible Path (Can Be Cut) ──────────────────┘
```

**Critical Path:** Phases 0–7 (foundation to recommendation).
**Compressible Path:** Phases 8–10 (workload, reporting, hardening).

---

## Work Breakdown by Phase

### Phase 0: Repository Foundation

**Duration:** 1–2 weeks | **Status:** TODO

**Objectives:**
- Initialize repository with all tooling.
- Make `localbench --help` work.
- Set up CI/testing framework.

#### Tasks

| ID | Task | Status | Owner | Est. | Deps | Notes |
|----|------|--------|-------|-----|------|-------|
| 0.1 | Initialize GitHub repo | TODO | You | 30m | None | Create `src/`, `tests/`, `docs/` |
| 0.2 | Create `pyproject.toml` | TODO | You | 2h | None | Pin all versions in lock file |
| 0.3 | Add Pydantic v2 config | TODO | You | 1h | 0.2 | Validation setup |
| 0.4 | Add Typer CLI skeleton | TODO | You | 2h | 0.2 | Basic `--help`, `--version` |
| 0.5 | Add pytest + fixtures | TODO | You | 2h | 0.2 | Mock Ollama client fixture |
| 0.6 | Add GitHub Actions CI | TODO | You | 1h | 0.1 | Run tests on push (optional) |
| 0.7 | Add `.gitignore`, LICENSE | TODO | You | 1h | 0.1 | Standard OSS setup |
| 0.8 | First commit | TODO | You | 30m | 0.7 | Clean, focused message |

**Exit Criteria:**
- [ ] `pytest` runs and passes (no tests yet, framework works).
- [ ] `pip install -e .` succeeds.
- [ ] `localbench --help` shows output.
- [ ] Git history is clean (focused commits).

**Risk Flags:**
- **Dependency conflicts:** Test on actual dev machine early.
- **Python version:** Verify 3.10+ before starting.

**Success Metrics:**
- Package imports without errors.
- CLI is responsive.
- Tests execute in <5s.

---

### Phase 1: Hardware + Ollama Runtime

**Duration:** 2–3 weeks | **Status:** TODO | **Depends On:** Phase 0

**Objectives:**
- Establish reliable inference.
- Build runtime abstraction.
- Implement model discovery.

#### Priority Tasks

| ID | Task | Status | Owner | Est. | Deps | Notes |
|----|------|--------|-------|-----|------|-------|
| 1.1 | System profiler module | TODO | You | 3h | 0.8 | CPU, RAM, GPU via psutil |
| 1.2 | LocalModel protocol | TODO | You | 1h | 0.8 | Abstract interface |
| 1.3 | OllamaModel class | TODO | You | 3h | 1.2 | HTTP client, caching |
| 1.4 | OllamaRuntime class | TODO | You | 2h | 1.3 | Health check, discovery |
| 1.5 | ModelRegistry | TODO | You | 1h | 1.4 | Central model store |
| 1.6 | `models` CLI command | TODO | You | 1h | 1.5 | List models table |
| 1.7 | `ask` CLI command (basic) | TODO | You | 2h | 1.5 | Simple text generation |
| 1.8 | Unit tests (runtime) | TODO | You | 3h | 1.7 | Test health, discovery, gen |
| 1.9 | Integration tests | TODO | You | 2h | 1.8 | Mock Ollama responses |

**Exit Criteria:**
- [ ] `localbench models` lists models.
- [ ] `localbench ask "What is X?"` generates text (Ollama running).
- [ ] Health check correctly identifies Ollama down/up.
- [ ] All unit tests pass.
- [ ] No Ollama-specific code outside runtime module.

**Risk Flags:**
- **Ollama connectivity:** Network proxies can break httpx. Test early.
- **Model metadata inconsistency:** Handle missing fields gracefully.
- **Timeouts:** Set reasonable defaults (60s generation, 5s health check).

**Blocked If:**
- Ollama doesn't install on dev machine (work around with mock).

---

### Phase 2: Structured Generation

**Duration:** 1–2 weeks | **Status:** TODO | **Depends On:** Phase 1

**Objectives:**
- Add Pydantic validation and JSON parsing.
- Set up contracts for structured output.

#### Tasks

| ID | Task | Status | Owner | Est. | Deps | Notes |
|----|------|--------|-------|-----|------|-------|
| 2.1 | Generation schemas | TODO | You | 1h | 1.7 | Pydantic models |
| 2.2 | JSON extraction | TODO | You | 2h | 2.1 | Find `{...}` blocks |
| 2.3 | Pydantic validation | TODO | You | 1h | 2.2 | Parse JSON → model |
| 2.4 | StructuredGenerator | TODO | You | 2h | 2.3 | No retry yet |
| 2.5 | Unit tests (happy path) | TODO | You | 2h | 2.4 | Valid JSON works |
| 2.6 | Unit tests (failures) | TODO | You | 2h | 2.5 | Malformed, missing fields |

**Exit Criteria:**
- [ ] Valid JSON → parsed Pydantic object.
- [ ] Invalid JSON → caught with error details.
- [ ] Missing fields → clear validation error.
- [ ] 80%+ test coverage for generation module.

**Risk Flags:**
- **JSON extraction fragility:** Models may add text before/after JSON. Be liberal.
- **Pydantic strict mode:** Decide early: coerce types or reject strictly?

---

### Phase 3: Reliability / Retry Engine

**Duration:** 1 week | **Status:** TODO | **Depends On:** Phase 2

**Objectives:**
- Make structured generation fault tolerant.
- Implement retries with diagnostics.

#### Tasks

| ID | Task | Status | Owner | Est. | Deps | Notes |
|----|------|--------|-------|-----|------|-------|
| 3.1 | RetryConfig schema | TODO | You | 1h | 2.4 | Config model |
| 3.2 | Failure classification | TODO | You | 1h | 3.1 | json_parse, schema_val, etc |
| 3.3 | Diagnostic prompt builder | TODO | You | 1.5h | 3.2 | Include error feedback |
| 3.4 | Retry loop | TODO | You | 2h | 3.3 | Bounded, observable, logged |
| 3.5 | Attempt logging | TODO | You | 1h | 3.4 | AttemptRecord storage |
| 3.6 | Retry tests | TODO | You | 3h | 3.5 | All failure types |
| 3.7 | Integration test | TODO | You | 1h | 3.6 | Max retries exhausted |

**Exit Criteria:**
- [ ] Malformed JSON on attempt 1 → retry attempt 2.
- [ ] All attempts logged.
- [ ] Non-recoverable errors do NOT retry.
- [ ] Max retries reached → clear error.
- [ ] `StructuredGenerationResult` shows full attempt history.

**Risk Flags:**
- **Infinite loops:** Use absolute `max_attempts=3`.
- **Diagnostic prompt quality:** Test that diagnostics improve output.

---

### Phase 4: Benchmark Dataset

**Duration:** 2–3 weeks | **Status:** TODO | **Depends On:** Phase 1 (to test cases)

**Objectives:**
- Build fair, versioned benchmark dataset.
- Write reference answers and evaluation rules.

#### Tasks

| ID | Task | Status | Owner | Est. | Deps | Notes |
|----|------|--------|-------|-----|------|-------|
| 4.1 | Define categories | TODO | You | 2h | None | e.g., conceptual, code, math |
| 4.2 | Write benchmark cases | TODO | You | 6h | 4.1 | 20–30 cases with answers |
| 4.3 | Define eval strategy | TODO | You | 3h | 4.2 | keyword, exact, numeric, judge |
| 4.4 | BenchmarkCase schema | TODO | You | 1h | 4.3 | Pydantic model |
| 4.5 | EvaluationConfig schema | TODO | You | 1h | 4.4 | Strategy-specific fields |
| 4.6 | Dataset JSONL (v1.0.0) | TODO | You | 1h | 4.5 | Serialize all cases |
| 4.7 | Dataset validation tests | TODO | You | 2h | 4.6 | Load, validate, spot-check |
| 4.8 | README methodology | TODO | You | 1h | 4.7 | Document dataset |

**Exit Criteria:**
- [ ] 20–30 cases written with reference answers.
- [ ] Evaluation strategy defined per case.
- [ ] Dataset loads without errors.
- [ ] All cases have valid schema.
- [ ] Dataset JSONL checked in as `data/benchmark_v1.0.0.jsonl`.

**Risk Flags:**
- **Weak evaluation:** Test strategy on 3–5 cases first.
- **Biased dataset:** Avoid architecture-specific questions.
- **Too ambitious:** Start with 20, expand to 30+ only if time permits.

**Deferred Decisions:**
- LLM-as-judge: Implement later if needed.

---

### Phase 5: Benchmark Runner

**Duration:** 2 weeks | **Status:** TODO | **Depends On:** Phase 3, Phase 4

**Objectives:**
- Execute identical benchmark on multiple models.
- Produce raw outputs and case results.

#### Tasks

| ID | Task | Status | Owner | Est. | Deps | Notes |
|----|------|--------|-------|-----|------|-------|
| 5.1 | BenchmarkConfig schema | TODO | You | 1h | 4.5 | Config model |
| 5.2 | BenchmarkRunner skeleton | TODO | You | 1h | 5.1 | High-level run() |
| 5.3 | Warm-up sequence | TODO | You | 1h | 5.2 | 3–5 dummy requests |
| 5.4 | Case execution loop | TODO | You | 2h | 5.3 | For each model, case |
| 5.5 | Timing measurement | TODO | You | 1h | 5.4 | Record latency |
| 5.6 | Output persistence | TODO | You | 1.5h | 5.5 | Write raw_outputs.jsonl |
| 5.7 | Case result persistence | TODO | You | 1.5h | 5.6 | Write case_results.jsonl |
| 5.8 | Error handling | TODO | You | 1h | 5.7 | Single case fails, continue |
| 5.9 | Integration tests | TODO | You | 2h | 5.8 | Mock models, verify outputs |

**Exit Criteria:**
- [ ] BenchmarkRunner.run() executes and completes.
- [ ] Two models benchmarked on 10+ cases.
- [ ] raw_outputs.jsonl contains all text.
- [ ] case_results.jsonl contains timing and status.
- [ ] Single case failure doesn't crash benchmark.
- [ ] All metrics recorded (latency minimum).

**Risk Flags:**
- **Benchmark too slow:** Reduce case count if needed.
- **Model crashes:** Record error, continue.

---

### Phase 6: Evaluation + Resource Metrics

**Duration:** 1–2 weeks | **Status:** TODO | **Depends On:** Phase 4, Phase 5

**Objectives:**
- Score case results.
- Measure CPU/RAM/VRAM usage.

#### Tasks

| ID | Task | Status | Owner | Est. | Deps | Notes |
|----|------|--------|-------|-----|------|-------|
| 6.1 | EvaluationResult schema | TODO | You | 1h | 5.9 | Score, strategy, details |
| 6.2 | Keyword evaluator | TODO | You | 1.5h | 6.1 | Count required keywords |
| 6.3 | Exact match evaluator | TODO | You | 1h | 6.2 | String equality |
| 6.4 | Numeric evaluator | TODO | You | 1.5h | 6.3 | Tolerance around ref number |
| 6.5 | ResourceMonitor | TODO | You | 2h | 6.4 | psutil sampling |
| 6.6 | Peak RSS calculation | TODO | You | 1h | 6.5 | Max memory during inference |
| 6.7 | CPU utilization | TODO | You | 1h | 6.6 | Avg CPU% during inference |
| 6.8 | Evaluation tests | TODO | You | 2h | 6.7 | Test each strategy |
| 6.9 | Metric tests | TODO | You | 1h | 6.8 | Peak stats from samples |

**Exit Criteria:**
- [ ] Each case gets quality score (0.0–1.0).
- [ ] Peak RSS recorded per case.
- [ ] CPU% recorded per case.
- [ ] Evaluation strategy correctly applied.
- [ ] 80%+ test coverage.

**Risk Flags:**
- **Resource measurement noise:** Document methodology clearly.
- **LLM-as-judge bias:** Label clearly if used.

---

### Phase 7: Comparison + Recommendation

**Duration:** 1–2 weeks | **Status:** TODO | **Depends On:** Phase 6

**Objectives:**
- Turn metrics into recommendations.
- Implement constraint filtering and ranking.

#### Tasks

| ID | Task | Status | Owner | Est. | Deps | Notes |
|----|------|--------|-------|-----|------|-------|
| 7.1 | RecommendationConstraints schema | TODO | You | 1h | 6.9 | Constraints model |
| 7.2 | Constraint validator | TODO | You | 1h | 7.1 | Validate constraints |
| 7.3 | Constraint filtering | TODO | You | 1.5h | 7.2 | Remove violating models |
| 7.4 | Ranking algorithm | TODO | You | 1.5h | 7.3 | Score remaining models |
| 7.5 | Explanation generator | TODO | You | 2h | 7.4 | Why selected, why rejected |
| 7.6 | RecommendationResult schema | TODO | You | 1h | 7.5 | Result model |
| 7.7 | RecommendationEngine | TODO | You | 1.5h | 7.6 | High-level generate() |
| 7.8 | `compare` command | TODO | You | 1h | 7.7 | Show results table |
| 7.9 | `recommend` command | TODO | You | 1h | 7.8 | Show recommendation |
| 7.10 | Edge case tests | TODO | You | 2h | 7.9 | No models qualify, etc |

**Exit Criteria:**
- [ ] Recommendation is explainable.
- [ ] Rejected models show violated constraints.
- [ ] Edge case: no models qualify → clear message.
- [ ] Compare table shows key metrics.

**Risk Flags:**
- **Ranking complexity:** Start simple, don't over-engineer.
- **Impossible constraints:** Explain clearly.

---

### Phase 8: Education Workload

**Duration:** 2 weeks | **Status:** TODO | **Depends On:** Phase 1, Phase 3

**Objectives:**
- Demonstrate real-world utility.
- Implement Q&A and quiz generation.

#### Tasks

| ID | Task | Status | Owner | Est. | Deps | Notes |
|----|------|--------|-------|-----|------|-------|
| 8.1 | StudyContext schema | TODO | You | 1h | 7.10 | Document model |
| 8.2 | PDF ingestion | TODO | You | 1.5h | 8.1 | PyMuPDF extraction |
| 8.3 | Text ingestion | TODO | You | 1h | 8.2 | Read UTF-8 text |
| 8.4 | Document normalization | TODO | You | 1h | 8.3 | Remove headers, etc |
| 8.5 | Simple retrieval | TODO | You | 1h | 8.4 | Keyword search (no vectors) |
| 8.6 | Q&A | TODO | You | 1.5h | 8.5 | Answer questions |
| 8.7 | Quiz schemas | TODO | You | 1h | 8.6 | Question, Quiz models |
| 8.8 | Quiz generation | TODO | You | 1.5h | 8.7 | Structured generation |
| 8.9 | Quiz CLI | TODO | You | 1h | 8.8 | Interactive terminal quiz |
| 8.10 | Study command | TODO | You | 1h | 8.9 | Load doc, offer options |
| 8.11 | Study tests | TODO | You | 2h | 8.10 | Test ingestion, gen |

**Exit Criteria:**
- [ ] Load PDF and ask questions.
- [ ] Get reasonable answers (no cloud API).
- [ ] Generate quiz with 5 questions.
- [ ] Quiz validates (Pydantic).
- [ ] `localbench study notes.pdf` works end-to-end.

**Risk Flags:**
- **Retrieval limitations:** Keyword search won't handle paraphrasing. Document.
- **Quiz hallucination:** Model may generate unsupported questions. Acceptable.
- **Scope creep:** MVP is Q&A + quiz. Skip summarization/flashcards.

**DEFERRED:** Summarization, flashcards (P1+).

---

### Phase 9: Reporting

**Duration:** 1 week | **Status:** TODO | **Depends On:** Phase 6, Phase 7

**Objectives:**
- Make results publishable and understandable.

#### Tasks

| ID | Task | Status | Owner | Est. | Deps | Notes |
|----|------|--------|-------|-----|------|-------|
| 9.1 | ReportGenerator class | TODO | You | 1h | 7.10 | High-level generate() |
| 9.2 | Markdown summary table | TODO | You | 1h | 9.1 | Metrics table |
| 9.3 | Matplotlib latency chart | TODO | You | 1h | 9.2 | Bar chart |
| 9.4 | Matplotlib resource chart | TODO | You | 1h | 9.3 | Bar chart |
| 9.5 | Methodology section | TODO | You | 1.5h | 9.4 | Explain how run |
| 9.6 | Limitations section | TODO | You | 1h | 9.5 | Be honest |
| 9.7 | CSV export | TODO | You | 1h | 9.6 | All case results |
| 9.8 | Report metadata | TODO | You | 1h | 9.7 | Reproducibility info |
| 9.9 | README results | TODO | You | 1h | 9.8 | Actual benchmark output |

**Exit Criteria:**
- [ ] BenchmarkRun produces Markdown report.
- [ ] Report includes summary table and charts.
- [ ] Methodology documented.
- [ ] Limitations section present.
- [ ] README can be copied from actual run.

**Risk Flags:**
- **Chart readability:** Ensure axes labeled, legends clear.
- **Report length:** Keep concise.

---

### Phase 10: Hardening + Release

**Duration:** 1 week | **Status:** TODO | **Depends On:** Phases 0–9

**Objectives:**
- Make production-ready.
- Tag first release.

#### Tasks

| ID | Task | Status | Owner | Est. | Deps | Notes |
|----|------|--------|-------|-----|------|-------|
| 10.1 | Expand test coverage | TODO | You | 3h | 9.9 | Target 70%+ |
| 10.2 | Failure path audit | TODO | You | 2h | 10.1 | Ollama down, etc |
| 10.3 | Privacy audit | TODO | You | 1h | 10.2 | No prompt logging |
| 10.4 | Reproducibility audit | TODO | You | 1h | 10.3 | Same run twice |
| 10.5 | End-to-end test | TODO | You | 1h | 10.4 | Full workflow |
| 10.6 | Error review | TODO | You | 1h | 10.5 | All messages actionable |
| 10.7 | Docs polish | TODO | You | 1.5h | 10.6 | README, API docs |
| 10.8 | Version bump | TODO | You | 30m | 10.7 | v0.1.0 |
| 10.9 | Release checklist | TODO | You | 1h | 10.8 | Final verification |

**Exit Criteria:**
- [ ] All unit tests pass.
- [ ] All integration tests pass.
- [ ] No crashes on failure paths.
- [ ] No unexpected prompts in logs.
- [ ] Reproducible results.
- [ ] README has actual results.
- [ ] Repository tagged v0.1.0.

**Risk Flags:**
- **Coverage gaps:** Don't aim for 100%; focus on critical paths.
- **Flaky tests:** Mock timing-dependent tests.

---

## Timeline Summary

| Phase | Task | Duration | Cumulative | Critical? |
|-------|------|----------|-----------|-----------|
| 0 | Foundation | 1–2w | 1–2w | YES |
| 1 | Runtime | 2–3w | 3–5w | YES |
| 2 | Structured Gen | 1–2w | 4–7w | YES |
| 3 | Reliability | 1w | 5–8w | YES |
| 4 | Dataset | 2–3w | 7–11w | YES |
| 5 | Runner | 2w | 9–13w | YES |
| 6 | Metrics | 1–2w | 10–15w | YES |
| 7 | Recommendation | 1–2w | 11–17w | YES |
| 8 | Workload | 2w | 13–19w | NO (can cut) |
| 9 | Reporting | 1w | 14–20w | NO (can cut) |
| 10 | Hardening | 1w | 15–21w | NO (can defer) |

**Realistic Total:** 15–21 weeks (~4–5 months).
**Available:** 26 weeks (until January 2027 placement season).
**Buffer:** 5–11 weeks.

---

## Slip Scenarios

### Scenario 1: Phase 0 Takes 3 Weeks
**Action:**
- Don't fall back. Use standard Python venv.
- If dependency conflicts, cut non-critical deps temporarily.
- Move forward with Phase 1; resolve in parallel.

### Scenario 2: Phase 4 Dataset Takes 4 Weeks
**Action:**
- Reduce target to 20 cases (quality > quantity).
- Focus on one category deeply.
- Expand post-release as P1 work.

### Scenario 3: Phase 5 Benchmark Too Slow
**Action:**
- Use 10 cases for dev, 20–30 for final release.
- Parallelize case execution if needed.

### Scenario 4: Phase 8 Scope Creep
**Action:**
- MVP: Q&A only. Skip quiz generation initially.
- Move quiz to P1 feature.
- Benchmark engine is the priority.

### Scenario 5: Hit Wall in Phase 6 (Resource Measurement)
**Action:**
- Simplify to peak RSS + CPU%. Skip GPU/VRAM for MVP.
- Document as P1 feature.

---

## Decision Log

### Major Decisions

| Date | Decision | Reasoning | Status |
|------|----------|-----------|--------|
| 2026-08-13 | Benchmark engine is core product | Differentiates from Ollama wrapper | APPROVED |
| 2026-08-13 | CLI-first (no web UI in MVP) | Keep focus on engineering | APPROVED |
| 2026-08-13 | No cloud LLM judge in MVP | Preserves offline premise | APPROVED |
| 2026-08-13 | No LangChain dependency | Avoid unnecessary abstraction | APPROVED |
| 2026-08-13 | Recommendation must be constraint-aware | Avoids claiming universal "best" | APPROVED |
| 2026-08-14 | Start with 20 cases, expand to 30+ if time | Quality over breadth | APPROVED |
| 2026-08-14 | Study workload is P1 (lower priority) | Benchmark engine is MVP focus | APPROVED |
| 2026-08-14 | Use deterministic eval for most cases | Avoid LLM judge bias | APPROVED |

### Pending Decisions

| Question | Status | Target Decision Date |
|----------|--------|---------------------|
| Which models to benchmark? | PENDING | Before Phase 1 starts |
| Semantic eval (judge) strategy? | PENDING | Before Phase 6 starts |
| GitHub actions required? | OPTIONAL | End of Phase 0 |
| Vector retrieval for study assistant? | NO (out of scope) | – |

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner | Status |
|------|--------|-----------|-----------|-------|--------|
| Ollama integration fragile | High | Medium | Test early, have fallback models | You | ACTIVE |
| Dataset weak or biased | High | Medium | Define eval strategy before writing cases | You | ACTIVE |
| Structured generation retries don't work | High | Low | Unit test with malformed inputs | You | ACTIVE |
| Scope creep (education workload) | Medium | High | Define MVP clearly, use feature flags | You | ACTIVE |
| Timeline slips (total >21 weeks) | Medium | Medium | Aggressive prioritization, cut P1 | You | ACTIVE |
| Results underwhelming (models similar) | Low | Medium | Document trade-offs, explain recommendations | You | ACCEPTED |

---

## Metrics & KPIs

### Development Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Test coverage (core modules) | 70%+ | 0% | NOT STARTED |
| Cyclomatic complexity | <10 avg | – | TBD |
| Build time (full test suite) | <30s | – | TBD |
| Documentation coverage | 100% | 0% | NOT STARTED |

### Portfolio Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Benchmark cases | 20–30 | 0 | NOT STARTED |
| Models benchmarked | 2–3 | 0 | NOT STARTED |
| Evaluation strategies | 3+ | 0 | NOT STARTED |
| Test pass rate | 100% | 0% | NOT STARTED |

---

## Success Criteria (By Phase)

| Phase | Success | Verification |
|-------|---------|--------------|
| 0 | `pytest` works | `pytest --version` returns output |
| 1 | `localbench models` lists models | Run command, see table |
| 2 | JSON → Pydantic object | Unit test passes |
| 3 | Retry on failure | Malformed JSON test passes |
| 4 | Dataset loads | `pytest tests/benchmark/` passes |
| 5 | Benchmark produces JSONL | Check case_results.jsonl exists |
| 6 | Metrics calculated | Unit tests verify calculations |
| 7 | Recommendation explains itself | Run `recommend`, see reasoning |
| 8 | Q&A works on PDF | Load PDF, ask question, get answer |
| 9 | Report is readable | Run benchmark, check report.md |
| 10 | Everything works | Full end-to-end succeeds |

---

## Communication & Escalation

### Status Updates
- **Weekly:** Self-review against timeline.
- **Bi-weekly:** If slipping, adjust scope or timeline.
- **Before major phases:** Verify dependencies are ready.

### Escalation
- **Phase slip >1 week:** Cut scope (move to P1).
- **Blocker (e.g., Ollama won't run):** Work around or defer.
- **Test coverage <50%:** Increase focus on testing.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-08-13 | Initial tracker |
| 2.0 | 2026-08-14 | Advanced version: detailed tasks, dependencies, risk register, success metrics |

