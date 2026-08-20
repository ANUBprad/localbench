# LocalBench --- Complete Documentation Index v3

**Last Updated:** 2026-08-19  
**Status:** Comprehensive documentation suite for research platform

---

## Overview

This documentation suite defines **LocalBench**, a local-first experimentation platform for evaluating whether small, locally deployable language models can be specialized for narrowly scoped real-world workloads and achieve competitive downstream performance while reducing computational resources.

**Core research question:**
> Can a small, locally deployable language model specialized for a specific real-world workload achieve competitive downstream performance against larger general-purpose local models while using substantially fewer computational resources?

---

## Documentation Structure

The documentation is organized into **8 comprehensive guides** covering different aspects of the platform:

### 1. **ARCHITECTURE.md** — System Design & Component Responsibilities
**Purpose:** Technical blueprint for the entire system.

**Key sections:**
- System overview (5-layer architecture)
- Module responsibilities (CLI, Application, Runtime, Workload)
- Data flow (benchmark execution, generation with retry, retrieval)
- Module interactions & dependency graph
- Integration contracts between layers
- Error handling & resilience
- Extensibility patterns
- Testing strategy

**Audience:** Software engineers, architects  
**When to read:** Before implementing any component

**Key takeaway:** LocalBench separates inference (Runtime) from task logic (Workload) to maintain modularity and enable multiple workloads.

---

### 2. **DATASET_SPECIFICATION.md** — Dataset Schema, Collection & Versioning
**Purpose:** Define dataset structure, collection methodology, and quality assurance.

**Key sections:**
- Dataset purpose & scope (code semantic retrieval)
- Core schemas (BenchmarkCase, SemanticLabel, Query, Relevance)
- Collection methodology (repository selection, code extraction, labeling)
- Semantic labeling (human & AI-assisted)
- Query generation (developer-style searches)
- Dataset versioning & immutability
- Leakage prevention (repository-disjoint splits)
- Dataset validation (schema, statistical, manual review)
- Artifact layout & distribution

**Audience:** Data engineers, dataset managers, researchers  
**When to read:** During Phase 4 (dataset construction)

**Key takeaway:** Repository-disjoint splits prevent leakage; frozen test set ensures reproducibility.

**Dataset statistics (v1.0.0):**
- 450 code units (Python functions)
- 45 test queries
- 3 repositories (train/val/test split)
- Train: 225, Validation: 112, Test: 113

---

### 3. **EVALUATION_PROTOCOL.md** — Benchmark Methodology & Fairness
**Purpose:** Define the benchmark protocol, metrics, and comparison framework.

**Key sections:**
- Evaluation philosophy (downstream task matters most)
- Benchmark protocol (pre-benchmark, warm-up, execution, retrieval phases)
- Metric definitions (Hit@K, MRR, latency, throughput, memory, reliability)
- Fair comparison framework (identical conditions for comparable models)
- Failure handling & taxonomy
- Statistical rigor (t-tests, confidence intervals)
- Benchmark report template
- Ablation study design
- Reproducibility checklist

**Audience:** Researchers, benchmarkers, data scientists  
**When to read:** Before running benchmarks (Phase 5+)

**Key takeaway:** Retrieve-quality matters more than text quality; all models evaluated identically.

**Primary metrics:**
- Hit@1, Hit@3, Hit@5, Hit@10 (retrieval success)
- MRR (ranking quality)
- Latency, Memory, Throughput (resources)

---

### 4. **TRAINING_SPECIFICATION.md** — Fine-Tuning & Specialization
**Purpose:** Define training methodology, hyperparameters, and specialization approach.

**Key sections:**
- Training objective (specialize small model via LoRA)
- Base model selection (phi-3-mini, phi-3, mistral, gemma candidates)
- Training dataset construction (from Phase 4)
- Prompt template & consistency
- Fine-tuning approach (LoRA = parameter-efficient)
- LoRA configuration (rank, alpha, dropout, target modules)
- Training configuration (learning rate, epochs, batch size)
- Training procedure & artifacts
- Ablation study design (training data size, LoRA rank)
- Model evaluation during training (validation set, early stopping)
- Inference-time deployment & resource profile
- Failure handling & troubleshooting
- Reproducibility checklist

**Audience:** ML engineers, practitioners  
**When to read:** During Phase 7 (specialization)

**Key takeaway:** LoRA is efficient; adds ~1% overhead; baseline required first.

**Training configuration:**
- Base model: phi-3-mini (0.5B)
- LoRA: rank=8, alpha=16
- Epochs: 3
- Learning rate: 5e-4
- Batch size: 16
- Seed: 42

---

### 5. **TESTING_STRATEGY.md** — Unit, Integration & System Tests
**Purpose:** Define comprehensive testing approach for quality assurance.

**Key sections:**
- Testing philosophy (failure paths first-class)
- Test structure (unit, integration, system organization)
- Unit tests (configuration, runtime, workload, reporting)
- Integration tests (benchmark flow, generation+retry, retrieval)
- System tests (CLI, artifact persistence, reproducibility)
- Fixture design (mocks, datasets, expected outputs)
- Coverage goals (85%+ target)
- Test execution (local & CI/CD)
- Key principles

**Audience:** QA engineers, developers  
**When to read:** During implementation (all phases)

**Key takeaway:** Mock external dependencies; test failure paths; aim for 85%+ coverage.

**Test coverage targets:**
- Config: 95%
- Runtime adapter: 90%
- Generation validation: 95%
- Workload evaluation: 90%
- CLI: 80%
- Reporting: 85%

---

### 6. **DEPLOYMENT_GUIDE.md** — Installation, Configuration & Usage
**Purpose:** Operational guide for deploying and using LocalBench.

**Key sections:**
- Prerequisites (system requirements, software dependencies)
- Installation (clone, pip install, Ollama setup)
- Configuration (YAML, environment variables, CLI overrides)
- Quick start (verify, test inference)
- Running benchmarks (baseline, specialized, comparison)
- Workload management (list, download, validate)
- Advanced usage (custom profiles, debugging, exporting)
- Troubleshooting (Ollama, models, OOM, GPU)
- Performance tuning (inference optimization, disk space)
- Data privacy (local-first, no telemetry)
- Reproducibility (environment report, archives)
- Monitoring & logging
- Typical workflow examples

**Audience:** End users, operators  
**When to read:** Before running benchmarks

**Key command examples:**
```bash
localbench system                          # Display hardware
localbench models                          # List models
localbench benchmark                       # Run full benchmark
localbench compare run1 run2                # Compare results
localbench recommend --min-hit-at-10 0.85  # Get recommendation
```

---

### 7. **RESEARCH_METHODOLOGY.md** — Scientific Integrity & Experimental Design
**Purpose:** Document research rigor, bias prevention, and limitations.

**Key sections:**
- Research framework (question, hypothesis, falsifiability)
- Experimental design (variables, groups, controls)
- Scientific rigor (baseline-first, test immutability, repository-disjoint)
- Statistical methodology (sample size, significance testing, multiple comparisons)
- Validity threats (internal, external, construct)
- Reproducibility standards (required information, checklist)
- Conflict of interest & bias (development bias, model selection bias)
- Ethical considerations (code privacy, responsible claims)
- Publication & dissemination (what to publish, transparent reporting)
- Key principles

**Audience:** Researchers, peer reviewers  
**When to read:** Before publishing results

**Key principles:**
- Hypothesis stated before results
- Negative results are valid
- Test set frozen after Phase 4
- Repository-disjoint splits prevent leakage
- Limitations explicitly acknowledged

---

### 8. **README_DOCUMENTATION.md** — This Document
**Purpose:** Master index and navigation guide.

---

## Quick Navigation

### By Role

**Software Engineer:**
→ Start with ARCHITECTURE.md  
→ Then: TESTING_STRATEGY.md, DEPLOYMENT_GUIDE.md

**Data Scientist/Researcher:**
→ Start with EVALUATION_PROTOCOL.md  
→ Then: DATASET_SPECIFICATION.md, RESEARCH_METHODOLOGY.md

**ML Engineer (Training):**
→ Start with TRAINING_SPECIFICATION.md  
→ Then: DATASET_SPECIFICATION.md, EVALUATION_PROTOCOL.md

**DevOps/Operator:**
→ Start with DEPLOYMENT_GUIDE.md  
→ Then: ARCHITECTURE.md (understand components)

**Project Manager:**
→ Start with this README  
→ Then: Original PRD, Implementation Plan, Tracker

---

### By Phase

| Phase | Key Documents |
|-------|----------------|
| **Phase 0** (Research Reset) | RESEARCH_METHODOLOGY.md |
| **Phase 1** (Runtime) | ARCHITECTURE.md, DEPLOYMENT_GUIDE.md |
| **Phase 2** (Validation) | TESTING_STRATEGY.md |
| **Phase 3** (Retry/Recovery) | ARCHITECTURE.md |
| **Phase 4** (Dataset) | DATASET_SPECIFICATION.md |
| **Phase 5** (Retrieval) | EVALUATION_PROTOCOL.md, ARCHITECTURE.md |
| **Phase 6** (Baseline) | EVALUATION_PROTOCOL.md, DEPLOYMENT_GUIDE.md |
| **Phase 7** (Specialization) | TRAINING_SPECIFICATION.md |
| **Phase 8** (Experiments) | TRAINING_SPECIFICATION.md, EVALUATION_PROTOCOL.md |
| **Phase 9–10** (Release) | RESEARCH_METHODOLOGY.md |

---

### By Topic

**System Architecture:**
- ARCHITECTURE.md (design, layers, modules)
- DEPLOYMENT_GUIDE.md (operational aspects)

**Data & Datasets:**
- DATASET_SPECIFICATION.md (schema, collection, versioning)
- EVALUATION_PROTOCOL.md (test set management)

**Benchmarking & Evaluation:**
- EVALUATION_PROTOCOL.md (metrics, protocol, fairness)
- DEPLOYMENT_GUIDE.md (running benchmarks)

**Model Training:**
- TRAINING_SPECIFICATION.md (fine-tuning, LoRA)
- DATASET_SPECIFICATION.md (training data)

**Quality Assurance:**
- TESTING_STRATEGY.md (unit, integration, system tests)
- EVALUATION_PROTOCOL.md (metric validation)

**Research Rigor:**
- RESEARCH_METHODOLOGY.md (scientific integrity)
- EVALUATION_PROTOCOL.md (experimental design)

**Operations & Deployment:**
- DEPLOYMENT_GUIDE.md (installation, configuration, usage)
- ARCHITECTURE.md (understanding components)

---

## Key Concepts Across Documents

### 1. Layers of Architecture
```
CLI (command interface)
  ↓
Application (orchestration)
  ↓
Runtime (inference provider-agnostic)
  ↓
Workload (task-specific)
```

### 2. Data Flow
```
Code Unit → Generation → Validation & Retry → Artifact Store
                                                    ↓
                                            Retrieval Index
                                                    ↓
                                            Query Evaluation
                                                    ↓
                                            Hit@K, MRR Metrics
```

### 3. Core Metrics
- **Primary:** Hit@1, Hit@3, Hit@5, Hit@10, MRR (retrieval quality)
- **Secondary:** Latency, RAM, Throughput (resources)
- **Supporting:** Validity, Reliability (robustness)

### 4. Versioning Scheme
- **Dataset:** MAJOR.MINOR.PATCH (v1.0.0 frozen after Phase 4)
- **Workload:** MAJOR.MINOR.PATCH (v1.0.0 = code retrieval)
- **Protocol:** MAJOR.MINOR.PATCH (v3.0.0 = benchmark protocol)
- **Models:** name + checkpoint ID (phi-3-mini-lora-full)

### 5. Safety Principles
- **Reproducibility:** All configs recorded, raw data preserved
- **Fairness:** Identical conditions for comparable models
- **Transparency:** Failures visible, limitations acknowledged
- **Privacy:** Local-first, no telemetry, code not logged

---

## Document Cross-References

### ARCHITECTURE.md references:
- TESTING_STRATEGY.md (testing strategy for each layer)
- DEPLOYMENT_GUIDE.md (CLI commands, configuration)
- RESEARCH_METHODOLOGY.md (ethical considerations)

### DATASET_SPECIFICATION.md references:
- EVALUATION_PROTOCOL.md (test set immutability)
- TRAINING_SPECIFICATION.md (training data source)
- RESEARCH_METHODOLOGY.md (repository-disjoint splits)

### EVALUATION_PROTOCOL.md references:
- DATASET_SPECIFICATION.md (test set definition)
- DEPLOYMENT_GUIDE.md (running benchmarks)
- RESEARCH_METHODOLOGY.md (statistical methodology)

### TRAINING_SPECIFICATION.md references:
- DATASET_SPECIFICATION.md (training data)
- EVALUATION_PROTOCOL.md (evaluation during training)
- TESTING_STRATEGY.md (training tests)

### TESTING_STRATEGY.md references:
- ARCHITECTURE.md (module-level tests)
- All other docs (test fixtures based on schemas)

### DEPLOYMENT_GUIDE.md references:
- ARCHITECTURE.md (understanding components)
- EVALUATION_PROTOCOL.md (running benchmarks)
- RESEARCH_METHODOLOGY.md (reproducibility)

### RESEARCH_METHODOLOGY.md references:
- EVALUATION_PROTOCOL.md (experimental design)
- DATASET_SPECIFICATION.md (validity threats)
- All docs (reproducibility requirements)

---

## Implementation Roadmap

### Immediate (Week 1)
- [ ] Review ARCHITECTURE.md for system design
- [ ] Audit existing Phase 0 codebase against v3
- [ ] Set up development environment per DEPLOYMENT_GUIDE.md

### Short-term (Phase 1–2, Weeks 2–6)
- [ ] Implement runtime layer per ARCHITECTURE.md
- [ ] Implement tests per TESTING_STRATEGY.md
- [ ] Document configuration options

### Medium-term (Phase 4–6, Weeks 7–12)
- [ ] Build dataset per DATASET_SPECIFICATION.md
- [ ] Implement retrieval pipeline
- [ ] Run baseline benchmarks per EVALUATION_PROTOCOL.md

### Long-term (Phase 8–10, Weeks 13+)
- [ ] Fine-tune model per TRAINING_SPECIFICATION.md
- [ ] Run ablations per EVALUATION_PROTOCOL.md
- [ ] Document findings per RESEARCH_METHODOLOGY.md
- [ ] Publish results

---

## Success Criteria

Each phase has success criteria defined in its primary document:

**Phase 1 (Runtime):** ARCHITECTURE.md §9
- [ ] `localbench system` works
- [ ] `localbench models` works
- [ ] `localbench ask` works with Ollama
- [ ] Tests pass

**Phase 4 (Dataset):** DATASET_SPECIFICATION.md §10
- [ ] Dataset loaded and validated
- [ ] Repository-disjoint splits verified
- [ ] Ground truth labeled
- [ ] Dataset v1.0.0 frozen

**Phase 6 (Baseline):** EVALUATION_PROTOCOL.md §7
- [ ] Baseline metrics computed for all models
- [ ] Failures classified and analyzed
- [ ] Report generated
- [ ] Artifacts persisted

**Phase 8 (Experiments):** TRAINING_SPECIFICATION.md §11
- [ ] Specialization hypothesis tested
- [ ] Ablations completed
- [ ] Reproducibility verified

---

## Common Questions Answered

**Q: How do I run a benchmark?**  
A: See DEPLOYMENT_GUIDE.md §5 "Running Benchmarks"

**Q: How do I understand the architecture?**  
A: Start with ARCHITECTURE.md §1, then §2 (modules)

**Q: What are the main evaluation metrics?**  
A: See EVALUATION_PROTOCOL.md §3 "Metrics"

**Q: How is the dataset constructed?**  
A: See DATASET_SPECIFICATION.md §4 "Collection Methodology"

**Q: How do I train a specialized model?**  
A: See TRAINING_SPECIFICATION.md §6 "Training Procedure"

**Q: What are the research integrity principles?**  
A: See RESEARCH_METHODOLOGY.md §3–7

**Q: How do I test my implementation?**  
A: See TESTING_STRATEGY.md (entire document)

**Q: How do I ensure reproducibility?**  
A: See RESEARCH_METHODOLOGY.md §6 or DEPLOYMENT_GUIDE.md §11

---

## Contributing to Documentation

When updating documentation:

1. **Update all cross-references** in related documents
2. **Maintain consistent formatting** (headers, code blocks, tables)
3. **Keep version numbering aligned** (all v3)
4. **Include "Last Updated" timestamp**
5. **Document changes in CHANGELOG.md** (if applicable)

---

## Additional Resources

**Original LocalBench Documents:**
- PRD (Product Requirements Document) — Product vision & goals
- Implementation Plan — Phase-by-phase breakdown
- Tracker — Real-time status & milestones
- Technical Spec — Previous technical details (superseded by ARCHITECTURE.md)
- Design Direction — Visual & UX guidance
- App Flow — User journeys
- Rules — Engineering & research rules

---

## Contact & Support

For questions or clarifications:
- **Architecture:** See ARCHITECTURE.md §2 (modules) and §3 (data flow)
- **Research approach:** See RESEARCH_METHODOLOGY.md
- **Operational issues:** See DEPLOYMENT_GUIDE.md §8 (troubleshooting)
- **Test failures:** See TESTING_STRATEGY.md

---

## Summary

This 8-document suite provides comprehensive guidance for building, benchmarking, and understanding LocalBench. Each document serves a specific purpose and audience while maintaining alignment with the overall research vision.

**The core principle:** Local-first, reproducible research on specialized small language models through rigorous experimentation and transparent reporting.

---

**Last Updated:** 2026-08-19  
**LocalBench Version:** 3.0.0  
**Status:** Complete documentation suite for Phase 0 research reset

