# LocalBench --- Research Methodology & Scientific Integrity v3

**Status:** Foundational research guidelines  
**Last Updated:** 2026-08-19  
**Role:** Documents research integrity principles, experimental design decisions, and limitations of findings.

---

## 1. Research Framework

### 1.1 Research Question

> **Can a small, locally deployable language model specialized for a specific real-world workload achieve competitive downstream performance against larger general-purpose local models while using substantially fewer computational resources?**

This question is **falsifiable**:
- Affirmative answer: Specialization provides measurable retrieval quality improvement
- Negative answer: Specialization does not improve performance (null result is valid)
- Mixed answer: Specialization helps for some model sizes/training data sizes but not others

---

### 1.2 Hypothesis

**Primary:** A small local model specialized for code semantic representation should improve substantially over its zero-shot baseline and may approach the downstream retrieval performance of larger general-purpose local models while requiring less RAM, lower latency, and fewer compute resources.

**Sub-hypotheses:**
1. Specialization improves zero-shot baseline (H1)
2. Specialization improvements scale with training data (H2)
3. Specialized small model can match large model quality (H3)
4. Specialization maintains computational efficiency (H4)

---

## 2. Experimental Design

### 2.1 Variables

**Independent variables (what we change):**
- Model size (0.5B, 3B, 7B, 9B)
- Training data (none, 50 ex, 112 ex, 225 ex)
- Fine-tuning method (LoRA rank 4, 8, 16; full fine-tune)
- Hardware (GPU present vs CPU-only)

**Dependent variables (what we measure):**
- Hit@K metrics (primary)
- MRR (primary)
- Latency (secondary)
- Peak RAM (secondary)
- Model footprint (secondary)
- Reliability/failure rate (supporting)

**Controlled variables (kept constant):**
- Dataset version (1.0.0)
- Test split (frozen, immutable)
- Workload definition (code retrieval)
- Prompt template
- Temperature (0.3)
- Max tokens (256)
- Embedding model (for retrieval)

---

### 2.2 Experimental Groups

**Group A: Baseline Models (zero-shot)**
- phi-3-mini (0.5B)
- phi-3 (3.8B)
- mistral-7b (7B)
- gemma-2 (9B)

**Group B: Specialized Small Model**
- phi-3-mini fine-tuned on full train set

**Group C: Ablation Studies (optional, Phase 7)**
- phi-3-mini fine-tuned on subset 1 (50 examples)
- phi-3-mini fine-tuned on subset 2 (112 examples)
- phi-3-mini fine-tuned on full set (225 examples)

---

## 3. Scientific Rigor

### 3.1 Baseline-First Principle

**Rule:** Establish baseline benchmarks before training any specialized model.

**Rationale:**
- Prevents confirmation bias (assuming specialization will help)
- Establishes ground truth for comparison
- Allows fair assessment of specialization impact

**Implementation:**
- Phase 5 (baseline) must complete before Phase 6 (specialization)
- Baseline metrics frozen; no tuning against baseline afterward

---

### 3.2 Test Set Immutability

**Rule:** Once final test set is frozen (end of Phase 4), it is never modified or tuned against.

**Prohibited actions:**
- Adjusting prompts based on test case failures
- Retraining models based on test set performance
- Removing "hard" test cases to improve metrics
- Rebalancing test set based on results

**Rationale:**
- Prevents overfitting
- Ensures generalization
- Maintains integrity of downstream evaluation

---

### 3.3 Repository-Disjoint Splits

**Rule:** Train, validation, and test splits are repository-disjoint.

**Implementation:**
```
Repository A → Train split
Repository B → Validation split
Repository C → Test split
```

**Rationale:**
- Prevents code memorization
- Ensures transfer to novel codebases
- Reflects real-world deployment (generalizing to unseen repositories)

---

### 3.4 Negative Results Are Valid

**Principle:** Null results are scientifically valuable.

**Examples of valid null results:**
- "Specialization with 225 examples does not improve over baseline"
- "LoRA with rank 8 does not differ from rank 4"
- "Retrieval performance plateaus at 112 training examples"

**How to report null results:**
```markdown
## Hypothesis H1: Specialization Improves Baseline
**Result:** NOT SUPPORTED

Specialized phi-3-mini achieved Hit@10 = 0.901 vs baseline 0.900.
95% CI: [-0.02, 0.04]. This difference is not statistically significant.
Conclusion: Evidence insufficient to support specialization benefit.
```

---

## 4. Statistical Methodology

### 4.1 Sample Size & Power

**For Phase 5 (baseline):**
- Test set: 113 cases
- Sufficient for computing reliable Hit@K, MRR
- No formal power analysis needed (single-run benchmark)

**For Phase 7 (comparison):**
- Same 113 test cases used across all models
- Paired comparison (matched on test cases)
- Effect size detected at typical 0.05 significance level

**Assumptions:**
- Hit@K approximately normal for aggregate queries
- MRR approximately normal
- Statistical tests are exploratory (not confirmatory)

---

### 4.2 Significance Testing

**When applicable:**
- Comparing two models on identical test set
- Test set size > 30 cases (satisfied: 113 > 30)
- Metric has natural variance (Hit@K, MRR do)

**Method: Paired t-test**
```python
from scipy import stats

# Data: Hit@10 for each query
baseline_hits = [...]  # 45 queries
specialized_hits = [...]

# Paired t-test (same queries, two models)
t_stat, p_value = stats.ttest_rel(specialized_hits, baseline_hits)

# Report with confidence interval
effect = np.mean(specialized_hits) - np.mean(baseline_hits)
se = np.std(specialized_hits - baseline_hits) / np.sqrt(len(baseline_hits))
ci = (effect - 1.96*se, effect + 1.96*se)

print(f"Effect: {effect:.3f} (95% CI: {ci})")
print(f"p-value: {p_value:.4f}")
```

**Interpretation:**
- p < 0.05: Statistically significant difference
- p ≥ 0.05: No significant difference (or insufficient evidence)
- Confidence interval contains 0: Consistent with null hypothesis

---

### 4.3 Multiple Comparisons

**Issue:** If testing many hypotheses, false positive rate increases.

**Approach:**
- Identify primary hypotheses (H1: specialization improves baseline)
- Secondary/exploratory hypotheses are clearly labeled
- Bonferroni correction considered for multiple tests

**Example:**
```python
# Primary hypothesis (alpha = 0.05)
t_stat, p_value = ttest_rel(specialized, baseline)

# Secondary hypothesis (exploratory, not multiple comparison corrected)
# (Clearly label in report)
```

---

## 5. Validity Threats & Limitations

### 5.1 Internal Validity

**Threat:** The observed effect is due to the treatment (specialization), not confounds.

| Threat | Mitigation |
|--------|-----------|
| Instrumentation change | Identical benchmark protocol across runs |
| History (time-based changes) | Same test set, frozen ground truth |
| Selection bias | Repository-disjoint splits |
| Maturation | Single pass benchmark (no learning curves) |

---

### 5.2 External Validity

**Threat:** Findings generalize beyond the specific test set/models.

**Known limitations:**

1. **Repository specificity**
   - Dataset: 3 Python repositories (450 functions)
   - Generalizability: Other Python projects, unknown
   - Different domains (web, ML, CLI) may behave differently
   - Other languages untested

2. **Model specificity**
   - Primary: phi-3-mini (0.5B)
   - Tested: phi-3, mistral-7b, gemma-2
   - Specialized architecture effects unknown
   - Larger models (>20B) untested

3. **Task specificity**
   - Task: Code semantic retrieval
   - Other code tasks (bug detection, summarization) unknown
   - Non-code domains untested

4. **Training data specificity**
   - Size: 225 examples (small)
   - Quality: Well-documented code (bias toward high-quality repositories)
   - Domain: Primarily Python (language bias)

**Scope statement:**
> These findings are specific to code semantic retrieval using Python codebases and small local models (≤7B). Generalization to other languages, tasks, or model families requires additional research.

---

### 5.3 Construct Validity

**Threat:** Metrics measure what we intend to measure.

| Metric | What it measures | What it doesn't measure |
|--------|------------------|------------------------|
| Hit@10 | Presence of relevant code in top-10 | Ranking quality (MRR handles this) |
| MRR | Average rank of first relevant result | Absolute quality of descriptions |
| Latency | Wall-clock generation time | User perception/interactivity |
| RAM | Peak process memory | Full system memory impact |

**Mitigation:**
- Measure downstream utility (retrieval), not surface quality
- Multiple metrics provide complementary views
- Limitations acknowledged explicitly

---

## 6. Reproducibility Standards

### 6.1 Information Required for Reproduction

Every result must include:

```json
{
  "experiment_id": "baseline_20260930",
  "research_question": "...",
  "hypothesis": "...",
  "workload": "code-retrieval-v1",
  "workload_version": "1.0.0",
  "dataset": {
    "version": "1.0.0",
    "code_units": 450,
    "train": 225,
    "validation": 112,
    "test": 113
  },
  "models": [
    {
      "name": "phi-3-mini",
      "parameter_count": "0.5B",
      "quantization": "Q4_K_M",
      "checkpoint": "ollama/phi-3-mini:latest"
    }
  ],
  "benchmark_protocol": "3.0.0",
  "hardware": {
    "cpu": "Intel Core i7-11700K",
    "ram_gb": 32,
    "gpu": "NVIDIA RTX 3090",
    "vram_gb": 24,
    "os": "Ubuntu 22.04"
  },
  "software": {
    "python": "3.10.11",
    "ollama": "0.29.0",
    "localbench": "3.0.0",
    "git_commit": "abc123def456..."
  },
  "configuration": {
    "temperature": 0.3,
    "max_tokens": 256,
    "seed": 42,
    "warmup_cases": 5
  },
  "results": {
    "hit_at_1": 0.62,
    "hit_at_10": 0.90,
    "mrr": 0.76,
    "latency_ms": 720,
    "peak_ram_mb": 1128,
    "failure_rate": 0.03
  },
  "artifacts": {
    "raw_outputs": "results/.../raw_outputs.jsonl",
    "metrics": "results/.../metrics.json",
    "report": "results/.../report.md"
  }
}
```

---

### 6.2 Reproducibility Checklist

- [ ] Dataset version immutable and archived
- [ ] Model identifiers exact (including quantization)
- [ ] Hardware configuration documented
- [ ] Software versions recorded
- [ ] Random seed fixed and reported
- [ ] All hyperparameters recorded
- [ ] Raw outputs preserved
- [ ] Metrics reproducible from raw outputs
- [ ] Git commit hash for code version
- [ ] README documents methodology
- [ ] Limitations clearly stated

---

## 7. Conflict of Interest & Bias

### 7.1 Potential Biases

**Development bias:** The researchers building LocalBench also design the experiments.

**Mitigation:**
- Hypothesis stated before results
- Null results accepted
- External review of methods/findings
- Separate roles (experimentalist vs. analyst)

---

### 7.2 Model Selection Bias

**Threat:** Choosing models that are easier to specialize.

**Mitigation:**
- Model selection based on availability/popularity, not cherry-picking
- Rationale documented
- Ablations explore effect of model size

---

## 8. Ethical Considerations

### 8.1 Code Privacy

**Principle:** Preserve privacy of evaluated code.

**Implementation:**
- No source code stored in published results (except metadata)
- No unique code snippets in reports
- Anonymized repository names

---

### 8.2 Responsible Claims

**Avoid:**
- "Small models are better" (oversimplification)
- "This proves specialization always helps" (overgeneralization)
- Comparisons to proprietary/unavailable models

**Instead:**
- "Under these specific conditions, specialization improved retrieval"
- "Effect size is modest but statistically significant"
- Acknowledge limitations

---

## 9. Publication & Dissemination

### 9.1 What to Publish

**Publishable:**
- Full methodology (reproducible)
- Raw results and artifacts
- Limitations and caveats
- Negative/null results

**Not for publication (privacy):**
- Proprietary source code
- Sensitive repository names
- User data

---

### 9.2 Transparent Reporting

**Template for results publication:**

```markdown
# LocalBench: Code Semantic Retrieval Specialization Study

## Abstract
[Clear summary of research question, method, results, limitations]

## Methods
- Workload definition
- Dataset composition
- Baseline protocol
- Specialization approach
- Evaluation metrics
- Statistical methods

## Results
- Baseline benchmarks
- Specialization impact
- Ablation studies
- Failure analysis

## Discussion
- Findings vs. hypothesis
- Limitations
- Generalizability
- Future work

## Reproducibility
- All code in GitHub: [repo]
- Dataset: [link]
- Artifact archive: [link]
- Full configuration: [config file]

## Limitations
- Repository specificity
- Model scope
- Task scope
- Training data size
- Single-run nature

## Conclusion
[Restrained conclusion scoped to findings]
```

---

## 10. Key Principles

1. **Hypothesis first** — State predictions before observing results
2. **Baseline is gold** — Don't tune against test set
3. **Negative results matter** — Null results are published
4. **Limits are real** — Acknowledge scope and assumptions
5. **Transparency is default** — All methods, data, code available
6. **Reproduction is possible** — Full information for re-running study
7. **Claims are scoped** — Tied to specific task/dataset/models

