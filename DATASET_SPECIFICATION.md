# LocalBench --- Dataset Specification v3

**Status:** Blueprint for Phase 4 implementation
**Last Updated:** 2026-08-20
**Role:** Defines dataset schema, collection methodology, versioning, leakage prevention, and validation procedures.

---

## 1. Dataset Purpose & Scope

The LocalBench dataset exists to support the flagship **code semantic retrieval** workload. It enables:

1. **Semantic artifact generation** — Generate descriptions of code methods
2. **Downstream evaluation** — Test retrieval quality via real developer queries
3. **Specialization training** — Fine-tune a small model on code semantics
4. **Ablation studies** — Evaluate impact of training data size

The dataset is **repository-disjoint** (train/val/test split by repository group) to prevent leakage between stages.

---

## 2. Dataset Structure

### 2.1 High-Level Organization

```
dataset/
├── meta/
│   ├── version.json          # {version: "1.0.0", date: "..."}
│   ├── schema.json           # Dataset schema definition
│   └── stats.json            # Dataset statistics
├── repositories/
│   ├── repo_001/
│   │   ├── metadata.json     # Repository info & provenance
│   │   ├── functions.jsonl   # Code units extracted
│   │   └── source/           # (Optional) Original source files
│   ├── repo_002/
│   └── ...
├── splits/
│   ├── train.jsonl           # Training cases (repository-disjoint)
│   ├── validation.jsonl      # Validation cases
│   └── test.jsonl            # Test cases (frozen)
├── queries/
│   ├── test_queries.jsonl    # Developer-style queries for evaluation
│   └── query_relevance.jsonl # Ground truth: query → relevant code units
└── README.md
```

---

## 3. Core Schemas

### 3.1 Code Unit (BenchmarkCase)

A **code unit** is a single function or method with context.

```json
{
  "id": "repo001_py_class_PaymentProcessor_method_process_retry",
  "repository": "repo001",
  "language": "python",
  "file_path": "payment/processor.py",
  "symbol": "PaymentProcessor.process_retry",
  "symbol_type": "method",
  "source_code": "def process_retry(self, transaction_id, max_attempts=3):\n    \"\"\"Retry a failed payment transaction with exponential backoff.\"\"\"\n    ...",
  "context": {
    "class_name": "PaymentProcessor",
    "module_docstring": "Payment transaction processing...",
    "imports": ["time", "logging"],
    "parent_methods": ["__init__", "validate_transaction"]
  },
  "source_url": "https://github.com/example/repo/blob/main/payment/processor.py#L42",
  "split": "test",
  "is_public": true,
  "docstring": "Retry a failed payment transaction with exponential backoff.",
  "source_file_lines": 42,
  "extracted_at": "2026-08-19T10:30:00Z"
}
```

**Fields:**
- `id` — Globally unique code unit identifier (repo + symbol path)
- `repository` — Source repository identifier
- `language` — Programming language (python, java, js, etc.)
- `file_path` — Relative path within repository
- `symbol` — Qualified name (e.g., `ClassName.method_name`)
- `source_code` — Full function/method body
- `context` — Surrounding class/module context
- `split` — {train, validation, test}
- `is_public` — True if method/function is public API
- `extracted_at` — ISO timestamp

**Constraints:**
- Source code must be syntactically valid
- Symbol must match extraction
- Split must match repository group assignment

---

### 3.2 Semantic Label

A **semantic label** is human-assigned or AI-generated metadata for a code unit.

```json
{
  "code_unit_id": "repo001_py_class_PaymentProcessor_method_process_retry",
  "description": "Retries a failed payment transaction using exponential backoff with configurable maximum attempts. Logs each retry attempt and raises an exception after exhausting retries.",
  "summary": "Payment retry with exponential backoff",
  "concepts": [
    "retry logic",
    "exponential backoff",
    "transaction processing",
    "error handling"
  ],
  "input_types": ["int (transaction_id)", "int (max_attempts)"],
  "output_type": "bool",
  "side_effects": ["Logs to payment logger", "Updates transaction status in database"],
  "created_by": "human",
  "label_version": "1.0.0"
}
```

**Fields:**
- `description` — Detailed explanation of what the function does
- `summary` — One-sentence summary
- `concepts` — Semantic tags (for cross-function retrieval)
- `input_types` — Parameter types and meanings
- `output_type` — Return type
- `side_effects` — State modifications (logging, DB updates, etc.)
- `created_by` — {human, model_generated, hybrid}
- `label_version` — Versioning for label schema

---

### 3.3 Query Case

A **query case** represents a developer looking for code.

```json
{
  "id": "query_test_0001",
  "query": "Where are payment transactions retried when they fail?",
  "query_style": "natural",
  "query_intent": "find_error_handling",
  "relevant_code_units": [
    "repo001_py_class_PaymentProcessor_method_process_retry"
  ],
  "related_concepts": [
    "retry logic",
    "exponential backoff",
    "failure recovery"
  ],
  "split": "test",
  "difficulty": "medium",
  "created_at": "2026-08-19T10:30:00Z"
}
```

**Fields:**
- `id` — Unique query identifier
- `query` — Natural language question
- `query_style` — {natural, technical, verbose, concise}
- `query_intent` — {find_implementation, find_bug, find_optimization, etc.}
- `relevant_code_units` — Ground truth: list of IDs
- `related_concepts` — Semantic tags to guide understanding
- `split` — {test} (queries only used in evaluation)
- `difficulty` — {easy, medium, hard}

**Constraints:**
- At least one relevant code unit must exist
- Relevant units must be from the same split
- Query must be realistically searchable

---

### 3.4 Query-Relevance Mapping

A **relevance relationship** maps queries to code units.

```json
{
  "query_id": "query_test_0001",
  "code_unit_id": "repo001_py_class_PaymentProcessor_method_process_retry",
  "relevance_score": 1.0,
  "relevance_label": "direct_match",
  "explanation": "This method directly implements the retry logic described in the query."
}
```

**Relevance labels:**
- `direct_match` (1.0) — Exact match to query
- `highly_relevant` (0.8) — Closely related, answers query
- `related` (0.5) — Tangentially relevant
- `not_relevant` (0.0) — Irrelevant

---

## 4. Collection Methodology

### 4.1 Repository Selection

**Criteria for source repositories:**

1. **Language:** Python, Java, JavaScript, TypeScript (start with Python)
2. **Scale:** 2k–50k lines of code (manageable, realistic)
3. **Quality:** Well-maintained, meaningful docstrings/comments
4. **Diversity:** Different domains (web, CLI, data processing, etc.)
5. **License:** Permissive (MIT, Apache 2.0, etc.); avoid GPL
6. **Availability:** Publicly available; no private/proprietary code

**Candidate sources:**
- Popular open-source projects (e.g., pandas, requests, FastAPI)
- Well-documented libraries
- Projects with good test coverage (indicates clarity)

**Minimum dataset:**
- 3–5 repositories
- 300–500 total methods/functions
- Train: 50%, Validation: 25%, Test: 25% (by repository group, not case count)

---

### 4.2 Code Unit Extraction

**Extraction process:**

1. **Source parsing** — Parse repository into AST using Python `ast` (stdlib)
2. **Function/method discovery** — Identify all functions and methods
3. **Filtering** — Exclude:
   - Private methods (single underscore prefix, unless pedagogically useful)
   - Generated code
   - Vendor code
   - Tests and test utilities
   - Lambdas and trivial single-liners (< 3 source lines)
   - Nested functions (only top-level functions and class methods are extracted)
   - Code units exceeding 100 source lines
4. **Context collection** — Capture class/module context
5. **Validation** — Ensure source code is valid, symbol matches
6. **Normalization** — Consistent formatting, encoding (UTF-8)
7. **Deduplication** — Exact duplicates detected via content hashing

**Code unit bounds (v1):**
- Minimum: 3 source lines (functions/methods shorter than this are excluded)
- Maximum: 100 source lines (functions/methods longer than this are excluded)

**Tools:**
- AST parsing: `ast` (Python stdlib) — v1 scope
- Duplicate detection: Content hashing (SHA256 of normalized source code)

---

### 4.3 Semantic Labeling

Semantic labels describe what each code unit does. They exist for all code units across all splits (train, validation, test). Labels are used for training data (code → description pairs). Retrieval queries (§4.4) are generated from source code only, without labels.

**Human labeling process:**

1. **Reviewer reads** code unit (source + context)
2. **Writes description** — What does it do? Why?
3. **Extracts concepts** — Semantic tags (retry, caching, validation, etc.)
4. **Notes side effects** — State modifications
5. **Labels parameters/return** — Input/output types and meanings

**AI-assisted labeling:**

1. **Template generation** — Use a prompt to generate initial label
2. **Human review** — Annotator accepts/revises
3. **Consensus** — Multiple annotators if high-quality label desired

**Quality gates:**
- Descriptions ≥ 20 words, ≤ 256 words
- ≥ 2 concepts, ≤ 10 concepts
- Descriptions must reference code specifics (not generic)

**Distinction from retrieval queries:** Semantic labels describe code units in detail. Retrieval queries (§4.4) are short developer-style questions used only for evaluation. These are separate artifacts with separate generation processes.

---

### 4.4 Retrieval Query Generation

Retrieval queries are short developer-style questions used exclusively for evaluation. They are a **separate artifact** from semantic labels.

**v1 query design (frozen):**
- 45 retrieval queries total
- All 45 queries belong to the **test split only**
- Train and validation splits contain **no retrieval queries**
- Queries are generated against test code units (source code only; semantic labels are NOT provided to the query generator — see §4.4.1)

**Query generation methodology:**

1. A **dedicated local query-generation model** generates candidate queries
2. This model must be **separate from all benchmark models** (the models being evaluated)
3. The query-generation model must **never** be one of the models under evaluation
4. A human reviewer **validates and finalizes** all queries before freeze

**Reproducibility requirements (mandatory):**
- Model name and version of the query-generation model must be recorded
- Prompt template version must be recorded
- Deterministic seed must be recorded
- All three are stored in dataset metadata

**Query generation guidelines:**
- Use question or imperative form
- Reference actual concepts from the code (not generic)
- Vary query style (natural, technical, verbose, concise)
- Ensure query is unambiguous for humans
- A developer should find the relevant code unit in top-10 results
- Avoid queries that could match multiple unrelated units

**Style diversity:** The guidelines encourage varying query style (natural, technical, verbose, concise) but do not impose numerical distribution quotas. The resulting distribution across the 45 final queries is recorded in dataset statistics (§8.1).

### 4.4.1 Query-Generation Model Selection (Decision Record)

**Selected model:** `qwen2.5-coder:7b` (4.7GB, local Ollama deployment)

**Selection methodology:** Controlled smoke test on 2026-08-21. Five Ollama models evaluated on the same prompt — generating a natural-language query from a function signature + docstring without leaking identifiers.

| Model | Size | Latency | Output Quality | Verdict |
|---|---|---|---|---|
| qwen2.5-coder:7b | 4.7GB | ~49s | Clean natural query, no identifier leaks | **Selected** |
| llama3:latest | 4.7GB | ~27s | Clean but slightly verbose | Acceptable alternative |
| mistral:latest | 4.4GB | ~21s | Leaked function name + parameter names | Rejected |
| gemma4:latest | 9.6GB | >90s timeout | Timed out | Rejected |
| qwen3:latest | 5.2GB | >120s timeout | Timed out | Rejected |

**Rationale:**
- Code-specialized model produces queries grounded in functionality, not identifiers
- No leakage of function names, parameter names, or class names in output
- Moderate size (4.7GB) fits resource constraints
- ~49s latency acceptable for 45-query batch generation

**Independence guarantee:** `qwen2.5-coder:7b` is **never** used as a benchmark model (models being evaluated for retrieval performance). It serves exclusively as dataset infrastructure.

**SemanticLabel visibility:** The query generator receives **source code only** — it does NOT receive SemanticLabels. Labels are a separate artifact used for training, not query generation. This prevents circular dependency between label generation and query generation.

**Reproducibility metadata stored in `meta/version.json`:**
- `model_name`: "qwen2.5-coder:7b"
- `model_version`: "7b"
- `prompt_template_version`: "1.0.0"
- `seed`: 42
- `generation_params`: {temperature: 0.7, top_p: 0.9}

### 4.4.2 Candidate Coverage & Selection Pipeline

**Coverage principle:** Generate one initial candidate for every eligible test CodeUnit. Do NOT pre-select a subset of CodeUnits before candidate generation.

**Full pipeline:**

```
113 test CodeUnits
        ↓
candidate generation (one candidate per CodeUnit)
        ↓
automated validation (schema + leakage checks)
        ↓
human review (benchmark-blind — see §4.4.4)
        ↓
eligible candidate pool (may be < 113)
        ↓
deterministic seed-42 selection → 45 final queries
        ↓
ground-truth relevance assignment (after freeze — see §4.5)
```

**Selection method:** The final 45 queries are selected deterministically from the eligible candidate pool using seed=42. Selection must not observe benchmark results. The eligible pool must contain at least 45 candidates; if fewer than 45 candidates pass review, the dataset is incomplete and must be regenerated with adjusted parameters.

**Stratification:** The canonical documentation does not mandate numerical stratification quotas by repository, symbol type, or query style. Diversity is encouraged (§4.4 guidelines) but not enforced via quotas. The resulting distribution is recorded in dataset statistics.

### 4.4.3 Rejection & Regeneration Policy

**Automated rejection triggers:**
1. Schema validation failure (CandidateQuery does not parse)
2. Leakage detection failure (check_query_leakage returns LEAK_DETECTED)
3. Empty or trivially short query text

**Bounded regeneration:** Maximum 3 generation attempts per CodeUnit. The existing Phase 3 retry infrastructure handles structured-generation failures (malformed JSON, missing fields). Semantic rejection (leakage, quality) is a separate post-generation decision.

**Audit trail:** For every failed candidate, regardless of attempt number, retain:
- Candidate text
- Attempt number
- Rejection reason (automated or human)
- Automated validation result
- Leakage check result
- Generation metadata (model, seed, params)

Failed candidates are never silently deleted. If all 3 attempts fail, the CodeUnit is marked as having no eligible candidate and excluded from the selection pool.

**Semantic rejection criteria (human review):**
- Query is incoherent or grammatically broken
- Query does not reference identifiable code concepts
- Query is too generic (would match many unrelated units)
- Query leaks identifiers despite automated checks

### 4.4.4 Human Review (Benchmark-Blind)

Human review is mandatory before final query selection (§4.4 step 4). The reviewer must operate under **benchmark-result blindness**.

**Reviewer may see:**
- Candidate query text
- Target CodeUnit (source code + context)
- Permitted repository/source context
- Automated validation results (schema pass/fail, leakage pass/fail)

**Reviewer must NOT see:**
- Benchmark model results (Hit@K, MRR, etc.)
- Model rankings or performance comparisons
- Information that could encourage selecting queries favorable to a particular benchmark model
- Ground-truth relevance scores (assigned after review)

**Review criteria (all must pass for acceptance):**

| # | Criterion | Maps to |
|---|-----------|---------|
| 1 | Understandable — query is grammatically correct and clear | §4.4 guidelines |
| 2 | Behaviorally relevant — query describes real code behavior | §4.4 "reference actual concepts" |
| 3 | Sufficiently specific — not generic, grounded in this CodeUnit | §4.4 "avoid queries that match multiple unrelated units" |
| 4 | Unambiguous — one clear interpretation | §4.4 "unambiguous for humans" |
| 5 | No implementation leakage — no function/class/parameter names | §4.4.1 SemanticLabel visibility + leakage check |
| 6 | Developer could locate code — relevant unit reachable in top-10 | §4.4 "developer should find relevant code in top-10" |

**Outcome:** Accept or reject with documented reason. Rejected candidates trigger regeneration (§4.4.3). Maximum review iterations are bounded by the 3-attempt regeneration limit.

---

### 4.4.5 Final-45 Selection Procedure

**Status:** Frozen methodology resolution (Phase 4F-I-C1, 2026-08-23). Codifies the §4.4.2 requirement ("deterministically … using seed=42") into an exact, reproducible algorithm. Selection remains benchmark-blind per §4.4.4.

**Eligible pool.** A candidate enters the eligible pool only if all of the following hold:

1. Structured/schema validation passed.
2. Leakage screening passed (`check_query_leakage`).
3. Non-trivial-query check passed (query text non-empty).
4. The target CodeUnit belongs to the canonical test split.
5. Its `code_unit_id` is unique within the pool.

Failure records are never eligible. If any duplicate `code_unit_id` is encountered during selection, selection aborts with an error; duplicates are never silently resolved.

**Canonical order.** The eligible pool is sorted by `code_unit_id` ascending using Python string ordering. This sorted sequence is the sole input order to sampling.

**PRNG.** Python standard-library `random.Random(42)`. No NumPy RNG, hash randomization, system entropy, timestamps, or unordered-set iteration may influence selection.

**Sampling.** Exactly 45 candidates are drawn with:

```python
random.Random(42).sample(ordered_eligible_candidates, 45)
```

This is uniform sampling without replacement. Because IDs are unique and the ordering is canonical, no secondary tie-break rule is required.

**Prohibited influences.** No repository quotas, no query-style quotas, no symbol-type quotas, no length quotas, no manually chosen candidates, and no observation of benchmark results or model performance. Resulting repository/style distributions are observations recorded after the draw, never constraints.

**Selection record.** Every selection execution writes a reproducible record containing at minimum:

```json
{
  "selection_version": "<string>",
  "seed": 42,
  "prng": "python.random.Random",
  "python_version": "<runtime version>",
  "sampling_method": "sample_without_replacement",
  "canonical_order": "code_unit_id_lexicographic_ascending",
  "eligible_candidate_count": 0,
  "eligible_pool_sha256": "<sha256>",
  "selected_count": 45,
  "selected_code_unit_ids": ["<id>", "..."],
  "selected_candidate_ids": ["<id>", "..."],
  "selected_repository_distribution": {"<repo>": <int>},
  "selected_query_style_distribution": {"<style>": <int>},
  "generation_source_commit": "<commit>",
  "selection_created_utc": "<iso8601>"
}
```

Selected IDs are recorded explicitly so the artifact is independently auditable. The record must NOT contain benchmark metrics, model rankings, Hit@K/MRR/latency values, or ground-truth relevance.

**Human-review scope.** The documented procedure remains review-all-before-selection (§4.4.2, §4.4.4): every candidate in the eligible pool is reviewed before the deterministic draw. At the 2026-08-23 artifact state the automated-pass pool holds 2,077 candidates (versus the original 113-unit planning figure), so full pre-selection review is a substantial manual effort. Any scope reduction (for example, reviewing only the eventual 45) requires an explicit, documented amendment BEFORE selection executes; no such amendment is currently approved.

---

### 4.5 Ground-Truth Labeling

**Timing:** Ground-truth relevance assignment occurs **after** final query selection and freeze. Never before.

**Blindness:** Ground-truth assignment must not be influenced by benchmark model outputs. Relevance is assessed based on the query text and the CodeUnit source code alone.

**Creating relevance relationships:**

1. For each query, assign relevant code unit(s)
2. Assess relevance score (0.0–1.0)
3. Explain why (direct match, related, etc.)
4. Handle edge cases:
   - Multi-code-unit queries → all relevant units listed
   - Ambiguous queries → disambiguate or remove

**Validation:**
- No query with zero relevant units
- No code unit with zero relevant queries (unless validation-only)
- Relevance symmetry checked (reciprocal relationships make sense)

---

## 5. Dataset Versioning

### 5.1 Version Identifier

Dataset version: `MAJOR.MINOR.PATCH`

```json
{
  "version": "1.0.0",
  "release_date": "2026-09-30",
  "repositories": ["repo001", "repo002", "repo003"],
  "repository_commits": {
    "repo001": "abc1234",
    "repo002": "def5678",
    "repo003": "ghi9012"
  },
  "total_code_units": 450,
  "train_cases": 225,
  "validation_cases": 112,
  "test_cases": 113,
  "total_queries": 45,
  "split_seed": 42,
  "query_generation": {
    "model_name": "qwen2.5-coder:7b",
    "model_version": "7b",
    "prompt_template_version": "1.0.0",
    "seed": 42,
    "generation_params": {
      "temperature": 0.7,
      "top_p": 0.9
    }
  },
  "schema_version": "1.0.0",
  "parser": "python_ast",
  "frozen": true
}
```

**Version increments:**
- **MAJOR** — Schema changes (new fields, removed fields, renamed fields)
- **MINOR** — Content changes (new repositories, new queries, updated labels)
- **PATCH** — Corrections (typo fixes, labeling corrections, validation fixes)

---

### 5.2 Immutability & Freezing

**Freeze points:**

1. **Initial dataset freeze** (end of Phase 4) → `dataset-v1.0.0`
2. **Before baseline benchmark** → Test set is frozen
3. **Before specialization** → No test set changes

**Frozen properties:**
- Test split is immutable
- Queries are immutable
- Ground-truth relevance is immutable

**Mutable (only before freeze):**
- Train/validation content
- Semantic labels (for train/val only)
- Metadata and documentation

---

## 6. Leakage Prevention

### 6.1 Repository-Disjoint Splits

**Key principle:** Train and test repositories are completely separate.

```
Repository A → Train split
Repository B → Validation split
Repository C → Test split
```

**Rationale:**
- Models cannot memorize test code from training data
- Prevents overfitting to specific repositories
- Ensures generalization to new codebases

---

### 6.2 Cross-Validation Boundaries

During specialization ablations:

- **Train set:** Used only for fine-tuning
- **Validation set:** Used for hyperparameter tuning, early stopping
- **Test set:** Used only for final evaluation (frozen, never tuned against)

**Forbidden:**
- Tuning prompts against test queries
- Selecting models based on test retrieval metrics
- Running queries through test code for label inspiration

---

### 6.3 Data Access Control

**During baseline phase:**
- Trainers work only on train split
- Benchmarkers access full dataset (needed for ground truth)
- Evaluators access only test split

**During specialization:**
- Fine-tuners access only train split
- Evaluators access test split (never used during training)

---

## 7. Dataset Validation

### 7.1 Schema Validation

All records must pass Pydantic validation:

```python
class BenchmarkCase(BaseModel):
    id: str
    repository: str
    language: str
    symbol: str
    source_code: str
    split: Literal["train", "validation", "test"]
    # ... other fields

class Query(BaseModel):
    id: str
    query: str
    relevant_code_units: List[str]
    split: Literal["test"]
    # ... other fields
```

**Validation pipeline:**
1. Type checking (Pydantic)
2. Uniqueness (IDs are globally unique)
3. Referential integrity (query.relevant_code_units exist)
4. Split consistency (all cases in split are from same repository group)

---

### 7.2 Statistical Validation

```python
# Check coverage
assert len(train_split) > 200, "Train split too small"
assert len(test_split) > 50, "Test split too small"

# Check relevance distribution
query_relevance_count = len([r for r in relevance if r.score > 0.5])
assert query_relevance_count / len(queries) > 0.8, "Low relevance coverage"

# Check split isolation
train_repos = {c.repository for c in train_split}
test_repos = {c.repository for c in test_split}
assert train_repos.isdisjoint(test_repos), "Repository leakage detected"
```

---

### 7.3 Manual Quality Review

**Sample validation:**
1. Randomly sample 50 code units from each split
2. For each unit:
   - Is the extracted source code correct?
   - Is the semantic label accurate and complete?
   - Would a developer find relevant queries?
3. Document any issues; flag units for correction

---

## 8. Dataset Statistics & Metadata

### 8.1 Descriptive Statistics

```json
{
  "code_units": {
    "total": 450,
    "by_language": {"python": 450},
    "by_split": {"train": 225, "validation": 112, "test": 113},
    "by_repository": {"repo001": 150, "repo002": 150, "repo003": 150},
    "avg_lines_of_code": 18.5,
    "avg_docstring_length": 42,
    "concept_coverage": 82
  },
  "queries": {
    "total": 45,
    "by_split": {"test": 45},
    "by_difficulty": {"easy": 15, "medium": 20, "hard": 10},
    "avg_relevant_per_query": 1.8,
    "query_coverage": 0.98
  },
  "relevance": {
    "total_relationships": 81,
    "direct_matches": 45,
    "highly_relevant": 30,
    "related": 6
  }
}
```

---

## 9. Dataset Artifacts

### 9.1 Deliverables

1. **Code units** — `repository/repo_*/functions.jsonl`
2. **Semantic labels** — Embedded in code unit JSON or separate `labels.jsonl`
3. **Splits** — `splits/{train,validation,test}.jsonl`
4. **Queries** — `queries/test_queries.jsonl`
5. **Relevance** — `queries/query_relevance.jsonl`
6. **Metadata** — `meta/version.json`, `meta/stats.json`
7. **README** — Methodology, schema, statistics, usage notes

### 9.2 Distribution

**Storage:**
- Local: `dataset/` directory in repository (if <500MB)
- Remote: Hosted on data server (if >500MB)
- Download: Script to fetch dataset with integrity check

**Reproducibility:**
- Git commit hash for version lock
- Checksums (SHA256) for artifact verification

---

## 10. Phase 4 Implementation Checklist

- [ ] Select 3–5 Python source repositories (2k–50k LoC each)
- [ ] Record exact commit/tag for each repository
- [ ] Extract code units (3–100 source lines, top-level only, no nested functions)
- [ ] Deduplicate via content hashing (SHA256)
- [ ] Create semantic labels (human-reviewed, all splits)
- [ ] Generate candidate queries for all eligible test CodeUnits using dedicated query-generation model
- [ ] Run automated validation (schema + leakage checks) on all candidates
- [ ] Apply bounded regeneration (max 3 attempts) for failed candidates
- [ ] Human review all candidates (benchmark-blind)
- [ ] Select 45 final queries deterministically (seed=42) from eligible pool
- [ ] Record model version, prompt version, seed in metadata
- [ ] Assign ground-truth relevance (after query freeze, no benchmark influence)
- [ ] Create train/validation/test splits (repository-disjoint, seed=42)
- [ ] Validate schema (Pydantic)
- [ ] Validate referential integrity
- [ ] Manual quality review (50 samples)
- [ ] Compute and document statistics
- [ ] Freeze dataset v1.0.0
- [ ] Write dataset README with methodology
- [ ] Publish to version control with commit hash

---

## 11. Future Dataset Extensions

**Out of scope for v1.0.0:**

- Multiple programming languages (v1 = Python only)
- Large-scale dataset (100k+ units)
- Continuous query updates
- User-submitted queries

**Future capabilities:**

- Dataset v2.0 with additional repositories
- Additional workloads (e.g., bug detection, API usage)
- Query generation from user logs
- Cross-repository concept linking

---

## 12. Key Principles

1. **Repository-disjoint splits** — No code reuse between train and test
2. **Frozen test set** — Immutable after baseline phase
3. **Ground-truth is sacred** — Relevance labels never tuned against
4. **Versioning is explicit** — All changes tracked and versioned
5. **Schema is contracts** — Pydantic enforces correctness
6. **Human review is first-class** — Labels are validated by humans
7. **Leakage is unacceptable** — Detected and prevented systematically

