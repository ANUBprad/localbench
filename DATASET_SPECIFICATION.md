# LocalBench --- Dataset Specification v3

**Status:** Blueprint for Phase 3 implementation  
**Last Updated:** 2026-08-19  
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

1. **Source parsing** — Parse repository into AST
2. **Function/method discovery** — Identify all functions and methods
3. **Filtering** — Exclude:
   - Private methods (single underscore prefix, unless pedagogically useful)
   - Generated code
   - Tests and test utilities
   - Lambdas and trivial single-liners (< 3 lines)
4. **Context collection** — Capture class/module context
5. **Validation** — Ensure source code is valid, symbol matches
6. **Normalization** — Consistent formatting, encoding (UTF-8)

**Tools:**
- AST parsing: `ast` (Python), `tree-sitter` (multi-language)
- Code formatting: `black` for normalization
- Duplicate detection: Similarity hashing

---

### 4.3 Semantic Labeling

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

---

### 4.4 Query Generation

**Developer-style query generation:**

1. **Read code unit** and semantic label
2. **Generate 2–3 realistic queries** — How would a developer search for this?
   - "How do we retry failed payments?"
   - "Where is exponential backoff implemented?"
   - "Find the payment processor retry logic"
3. **Label query style** — Natural, technical, verbose, concise
4. **Label intent** — Finding error handling, optimization, etc.
5. **Assign difficulty** — Based on query specificity and semantic distance

**Query generation guidelines:**
- Use question or imperative form
- Reference actual concepts from the code (not generic)
- Vary query style (not all identical patterns)
- Ensure query is unambiguous for humans

**Validation:**
- A developer should find the relevant code unit in top-10 results
- Avoid queries that could match multiple unrelated units

---

### 4.5 Ground-Truth Labeling

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
  "total_code_units": 450,
  "train_cases": 225,
  "validation_cases": 112,
  "test_cases": 113,
  "total_queries": 45,
  "schema_version": "1.0.0",
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

1. **Initial dataset freeze** (end of Phase 3) → `dataset-v1.0.0`
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

## 10. Phase 3 Implementation Checklist

- [ ] Select 3–5 source repositories
- [ ] Extract code units (200+ functions)
- [ ] Create semantic labels (human-reviewed)
- [ ] Generate developer-style queries
- [ ] Assign ground-truth relevance
- [ ] Create train/validation/test splits (repository-disjoint)
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

