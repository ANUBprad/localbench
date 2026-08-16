# LocalBench --- Design Direction v2.0

**Status:** v2.0 (Advanced)\
**Revision Date:** August 2026

---

## Design Philosophy

LocalBench should feel like **an engineering tool, not a generic chatbot.**

**Primary Qualities (in priority order):**
1. **Technical** — Show measurements, not marketing.
2. **Clear** — Readable, scannable, no unnecessary decoration.
3. **Trustworthy** — Transparent about methodology and limitations.
4. **Data-oriented** — Tables, metrics, reproducible artifacts.
5. **Offline/private** — Visible assurance of local execution.
6. **Minimal** — Remove decoration that obscures information.
7. **Explainable** — Every recommendation has reasoning.

---

## 1. Design Principles

### Principle 1: Data Over Decoration

**DO:**
- Show tables with clear column headers.
- Use monospace for code and technical output.
- Preserve numerical precision (e.g., 1.234s, not "~1s").
- Show raw metrics alongside summaries.

**DON'T:**
- Use colored badges or emojis as primary information.
- Hide numbers behind visual bars (show the number).
- Use decorative borders or ASCII art.
- Abbreviate precise values ("~90%" vs "87.3%").

**Example (GOOD):**
```
Benchmark Results

Model       Quality  Latency  RAM
────────────────────────────────
Qwen 7B     87.3%    1.24s    6.2GB
Mistral 7B  72.1%    0.94s    5.8GB
Llama 13B   89.2%    2.11s   10.4GB
```

**Example (BAD):**
```
Benchmark Results ✨

🏆 Qwen: ████████ Quality | ⚡ Fast
🥈 Mistral: ██████ Quality | ⚡⚡ Very fast
🥉 Llama: ██████████ Quality | 🐢 Slow
```

### Principle 2: Explain Every Recommendation

**Every recommendation must include:**
- Why selected (what constraints passed?).
- What score/metrics mattered.
- Why others were rejected (specific violated constraints).
- Trade-offs (speed vs. quality, etc.).

**Example (GOOD):**
```
RECOMMENDATION

Qwen 7B

Why selected:
  ✓ Quality 87.3% >= 85% (meets minimum)
  ✓ RAM 6.2GB <= 8GB (within budget)
  ✓ Latency 1.24s <= 3s (acceptable)
  ✓ Highest quality among qualifying models

Why rejected:
  Mistral 7B: Quality 72.1% < 85% (below minimum)
  Llama 13B: RAM 10.4GB > 8GB (exceeds budget)
```

**Example (BAD):**
```
Recommendation: Qwen 7B

Reasoning: It's the best model for your needs.
```

### Principle 3: Never Hide Failed Cases

**Transparency Rule:** If a case failed or returned an error, show it.

**Examples:**
- ✓ "Benchmark completed. 28 of 30 cases passed. 2 cases had errors (see report)."
- ✓ "Case 'math-003': Model returned empty output. Error recorded in results."
- ✗ "Benchmark complete: Quality 87%." (Hides failures.)

### Principle 4: Separate Experimental Data from Presentation

**Raw Data (IMMUTABLE):**
- Every generated response.
- Every evaluation detail.
- Every resource sample.
- Stored in JSONL files.

**Derived Presentation (REGENERABLE):**
- Summary tables.
- Charts.
- Reports.
- These can be regenerated from raw data.

**CLI Design Implication:**
- Show summary, link to artifacts.
- `See full results in ./results/2026-08-14T143000Z/`

### Principle 5: Make Privacy Claims Factual

**DO:**
- "Inference runs locally."
- "No cloud LLM is required."
- "Documents remain on this machine."

**DON'T:**
- "Completely secure." (Not verifiable.)
- "100% private." (Absolute claims are false.)
- "Military-grade encryption." (Unnecessary and unverified.)

**Example (GOOD):**
```
Privacy Status

Inference:  LOCAL (Ollama on localhost)
Storage:    LOCAL (results in ./results/)
External:   NONE (no cloud APIs used)
```

---

## 2. Terminal UI Design

### Typography & Spacing

**Font:** Monospace (system default).

**Line Spacing:** Single-spaced in tables, blank lines between sections.

**Text Width:** Aim for 80 character lines (readable on mobile/small terminals).

### Table Design

**Good table:**
```
Model       Quality  Latency  RAM     CPU
────────────────────────────────────────
Qwen 7B      87.3%   1.24s    6.2GB  12%
Mistral 7B   72.1%   0.94s    5.8GB   8%
Llama 13B    89.2%   2.11s   10.4GB  18%

Notes: CPU % is average during inference.
```

**Bad table:**
```
┌─────────────┬──────────┐
│ Model       │ Quality  │
├─────────────┼──────────┤
│ Qwen 7B     │ 87.3%    │
│ Mistral 7B  │ 72.1%    │
│ Llama 13B   │ 89.2%    │
└─────────────┴──────────┘

[Too much decoration, hard to copy/paste]
```

### Error Messages

**Format:**
```
ERROR [CODE]: Description

Reason:
  Why this happened.

Action:
  What to do next.

Example:
  command here
```

**Example:**
```
ERROR [RUNTIME_UNAVAILABLE]: Ollama is not running

Reason:
  The LocalBench CLI couldn't connect to Ollama on localhost:11434.

Action:
  Start Ollama with:
    ollama serve

Then run your command again.

Docs:
  https://ollama.ai/docs
```

### Status Messages

**Use simple, scannable status:**
```
[✓] System profiled
[✓] Models discovered (3 available)
[✓] Dataset loaded (30 cases)
[⏳] Running benchmark... (2/3 models, 15/30 cases)
```

### Charts (matplotlib)

**Use matplotlib for simple charts:**

- Bar charts (model comparison).
- Line charts (latency trends if running multiple times).
- Avoid pie charts (hard to read exact percentages).
- Always include numeric labels on bars.

**Example (Good):**
```
Latency Comparison (ms)

1000 ├─────────────────────────────
     │
 800 ├────────┐
     │        │
 600 ├────────┼────────┐
     │        │        │
 400 ├────────┼────────┼────────┐
     │        │        │        │
 200 ├────────┼────────┼────────┼────────
     │  946   │  1240  │  2110  │
   0 └────────┴────────┴────────┴────────
        Mistral  Qwen   Llama

Note: Each bar shows average latency across all cases.
```

---

## 3. CLI Command Design

### Command Structure

```bash
localbench <command> [OPTIONS]
```

### Commands (MVP)

| Command | Purpose | Output |
|---------|---------|--------|
| `models` | List available models | Table of models with metadata |
| `ask` | Ask a model | Generated text + timing |
| `benchmark` | Run full benchmark | Summary + results path |
| `compare` | Compare latest results | Comparison table |
| `recommend` | Get model recommendation | Recommendation with explanation |
| `study` | Study assistant | Interactive menu |
| `system` | Show system info | System metadata table |

### Flag Conventions

**Global flags:**
- `--verbose` or `-v` — More detailed output.
- `--quiet` or `-q` — Suppress non-essential output.
- `--json` — Output as JSON (for scripting).

**Benchmark flags:**
- `--models <name1,name2>` — Which models to benchmark.
- `--dataset <version>` — Which dataset version.
- `--output <dir>` — Where to save results.

**Recommendation flags:**
- `--min-quality <0.0-1.0>` — Minimum quality threshold.
- `--max-ram <GB>` — Maximum RAM allowed.
- `--max-latency <ms>` — Maximum latency.

---

## 4. Information Hierarchy

### CLI Output Levels (MVP to detailed)

**Level 1: Summary** (default for `benchmark`)
```
Benchmark complete!

Results saved to: ./results/2026-08-14T143000Z/

Quick summary:
  Models benchmarked: 2
  Cases executed: 30
  Successful: 28
  Errors: 2

See detailed results in the directory above.
```

**Level 2: Table** (with `compare` or `--verbose`)
```
[Full comparison table with all metrics]
```

**Level 3: Raw Data** (inspect artifacts)
```
./results/2026-08-14T143000Z/
├── case_results.jsonl    [One line per case result]
├── raw_outputs.jsonl     [Complete model outputs]
├── summary.json          [Aggregated metrics]
└── report.md             [Human-readable report]
```

---

## 5. Report Design (Markdown)

### Report Structure

```markdown
# LocalBench Benchmark Report

**Run ID:** 2026-08-14T143000Z
**Date:** 2026-08-14 14:30:00 UTC
**Software Version:** 0.1.0
**Benchmark Version:** 0.1.0
**Dataset Version:** 1.0.0

## System Information

| Property | Value |
|----------|-------|
| OS | macOS |
| Architecture | arm64 |
| CPU | Apple M1 (8 cores) |
| RAM | 16 GB |
| Ollama | v0.2.0 |

## Results Summary

| Model | Quality | Latency | RAM | Throughput |
|-------|---------|---------|-----|------------|
| Qwen 7B | 87.3% | 1.24s | 6.2GB | 24.3 tok/s |
| Mistral 7B | 72.1% | 0.94s | 5.8GB | 32.1 tok/s |
| Llama 13B | 89.2% | 2.11s | 10.4GB | 21.8 tok/s |

## Methodology

### Evaluation Strategy

- **Conceptual questions** (10 cases): Keyword coverage.
- **Math problems** (5 cases): Numeric tolerance ±5%.
- **Code questions** (10 cases): Exact output match.
- **Q&A** (5 cases): Semantic similarity (judge-scored).

### Measurement Methodology

- **Quality:** Score per evaluation strategy.
- **Latency:** Average time per inference (ms).
- **RAM:** Peak process RSS during inference.
- **Throughput:** Tokens/second during generation.

### Warm-up

Each model received 3 warm-up requests before measurement.

### Constraints

Only one model per machine per benchmark run.
Hardware: Single CPU, no GPU.

## Limitations

1. **Sample Size:** 30 cases is small. Results may not generalize to unseen tasks.
2. **Hardware Specificity:** Results are specific to this machine.
3. **Model Versions:** Exact model versions recorded in metadata.
4. **Judge Scoring:** 5 cases use LLM-as-judge (Qwen 7B). Results may have bias.

## Detailed Results

See `case_results.jsonl` for complete per-case data.

## Artifacts

- `raw_outputs.jsonl` — Every generated response.
- `case_results.jsonl` — Scored results.
- `summary.json` — Aggregated metrics.
```

---

## 6. Accessibility

**Terminal Output Accessibility:**

- **Color:** Use color only for emphasis, not information (colorblind-friendly).
- **No emoji:** Use text (✓, ✗, ⏳) but ensure emoji fallbacks.
- **Clear contrast:** Readable on light and dark backgrounds.
- **Table alignment:** Use monospace; align numbers right.
- **Text sizing:** Readable at 80-column terminal width.
- **Keyboard:** All navigation via keyboard (no mouse required).

---

## 7. Future UI Directions (P2, Not MVP)

### Web Dashboard (Future)

If building a web interface, follow these principles:

**Dashboard Page:**
- System profile and available models.
- Latest benchmark summary.
- Recommended model (from latest run).

**Benchmark Page:**
- Comparison table (sortable columns).
- Latency and resource charts (interactive).
- Quality breakdown per category.
- Structured-output reliability metrics.

**Study Page:**
- Document selector.
- Q&A conversation interface.
- Quiz mode.
- Study progress tracking.

### Terminal UI Enhancements (Future)

- Interactive table viewer (`less`-like pagination).
- Model selection menu.
- Constraint input UI.
- Real-time benchmark progress.

---

## 8. Design Consistency Rules

### Naming Conventions

- **Commands:** All lowercase, hyphenated (e.g., `localbench run-benchmark`).
- **Flags:** Long form preferred (e.g., `--models`, not `-m`).
- **Output headers:** Title case (e.g., "Benchmark Results").
- **Metrics:** Consistent units (e.g., latency always in ms, RAM in GB).

### Color Usage (if used)

| Use | Color | Hex | Reason |
|-----|-------|-----|--------|
| Success | Green | #00AA00 | Standard |
| Error | Red | #CC0000 | Standard |
| Warning | Yellow | #CCAA00 | Standard |
| Info | Blue | #0088FF | Standard |
| Neutral | Default | (none) | No color for data |

**Rule:** Use color sparingly. Ensure text is readable on both light and dark backgrounds.

### Typography

- **Code/Technical:** Monospace (model names, paths, commands).
- **Headings:** Title Case, slightly larger.
- **Body:** Sentence case, clear grammar.
- **Table headers:** Title Case, centered.
- **Table data:** Left-aligned (numbers right-aligned).

---

## 9. Error & Status Design

### Status Indicators

```
[✓] Success
[✗] Failure
[⏳] In progress
[⚠] Warning
[?] Unknown
[~] Approximate
```

### Error Categories & Messages

| Category | Tone | Example |
|----------|------|---------|
| Config error | Instruct | "Please specify..." |
| User error | Guide | "Did you mean...?" |
| System error | Clear | "Ollama is not running." |
| Data error | Specific | "Field 'score' is required." |

---

## 10. Documentation Design

### README Structure

1. **Quick Start** (3 commands to benchmark).
2. **Features** (what it does).
3. **Installation** (pip install).
4. **Example Workflow** (step-by-step).
5. **Results** (actual benchmark output).
6. **Methodology** (how benchmark works).
7. **Limitations** (be honest).
8. **Contributing** (if open-source).

### Inline Documentation

- **Docstrings:** Every module, class, function.
- **Comments:** Explain why, not what (code is obvious).
- **Examples:** Show usage in docstrings.

---

## 11. Brand & Identity

### Project Identity

- **Name:** LocalBench (camelCase, not local-bench).
- **Tagline:** "Offline LLM Evaluation Platform."
- **Tone:** Technical, honest, no hype.
- **Values:** Transparency, reproducibility, privacy.

### Logo (Future)

If creating a logo, emphasize:
- "Local" (not cloud).
- "Measurement" (benchmark/ruler).
- "Offline" (closed system/private).

---

## 12. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-08-13 | Initial design doc |
| 2.0 | 2026-08-14 | Advanced version: detailed CLI design, accessibility, report structure, future roadmap |

