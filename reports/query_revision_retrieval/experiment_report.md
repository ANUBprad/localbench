# LocalBench: Source-Informed Query Revision Retrieval Experiment

**Experiment date:** 2026-11-04  
**Status:** Completed ? exploratory evaluation  
**Cohort:** Batch30 source-informed revisions, nine paired queries

## 1. Objective

Evaluate whether revising nine weak natural-language code-retrieval queries to describe their corresponding source-code behavior improves retrieval of their intended CodeUnits.

Each original query is paired with its revised version. Both versions use the same relevance target and the same retrieval corpus.

## 2. Experimental setup

| Setting | Value |
|---|---|
| Corpus | `dataset/splits/test.jsonl` |
| Corpus CodeUnits | 2,493 |
| Query count | 9 original + 9 revised |
| Relevance | One intended CodeUnit per query |
| Document representation | Source code |
| Lexical method | BM25 |
| Dense method | Cosine similarity over local embeddings |
| Embedding model | `nomic-embed-text:latest` |
| Embedding service | Local Ollama (`http://127.0.0.1:11434`) |
| Retrieval cutoff | Top 10 |
| Metrics | Hit@1, Hit@5, Hit@10, MRR |

Both runs used the same corpus SHA256, embedding model, representation and top-10 cutoff. The query JSONL files differ only in query wording; the IDs and relevance targets are paired.

## 3. Aggregate results

### BM25

| Metric | Original | Revised | Change |
|---|---:|---:|---:|
| hit_at_1 | 0.1111 | 0.7778 | +0.6667 |
| hit_at_5 | 0.2222 | 0.8889 | +0.6667 |
| hit_at_10 | 0.2222 | 0.8889 | +0.6667 |
| mrr | 0.1481 | 0.8000 | +0.6519 |

### Dense retrieval

| Metric | Original | Revised | Change |
|---|---:|---:|---:|
| hit_at_1 | 0.1111 | 0.8889 | +0.7778 |
| hit_at_5 | 0.3333 | 0.8889 | +0.5556 |
| hit_at_10 | 0.3333 | 1.0000 | +0.6667 |
| mrr | 0.1852 | 0.9074 | +0.7222 |

Hit@k is the fraction of queries whose intended CodeUnit appears within the first k results. MRR is the mean reciprocal rank of the first relevant result; a query with no relevant result in the returned ranking contributes zero.

## 4. Per-query target ranks

A rank of 1 is the best possible result. An em dash means the intended CodeUnit did not appear in the top 10.

| ID | BM25 original | BM25 revised | BM25 change | Dense original | Dense revised | Dense change |
|---|---:|---:|---|---:|---:|---|
| R002 | ? | 1 | Entered top 10 | ? | 1 | Entered top 10 |
| R004 | ? | 5 | Entered top 10 | ? | 1 | Entered top 10 |
| R008 | ? | 1 | Entered top 10 | ? | 1 | Entered top 10 |
| R010 | 3 | 1 | Improved by 2 | ? | 1 | Entered top 10 |
| R016 | ? | ? | No change | 3 | 6 | Regressed by 3 |
| R017 | ? | 1 | Entered top 10 | ? | 1 | Entered top 10 |
| R018 | 1 | 1 | No change | 3 | 1 | Improved by 2 |
| R023 | ? | 1 | Entered top 10 | 1 | 1 | No change |
| R029 | ? | 1 | Entered top 10 | ? | 1 | Entered top 10 |

## 5. Query pairs

### R002

**Target CodeUnit:** `repo003_py_examples_downloader_py__download_1ed25415a5d8`

**Original:** What Python function performs an HTTP transfer to a local file and updates a progress display using a progress bar?

**Revised:** How does the download function create a progress task for each URL and submit the file-copy operation to a thread pool?

### R004

**Target CodeUnit:** `repo003_py_examples_fullscreen_py__make_syntax_cd3922e3bad5`

**Original:** What function renders progress displays in a terminal/console using the rich library's panel and progress features, and returns the rendered output as a string?

**Revised:** How does make_syntax create a Rich Syntax object containing Python code with line numbers enabled?

### R008

**Target CodeUnit:** `repo003_py_examples_columns_py__get_content_c74911b4d127`

**Original:** How does the Python function serialize a JSON object and transfer it over HTTP to a local file?

**Revised:** How does get_content extract a user's first and last name and country from a nested dictionary and format them as Rich markup?

### R010

**Target CodeUnit:** `repo003_py_examples_dynamic_progress_py__run_steps_c321bf145d3d`

**Original:** How does the code update the console with a progress bar and render panels while iterating over items?

**Revised:** How does run_steps update a step-level progress task during each timed step and advance the app-level progress task when that step finishes?

### R016

**Target CodeUnit:** `repo003_py_examples_rainbow_py__RainbowHighlighter_highlight_5e87a4ea5e3d`

**Original:** Iterates over a collection and applies formatting to each item using the Rich library, then randomly selects a color for highlighting.

**Revised:** How does RainbowHighlighter.highlight iterate over a Rich Text object's characters and apply a randomly selected color style to each character?

### R017

**Target CodeUnit:** `repo003_py_benchmarks_benchmarks_py__TextSuite_time_align_center_unicode_heavy_4989c5ff2937`

**Original:** How does the code format text for display and alignment in a tabulated console rendering context?

**Revised:** How does time_align_center_unicode_heavy create a Rich Text object from Unicode-heavy text and call its align method with center alignment and a width three times len_lorem_ipsum?

### R018

**Target CodeUnit:** `repo003_py_examples_downloader_py__copy_url_830ff7c85cc9`

**Original:** What Python function transfers data over HTTP to a local file using a progress display and concurrent execution?

**Revised:** How does copy_url read an HTTP response in chunks, write the bytes to a local file, and update a progress task as data is received?

### R023

**Target CodeUnit:** `repo003_py_examples_print_calendar_py__print_calendar_9455f21eed10`

**Original:** How does the Python function process a list of dates to render a table in the console, ensuring each row is aligned and formatted correctly?

**Revised:** How does print_calendar build and print a twelve-month calendar for a specified year using Rich tables and columns?

### R029

**Target CodeUnit:** `repo003_py_examples_fullscreen_py__make_sponsor_message_629dc2a53560`

**Original:** What Python function is designed to render a table of items to the console using the Rich library?

**Revised:** How does make_sponsor_message build a Rich grid of sponsor links, center the content vertically and horizontally, and return it inside a rounded Panel with a styled border?

## 6. Interpretation

On this nine-query cohort, revised wording substantially improved the measured retrieval metrics for both BM25 and dense retrieval. The revised queries produced more top-10 target matches and higher MRR.

The result is consistent across two retrieval approaches: BM25 benefits from more specific lexical descriptions, while dense retrieval also benefits from the more explicit behavioral descriptions.

R016 (`RainbowHighlighter.highlight`) remains a diagnostic case. Its intended target was absent from BM25 top 10 in both runs. Dense retrieval moved from rank 3 to rank 6 after revision.

## 7. Limitations

- The cohort contains only nine queries and is not a random sample.
- The queries were selected because they had been flagged for revision.
- Revised wording was informed by inspection of the target source code.
- The source-grounding review was assistant-assisted, not independent human adjudication.
- Each query has one intended CodeUnit relevance label. Other potentially relevant CodeUnits are not counted as hits.
- The experiment measures retrieval against this corpus and configuration; it does not establish generalization to unseen repositories or queries.
- The comparison does not establish statistical significance.

Consequently, these results are preliminary evidence for this cohort, not a general estimate of LocalBench retrieval quality.

## 8. Provenance and artifact hashes

| Artifact | SHA256 |
|---|---|
| Corpus | `37b27223c1701126a8f24559e33212f66a5e70ce85fcbcefd01aa116dbe6ff0e` |
| Original query JSONL | `ef15b1a650779e33fc39a753e399651afcfa0d3381311599bed6e24e9d398a27` |
| Revised query JSONL | `93828ec89cbdefce16f928456758861de1af3e03a1ae06617ba968c3cddc646b` |
| Original result JSON | `0d57dab7643b13812274bde722586f60ee7f65f161e200dc645c6f5e3a270e11` |
| Revised result JSON | `a81ef71d0494d2483218a88cf74aac9579be0f3d98133cdacb265af40113a8ea` |
| Manual review CSV | `3672a69083abbb22c85b81e00c1baa6863f23775586f27c1cb77437a5870c050` |

## 9. Next steps

1. Preserve this experiment and its input/output files unchanged.
2. Build an independent evaluation cohort using new CodeUnits.
3. Have relevance labels independently adjudicated before retrieval evaluation.
4. Freeze the query set and evaluation protocol before running retrieval.
5. Compare BM25 and dense retrieval using the same corpus and configuration.
6. Report per-query results and limitations alongside aggregate metrics.

---

Generated from the experiment JSONL and JSON artifacts by a validation script. Metrics and ranks were read from the saved benchmark outputs.
