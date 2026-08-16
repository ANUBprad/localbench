# LocalBench

Offline-first, privacy-first local LLM benchmarking and model-selection platform.

## Features

- Run open-source LLMs through Ollama locally
- Standardized benchmark workloads
- Measure quality, performance, and resource usage
- Get model recommendations based on your hardware and constraints
- Study assistant with Q&A and quiz generation

## Installation

```bash
# Install from source
git clone https://github.com/localbench/localbench.git
cd localbench
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
# Show help
localbench --help

# Show version
localbench --version

# List available models (requires Ollama running)
localbench models

# Ask a model a question
localbench ask "What is paging in operating systems?"
```

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=localbench

# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/
```

## Architecture

LocalBench follows a layered architecture:

1. **CLI Layer** (Typer + Rich) - Command routing and output formatting
2. **Application Layer** - Study assistant, benchmark runner
3. **Generation Layer** - Structured validation, retry engine
4. **Runtime Layer** - Model abstraction (LocalModel protocol)
5. **Ollama Adapter** - HTTP client, model discovery

## Engineering Rules

- **Offline-first**: No cloud APIs, no fallback to cloud LLMs
- **Measurement before claims**: Never claim one model is "best" without context
- **Raw data first**: Immutable JSONL artifacts; summaries derived from raw data
- **Reproducibility**: Every run records enough metadata to reproduce
- **Model abstraction**: Only Ollama adapter knows about Ollama; everything else uses LocalModel protocol
- **Typed contracts**: Pydantic v2 with strict validation at all boundaries

## License

MIT License - see [LICENSE](LICENSE) for details.
