"""Command-line interface for LocalBench."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from localbench import __version__
from localbench.errors import LocalBenchError, OllamaUnavailableError
from localbench.profiling.hardware import get_system_profile
from localbench.runtime.ollama.adapter import OllamaAdapter
from localbench.runtime.registry import ModelRegistry

app = typer.Typer(
    name="localbench",
    help="Offline-first, privacy-first local LLM benchmarking.",
    add_completion=False,
)
console = Console()

OLLAMA_BASE_URL = "http://localhost:11434"


def version_callback(value: bool) -> None:
    if value:
        console.print(f"localbench version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    )
) -> None:
    """LocalBench CLI: benchmarking and model-selection platform."""


@app.command()
def system() -> None:
    """Display local hardware and runtime information."""
    profile = get_system_profile()

    table = Table(title="System Information", show_lines=True)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("OS", f"{profile.os} {profile.os_version}")
    table.add_row("Architecture", profile.architecture)
    table.add_row("Python", profile.python_version)
    table.add_row(
        "CPU",
        f"{profile.cpu.model} ({profile.cpu.physical_cores} cores, "
        f"{profile.cpu.logical_cores} logical)",
    )
    if profile.cpu.frequency_ghz:
        table.add_row("CPU Frequency", f"{profile.cpu.frequency_ghz} GHz")
    table.add_row(
        "Memory",
        f"{profile.memory.total_gb:.1f} GB total, "
        f"{profile.memory.available_gb:.1f} GB available",
    )
    if profile.gpus:
        for gpu in profile.gpus:
            vram = f" ({gpu.vram_bytes / (1024**3):.1f} GB)" if gpu.vram_bytes else ""
            table.add_row("GPU", f"{gpu.name}{vram}")
    else:
        table.add_row("GPU", "Not detected")

    console.print(table)


@app.command()
def models() -> None:
    """List available Ollama models."""
    try:
        adapter = OllamaAdapter(base_url=OLLAMA_BASE_URL)
        discovered = adapter.discover_models()
        adapter.close()
    except OllamaUnavailableError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if not discovered:
        msg = "No models found. Pull a model: ollama pull <model>"
        console.print(f"[yellow]{msg}[/yellow]")
        return

    registry = ModelRegistry()
    for model in discovered:
        registry.add(model)

    table = Table(title="Available Models")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")

    for model in registry.list_models():
        size = f"{model.size_gb:.1f} GB" if model.size_bytes else "Unknown"
        table.add_row(model.name, size)

    console.print(table)


@app.command()
def ask(
    model: str = typer.Option(
        ..., "--model", "-m", help="Model to use."
    ),
    prompt: str = typer.Argument(
        ..., help="The prompt to send."
    ),
) -> None:
    """Send a prompt to a local model and display the response."""
    try:
        adapter = OllamaAdapter(base_url=OLLAMA_BASE_URL)
    except OllamaUnavailableError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        from localbench.runtime.model import GenerationRequest

        request = GenerationRequest(prompt=prompt, model=model)
        result = adapter.generate(request)
        console.print(result.text)
    except LocalBenchError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        adapter.close()


review_app = typer.Typer(help="Human review workflow for the selected 45 queries.")
app.add_typer(review_app, name="review")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REVIEW_ARTIFACT_PATH = REPO_ROOT / "dataset" / "queries" / "review_artifact.json"
SELECTION_PATH = REPO_ROOT / "dataset" / "queries" / "final_45_selection.json"
CANDIDATES_PATH = REPO_ROOT / "dataset" / "queries" / "candidates.jsonl"
TEST_SPLIT_PATH = REPO_ROOT / "dataset" / "splits" / "test.jsonl"
TRAIN_SPLIT_PATH = REPO_ROOT / "dataset" / "splits" / "train.jsonl"
VALIDATION_SPLIT_PATH = REPO_ROOT / "dataset" / "splits" / "validation.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@review_app.command("build")
def review_build() -> None:
    """Build the benchmark-blind review artifact for the selected 45."""
    from localbench.workloads.code_retrieval.review import (
        ReviewArtifactError,
        build_review_artifact,
        validate_review_artifact,
    )

    console.print("Loading selection record …")
    with open(SELECTION_PATH, encoding="utf-8") as f:
        selection_record = json.load(f)

    console.print("Loading candidates …")
    all_candidates = _load_jsonl(CANDIDATES_PATH)
    candidates_by_id = {c["candidate_id"]: c for c in all_candidates}

    console.print("Loading test CodeUnits …")
    test_units = _load_jsonl(TEST_SPLIT_PATH)
    units_by_id = {u["id"]: u for u in test_units}
    test_code_unit_ids = {u["id"] for u in test_units}

    console.print("Loading train/val CodeUnits for leakage check …")
    train_units = _load_jsonl(TRAIN_SPLIT_PATH) if TRAIN_SPLIT_PATH.exists() else []
    validation_units = (
        _load_jsonl(VALIDATION_SPLIT_PATH) if VALIDATION_SPLIT_PATH.exists() else []
    )
    train_ids = {u["id"] for u in train_units}
    validation_ids = {u["id"] for u in validation_units}

    console.print("Building review artifact …")
    try:
        artifact = build_review_artifact(
            selection_record=selection_record,
            candidates_by_id=candidates_by_id,
            units_by_id=units_by_id,
            test_code_unit_ids=test_code_unit_ids,
            train_code_unit_ids=train_ids,
            validation_code_unit_ids=validation_ids,
        )
    except ReviewArtifactError as exc:
        console.print(f"[red]ERROR:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print("Validating review artifact …")
    errors = validate_review_artifact(artifact)
    if errors:
        for err in errors:
            console.print(f"  [red]VIOLATION:[/red] {err}")
        raise typer.Exit(1)

    REVIEW_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Review artifact written to {REVIEW_ARTIFACT_PATH}[/green]")
    console.print(f"  Items: {len(artifact['items'])}")
    all_pending = all(
        item["review"]["state"] == "pending" for item in artifact["items"]
    )
    console.print(f"  All pending: {all_pending}")


@review_app.command("status")
def review_status() -> None:
    """Show review progress for the selected 45."""
    from localbench.workloads.code_retrieval.review import review_progress

    if not REVIEW_ARTIFACT_PATH.exists():
        msg = "No review artifact found. Run 'localbench review build' first."
        console.print(f"[yellow]{msg}[/yellow]")
        raise typer.Exit(1)

    with open(REVIEW_ARTIFACT_PATH, encoding="utf-8") as f:
        artifact = json.load(f)

    progress = review_progress(artifact)
    table = Table(title="Review Progress")
    table.add_column("State", style="cyan")
    table.add_column("Count", justify="right")
    for state in ("pending", "accepted", "rejected"):
        table.add_row(state, str(progress[state]))
    table.add_row("total", str(progress["total"]))
    console.print(table)


@review_app.command("show")
def review_show(
    position: int = typer.Argument(..., help="Item position (1-45)."),
) -> None:
    """Display a single review item for human inspection."""
    if not REVIEW_ARTIFACT_PATH.exists():
        msg = "No review artifact found. Run 'localbench review build' first."
        console.print(f"[yellow]{msg}[/yellow]")
        raise typer.Exit(1)

    with open(REVIEW_ARTIFACT_PATH, encoding="utf-8") as f:
        artifact = json.load(f)

    if position < 1 or position > len(artifact["items"]):
        console.print(f"[red]Position must be 1–{len(artifact['items'])}[/red]")
        raise typer.Exit(1)

    item = artifact["items"][position - 1]
    target = item["target"]
    review = item["review"]

    console.print(f"[bold]Item {item['position']} of {len(artifact['items'])}[/bold]")
    console.print(f"  Candidate ID:  {item['candidate_id']}")
    console.print(f"  Code Unit ID:  {item['code_unit_id']}")
    console.print(f"  Query Style:   {item['query_style']}")
    console.print(f"  Query Intent:  {item['query_intent']}")
    console.print()
    console.print("[bold cyan]Query:[/bold cyan]")
    console.print(f"  {item['query']}")
    console.print()
    console.print("[bold cyan]Target CodeUnit:[/bold cyan]")
    console.print(f"  Repository:    {target['repository']}")
    console.print(f"  File:          {target['file_path']}")
    console.print(f"  Symbol:        {target['symbol']}")
    console.print(f"  Symbol Type:   {target['symbol_type']}")
    if target.get("docstring"):
        console.print(f"  Docstring:     {target['docstring']}")
    console.print()
    console.print("[bold cyan]Source Code:[/bold cyan]")
    console.print(target["source_code"])
    console.print()
    av = item["automated_validation"]
    console.print(
        f"[bold cyan]Automated Validation:[/bold cyan] "
        f"schema={'PASS' if av['validation_passed'] else 'FAIL'}, "
        f"leakage={'PASS' if av['leakage_passed'] else 'FAIL'}"
    )
    console.print()
    state_color = {"pending": "yellow", "accepted": "green", "rejected": "red"}
    color = state_color.get(review["state"], "white")
    state_str = review["state"]
    console.print(f"[bold]Review State:[/bold] [{color}]{state_str}[/{color}]")
    if review.get("notes"):
        console.print(f"  Notes: {review['notes']}")


if __name__ == "__main__":
    app()
