"""Command-line interface for LocalBench."""

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


if __name__ == "__main__":
    app()
