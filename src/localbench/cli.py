"""Command-line interface for LocalBench."""

from typing import Optional

import typer
from rich.console import Console

from localbench import __version__

app = typer.Typer(
    name="localbench",
    help="Offline-first, privacy-first local LLM benchmarking and model-selection platform.",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"localbench version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    )
) -> None:
    """LocalBench: Offline-first, privacy-first local LLM benchmarking and model-selection platform."""


if __name__ == "__main__":
    app()
