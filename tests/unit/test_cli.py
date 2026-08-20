"""Tests for CLI commands."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from localbench.cli import app
from localbench.errors import OllamaUnavailableError

runner = CliRunner()


class TestHelp:
    def test_help_shows_usage(self):
        """--help shows usage information."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "localbench" in result.output.lower()

    def test_system_help(self):
        """system --help shows command description."""
        result = runner.invoke(app, ["system", "--help"])
        assert result.exit_code == 0
        assert "system" in result.output.lower() or \
            "hardware" in result.output.lower()

    def test_models_help(self):
        """models --help shows command description."""
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()

    def test_ask_help(self):
        """ask --help shows command description."""
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "prompt" in result.output.lower()


class TestVersion:
    def test_version_flag(self):
        """--version shows version string."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestSystemCommand:
    @patch("localbench.cli.get_system_profile")
    def test_system_shows_hardware_info(self, mock_profile):
        """system command displays hardware information."""
        from localbench.profiling.hardware import (
            CPUInfo,
            MemoryInfo,
            SystemProfile,
        )

        mock_profile.return_value = SystemProfile(
            os="Linux",
            os_version="5.15.0",
            architecture="x86_64",
            python_version="3.10.12",
            cpu=CPUInfo(
                model="Intel Core",
                physical_cores=8,
                logical_cores=16,
                frequency_ghz=3.6,
            ),
            memory=MemoryInfo(
                total_bytes=32 * 1024**3,
                available_bytes=16 * 1024**3,
            ),
            gpus=[],
        )
        result = runner.invoke(app, ["system"])
        assert result.exit_code == 0
        assert "Linux" in result.output
        assert "x86_64" in result.output
        assert "32.0 GB" in result.output


class TestModelsCommand:
    @patch("localbench.cli.OllamaAdapter")
    def test_models_lists_available_models(self, mock_adapter_cls):
        """models command lists discovered models."""
        from localbench.runtime.model import ModelInfo

        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter
        mock_adapter.discover_models.return_value = [
            ModelInfo(name="phi-3-mini", size_bytes=2048000000),
            ModelInfo(name="mistral-7b", size_bytes=3800000000),
        ]

        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert "phi-3-mini" in result.output
        assert "mistral-7b" in result.output

    @patch("localbench.cli.OllamaAdapter")
    def test_models_handles_no_models(self, mock_adapter_cls):
        """models command handles empty model list."""
        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter
        mock_adapter.discover_models.return_value = []

        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert "No models found" in result.output or \
            "no models" in result.output.lower()

    @patch("localbench.cli.OllamaAdapter")
    def test_models_handles_ollama_unavailable(self, mock_adapter_cls):
        """models fails cleanly when Ollama is not running."""
        mock_adapter_cls.side_effect = OllamaUnavailableError()

        result = runner.invoke(app, ["models"])
        assert result.exit_code == 1
        assert "Ollama" in result.output or \
            "ollama" in result.output.lower()


class TestAskCommand:
    @patch("localbench.cli.OllamaAdapter")
    def test_ask_generates_response(self, mock_adapter_cls):
        """ask command generates and displays a response."""
        from localbench.runtime.model import GenerationResult

        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter
        mock_adapter.generate.return_value = GenerationResult(
            model="phi-3-mini",
            text="Binary search is an algorithm.",
            duration_ms=500.0,
        )

        result = runner.invoke(
            app,
            ["ask", "--model", "phi-3-mini", "What is binary search?"],
        )
        assert result.exit_code == 0
        assert "Binary search" in result.output

    @patch("localbench.cli.OllamaAdapter")
    def test_ask_handles_ollama_unavailable(self, mock_adapter_cls):
        """ask fails cleanly when Ollama is unavailable."""
        mock_adapter_cls.side_effect = OllamaUnavailableError()

        result = runner.invoke(
            app, ["ask", "--model", "phi-3-mini", "test"]
        )
        assert result.exit_code == 1

    def test_ask_requires_model(self):
        """ask command fails without --model option."""
        result = runner.invoke(app, ["ask", "test prompt"])
        assert result.exit_code != 0

    def test_ask_requires_prompt(self):
        """ask command fails without prompt argument."""
        result = runner.invoke(app, ["ask", "--model", "phi-3-mini"])
        assert result.exit_code != 0
