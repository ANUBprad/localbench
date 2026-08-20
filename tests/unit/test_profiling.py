"""Tests for system profiling module."""

from unittest.mock import patch

from localbench.profiling.hardware import (
    CPUInfo,
    GPUInfo,
    MemoryInfo,
    SystemProfile,
    get_system_profile,
)


def test_get_system_profile_returns_valid_data():
    """System profile returns a complete SystemProfile object."""
    profile = get_system_profile()
    assert isinstance(profile, SystemProfile)
    assert isinstance(profile.os, str)
    assert isinstance(profile.architecture, str)
    assert isinstance(profile.python_version, str)


def test_system_profile_required_fields_present():
    """All required fields are populated in the system profile."""
    profile = get_system_profile()
    assert profile.os
    assert profile.architecture
    assert profile.python_version
    assert profile.cpu.physical_cores > 0
    assert profile.cpu.logical_cores > 0
    assert profile.memory.total_bytes > 0
    assert profile.memory.available_bytes > 0


def test_cpu_info_fields():
    """CPUInfo dataclass has correct types."""
    cpu = CPUInfo(
        model="Test CPU",
        physical_cores=4,
        logical_cores=8,
        frequency_ghz=3.5,
    )
    assert cpu.model == "Test CPU"
    assert cpu.physical_cores == 4
    assert cpu.logical_cores == 8
    assert cpu.frequency_ghz == 3.5


def test_memory_info_properties():
    """MemoryInfo converts bytes to GB correctly."""
    mem = MemoryInfo(
        total_bytes=16 * 1024**3,
        available_bytes=8 * 1024**3,
    )
    assert mem.total_gb == 16.0
    assert mem.available_gb == 8.0


def test_gpu_info_fields():
    """GPUInfo dataclass has correct types."""
    gpu = GPUInfo(name="Test GPU", vram_bytes=8 * 1024**3)
    assert gpu.name == "Test GPU"
    assert gpu.vram_bytes == 8 * 1024**3


def test_system_profile_gpus_default_empty():
    """SystemProfile defaults to empty GPU list."""
    profile = SystemProfile(
        os="Linux",
        os_version="5.0",
        architecture="x86_64",
        python_version="3.10.0",
        cpu=CPUInfo(
            model="CPU", physical_cores=4, logical_cores=8
        ),
        memory=MemoryInfo(
            total_bytes=8 * 1024**3,
            available_bytes=4 * 1024**3,
        ),
    )
    assert profile.gpus == []


@patch("localbench.profiling.hardware._detect_gpus")
def test_get_system_profile_uses_psutil(mock_detect):
    """get_system_profile calls psutil for CPU and memory info."""
    mock_detect.return_value = []
    profile = get_system_profile()
    assert profile.cpu.logical_cores > 0
    assert profile.memory.total_bytes > 0
    mock_detect.assert_called_once()


@patch("httpx.get", side_effect=Exception("Ollama unavailable"))
def test_system_profile_works_without_ollama(mock_httpx):
    """System profiler returns CPU/RAM/OS even when Ollama is not running."""
    profile = get_system_profile()
    assert profile.os
    assert profile.cpu.logical_cores > 0
    assert profile.memory.total_bytes > 0
    assert profile.gpus == []
