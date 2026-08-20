"""System hardware and runtime profiling."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

import psutil


@dataclass
class CPUInfo:
    """CPU hardware details."""

    model: str
    physical_cores: int
    logical_cores: int
    frequency_ghz: float | None = None


@dataclass
class MemoryInfo:
    """System memory details."""

    total_bytes: int
    available_bytes: int

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)

    @property
    def available_gb(self) -> float:
        return self.available_bytes / (1024**3)


@dataclass
class GPUInfo:
    """GPU hardware details (if detectable)."""

    name: str
    vram_bytes: int | None = None


@dataclass
class SystemProfile:
    """Complete system hardware and runtime profile."""

    os: str
    os_version: str
    architecture: str
    python_version: str
    cpu: CPUInfo
    memory: MemoryInfo
    gpus: list[GPUInfo] = field(default_factory=list)


def _detect_gpus() -> list[GPUInfo]:
    """Detect GPUs via Ollama's /api/ps endpoint.

    Falls back to empty list if Ollama is not running.
    This is a best-effort detection — GPU info is optional.
    """
    import httpx

    try:
        response = httpx.get("http://localhost:11434/api/ps", timeout=2.0)
        response.raise_for_status()
        data = response.json()
        gpus: list[GPUInfo] = []
        for gpu in data.get("gpus", []):
            gpus.append(
                GPUInfo(
                    name=gpu.get("name", "Unknown GPU"),
                    vram_bytes=gpu.get("vram"),
                )
            )
        return gpus
    except Exception:
        return []


def get_system_profile() -> SystemProfile:
    """Collect hardware and runtime information about the local system.

    Returns a SystemProfile with CPU, memory, GPU, and platform details.
    GPU detection is best-effort and requires Ollama to be running.
    """
    uname = platform.uname()
    cpu_freq = psutil.cpu_freq()

    return SystemProfile(
        os=uname.system,
        os_version=uname.release,
        architecture=uname.machine,
        python_version=platform.python_version(),
        cpu=CPUInfo(
            model=uname.processor or "Unknown",
            physical_cores=psutil.cpu_count(logical=False) or 0,
            logical_cores=psutil.cpu_count(logical=True) or 0,
            frequency_ghz=round(cpu_freq.max / 1000, 2) if cpu_freq else None,
        ),
        memory=MemoryInfo(
            total_bytes=psutil.virtual_memory().total,
            available_bytes=psutil.virtual_memory().available,
        ),
        gpus=_detect_gpus(),
    )
