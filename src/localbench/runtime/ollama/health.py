"""Ollama health checking."""

from __future__ import annotations

import httpx

OLLAMA_DEFAULT_URL = "http://localhost:11434"


def check_ollama_health(base_url: str = OLLAMA_DEFAULT_URL) -> bool:
    """Check if Ollama is reachable at the given endpoint.

    Returns True if Ollama responds, False otherwise.
    Does not raise exceptions — failures are mapped to False.
    """
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/", timeout=2.0)
        response.raise_for_status()
        return True
    except Exception:
        return False
