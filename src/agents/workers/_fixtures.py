"""Shared fixture-loading helpers for the MCP worker servers.

All worker tools are backed by deterministic JSON fixtures in
``data/fixtures/`` so the demo is reproducible without live API keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


# Project root is two levels up from src/agents/workers/
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_DIR = _REPO_ROOT / "data" / "fixtures"


def load_fixture(name: str) -> Dict[str, Any]:
    """Load ``data/fixtures/<name>.json`` and return the parsed dict."""
    path = _FIXTURE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalise_ticker(ticker: str) -> str:
    """Normalise a user-supplied ticker string to the fixture key convention."""
    return (ticker or "").strip().upper().replace(".NS", "").replace("$", "")


def lookup(data: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    """Return the entry for ``ticker`` or raise ``KeyError`` with a helpful message."""
    key = normalise_ticker(ticker)
    if key not in data:
        available = sorted(k for k in data.keys() if not k.startswith("_"))
        raise KeyError(
            f"Ticker '{ticker}' not found in fixtures. Available: {available}"
        )
    return data[key]
