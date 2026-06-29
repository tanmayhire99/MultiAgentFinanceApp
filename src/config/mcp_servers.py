"""Lifecycle management for the FinAI MCP worker servers.

Each worker is declared in :data:`MCP_SERVERS` and launched lazily on first
access via :func:`get_tools`. We keep a process-wide ``MultiServerMCPClient``
so subprocess reuse is automatic and we only pay the spawn cost once per
server lifetime.

Design goals:

* The FastAPI app must not block on subprocess startup at import time.
* Tools should be namespaced by server so the panel transcript clearly shows
  which worker a call went to (e.g. ``us_stock__get_fundamentals``).
* Failure to start one worker must not bring down the others.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


log = logging.getLogger("finai.mcp")


# Python interpreter used to launch worker subprocesses. In Docker this is
# always the same binary that runs the API; locally we fall back to sys.executable.
_PYTHON = os.environ.get("FINAI_MCP_PYTHON", sys.executable)

# Repo root so the `src` package is importable from the subprocess CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


MCP_SERVERS: Dict[str, Dict[str, Any]] = {
    "portfolio": {
        "command": _PYTHON,
        "args": ["-m", "src.mcp.portfolio_mcp"],
        "transport": "stdio",
        "cwd": _REPO_ROOT,
        "env": dict(os.environ),
    },
    "us_stock": {
        "command": _PYTHON,
        "args": ["-m", "src.mcp.us_stock_mcp"],
        "transport": "stdio",
        "cwd": _REPO_ROOT,
        "env": dict(os.environ),
    },
    "indian_stock": {
        "command": _PYTHON,
        "args": ["-m", "src.mcp.indian_stock_mcp"],
        "transport": "stdio",
        "cwd": _REPO_ROOT,
        "env": dict(os.environ),
    },
    "research": {
        "command": _PYTHON,
        "args": ["-m", "src.mcp.research_mcp"],
        "transport": "stdio",
        "cwd": _REPO_ROOT,
        "env": dict(os.environ),
    },
}


# ---------------------------------------------------------------------------
# Opt-in cross-project worker: the sibling `automated-trading` repo's READ-ONLY
# "quant" MCP server (backtesting). It runs in ITS OWN interpreter (Python 3.14
# venv) since it can't share FinAI's 3.12 env — hence a per-server ``command``.
# Enabled only when both env vars are set, so FinAI runs standalone by default
# (CI, tests). Exposes backtest/list-strategies only; no execution surface.
# ---------------------------------------------------------------------------
def quant_server_config(python: str, cwd: str) -> Dict[str, Any]:
    """Build the stdio MCP_SERVERS entry for the automated-trading quant server."""
    return {
        "command": python,
        "args": ["quant_mcp.py"],
        "transport": "stdio",
        "cwd": cwd,
        "env": dict(os.environ),
    }


_QUANT_PY = os.environ.get("QUANT_MCP_PYTHON", "").strip()
_QUANT_CWD = os.environ.get("QUANT_MCP_CWD", "").strip()
if _QUANT_PY and _QUANT_CWD:
    MCP_SERVERS["quant"] = quant_server_config(_QUANT_PY, _QUANT_CWD)
    log.info("quant MCP server enabled (cwd=%s)", _QUANT_CWD)


_client: Optional[MultiServerMCPClient] = None
_tools_cache: Optional[List[BaseTool]] = None
_init_lock = asyncio.Lock()


def _namespace_tools(server_name: str, tools: List[BaseTool]) -> List[BaseTool]:
    """Prefix tool names with ``<server>__`` so the origin is visible in traces.

    We only rename if the tool isn't already prefixed - some versions of
    ``langchain-mcp-adapters`` already namespace for us.
    """
    prefix = f"{server_name}__"
    renamed: List[BaseTool] = []
    for t in tools:
        if not t.name.startswith(prefix):
            t.name = f"{prefix}{t.name}"
        renamed.append(t)
    return renamed


async def _load_tools() -> List[BaseTool]:
    """Spawn every server listed in :data:`MCP_SERVERS` and collect their tools."""
    global _client
    _client = MultiServerMCPClient(MCP_SERVERS)

    all_tools: List[BaseTool] = []
    try:
        # Newer adapters expose a single ``get_tools()`` that fans out.
        tools = await _client.get_tools()
        # If the adapter didn't namespace, do it per tool using the server field
        # exposed on each tool's metadata. Fall back to tag-by-position if needed.
        for t in tools:
            all_tools.append(t)
    except Exception as e:  # pragma: no cover - defensive
        log.exception("MultiServerMCPClient.get_tools() failed: %s", e)
        raise

    # Best-effort namespacing: if tool names clash across servers, adapter
    # should already have renamed; this guarantees a visible prefix in logs.
    seen: Dict[str, int] = {}
    for t in all_tools:
        seen[t.name] = seen.get(t.name, 0) + 1
    if any(c > 1 for c in seen.values()):
        log.info("Applying fallback namespacing to MCP tools due to name clashes")
        # If we got here the adapter did not namespace - group by server via
        # sequential partitioning: we assume tools come back grouped per server
        # in MCP_SERVERS declaration order, which is true in the current adapter.
        all_tools = _partition_and_namespace(all_tools)
    return all_tools


def _partition_and_namespace(tools: List[BaseTool]) -> List[BaseTool]:
    """Fallback namespacing: partition tools by server declaration order."""
    result: List[BaseTool] = []
    if not tools:
        return result

    # Count tools per server module by running a dry import-less split:
    # infer partitioning from known per-server counts. Keep this in sync
    # with the ``@mcp.tool`` decorators in each worker module. If a tool
    # is added or removed, bump the corresponding count here.
    known_counts = {
        "portfolio": 6,      # list_supported_users, holdings, summary, sectors, risks, score
        "us_stock": 6,       # list, quote, fundamentals, growth, defensive, moat_signals
        "indian_stock": 9,   # us_stock 6 + warehouse: price_history, top_movers, sector_performance
        "research": 16,      # list, search_news, search_web, company_brief, catalysts,
                             # analyst_takes, search_historical_news, get_sec_filings,
                             # fetch_sec_document, extract_forward_claims, compare_claim_to_reality,
                             # get_indian_filings, fetch_indian_document, get_screener_snapshot,
                             # get_indian_concall_urls, get_indian_annual_reports
    }
    # Opt-in cross-project quant server (last in MCP_SERVERS when enabled).
    if "quant" in MCP_SERVERS:
        known_counts["quant"] = 2  # list_strategies, backtest_strategy
    expected = sum(known_counts.values())
    if len(tools) != expected:
        # Can't safely partition; just leave names as-is. Surface a loud
        # warning so future tool additions don't silently break the
        # namespacing contract.
        log.warning(
            "MCP tool count %d does not match expected %d; "
            "namespacing skipped (tool-name clashes may follow)",
            len(tools),
            expected,
        )
        return tools
    cursor = 0
    for server, count in known_counts.items():
        for t in tools[cursor : cursor + count]:
            t.name = f"{server}__{t.name}"
            result.append(t)
        cursor += count
    return result


async def get_tools() -> List[BaseTool]:
    """Return the shared, lazily-initialised list of MCP tools."""
    global _tools_cache
    if _tools_cache is not None:
        return _tools_cache
    async with _init_lock:
        if _tools_cache is not None:  # another coroutine populated it
            return _tools_cache
        log.info("Initialising MCP worker servers: %s", list(MCP_SERVERS))
        _tools_cache = await _load_tools()
        log.info("Loaded %d MCP tools: %s", len(_tools_cache), [t.name for t in _tools_cache])
        return _tools_cache


def tools_loaded_count() -> int:
    """Cheap, non-blocking count of loaded MCP tools (0 if not yet warmed).

    Safe for liveness/readiness probes — never spawns workers or does I/O.
    """
    return len(_tools_cache) if _tools_cache else 0


async def shutdown() -> None:
    """Terminate the MCP subprocesses cleanly (called on app shutdown)."""
    global _client, _tools_cache
    if _client is None:
        return
    try:
        close_fn = getattr(_client, "close", None)
        if callable(close_fn):
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result
    except Exception:  # pragma: no cover - best-effort
        log.exception("Error while closing MultiServerMCPClient")
    finally:
        _client = None
        _tools_cache = None
