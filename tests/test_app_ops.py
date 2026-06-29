"""Ops/deploy surface tests: the deepened /health readiness payload.

Calls the handler coroutine directly so we don't pull in an HTTP test client
(keeps the CI dep set light). The request-ID access-log middleware is verified
against the running server.
"""
from __future__ import annotations

import asyncio

from src import app as app_module


def test_health_payload_shape():
    body = asyncio.run(app_module.health_check())
    assert body["service"] == "finai"
    assert body["version"] == "3.0.0"
    assert body["status"] in ("healthy", "degraded")
    checks = body["checks"]
    for key in ("llm_key_configured", "mcp_tools_loaded", "mcp_warm",
                "warehouse_enabled", "quant_enabled"):
        assert key in checks, f"missing health check: {key}"
    assert isinstance(checks["mcp_tools_loaded"], int)


def test_health_status_reflects_key_presence(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "something")
    assert asyncio.run(app_module.health_check())["status"] == "healthy"
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert asyncio.run(app_module.health_check())["status"] == "degraded"
