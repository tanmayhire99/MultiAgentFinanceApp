"""Shared ``build_chat_model`` seam for the per-agent factory modules.

Every per-agent file (``research_agent.py``, ``us_stock_agent.py``, …)
builds its :class:`ChatOpenAI` via ``_model.build_chat_model(...)``
rather than importing :func:`src.personas.base.build_chat_model`
directly. Routing the call through this one module gives the test
suite a **single patch target** (``src.core.agents._model.build_chat_model``)
that covers every factory at once — without it, each split module would
bind its own copy of the name and tests would have to patch each one.

Call it as an attribute (``_model.build_chat_model(...)``), never
``from ._model import build_chat_model``, so the patched reference is
resolved at call time.
"""
from __future__ import annotations

from src.personas.base import build_chat_model

__all__ = ["build_chat_model"]
