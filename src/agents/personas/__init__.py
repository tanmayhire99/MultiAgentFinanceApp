"""Persona agents for the FinAI Investor Panel.

Each persona is a LangGraph ReAct agent with:

* a distinctive system prompt (voice + investment philosophy),
* access to the same pool of MCP worker tools,
* a structured output contract consumed by the panel supervisor.

The shared factory in :mod:`src.agents.personas.base` keeps the model
configuration, output schema and graph construction identical across
personas so the comparison between them is apples-to-apples.
"""
from .base import PersonaVerdict, build_persona_agent  # noqa: F401
from .buffett import buffett_agent  # noqa: F401
from .graham import graham_agent  # noqa: F401
from .moderator import moderator_open, moderator_synthesise  # noqa: F401
from .wood import wood_agent  # noqa: F401
