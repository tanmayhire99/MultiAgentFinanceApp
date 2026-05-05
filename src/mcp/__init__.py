"""MCP data-provider servers.

Each module in this package is a standalone MCP server that exposes a small,
domain-specific toolset. The planner-called agents in :mod:`src.core.agents`
consume these tools via the ``MultiServerMCPClient`` during their ReAct loop.
"""