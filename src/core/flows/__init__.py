"""Flow modules dispatched by :mod:`src.core.dispatcher`.

All non-trivial queries route through ``planner_pipeline.run``
(:mod:`src.core.flows.planner_pipeline`), which calls
:func:`src.core.pipeline.run_pipeline` to orchestrate a planner-
generated DAG of standalone :class:`~src.core.agents._base.ScopedAgent`
instances.

Two fast-path intents skip the planner entirely because they need
zero LLM calls:

* ``smalltalk.run``  — short conversational reply for greetings
* ``meta_help.run``  — curated capabilities answer in markdown
"""
from . import (
    meta_help,
    planner_pipeline,
    smalltalk,
)

__all__ = [
    "meta_help",
    "planner_pipeline",
    "smalltalk",
]