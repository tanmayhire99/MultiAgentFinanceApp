"""Joiner — inspects scratchpad after execution and decides what to do next.

This is **Phase 4** of the planner-first pipeline. After the executor
has run all steps to a terminal state, the joiner examines the results
and decides one of three actions:

* **finish** — results are sufficient; the pipeline should surface the
  synthesizer's output.
* **replan** — gaps remain (e.g. unmet dependencies, a critical step
  failed); the joiner produces ``additional_steps`` to patch the DAG.
  The pipeline re-enters Phase 3 with the extended plan.
* **abort** — unrecoverable (e.g. the synthesizer step itself failed,
  or we've exhausted the replan budget); the pipeline emits an honest
  error.

Rule-based v0
-------------
The first version is **fully deterministic** — no LLM call. The rules
are:

1. If the synthesizer step completed successfully → ``finish``.
2. If there are ``unmet_dependencies`` in the scratchpad AND the
   replan budget allows it → ``replan`` with steps to fill the gaps.
3. If a non-synthesizer step failed AND it has descendants that are
   not the synthesizer → ``replan`` with a replacement step, IF the
   replan budget allows it.
4. If the replan budget is exhausted → ``abort``.
5. If the synthesizer step failed/skipped and we can't replan →
   ``abort``.

The rule-based approach is fast (zero LLM cost) and predictable.
An LLM-powered joiner (which could reason about *which* agent should
fill a gap) is a future enhancement.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from src.core.types import (
    ExecutionState,
    JoinAction,
    JoinDecision,
    Plan,
    PlanStep,
    Scratchpad,
    StepResult,
    UnmetDependency,
)

log = logging.getLogger("finai.joiner")


def decide(state: ExecutionState) -> JoinDecision:
    """Return a :class:`JoinDecision` based on ``state``.

    This is the joiner's single entry point. The pipeline calls it
    after each executor run and acts on the returned decision.

    Priority order:
    1. Unmet dependencies + replan budget → replan
    2. Synthesizer completed → finish
    3. Synthesizer failed + replan budget → replan (retry synth)
    4. Failed non-synth steps + replan budget → replan
    5. No replan budget → abort
    6. No path forward → abort
    """
    plan = state.plan
    pad = state.scratchpad

    if pad.unmet_dependencies:
        if state.can_replan:
            return _replan_for_unmet(state)
        return JoinDecision(
            action="abort",
            reasoning=(
                f"{len(pad.unmet_dependencies)} unmet dependency/dependencies "
                f"remain but replan budget exhausted "
                f"(replan_count={state.replan_count}, "
                f"max={state.max_replans})."
            ),
        )

    synth_step = _find_synth_step(plan)
    if synth_step is not None:
        synth_result = pad.get(synth_step.id)
        if synth_result is not None and synth_result.status == "complete":
            return JoinDecision(
                action="finish",
                reasoning=(
                    f"Synthesizer step {synth_step.id} completed "
                    f"successfully with output."
                ),
            )
        if synth_result is not None and synth_result.status in ("failed", "skipped"):
            if state.can_replan:
                return _replan_for_synth_failure(state, synth_step)

    failed_non_synth = [
        s for s in plan.steps
        if s.agent != "synthesizer"
        and s.id in pad.results
        and pad.results[s.id].status == "failed"
    ]
    if failed_non_synth and state.can_replan:
        return _replan_for_failures(state, failed_non_synth)

    if not state.can_replan:
        if synth_step is not None:
            synth_result = pad.get(synth_step.id)
            if synth_result is not None and synth_result.status in ("failed", "skipped"):
                partial = ""
                if synth_result.status == "failed" and isinstance(
                    synth_result.output, dict
                ):
                    partial = synth_result.output.get("text", "") or ""
                return JoinDecision(
                    action="abort",
                    reasoning=(
                        f"Synthesizer step {synth_step.id} "
                        f"{synth_result.status} and replan budget "
                        f"exhausted (replan_count={state.replan_count}, "
                        f"max={state.max_replans})."
                        + (" Partial output available." if partial else "")
                    ),
                )
        return JoinDecision(
            action="abort",
            reasoning=(
                f"Replan budget exhausted (replan_count={state.replan_count}, "
                f"max={state.max_replans}) with unresolved failures."
            ),
        )

    return JoinDecision(
        action="abort",
        reasoning="No path forward — synthesizer missing or failed, "
        "no unmet dependencies to fill, and no failed non-synth "
        "steps to retry.",
    )


def _find_synth_step(plan: Plan) -> Optional[PlanStep]:
    """Return the synthesizer step (highest-id if multiple)."""
    candidates = [s for s in plan.steps if s.agent == "synthesizer"]
    if not candidates:
        return None
    return sorted(candidates, key=lambda s: -s.id)[0]


_NEXT_STEP_ID_SENTINEL = 1000


def _next_step_id(plan: Plan) -> int:
    """Return a step ID that doesn't collide with any existing step."""
    existing = {s.id for s in plan.steps}
    candidate = max(existing) + 1 if existing else _NEXT_STEP_ID_SENTINEL
    while candidate in existing:
        candidate += 1
    return candidate


def _replan_for_unmet(state: ExecutionState) -> JoinDecision:
    """Produce replan steps to address unmet dependencies."""
    plan = state.plan
    pad = state.scratchpad
    new_steps: List[PlanStep] = []
    next_id = _next_step_id(plan)
    replan_round = state.replan_count + 1

    for unmet in pad.unmet_dependencies:
        existing_ids = {s.id for s in plan.steps}
        desc = (
            f"Fill gap requested by step {unmet.requested_by_step_id}: "
            f"{unmet.reason}"
        )
        ns = PlanStep(
            id=next_id,
            description=desc[:400],
            agent=unmet.target_agent,
            tool_subset=[],
            depends_on=[],
            replan_round=replan_round,
        )
        new_steps.append(ns)
        next_id += 1

    synth_step = _find_synth_step(plan)
    if synth_step is not None and new_steps:
        new_dep_ids = [s.id for s in new_steps]
        updated_deps = list(set(synth_step.depends_on + new_dep_ids))
        synth_replacement = PlanStep(
            id=next_id,
            description=synth_step.description,
            agent="synthesizer",
            tool_subset=[],
            depends_on=updated_deps,
            replan_round=replan_round,
        )
        new_steps.append(synth_replacement)
        next_id += 1

    if not new_steps:
        return JoinDecision(
            action="abort",
            reasoning="Unmet dependencies exist but no valid replan "
            "steps could be constructed.",
        )

    return JoinDecision(
        action="replan",
        reasoning=(
            f"{len(pad.unmet_dependencies)} unmet dependency/dependencies; "
            f"adding {len(new_steps)} step(s) to fill gaps "
            f"(replan round {replan_round})."
        ),
        additional_steps=new_steps,
    )


def _replan_for_failures(
    state: ExecutionState,
    failed_steps: List[PlanStep],
) -> JoinDecision:
    """Produce replan steps to retry failed non-synthesizer steps."""
    plan = state.plan
    pad = state.scratchpad
    new_steps: List[PlanStep] = []
    next_id = _next_step_id(plan)
    replan_round = state.replan_count + 1

    for fs in failed_steps:
        retry = PlanStep(
            id=next_id,
            description=f"Retry: {fs.description}",
            agent=fs.agent,
            tool_subset=fs.tool_subset,
            depends_on=fs.depends_on,
            replan_round=replan_round,
        )
        new_steps.append(retry)
        next_id += 1

    dependents_of_failed: List[PlanStep] = []
    failed_ids = {fs.id for fs in failed_steps}
    for s in plan.steps:
        if s.id in failed_ids:
            continue
        if s.agent == "synthesizer":
            continue
        if any(dep in failed_ids for dep in s.depends_on):
            if pad.results.get(s.id, None) is not None and pad.results[s.id].is_terminal:
                if pad.results[s.id].status == "skipped":
                    retry_deps = []
                    for dep in s.depends_on:
                        if dep in failed_ids:
                            retry_id = _find_retry_for(
                                failed_step_id=dep, new_steps=new_steps
                            )
                            if retry_id is not None:
                                retry_deps.append(retry_id)
                            else:
                                retry_deps.append(dep)
                        else:
                            retry_deps.append(dep)
                    retry_s = PlanStep(
                        id=next_id,
                        description=f"Retry: {s.description}",
                        agent=s.agent,
                        tool_subset=s.tool_subset,
                        depends_on=retry_deps,
                        replan_round=replan_round,
                    )
                    dependents_of_failed.append(retry_s)
                    next_id += 1

    new_steps.extend(dependents_of_failed)

    synth_step = _find_synth_step(plan)
    if synth_step is not None and new_steps:
        all_retry_ids = [s.id for s in new_steps]
        updated_deps = list(set(
            [d for d in synth_step.depends_on if d not in failed_ids]
            + all_retry_ids
        ))
        synth_replacement = PlanStep(
            id=next_id,
            description=synth_step.description,
            agent="synthesizer",
            tool_subset=[],
            depends_on=updated_deps,
            replan_round=replan_round,
        )
        new_steps.append(synth_replacement)

    if not new_steps:
        return JoinDecision(
            action="abort",
            reasoning="Failed steps exist but no valid replan steps "
            "could be constructed.",
        )

    return JoinDecision(
        action="replan",
        reasoning=(
            f"{len(failed_steps)} failed step(s); adding "
            f"{len(new_steps)} replacement/retry step(s) "
            f"(replan round {replan_round})."
        ),
        additional_steps=new_steps,
    )


def _replan_for_synth_failure(
    state: ExecutionState,
    synth_step: PlanStep,
) -> JoinDecision:
    """Produce a replan step to retry a failed/skipped synthesizer."""
    next_id = _next_step_id(state.plan)
    replan_round = state.replan_count + 1
    retry_synth = PlanStep(
        id=next_id,
        description=f"Retry: {synth_step.description}",
        agent="synthesizer",
        tool_subset=[],
        depends_on=synth_step.depends_on,
        replan_round=replan_round,
    )
    return JoinDecision(
        action="replan",
        reasoning=(
            f"Synthesizer step {synth_step.id} failed/skipped; "
            f"adding retry step {next_id} "
            f"(replan round {replan_round})."
        ),
        additional_steps=[retry_synth],
    )


def _find_retry_for(
    *, failed_step_id: int, new_steps: List[PlanStep]
) -> Optional[int]:
    """Find the retry step ID for a failed step among new_steps."""
    for ns in new_steps:
        if ns.description.startswith("Retry:") and ns.replan_round > 0:
            if failed_step_id in ns.depends_on or True:
                return ns.id
    return None


__all__ = ["decide"]
