# `src/legacy/` — Preserved earlier designs

This folder holds proposed-but-not-shipped designs that we want to keep for
reference, not delete.

## Contents

* **`planner.py`** — A real `PlannerAgent` (LLM with `guided_json` schema → a
  `Plan` of typed `PlanStep`s). Each step has `(id, description, agent,
  inputs, success_criteria)`. This is the **plan-and-execute** pattern,
  adapted for our agent registry.
* **`router.py`** — A simpler `RouterAgent` with regex-first intent matching
  and an LLM fallback. Single-agent selection (does not produce multi-step
  plans).
* **`orchestrator.py`** — A LangGraph state graph that wires planner →
  prepare-data → route → execute → next-step into a finite execution loop
  with a `max_steps` cap.

## Why we kept them

The current production architecture (`src/core/router.py` +
`src/core/dispatcher.py` + `src/core/flows/*`) replaced these with **single
intent → single flow → all-tools-available** dispatch. That works for
demos but produces the "every research run also does claim analysis" problem
the user flagged.

The next architecture iteration will reintroduce a proper planner, so the
shape of `planner.py` here is the right starting point for that work.

## Status

* **Not imported** by any production module as of 2026-04-25.
* The `from src.tools.llm_client import LLMClient` import inside these
  files refers to a module that does not exist in the current tree — the
  files were drafted before that helper was built.
* If you want to revive any of this, port the relevant types into
  `src/core/` and rewire the imports.
