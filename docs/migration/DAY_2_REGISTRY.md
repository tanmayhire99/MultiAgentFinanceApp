# Day 2 — `src/core/agents/registry.py` (Agent catalog + policy gates)

> **Goal:** Define the catalog of available agents, declare which
> MCP tools each one owns, and gate the expensive ones behind
> structural validators that the planner cannot bypass.

## Files touched

| File | New / Modified | Lines | Why |
|---|---|---|---|
| `src/core/agents/__init__.py` | **NEW** | ~40 | Package exports |
| `src/core/agents/registry.py` | **NEW** | ~620 | `PolicyGate`, `AgentDefinition`, `AgentRegistry`, the canonical 8-agent catalog |
| `tests/test_registry.py` | **NEW** | ~450 | 40 tests covering construction, validation, gates, invariants |

## What was added

### Eight canonical agents

| Name | Tools owned | Policy gate |
|---|---|---|
| `research_agent` | 11 (research__*) | none |
| `us_stock_agent` | 6 (us_stock__*) | none |
| `indian_stock_agent` | 6 (indian_stock__*) | none |
| `filings_agent` | 5 (research__sec_*, research__indian_filings*) | none |
| `portfolio_agent` | 6 (portfolio__*) | none |
| `synthesizer` | 0 (LLM-only) | none |
| **`claim_agent`** | 2 (extract / compare) | **`wants_claim_tracking`** |
| **`panel_agent`** | 0 (spawns personas internally) | **`wants_panel_debate`** |

Total tool ownership = **34 namespaced MCP tools** with no overlaps
(`tool_owner()` is bijective except for the synthesizer / panel,
which own no tools).

### `PolicyGate`

A Pydantic model that **boolean-checks intent flags** set by the
classifier LLM. No regex, no phrase whitelists, no text matching —
just `any(intent_flags.get(flag, False) for flag in self.required_intent_flags)`.

```python
PolicyGate(
    description="Claim tracking is opt-in.",
    required_intent_flags=["wants_claim_tracking"],
    hard_block_unless_match=True,  # vs advisory mode
)
```

A field-level validator rejects unknown flag names against
`KNOWN_INTENT_FLAGS` (defined in `src/core/types.py`), so a typo like
`"wants_claim_traking"` fails at **import time**, not on the first
query that hits the gate.

### `AgentDefinition`

```python
AgentDefinition(
    name="research_agent",       # snake_case, validated
    title="Research Agent",
    description="Does research over the web.",
    tools=("research__search_news", ...),
    role_hint="any web-search query",
    policy_gate=None,             # or PolicyGate(...)
)
```

`name` validator enforces snake_case; `description` has min-length;
`extra='forbid'` so future hand-written definitions can't drift.

### `AgentRegistry`

Holds the agents in a list + builds `_by_name` and
`_by_tool` lookup maps at construction. Construction-time invariants:

* No duplicate agent names → `ValueError`
* No tool claimed by two agents → `ValueError`

API:

* `get(name) -> Optional[AgentDefinition]`
* `tool_owner(tool_name) -> Optional[str]`
* `gated_agents() -> List[AgentDefinition]`
* `validate_step(step, intent_flags) -> List[str]` (errors)
* `validate_plan(plan, intent_flags) -> List[str]`
* `planner_catalog_text() -> str` — markdown for the planner's
  system prompt; ungated agents listed first, gated ones marked with
  `⚠️ POLICY GATE` so the planner sees the constraint inline.

### `tests/test_registry.py`

40 unittest tests across six classes:

* `PolicyGateTests` — open / single-flag / multi-flag (OR) /
  missing-flag / unknown-flag-rejected / explain string format /
  advisory mode (8 tests)
* `AgentDefinitionTests` — valid construction, snake_case
  enforcement, description length, extra=forbid, frozen (5)
* `AgentRegistryConstructionTests` — duplicate names rejected,
  duplicate tools rejected, minimal registry (3)
* `RegistryValidationTests` — unknown agent, tool not owned, valid
  step, synthesizer no-tool, plan-level error aggregation (5)
* `CanonicalPolicyGateTests` — claim_agent / panel_agent blocked
  without flag, allowed with flag, unrelated flags don't open gates,
  both gates independent, missing flag treated as False (7)
* `CanonicalRegistryInvariantTests` — 34-tool count, no dupes,
  claim tools owned by claim_agent, gated set is exactly two,
  required-flag pinning, planner_catalog text invariants (12)

## Important moments during Day 2

### Bug 1: `.append()` on a set

The first iteration of the duplicate-tool detector did:

```python
seen, dupes = set(), []
for tool in agent.tools:
    if tool in seen:
        dupes.append(tool)
    seen.append(tool)  # BUG: sets don't have .append
```

Caught immediately by `tests/test_registry.py::test_duplicate_tool_ownership_rejected`. Fixed to `seen.add(tool)`.

### Bug 2: regex coverage gap (and the larger pivot)

The original `PolicyGate` used `required_phrases` + `required_patterns`
(regex) to decide whether the user's query was asking for claim
tracking. Tests caught a coverage gap — `"made good on"` matched but
`"making good on"` didn't. Fixed by extending the regex to
`(made|make|makes|making) good on`.

**Then the user pushed back fundamentally:** _"Why are we having
regular expressions in our code? isn't it rule based and not dynamic?"_

This was correct. The whole point of the migration is to be
LLM-driven, not rule-based. A regex fundamentally can't capture all
the natural-language ways a user might ask for claim tracking, and
adding more regexes is just rebuilding a worse classifier.

### The refactor: text matching → intent flags

`PolicyGate` was rewritten to take `required_intent_flags: List[str]`
instead of phrases/patterns. The classifier LLM (Phase 1) is the
**only** place where natural-language understanding happens; it sets
boolean flags like `wants_claim_tracking`. The registry's gate is a
pure structural check: _did the classifier set this flag?_

This preserves the two-tier control plane:

1. **Classifier LLM (Phase 1)** — semantic understanding of the
   query → `intent_flags: Dict[str, bool]`
2. **Registry gate (Phase 2)** — structural enforcement: the planner
   may include this agent only if the flag is True

`KNOWN_INTENT_FLAGS` (in `src/core/types.py`) is the shared
vocabulary between the two phases. Adding a new flag is a one-step
change.

Files affected by the refactor:

* `src/core/types.py` — added `KNOWN_INTENT_FLAGS` frozenset with 6
  flags: `wants_claim_tracking`, `wants_panel_debate`, `wants_filings`,
  `wants_portfolio_data`, `wants_historical_news`, `wants_deep_research`
* `src/core/agents/registry.py` — `PolicyGate` rewritten;
  `validate_step` / `validate_plan` signatures changed from
  `query: str` → `intent_flags: Dict[str, bool]`
* `tests/test_registry.py` — all gate-related tests rewritten to
  pass `intent_flags` dicts instead of query strings

## Design decisions worth remembering

### Why two gated agents specifically?

`claim_agent` and `panel_agent` are both **deliberately expensive**:

* `claim_agent` runs 1 LLM call per claim and 1 per verdict — a
  query that triggers it on a 4-claim transcript is 8+ extra LLM
  calls.
* `panel_agent` runs 60-180 seconds (3 personas × multi-round debate
  + moderator).

The other 6 agents are cheap enough that running them when not
strictly needed costs ~no time. Gating only the expensive ones is
the right precision/recall tradeoff.

### Why "OR" semantics for `required_intent_flags`?

If a gate has multiple required flags, **any** one being True opens
it. This lets us write gates like:

```python
required_intent_flags=["wants_claim_tracking", "wants_filings"]
# → opens for either "verify guidance" OR "show me the 10-K"
```

AND-semantics ("all flags must be True") would be more restrictive,
but in practice the classifier almost never co-fires unrelated flags,
so OR is the sensible default. We can add `mode: "or"|"and"` later
if a real case calls for AND.

### Why is the planner catalog text in the registry, not in the planner?

The planner LLM's system prompt is mostly the agent catalog. Letting
the planner module own that text means changes to gate semantics
have to be mirrored in two places. By putting `planner_catalog_text()`
on the registry, the planner can just call it — single source of truth.

## Test results

```
$ docker exec finai-api python -m unittest tests.test_registry -v
[...]
Ran 40 tests in 0.003s
OK
```

All 40 tests passing.

## Snapshot

End-of-Day-2 file content:

* `docs/migration/snapshots/day-2/src/core/types.py` (with
  `KNOWN_INTENT_FLAGS` added; **no** `UnmetDependency` yet)
* `docs/migration/snapshots/day-2/src/core/agents/__init__.py`
* `docs/migration/snapshots/day-2/src/core/agents/registry.py`
* `docs/migration/snapshots/day-2/tests/test_types.py`
* `docs/migration/snapshots/day-2/tests/test_registry.py`

To restore Day 2's exact state:

```bash
cp docs/migration/snapshots/day-2/src/core/types.py src/core/types.py
cp docs/migration/snapshots/day-2/src/core/agents/__init__.py src/core/agents/__init__.py
cp docs/migration/snapshots/day-2/src/core/agents/registry.py src/core/agents/registry.py
cp docs/migration/snapshots/day-2/tests/test_types.py tests/test_types.py
cp docs/migration/snapshots/day-2/tests/test_registry.py tests/test_registry.py
# And remove anything Day 3 added:
rm -f src/core/agents/_base.py tests/test_scoped_agent.py
```
