# Migration Log — Planner-First Multi-Agent Refactor

This folder is the **change log** for the migration from the demo-era
flow-based architecture (Phase 11–14) to the planner-first multi-agent
architecture described in [`../MULTI_AGENT_ARCHITECTURE.md`](../MULTI_AGENT_ARCHITECTURE.md).

Every "day" of the migration gets:

1. A narrative document — `DAY_N_<topic>.md` — explaining what was
   touched, why, and how the changes fit together.
2. A snapshot directory — `snapshots/day-N/` — containing copies of
   the actual files as they existed at the **end** of that day. This
   is the bullet-proof rollback path: you can always copy a snapshot
   back into `src/` or `tests/` to get the file as it was.
3. A git tag — `migration/day-N` — pointing at the commit that ended
   that day. (Tags exist for every day from the day this folder was
   created onwards; for retroactive days the snapshot is the rollback
   path.)

## Why three layers of safety?

- **Narrative doc** — answers "what changed and why?" for a future
  reader (you in two weeks, or a teammate). Includes file paths,
  function names, and key code excerpts.
- **Snapshot** — answers "what did the file actually look like?"
  Survives if git history is rewritten or if the change was made in
  multiple commits.
- **Git tag** — answers "let me try the whole tree from that day,
  including config and dependencies". Cleanest rollback when you want
  the **whole project state** at that point, not just one file.

## Day index

| Day | Title | Git tag | Files touched |
|---|---|---|---|
| 0 | [Baseline (pre-migration)](DAY_0_BASELINE.md) | _(precedes `migration/day-3`)_ | n/a — describes existing code |
| 1 | [`src/core/types.py`](DAY_1_TYPES.md) — Plan, PlanStep, Scratchpad, etc. | rolled into `migration/day-3` | 2 new files |
| 2 | [`src/core/agents/registry.py`](DAY_2_REGISTRY.md) — Agent catalog + policy gates | rolled into `migration/day-3` | 3 new files |
| 3 | [`src/core/agents/_base.py`](DAY_3_SCOPED_AGENT.md) — `ScopedAgent` runtime wrapper | **`migration/day-3`** | 2 new files + 3 modified |
| 4 | [`src/core/agents/_factories.py`](DAY_4_SLICE_FACTORIES.md) — Per-agent ScopedAgent factories (slice subset: research / filings / claim / synthesizer) | **`migration/day-4-slice-factories`** | 2 new files + 2 modified |
| 4b | [Panel-slice factories + `PanelScopedAgent`](DAY_4B_PANEL_FACTORIES.md) — adds factories for us_stock / indian_stock / portfolio / panel; `_FACTORY_MAP` now covers every registry agent | **`migration/day-4b-panel-factories`** | 1 new file + 3 modified |

> Days 1 and 2 don't have their own git tags because the source code
> for those days was committed all at once on Day 3 (the project had
> been uncommitted up to that point). Per-file snapshots in
> `snapshots/day-1/` and `snapshots/day-2/` provide rollback for
> those days. Days 4+ each get their own tag.

### Vertical slice work (Days 4-10)

Days 4-10 of the migration are being delivered as a "vertical slice"
that compresses ~7 days of architecture work into 5 stages, each with
its own git tag. Each stage exercises end-to-end the parts of the
architecture it covers.

| Stage | Tag | What lands |
|---|---|---|
| 1 | [`migration/day-4-slice-factories`](DAY_4_SLICE_FACTORIES.md) | ✅ Factories for the 4 agents the claim-tracker slice needs |
| 2 | [`migration/day-6-7-slice-engine`](DAY_6_7_SLICE_ENGINE.md) | ✅ planner + sequential DAG executor + pipeline orchestrator (28 tests) |
| 3 | [`migration/day-10-claim-slice`](DAY_10_CLAIM_SLICE.md) | ✅ `/planner` opt-in dispatcher wiring; planner pipeline reachable end-to-end (20 tests) |
| 4 | [`migration/day-4b-panel-factories`](DAY_4B_PANEL_FACTORIES.md) | ✅ Factories for the 4 panel-slice agents (us_stock / indian_stock / portfolio / panel + `PanelScopedAgent` subclass that drives the multi-round persona debate; 15 new tests, 204 total) |
| 5 | `migration/day-10b-panel-slice` | `/planner` panel queries route through new pipeline |

## How to roll back

There are three ways to roll back, ordered from least-invasive to most:

### 1. Roll back ONE file

You want to undo a change to a single file (e.g. you don't like the
new `_base.py`):

```bash
# from the repo root
cp docs/migration/snapshots/day-2/src/core/agents/__init__.py src/core/agents/__init__.py
# or, to remove the file entirely (it was new on Day N):
rm src/core/agents/_base.py
```

The snapshots mirror the project layout, so the relative path under
`docs/migration/snapshots/day-N/` is the same as in the repo.

### 2. Roll back ONE day's changes

You want to undo everything Day N introduced. Each day-N doc lists
the files that day touched — see "Files touched" at the top of each
doc. For each one:

* If the file existed before Day N → copy the previous day's snapshot
  back into the repo.
* If the file was created on Day N → delete it.

> **Tip:** Day docs explicitly mark each file as **NEW** (created
> that day) or **MODIFIED** (existed before, edited that day) so this
> is unambiguous.

### 3. Roll back to a tagged whole-tree state

```bash
# View tags
git tag --list 'migration/*'

# Inspect what a day's tree looks like
git checkout migration/day-3 -- src/    # apply just src/ at that day's state
# or:
git checkout migration/day-3            # checkout the whole tree (detached HEAD)
git checkout main                       # come back

# To create a working branch starting from a day's state:
git checkout -b experiment-from-day-3 migration/day-3
```

## How to add a new day going forward

When we start "Day N", the workflow is:

1. **Plan** — write a short note at the top of a new
   `DAY_N_<topic>.md` describing the goal.
2. **Implement** — write the code. As you go, update the narrative
   doc with "what we touched" and "why".
3. **Test** — run the test suite. Add a "Test results" section to
   the doc.
4. **Snapshot** — copy every file the day touched into
   `docs/migration/snapshots/day-N/`, mirroring the repo layout.
5. **Commit** — `git commit` with a message starting with
   `migration: day N - <topic>`.
6. **Tag** — `git tag migration/day-N`.

For convenience, a helper script will be added at
`scripts/snapshot_day.sh` once we have a stable enough workflow
(see Day 4+).

## Conventions

* **Day boundaries** correspond to a coherent unit of work, not a
  literal calendar day. A day might span an afternoon or two
  sessions; what matters is that the work shipped together with
  passing tests.
* **Each day must end green.** No partial commits with broken tests.
  If a day can't finish on time, leave a "next steps" section in the
  doc, snapshot only the working subset, and start the unfinished
  work on Day N+1.
* **Doc filename convention**: `DAY_<N>_<SHORT_TOPIC>.md` where the
  topic is upper-snake-case and matches the work's primary subject
  (e.g. `DAY_3_SCOPED_AGENT.md`).
* **Snapshot policy**: include only the files the day **created** or
  **modified**. Don't snapshot the whole repo.
