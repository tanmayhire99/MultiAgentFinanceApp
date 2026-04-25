# FIXES — out-of-band changes between migration days

This file logs bug fixes and small improvements that don't fit into a
specific migration day's narrative — usually because they patch
pre-existing demo code rather than the new planner-first stack.

Each entry has the same shape as a day doc (smaller scale): files
touched, why, smoke-test results. Snapshots of the affected files at
the end of the fix go into `snapshots/fix-N-<topic>/` mirroring the
repo layout.

## Index

| # | Title | Tag | Files touched |
|---|---|---|---|
| 1 | [`meta_help` flow — fix the "I can't access the internet" hallucination](#fix-1-meta_help-flow) | `migration/fix-1-meta-help` | 4 files |
| 2 | [Hide developer trace + add `smalltalk` flow + conditional disclaimer](#fix-2-quieter-ux) | `migration/fix-2-quieter-ux` | 6 files |

---

## Fix 1: `meta_help` flow

**Tag:** `migration/fix-1-meta-help`

### Symptom

User asked _"hey what are your capabilities?"_ in the demo. The system
replied:

> _I'm a conversational AI built to help with a wide range of tasks,
> including: Answering questions on many topics such as finance,
> economics, science, history, and everyday life… I don't have
> real-time internet access, so I can't retrieve live market data
> or browse current news…_

This is **factually wrong** and embarrassing for a demo. FinAI has:

* Yahoo Finance live quotes (real-time prices, fundamentals)
* Tavily / DuckDuckGo web search
* SEC EDGAR full-text filings
* BSE / Screener / NSE for Indian markets
* The 3-persona investor panel
* Claim-tracking sub-system

None of which the answer mentioned.

### Root cause

Two independent bugs combined:

1. **No `meta_help` intent.** The router had 5 intents
   (`portfolio_analysis`, `stock_research`, `deep_stock_research`,
   `topic_research`, `educational`). Meta-questions about the system
   itself fell into `educational` because they happened to start with
   "what".
2. **`educational.py` has a CONCEPT prompt.** Its system prompt told
   the LLM _"You are a clear, trustworthy financial educator. The
   user has asked a CONCEPT question."_ Given a non-concept query,
   the LLM hallucinated a generic ChatGPT-style answer — including
   the false claim about no internet access.

### Files touched

| File | Change | Lines |
|---|---|---|
| `src/core/flows/meta_help.py` | **NEW** | ~110 |
| `src/core/flows/__init__.py` | **MODIFIED** | +6 (export new flow) |
| `src/core/router.py` | **MODIFIED** | +90 (intent constant, system-prompt section, `_detect_meta_help` fast-path, decision-card footer fix) |
| `src/core/dispatcher.py` | **MODIFIED** | +5 (register flow in `_FLOW_MAP`, doc-update) |

### What the fix does

* Adds a 6th intent `meta_help` to `INTENTS` and `INTENT_LABELS`.
* Adds a section to the router's LLM system prompt describing meta
  vs educational disambiguation: _"meta_help is meta-questions about
  FinAI; educational is concept questions about finance"_.
* Adds `_detect_meta_help()` — a deterministic regex fast-path that
  short-circuits the LLM call for unambiguous phrasings like
  _"what can you do"_, _"your capabilities"_, _"who are you"_,
  _"tell me about FinAI"_, etc. **This is a perf optimisation, not a
  semantic decision** — non-matching queries still go through the
  LLM. The fast-path also runs in `_safe_fallback` so the routing
  is correct even if the LLM is offline.
* New `src/core/flows/meta_help.py` flow that emits a curated
  hand-written markdown response describing FinAI's actual
  capabilities. **Zero LLM calls** — eliminates the hallucination
  surface entirely. Mentions every flow, every data source, the
  persona panel, and the explicit limits ("no investment advice").
* Decision-card footer text fixed: was hard-coded "from four
  options"; now derives the count from `INTENTS` and shows a
  different footer when the fast-path bypassed the LLM.

### Smoke tests run

1. `_detect_meta_help` unit-test — 23/23 routing decisions correct,
   including 8 negative cases (concept questions, ticker queries,
   etc. NOT classified as meta_help).
2. `classify_query` end-to-end — all 4 representative queries
   (`"hey what are your capabilities?"`, `"what can you do?"`,
   `"tell me about WDC"`, `"what is compound interest?"`) routed
   correctly.
3. `meta_help.run` flow integration — 4013-character markdown
   response covers FinAI / Buffett / Wood / Graham / SEC EDGAR /
   BSE / Yahoo Finance / all 5 query types / "no investment advice"
   limit.
4. Live FastAPI test — restart `finai-api`, hit the same buggy query
   from the bug report. Now responds with the curated markdown
   instead of the hallucination.
5. Existing migration test suite — 117/117 still pass
   (no regressions in `test_types`, `test_registry`,
   `test_scoped_agent`).

### Performance impact

* Meta queries that match the fast-path: **~50× faster**
  (no LLM call → static markdown emit).
* Non-meta queries: zero impact (fast-path returns `None`, LLM
  classifier runs as before).

### Snapshots

End-of-fix file content:

* `docs/migration/snapshots/fix-1-meta-help/src/core/flows/meta_help.py`
* `docs/migration/snapshots/fix-1-meta-help/src/core/flows/__init__.py`
* `docs/migration/snapshots/fix-1-meta-help/src/core/router.py`
* `docs/migration/snapshots/fix-1-meta-help/src/core/dispatcher.py`

To restore this fix's exact state:

```bash
cp -r docs/migration/snapshots/fix-1-meta-help/src/* src/
```

Or via git:

```bash
git checkout migration/fix-1-meta-help
```

---

## Fix 2: Quieter UX

**Tag:** `migration/fix-2-quieter-ux`

### Symptom

Every response — including casual ones like "hi" — was wrapped in
the same templated framing:

```
_Routing your query through the intent classifier…_

🎯 Query Classification
| Intent | educational |
| ... |

[the actual answer]

⚠️ Disclaimer — This response is an educational, multi-agent
analysis...
```

Two specific complaints:

1. The classification card is developer-facing (which flow / which
   tickers / which intent), but it's shown to every user on every
   response. **Repetitive and noisy.**
2. Casual messages like "hi" or "thanks" got routed to the
   educational flow, which prefaced its response with `# 🎓 Educational
   Answer` + an italic dev note + a SEBI disclaimer. **Disproportionate
   for chitchat.**

The user's framing of the bug was clear: _"Look at Claude — it does
not print the same stuff again and again. If the query is generic, it
talks generically. If it is something specific, then it calls agents
and subagents. No predetermined format."_

### Root cause

Three independent design choices combined poorly for the casual-query
case:

1. `src/core/dispatcher.py` unconditionally emitted the routing intro
   + classification card around every flow's output.
2. The disclaimer footer was unconditional too — every flow got a
   "consult a SEBI-registered advisor" footer regardless of whether
   the response was advisory.
3. There was no `smalltalk` intent. Casual messages fell into
   `educational`, which has a CONCEPT system prompt — fine for
   "what is beta?", weird for "hi".

### Files touched

| File | Change | Lines |
|---|---|---|
| `src/core/dispatcher.py` | **MODIFIED** | +60 (verbose_trace flag, /trace prefix detection, conditional disclaimer logic, intent set for "real finance flows") |
| `src/core/router.py` | **MODIFIED** | +90 (`smalltalk` intent + label + LLM-prompt section + `_detect_smalltalk` fast-path + safe_fallback wiring) |
| `src/core/flows/smalltalk.py` | **NEW** | ~180 (hybrid regex-static + LLM fallback) |
| `src/core/flows/__init__.py` | **MODIFIED** | +6 (export the new flow) |
| `src/core/flows/educational.py` | **MODIFIED** | -16 (drop banner header + italic dev-trace note) |
| `docs/migration/FIXES.md` | **MODIFIED** | +this section |

### What the fix does

#### 1. Hide developer trace by default

Two new toggles control whether the routing intro + classification
card appear:

```python
# .env
FINAI_VERBOSE_TRACE=1     # global default-on (for development / live demos)
```

```text
/trace what can you do?    # one-shot enable for this single message
```

When neither is set, the dispatcher silently classifies the query
(still logged to stdout), then jumps straight into the flow's
output. The user sees a clean answer.

#### 2. Conditional disclaimer

The `⚠️ Disclaimer` footer now appears only for intents in
`_FINANCE_FLOW_INTENTS = {portfolio_analysis, stock_research,
deep_stock_research, topic_research}`. `educational`, `meta_help`,
and `smalltalk` no longer get the SEBI footer — none of them
produce advisory-shaped content.

#### 3. New `smalltalk` flow with hybrid behaviour

`src/core/flows/smalltalk.py` handles greetings / thanks /
acknowledgments / goodbyes:

* **Static fast-path** for the most common phrasings (regex match
  → curated reply). E.g. `"hi"` → `"Hey 👋 — what would you like
  to look at? I can pull a stock..."`. **Zero LLM cost.**
* **LLM fallback** for nuanced casual messages the classifier
  routed here but the regex didn't match. Tight system prompt,
  `max_tokens=200`, streamed.

The router has a matching `_detect_smalltalk()` fast-path so the
classifier itself short-circuits for obvious greetings. Symmetric
with the `_detect_meta_help` introduced in Fix 1.

#### 4. Educational flow cleanup

Removed:

```python
yield {"type": "header", "text": "# 🎓 Educational Answer\n\n"}
yield {"type": "text", "text": "_This query was classified as a concept question, so no Portfolio, Stock, or Research Agent is being called..._\n\n"}
```

Both were dev-facing trace bleeding into the user response. The flow
now streams the answer directly, the way Claude would.

### Smoke tests run

| Query | Expected intent | Trace shown? | Disclaimer? | Result |
|---|---|---|---|---|
| `hi` | smalltalk | No | No | ✓ static greeting only |
| `thanks` | smalltalk | No | No | ✓ static thanks only |
| `/trace hi` | smalltalk | YES (forced) | No | ✓ classification card + greeting |
| `hey what are your capabilities?` | meta_help | No | No | ✓ curated markdown, no disclaimer |
| `what is beta?` | educational | No | No | ✓ direct answer (no banner header) |
| `tell me about WDC` | stock_research | No | YES | ✓ full research + disclaimer footer |

Plus the existing 117-test migration suite — all pass. No regressions.

### Snapshots

End-of-fix file content:

* `docs/migration/snapshots/fix-2-quieter-ux/src/core/dispatcher.py`
* `docs/migration/snapshots/fix-2-quieter-ux/src/core/router.py`
* `docs/migration/snapshots/fix-2-quieter-ux/src/core/flows/__init__.py`
* `docs/migration/snapshots/fix-2-quieter-ux/src/core/flows/smalltalk.py`
* `docs/migration/snapshots/fix-2-quieter-ux/src/core/flows/educational.py`

### Things deliberately NOT in this fix

* **Per-flow narration cleanup.** The stock_research / portfolio_analysis
  flows have their own internal dev-style narration (multiple
  `🔗 Orchestrator → Stock Agent` lines, `_For each ticker we will: 1...`,
  etc.). Those are part of the demo's "show the agents working"
  story and the user explicitly asked to leave those alone unless
  raised separately. A future fix can revisit if needed.
* **Random variation in static smalltalk replies.** Each category
  has a single hand-written reply. Adding rotation is easy
  (`random.choice` over a list) but the user feedback emphasised
  "be consistent like Claude", and Claude's "hi" isn't randomised.
