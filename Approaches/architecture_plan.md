# Multi-Agent System Architecture Plan
## Domain-Agnostic Hybrid Framework: LangGraph + CrewAI

**Version:** 2.0  
**Type:** Architecture-Only Plan (Domain Agnostic)  
**Last Updated:** March 15, 2026  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Framework Selection and Rationale](#2-framework-selection-and-rationale)
3. [Hybrid Architecture Design](#3-hybrid-architecture-design)
4. [Orchestration Patterns](#4-orchestration-patterns)
5. [Agent Hierarchy and Delegation](#5-agent-hierarchy-and-delegation)
6. [Debate Committee Architecture](#6-debate-committee-architecture)
7. [State Management](#7-state-management)
8. [Communication Protocols](#8-communication-protocols)
9. [Resilience Patterns](#9-resilience-patterns)
10. [Observability and Monitoring](#10-observability-and-monitoring)
11. [Scaling Considerations](#11-scaling-considerations)
12. [Implementation Guidelines](#12-implementation-guidelines)

---

## 1. Executive Summary

### 1.1 Problem Statement

Complex tasks exceed the capability of single agents. Multi-agent systems coordinate specialized agents to solve problems that require diverse expertise, parallel processing, and structured deliberation. However, building production-grade multi-agent systems requires solving coordination, state management, error handling, and debate mechanisms.

### 1.2 Solution: Hybrid LangGraph + CrewAI Architecture

This plan proposes a **hybrid architecture** combining:
- **LangGraph** for stateful orchestration and workflow control
- **CrewAI** for role-based agent teams and delegation

### 1.3 Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Orchestration | LangGraph | State machines, conditional routing, checkpointing |
| Agent Teams | CrewAI | Role definitions, task delegation, natural collaboration |
| Coordination | Hierarchical | Research shows hierarchical outperforms flat coordination |
| Debate | Structured Committees | Prevents herding, surfaces diverse perspectives |
| State | TypedDict + Persistence | Explicit state schema with recovery capability |
| Communication | Structured + NL Hybrid | Reduces context bloat, enables audit |

---

## 2. Framework Selection and Rationale

### 2.1 Framework Comparison Matrix

| Capability | LangGraph | CrewAI | AutoGen | Custom |
|------------|-----------|--------|---------|--------|
| **State Management** | Excellent | Good | Good | Full control |
| **Workflow Control** | Excellent | Good | Medium | Full control |
| **Agent Roles** | Good | Excellent | Good | Full control |
| **Parallel Execution** | Native | Via tasks | Native | Custom |
| **Conditional Routing** | Excellent | Limited | Good | Custom |
| **Checkpointing** | Native | Limited | Custom | Custom |
| **Learning Curve** | Medium | Low | Medium | High |
| **Production Ready** | Yes | Emerging | Yes | Varies |

### 2.2 Why LangGraph for Orchestration

LangGraph excels at:
1. **Explicit State Machines**: Define states, transitions, and conditions clearly
2. **Conditional Routing**: Route based on intermediate results dynamically
3. **Persistence**: Built-in checkpointing for resumable workflows
4. **Observability**: Integration with LangSmith for tracing
5. **Graph-based Flow**: Natural representation of complex workflows

**Research Support:**
> "LangGraph gives you the most control over complex branching workflows... graph-based orchestration with explicit state management." (Agent Patterns, 2026)

### 2.3 Why CrewAI for Agent Teams

CrewAI excels at:
1. **Role-based Agents**: Natural definition of agent personas and capabilities
2. **Task Delegation**: Agents can delegate subtasks to others
3. **Collaboration Patterns**: Sequential, hierarchical, and asynchronous execution
4. **Tool Integration**: Easy attachment of tools to agents
5. **Memory Systems**: Built-in short-term and long-term memory

**Research Support:**
> "CrewAI optimizes for multi-agent role orchestration... role-based collaboration with defined tasks." (Agent Patterns, 2026)

### 2.4 Why Hybrid Approach

Combining LangGraph and CrewAI provides:
- **Best of Both**: Workflow control + Role-based collaboration
- **Production-Proven**: Both frameworks are battle-tested
- **Flexibility**: Use CrewAI for agent definitions, LangGraph for orchestration
- **Industry Trend**: "Most advanced production systems use hybrid architectures" (Zylos Research, 2026)

---

## 3. Hybrid Architecture Design

### 3.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: ORCHESTRATION (LangGraph)                    │
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │
│  │   Planner   │──▶│   Router    │──▶│  Dispatcher │──▶│  Aggregator │   │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   │
│                                                                              │
│  Responsibilities:                                                           │
│  - Query understanding and intent classification                             │
│  - Execution plan generation                                                 │
│  - Conditional routing based on complexity                                   │
│  - State management and checkpointing                                        │
│  - Result synthesis                                                          │
│                                                                              │
│  State Graph: Planning → Level 0 → Level 1 → Level 2 → Synthesis           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 2: AGENT TEAMS (CrewAI)                         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    LEVEL 0: DATA GATHERING                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │    │
│  │  │  Agent A │  │  Agent B │  │  Agent C │  │  Agent D │          │    │
│  │  │  [Role]  │  │  [Role]  │  │  [Role]  │  │  [Role]  │          │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │    │
│  │         Parallel Execution with semaphore                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    LEVEL 1: ANALYSIS                                │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │    │
│  │  │ Analyst 1│  │ Analyst 2│  │ Analyst 3│  │ Analyst 4│          │    │
│  │  │ [Role A] │  │ [Role B] │  │ [Role C] │  │ [Role D] │          │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │    │
│  │         Parallel Execution with isolated context                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    LEVEL 2: DEBATE COMMITTEE                        │    │
│  │  ┌──────────────────┐         ┌──────────────────┐                │    │
│  │  │   Committee A    │◀───────▶│   Committee B    │                │    │
│  │  │   (Pro-side)     │         │   (Con-side)     │                │    │
│  │  │  ┌────────────┐  │         │  ┌────────────┐  │                │    │
│  │  │  │ Facilitator│  │         │  │ Facilitator│  │                │    │
│  │  │  │ Advocate 1 │  │         │  │ Advocate 1 │  │                │    │
│  │  │  │ Advocate 2 │  │         │  │ Advocate 2 │  │                │    │
│  │  │  └────────────┘  │         │  └────────────┘  │                │    │
│  │  └────────┬─────────┘         └────────┬─────────┘                │    │
│  │           └─────────────┬──────────────┘                          │    │
│  │                         ▼                                          │    │
│  │                ┌──────────────────┐                                │    │
│  │                │    Synthesizer   │                                │    │
│  │                │    (Consensus)   │                                │    │
│  │                └──────────────────┘                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 3: INFRASTRUCTURE                                │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   State      │  │  Resilience  │  │ Observability│  │  Compliance  │   │
│  │   Store      │  │    Layer     │  │    Layer     │  │    Layer     │   │
│  │              │  │              │  │              │  │              │   │
│  │ • PostgreSQL │  │ • Circuit    │  │ • Tracing    │  │ • Validation │   │
│  │ • Checkpoint │  │   Breakers   │  │ • Metrics    │  │ • Audit Trail│   │
│  │ • Recovery   │  │ • Retry      │  │ • Alerting   │  │ • Logging    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Responsibility Separation

| Layer | Framework | Responsibility |
|-------|-----------|----------------|
| Orchestration | LangGraph | State graph, routing, checkpointing |
| Agent Teams | CrewAI | Agent definitions, roles, delegation |
| Infrastructure | Both + Custom | Persistence, resilience, monitoring |

---

## 4. Orchestration Patterns

### 4.1 Pattern 1: Sequential Pipeline

**Use When:** Tasks have clear dependencies, each step needs previous step's output.

```
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ Step 1 │───▶│ Step 2 │───▶│ Step 3 │───▶│ Result │
└────────┘    └────────┘    └────────┘    └────────┘
```

**Implementation:**
```python
# LangGraph StateGraph with sequential edges
graph = StateGraph(State)
graph.add_node("step1", step1_agent)
graph.add_node("step2", step2_agent)
graph.add_node("step3", step3_agent)

graph.add_edge("step1", "step2")
graph.add_edge("step2", "step3")
graph.add_edge("step3", END)
```

### 4.2 Pattern 2: Parallel Fan-Out/Fan-In

**Use When:** Multiple independent tasks can run simultaneously.

```
                    ┌────────┐
                 ┌──│ Agent A │──┐
                 │  └────────┘  │
┌────────┐       │  ┌────────┐  │       ┌────────────┐
│ Input  │───────┼──│ Agent B │──┼──────▶│ Aggregator │
└────────┘       │  └────────┘  │       └────────────┘
                 │  ┌────────┐  │
                 └──│ Agent C │──┘
                    └────────┘
```

**Implementation:**
```python
# LangGraph parallel execution
async def parallel_executor(state: State):
    agents = ["agent_a", "agent_b", "agent_c"]
    tasks = [execute_agent(agent, state) for agent in agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    
    return {"results": successful, "failures": failed}
```

### 4.3 Pattern 3: Conditional Routing

**Use When:** Next step depends on intermediate results.

```
                    ┌─────────────┐
                 ┌──│   Path A    │
                 │  └─────────────┘
┌────────┐       │
│ Router │───────┤  ┌─────────────┐
└────────┘       ├──│   Path B    │
                 │  └─────────────┘
                 │  ┌─────────────┐
                 └──│   Path C    │
                    └─────────────┘
```

**Implementation:**
```python
# LangGraph conditional edges
def route_by_complexity(state: State) -> str:
    if state["complexity"] < 4:
        return "simple_path"
    elif state["complexity"] < 7:
        return "standard_path"
    else:
        return "complex_path"

graph.add_conditional_edges(
    "router",
    route_by_complexity,
    {
        "simple_path": "level_0_only",
        "standard_path": "level_0_to_1",
        "complex_path": "full_execution"
    }
)
```

### 4.4 Pattern 4: Hierarchical Delegation

**Use When:** Complex tasks need decomposition by a manager agent.

```
                    ┌─────────────┐
                    │   Manager   │
                    │   Agent     │
                    └──────┬──────┘
                           │ delegates
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Worker 1 │ │ Worker 2 │ │ Worker 3 │
        └──────────┘ └──────────┘ └──────────┘
              │            │            │
              └────────────┴────────────┘
                           │
                    ┌──────▼──────┐
                    │   Review    │
                    │   & Synch   │
                    └─────────────┘
```

**Implementation:**
```python
# CrewAI hierarchical process
from crewai import Agent, Task, Crew, Process

manager = Agent(
    role="Manager",
    goal="Coordinate worker agents",
    allow_delegation=True
)

worker1 = Agent(role="Worker 1", goal="Execute subtask 1")
worker2 = Agent(role="Worker 2", goal="Execute subtask 2")

crew = Crew(
    agents=[manager, worker1, worker2],
    tasks=[planning_task, execution_task],
    process=Process.hierarchical
)
```

### 4.5 Pattern 5: Debate Committee

**Use When:** Multiple perspectives needed to reach balanced conclusion.

```
┌─────────────────────────────────────────────────────────────┐
│                    DEBATE COMMITTEE                          │
│                                                              │
│   ┌─────────────────┐         ┌─────────────────┐         │
│   │  PRO COMMITTEE  │         │  CON COMMITTEE  │         │
│   │  ┌───────────┐  │         │  ┌───────────┐  │         │
│   │  │ Facilitator│  │         │  │ Facilitator│  │         │
│   │  │  Advocate  │  │         │  │  Advocate  │  │         │
│   │  │  Researcher│  │         │  │  Researcher│  │         │
│   │  └───────────┘  │         │  └───────────┘  │         │
│   └────────┬────────┘         └────────┬────────┘         │
│            │                           │                    │
│            │    ┌─────────────┐        │                    │
│            └───▶│  ROUNDS     │◀───────┘                    │
│                 │  1. Opening │                             │
│                 │  2. Rebuttal│                             │
│                 │  3. Closing │                             │
│                 └──────┬──────┘                             │
│                        ▼                                     │
│                 ┌─────────────┐                             │
│                 │ Synthesizer │                             │
│                 │ (Consensus) │                             │
│                 └─────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Agent Hierarchy and Delegation

### 5.1 Hierarchical Delegation Theory

**Research Finding:**
> "Hierarchical delegation consistently outperforms flat coordination on complex tasks... clear chains of responsibility, simplified debugging, efficient use of specialization." (Zylos Research, 2026)

**Key Principles:**

1. **Single Responsibility**: Each agent has one well-defined role
2. **Clear Authority**: Managers delegate, workers execute
3. **Information Hiding**: Workers don't need to know full system state
4. **Graceful Degradation**: System functions even if some agents fail

### 5.2 Three-Level Hierarchy

```
LEVEL 2: SYNTHESIS & DECISION
├── Synthesizer Agent
└── Decision Agent

LEVEL 1: SPECIALIZED ANALYSIS
├── Analyst Type A
├── Analyst Type B
├── Analyst Type C
└── Risk Assessor

LEVEL 0: DATA GATHERING
├── Source Agent 1
├── Source Agent 2
├── Source Agent 3
└── Validation Agent
```

### 5.3 Delegation Protocol

**Manager Agent Responsibilities:**
1. Receive high-level task
2. Decompose into subtasks
3. Assign subtasks to workers based on capability
4. Monitor worker progress
5. Aggregate and synthesize results
6. Handle worker failures (retry, reassign, fallback)

**Worker Agent Responsibilities:**
1. Receive specific subtask
2. Execute using assigned tools
3. Return structured output
4. Report errors immediately

### 5.4 Delegation Decision Tree

```
Task Received
    │
    ├── Can single agent handle? ──YES──▶ Execute directly
    │
    NO
    │
    ├── Is task decomposable? ──NO──▶ Escalate to human
    │
    YES
    │
    ├── Identify subtasks
    │
    ├── Map to agent capabilities
    │
    ├── Determine dependencies
    │
    ├── Create execution plan
    │
    └── Execute with delegation
```

---

## 6. Debate Committee Architecture

### 6.1 Why Debate Committees?

**Research Finding:**
> "Multi-agent debate reduces hallucinations by 40% and improves decision quality by forcing explicit justification of positions." (xDebate, 2026)

**Problems with Single-Agent Decisions:**
- Confirmation bias
- Missing edge cases
- Overconfidence
- No diversity of perspective

**Debate Committee Benefits:**
- Forces consideration of opposing views
- Surfaces hidden risks
- Produces calibrated confidence
- Enables explicit disagreement documentation

### 6.2 Debate Committee Structure

```
┌──────────────────────────────────────────────────────────────────┐
│                     DEBATE COMMITTEE                              │
│                                                                   │
│  COMPONENTS:                                                      │
│                                                                   │
│  1. PRO-SIDE COMMITTEE          2. CON-SIDE COMMITTEE            │
│     ├── Facilitator                ├── Facilitator               │
│     ├── Primary Advocate           ├── Primary Advocate          │
│     ├── Supporting Advocate        ├── Supporting Advocate       │
│     └── Researcher                 └── Researcher                │
│                                                                   │
│  3. DEBATE PROTOCOL                                               │
│     ├── Round 1: Opening Statements                              │
│     ├── Round 2: Rebuttal                                         │
│     ├── Round 3: Clarification                                    │
│     └── Round 4: Closing                                          │
│                                                                   │
│  4. SYNTHESIZER                                                   │
│     ├── Extract consensus points                                  │
│     ├── Document disagreements                                    │
│     └── Produce final recommendation                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 Debate Protocol

**Round 1: Opening Statements (Pro → Con)**
- Pro presents primary case with evidence
- Con presents counter-case with evidence
- Each acknowledges strongest opposing argument

**Round 2: Rebuttal**
- Pro challenges Con's arguments
- Con challenges Pro's arguments
- Both adjust conviction if warranted

**Round 3: Clarification**
- Pro asks Con clarifying questions
- Con asks Pro clarifying questions
- Both provide evidence-based answers

**Round 4: Closing**
- Pro summarizes final position
- Con summarizes final position
- Both state final conviction level

### 6.4 Consensus Synthesis

**Synthesizer Responsibilities:**
1. Analyze debate transcript
2. Extract points of agreement
3. Document remaining disagreements with impact assessment
4. Produce recommendation with confidence range
5. Include action items and risk warnings

**Consensus Categories:**

| Category | Definition | Handling |
|----------|------------|----------|
| **Strong Consensus** | Both sides agree with high conviction | Include as fact |
| **Weak Consensus** | Both sides agree with low conviction | Include with uncertainty note |
| **Managed Disagreement** | Disagreement acknowledged, impact low | Document both views |
| **Critical Disagreement** | Disagreement affects outcome | Flag for human review |

### 6.5 Diverse Reasoning Strategies (DMAD)

**Research Finding:**
> "DMAD outperforms standard MAD by requiring each agent to use different reasoning styles, breaking 'fixed mental sets'." (ICLR-25)

**Reasoning Strategy Assignment:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Backward Reasoning** | Start from goal, work backward | Goal-driven tasks |
| **Step-by-Step** | Sequential processing | Sequential problems |
| **Example-Based** | Learn from similar cases | Pattern matching |
| **Symbolic** | Apply formal rules | Structured data |
| **First Principles** | Build from fundamentals | Novel problems |
| **Counterfactual** | "What if X were different?" | Risk assessment |

**Implementation:**
Each agent in a debate is assigned a unique reasoning strategy to prevent herding.

---

## 7. State Management

### 7.1 State Schema Design

**TypedDict State Definition:**
```python
from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict, total=False):
    # Core identification
    thread_id: str
    query: str
    
    # Execution state
    current_level: int
    execution_plan: Dict[str, Any]
    
    # Level outputs (isolated per level)
    level_0_output: Dict[str, Any]
    level_1_output: Dict[str, Any]
    level_2_output: Dict[str, Any]
    
    # Debate state
    debate_round: int
    pro_case: str
    con_case: str
    debate_history: List[Dict]
    
    # Consensus
    consensus_points: List[str]
    disagreements: List[Dict]
    final_recommendation: Dict[str, Any]
    
    # Error tracking
    failed_agents: List[str]
    retry_count: Dict[str, int]
    
    # Context management
    context_summary: Optional[str]
    key_facts: List[str]
    compression_applied: bool
    
    # Final output
    final_response: Dict[str, Any]
```

### 7.2 State Flow Between Levels

```
LEVEL 0 OUTPUT
    │
    ├── Store in level_0_output
    │
    ├── Extract key facts
    │
    ├── Compress context
    │
    └── Pass to Level 1
        │
        ├── Level 1 agents read compressed context
        │
        ├── Level 1 agents execute independently
        │
        └── Store in level_1_output
            │
            ├── Compress Level 0 + Level 1
            │
            └── Pass to Level 2 Debate
                │
                ├── Debate uses full analysis context
                │
                └── Store in level_2_output
                    │
                    └── Synthesizer reads all levels
                        │
                        └── Produce final_response
```

### 7.3 Checkpointing Strategy

**When to Checkpoint:**
1. Before each level transition
2. After each major agent execution
3. Before debate rounds
4. After consensus synthesis

**Checkpoint Contents:**
- Full state snapshot
- Timestamp
- Execution position
- Error information (if any)

**Recovery Protocol:**
1. Load last checkpoint
2. Resume from checkpoint position
3. Skip completed agents
4. Continue execution

### 7.4 Context Compaction

**Research Finding:**
> "Context compaction at level boundaries reduces tokens by 67% compared to single-context approaches." (LangChain, 2026)

**Compaction Strategy:**
1. Preserve recent messages (last 10%)
2. Summarize older messages
3. Extract key facts
4. Store full history in checkpoint

**Compaction Trigger:**
- Token count > 85% of context limit
- Level transition (automatic)
- Agent request (autonomous)

---

## 8. Communication Protocols

### 8.1 Structured Message Format

**Standard Message Schema:**
```python
class AgentMessage(BaseModel):
    id: str
    timestamp: datetime
    sender: str
    receiver: str
    message_type: Literal["request", "response", "broadcast"]
    content: Dict[str, Any]
    correlation_id: Optional[str]
    priority: Literal["low", "medium", "high"]
```

### 8.2 Communication Patterns

**Pattern 1: Request-Response**
```
Agent A ────request────▶ Agent B
Agent A ◀───response──── Agent B
```

**Pattern 2: Broadcast**
```
Agent A ────broadcast───▶ All Agents
                            ├── Agent B receives
                            ├── Agent C receives
                            └── Agent D receives
```

**Pattern 3: Pub/Sub**
```
Topic: "analysis_complete"
    ├── Agent B subscribed
    └── Agent C subscribed

Agent A publishes to "analysis_complete"
    ├── Agent B receives
    └── Agent C receives
```

### 8.3 Shared Global State

**Research Finding:**
> "Structured reports as the main channel reduce context length and support auditing." (TradingAgents, AAAI-25)

**State Structure:**
```python
class GlobalState:
    artifacts: Dict[str, Artifact]
    
    def publish(self, artifact: Artifact):
        self.artifacts[artifact.id] = artifact
        
    def subscribe(self, artifact_type: str) -> List[Artifact]:
        return [a for a in self.artifacts.values() 
                if a.type == artifact_type]
```

**Benefits:**
- Agents query only what they need
- No full transcript needed
- Deterministic extraction
- Supports auditing

### 8.4 Hybrid Communication Protocol

**Best Practice:** Combine structured and natural language communication.

**Structured Components:**
- Agent reports (JSON schemas)
- State artifacts
- Decision logs
- Audit trails

**Natural Language Components:**
- Debate arguments
- Clarification questions
- Justifications

**Implementation:**
```python
# Structured report
analysis_report = AnalysisReport(
    agent_name="analyst_a",
    signal="positive",
    confidence=0.82,
    key_findings=["finding1", "finding2"],
    rationale="Natural language explanation..."
)

# Debate uses NL
debate_argument = """
Based on the analysis, the evidence suggests...
However, I acknowledge the opposing view...
"""
```

---

## 9. Resilience Patterns

### 9.1 Five-Layer Defense Strategy

**Layer 1: Retry with Exponential Backoff + Jitter**

```python
async def execute_with_retry(agent, state, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await agent.run(state)
        except TransientError:
            delay = min(60, 1 * (2 ** attempt)) + random.uniform(0.1, 0.3) * delay
            await asyncio.sleep(delay)
    raise MaxRetriesExceeded(agent.name)
```

**Layer 2: Error Classification**

| Error Type | Action |
|------------|--------|
| Transient (429, 503) | Retry with backoff |
| Permanent (401, 404) | Fail fast |
| Content Policy | Fallback provider |
| Capability | Reassign to different agent |

**Layer 3: Circuit Breaker**

```
States: CLOSED (normal) ──failures > threshold──▶ OPEN (fail fast)
                                │
                                │ cooldown expired
                                ▼
                          HALF_OPEN (test)
                                │
                                │ success
                                ▼
                            CLOSED
```

**Layer 4: Multi-Provider Fallback**

```python
FALLBACK_CHAIN = [
    {"provider": "primary", "model": "model-a"},
    {"provider": "secondary", "model": "model-b"},
    {"provider": "fallback", "model": "model-c"},
]
```

**Layer 5: Graceful Degradation**

```python
async def execute_with_degradation(state):
    try:
        return await full_analysis(state)
    except Exception:
        try:
            return await partial_analysis(state)
        except Exception:
            return await cached_response(state)
```

### 9.2 Failure Handling Decision Tree

```
Agent Fails
    │
    ├── Transient error? ──YES──▶ Retry with backoff
    │
    NO
    │
    ├── Permanent error? ──YES──▶ Fail fast, log error
    │
    NO
    │
    ├── Can reassign? ──YES──▶ Delegate to different agent
    │
    NO
    │
    ├── Have cache? ──YES──▶ Return cached result
    │
    NO
    │
    └── Return graceful failure with partial results
```

### 9.3 Circuit Breaker Configuration per Tier

| Tier | Failure Threshold | Recovery Timeout | Half-Open Probes |
|------|-------------------|------------------|-------------------|
| Data Gathering | 5 failures | 60 seconds | 3 |
| Analysis | 4 failures | 45 seconds | 2 |
| Debate | 3 failures | 30 seconds | 1 |

---

## 10. Observability and Monitoring

### 10.1 Tracing Requirements

**Trace Structure:**
```python
class Trace:
    trace_id: str
    parent_id: Optional[str]
    agent_name: str
    step_name: str
    start_time: datetime
    end_time: datetime
    status: Literal["started", "completed", "failed"]
    input_tokens: int
    output_tokens: int
    confidence: float
    reasoning_strategy: str
    tools_called: List[str]
    error: Optional[str]
```

**Full Delegation Chain Tracing:**
- Every agent execution logged
- Every handoff recorded
- Context at each step captured
- Decision rationale preserved

### 10.2 Metrics Collection

**System Metrics:**
| Metric | Target |
|--------|--------|
| Success rate | >99% |
| P50 latency | <5s |
| P99 latency | <30s |
| Token efficiency | 67% reduction |
| Circuit breaker trips | <1/hour |

**Agent Metrics:**
| Metric | Description |
|--------|-------------|
| Execution count | Times agent executed |
| Success count | Times agent succeeded |
| Average latency | Mean execution time |
| Token usage | Input + output tokens |
| Confidence distribution | Histogram of confidence scores |

**Debate Metrics:**
| Metric | Description |
|--------|-------------|
| Rounds to consensus | Average rounds needed |
| Consensus rate | % reaching consensus |
| Disagreement severity | Distribution by impact |

### 10.3 Alerting Thresholds

| Alert | Condition | Severity |
|-------|-----------|----------|
| Low success rate | <95% | Error |
| High latency P99 | >30s | Warning |
| Circuit breaker trips | >5/hour | Error |
| Fallback usage | >10% | Warning |
| Compliance violation | >0 | Critical |

### 10.4 Dashboard Components

1. **Request Overview**: Success/failure rates, latency distribution
2. **Agent Performance**: Per-agent metrics, bottlenecks
3. **Debate Monitoring**: Rounds, consensus rate, disagreements
4. **State Health**: Checkpoint success, compression ratio
5. **Error Tracking**: Error types, frequency, affected agents

---

## 11. Scaling Considerations

### 11.1 Horizontal Scaling

**Challenge:** As agent count increases, coordination overhead grows.

**Solution: Hierarchical Scaling**

```
Single Manager Agent (handles 5-10 workers)
    │
    ├── Too many workers? ──▶ Add mid-level managers
    │
    └── Each mid-level manager handles 5-10 workers
```

**Research Finding:**
> "Irregular graphs outperform regular structures for complex tasks." (MacNet, ICLR-25)

### 11.2 Vertical Scaling

**Challenge:** Some agents need more compute (e.g., complex analysis).

**Solution: Model Tiering**

| Agent Type | Model | Latency | Cost |
|------------|-------|---------|------|
| Simple classification | Small model | Fast | Low |
| Standard analysis | Medium model | Medium | Medium |
| Complex reasoning | Large model | Slow | High |
| Debate facilitator | Largest model | Slowest | Highest |

### 11.3 Token Budget Management

**Budget Allocation:**
```
Total Budget: 100,000 tokens

Level 0: 40,000 (data gathering)
    ├── Compression: 10,000 tokens
    
Level 1: 30,000 (analysis)
    ├── Compression: 10,000 tokens
    
Level 2: 20,000 (debate)

Synthesis: 10,000 (final output)
```

**Budget Enforcement:**
- Track tokens per level
- Compress when approaching budget
- Abort if budget exceeded

---

## 12. Implementation Guidelines

### 12.1 Directory Structure

```
project/
├── src/
│   ├── orchestration/
│   │   ├── state_graph.py      # LangGraph definitions
│   │   ├── nodes.py            # Node functions
│   │   ├── edges.py            # Edge conditions
│   │   └── checkpointer.py     # State persistence
│   │
│   ├── agents/
│   │   ├── base_agent.py       # Base agent class
│   │   ├── level_0/            # Data gathering agents
│   │   ├── level_1/            # Analysis agents
│   │   └── level_2/            # Debate agents
│   │
│   ├── debate/
│   │   ├── committee.py        # Debate committee
│   │   ├── facilitator.py      # Facilitator logic
│   │   ├── synthesizer.py      # Consensus extraction
│   │   └── protocol.py         # Debate protocol
│   │
│   ├── state/
│   │   ├── schema.py           # TypedDict definitions
│   │   ├── store.py            # State persistence
│   │   └── compactor.py        # Context compaction
│   │
│   ├── resilience/
│   │   ├── retry.py            # Retry logic
│   │   ├── circuit_breaker.py  # Circuit breaker
│   │   └── fallback.py         # Fallback handling
│   │
│   └── observability/
│       ├── tracer.py           # Tracing
│       ├── metrics.py          # Metrics collection
│       └── alerts.py           # Alerting
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── config/
    ├── agents.yaml             # Agent configurations
    ├── debate.yaml             # Debate protocol config
    └── system.yaml             # System settings
```

### 12.2 Configuration Example

```yaml
# config/system.yaml
orchestration:
  framework: langgraph
  checkpoint_enabled: true
  checkpoint_backend: postgresql
  
agent_teams:
  framework: crewai
  process: hierarchical
  
execution:
  levels:
    - name: data_gathering
      parallel: true
      max_concurrent: 5
    - name: analysis
      parallel: true
      max_concurrent: 4
    - name: debate
      parallel: false
      sequential: true
      
debate:
  rounds: 4
  pro_committee_size: 3
  con_committee_size: 3
  
resilience:
  retry:
    max_attempts: 3
    base_delay: 1.0
    max_delay: 60.0
  circuit_breaker:
    thresholds:
      data_gathering: 5
      analysis: 4
      debate: 3
```

### 12.3 Getting Started Checklist

**Phase 1: Foundation**
- [ ] Define state schema (TypedDict)
- [ ] Implement basic LangGraph flow
- [ ] Create agent base classes (CrewAI)
- [ ] Set up checkpointing

**Phase 2: Core Execution**
- [ ] Implement parallel executor
- [ ] Add error handling
- [ ] Implement retry logic
- [ ] Add circuit breakers

**Phase 3: Debate Committee**
- [ ] Implement facilitator
- [ ] Implement debate protocol
- [ ] Implement synthesizer
- [ ] Add consensus extraction

**Phase 4: Production Readiness**
- [ ] Add tracing
- [ ] Add metrics
- [ ] Add alerting
- [ ] Add compliance validation

---

## Summary

This architecture plan provides a domain-agnostic blueprint for building production-grade multi-agent systems using a hybrid LangGraph + CrewAI approach.

**Key Architectural Components:**
1. **Layered Architecture**: Orchestration → Agent Teams → Infrastructure
2. **Hierarchical Delegation**: Manager → Worker coordination
3. **Structured Debate**: Pro vs Con committees with facilitation
4. **State Management**: TypedDict schemas with checkpointing
5. **Five-Layer Resilience**: Retry → Classification → Circuit Breaker → Fallback → Degradation
6. **Hybrid Communication**: Structured reports + Natural language debates
7. **Comprehensive Observability**: Tracing, metrics, alerting

**Research-Backed Decisions:**
- Hierarchical coordination outperforms flat (Zylos, 2026)
- Structured debate reduces hallucinations by 40% (xDebate, 2026)
- Context compaction saves 67% tokens (LangChain, 2026)
- Diverse reasoning strategies improve outcomes (DMAD, ICLR-25)
- Sparse communication reduces herding (EMNLP-24)

This plan serves as the architectural foundation that can be adapted to any domain requiring complex multi-agent coordination.
