# Integration Plan: CrewAI + LangGraph Hybrid Architecture

## Executive Summary

This plan outlines how to integrate **CrewAI's task delegation and role-based collaboration** with the existing **LangGraph orchestration** to create a more powerful, parallel, and hierarchical multi-agent financial advisory system.

---

## Current Architecture Analysis

### Existing LangGraph Flow

```
START → Planner → Prepare Data → Route → Execute → Next → END
                                    ↑              ↓
                                    └──────────────┘
```

### Strengths
- Clean state management via `GraphState` TypedDict
- Conditional routing with `_should_continue` and `_should_prepare_data`
- Structured planning via `PlannerAgent` with guided JSON
- Sequential step execution with scratchpad logging

### Limitations
1. **Sequential execution only** - agents run one at a time
2. **No parallelism** - independent tasks cannot run concurrently
3. **No agent collaboration** - agents cannot delegate to or query other agents
4. **Single agent per step** - cannot synthesize multiple perspectives
5. **No debate/consensus mechanisms** - no way to resolve conflicting analyses

---

## Proposed Hybrid Architecture

### Core Idea

Use **LangGraph as the orchestration backbone** and **CrewAI for agent-level task delegation**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LangGraph Orchestration Layer                   │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────────┐  │
│  │ Planner │───▶│ Intent   │───▶│ Parallel  │───▶│ Synthesis     │  │
│  │         │    │ Classifier│   │ Executor  │    │ & Response    │  │
│  └─────────┘    └──────────┘    └────┬─────┘    └───────────────┘  │
│                                       │                              │
│  ┌────────────────────────────────────┼───────────────────────────┐│
│  │                    CrewAI Delegation Layer                      ││
│  │                                    ▼                            ││
│  │   ┌─────────────────────────────────────────────────────────┐  ││
│  │   │              LEVEL 0: Data Gathering (Parallel)          │  ││
│  │   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │  ││
│  │   │  │ Upstox  │ │WebSearch│ │ NewsAPI │ │Filings  │        │  ││
│  │   │  │ (Crew)  │ │ (Crew)  │ │ (Crew)  │ │ (Crew)  │        │  ││
│  │   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │  ││
│  │   └─────────────────────────────────────────────────────────┘  ││
│  │                              │                                  ││
│  │                              ▼                                  ││
│  │   ┌─────────────────────────────────────────────────────────┐  ││
│  │   │           LEVEL 1: Specialized Analysis (Parallel)       │  ││
│  │   │  ┌───────────┐ ┌───────────┐ ┌───────────┐              │  ││
│  │   │  │ Technical │ │Fundamental│ │ Sentiment │              │  ││
│  │   │  │ Analyst   │ │ Analyst   │ │ Analyst   │              │  ││
│  │   │  │ [RSA-A]   │ │ [RSA-B]   │ │ [RSA-C]   │              │  ││
│  │   │  └───────────┘ └───────────┘ └───────────┘              │  ││
│  │   └─────────────────────────────────────────────────────────┘  ││
│  │                              │                                  ││
│  │                              ▼                                  ││
│  │   ┌─────────────────────────────────────────────────────────┐  ││
│  │   │              LEVEL 2: Debate Committee (Crew)            │  ││
│  │   │  ┌────────────────┐     ┌────────────────┐              │  ││
│  │   │  │  Bull Committee│◀───▶│ Bear Committee │              │  ││
│  │   │  │  (Facilitator) │     │  (Facilitator) │              │  ││
│  │   │  └────────┬───────┘     └───────┬────────┘              │  ││
│  │   │           └──────────┬──────────┘                       │  ││
│  │   │                      ▼                                  │  ││
│  │   │            ┌─────────────────┐                          │  ││
│  │   │            │   Consensus     │                          │  ││
│  │   │            │   Synthesizer   │                          │  ││
│  │   │            └─────────────────┘                          │  ││
│  │   └─────────────────────────────────────────────────────────┘  ││
│  └──────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Integration Plan

### Phase 1: Add Parallel Execution to LangGraph (Week 1-2)

#### 1.1 Create Parallel Executor Node

```python
# src/core/parallel_executor.py

import asyncio
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class AgentTask:
    agent_key: str
    inputs: Dict[str, Any]
    reasoning_strategy: str  # RSA assignment

class ParallelExecutor:
    def __init__(self, max_concurrent: int = 5, timeout: float = 30.0):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_level(
        self, 
        tasks: List[AgentTask],
        state: GraphState
    ) -> Dict[str, Any]:
        """Execute all tasks at a level concurrently."""
        async def run_with_semaphore(task: AgentTask):
            async with self.semaphore:
                agent = _resolve_agent(task.agent_key)
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(agent.run, state),
                        timeout=self.timeout
                    )
                    return {"agent": task.agent_key, "result": result, "status": "success"}
                except asyncio.TimeoutError:
                    return {"agent": task.agent_key, "error": "timeout", "status": "failed"}
        
        results = await asyncio.gather(*[run_with_semaphore(t) for t in tasks])
        return self._aggregate_results(results)
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate parallel results using weighted strategy."""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]
        
        return {
            "data": {r["agent"]: r["result"] for r in successful},
            "failures": failed,
            "success_rate": len(successful) / len(results) if results else 0
        }
```

#### 1.2 Update GraphState for Level-Based Execution

```python
# Updated GraphState in orchestrator.py

class LevelOutput(TypedDict):
    level_id: int
    agents_executed: List[str]
    results: Dict[str, Any]
    synthesis: Optional[str]

class GraphState(TypedDict, total=False):
    # Existing fields
    user_id: str
    query: str
    plan: Plan
    step_index: int
    
    # New fields for parallel execution
    execution_levels: List[List[str]]  # Agents per level
    current_level: int
    level_outputs: List[LevelOutput]
    parallel_results: Dict[str, Any]
    
    # Reasoning strategy assignments
    agent_strategies: Dict[str, str]  # agent_key -> reasoning_strategy
```

#### 1.3 Add Parallel Execution Nodes to Graph

```python
# In orchestrator.py

def _node_plan_parallel(self, state: GraphState) -> GraphState:
    """Enhanced planner that creates level-based execution plan."""
    plan = self.planner.plan_parallel(
        goal=state["query"],
        intent=state.get("intent"),
        available_agents=list(AGENT_REGISTRY.keys())
    )
    
    new = dict(state)
    new["execution_levels"] = plan.levels
    new["current_level"] = 0
    new["agent_strategies"] = plan.strategies
    return new

def _node_execute_level(self, state: GraphState) -> GraphState:
    """Execute all agents at current level in parallel."""
    level_idx = state["current_level"]
    agents_at_level = state["execution_levels"][level_idx]
    
    tasks = [
        AgentTask(
            agent_key=agent,
            inputs={"query": state["query"], "profile": state.get("profile")},
            reasoning_strategy=state["agent_strategies"].get(agent, "default")
        )
        for agent in agents_at_level
    ]
    
    results = await self.parallel_executor.execute_level(tasks, state)
    
    new = dict(state)
    new["level_outputs"].append({
        "level_id": level_idx,
        "agents_executed": agents_at_level,
        "results": results["data"],
        "synthesis": None
    })
    new["current_level"] = level_idx + 1
    return new

def _should_continue_levels(self, state: GraphState) -> str:
    """Check if more levels to execute."""
    if state["current_level"] >= len(state["execution_levels"]):
        return "synthesize"
    return "execute_level"
```

---

### Phase 2: Integrate CrewAI for Task Delegation (Week 3-4)

#### 2.1 Install and Configure CrewAI

```bash
pip install crewai crewai-tools
```

#### 2.2 Create CrewAI Agent Wrappers

```python
# src/crew/agents.py

from crewai import Agent
from typing import List

class CrewAgentFactory:
    """Factory for creating CrewAI agents from our agent registry."""
    
    REASONING_PROMPTS = {
        "backward": """Start from the target outcome and work backward.
        Identify: 1) What conditions make target achievable? 2) Key milestones?
        3) What could derail us?""",
        
        "step_by_step": """Analyze each factor sequentially, one at a time.
        Build conclusions step by step before synthesizing final view.""",
        
        "example_based": """Identify historical precedents before concluding.
        Compare current situation to past outcomes and derive lessons.""",
        
        "symbolic": """Express analysis using numerical relationships.
        Define variables, relationships, and calculate implied values.""",
        
        "counterfactual": """Consider alternative scenarios.
        What if assumptions change? Test sensitivity.""",
        
        "first_principles": """Decompose to intrinsic value components.
        Build up from fundamental truths rather than analogies."""
    }
    
    @classmethod
    def create_analyst_agent(
        cls,
        role: str,
        goal: str,
        backstory: str,
        reasoning_strategy: str,
        tools: List
    ) -> Agent:
        strategy_prompt = cls.REASONING_PROMPTS.get(reasoning_strategy, "")
        
        return Agent(
            role=role,
            goal=f"{goal}\n\nApply this reasoning approach:\n{strategy_prompt}",
            backstory=backstory,
            tools=tools,
            verbose=True,
            allow_delegation=False,  # Specialists don't delegate
            max_iter=15,
            memory=True
        )
    
    @classmethod
    def create_manager_agent(
        cls,
        role: str,
        goal: str,
        backstory: str,
        available_agents: List[str]
    ) -> Agent:
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            allow_delegation=True,
            max_iter=25,
            memory=True,
            # Manager can delegate to any of these agents
            delegation_targets=available_agents
        )
```

#### 2.3 Create CrewAI Tasks with Dependencies

```python
# src/crew/tasks.py

from crewai import Task
from typing import List, Dict, Any

class TaskFactory:
    """Factory for creating CrewAI tasks with proper context."""
    
    @classmethod
    def create_research_task(
        cls,
        description: str,
        agent: Agent,
        context_tasks: List[Task] = None
    ) -> Task:
        return Task(
            description=description,
            expected_output="Detailed research report with findings and sources",
            agent=agent,
            context=context_tasks or [],
            output_file="research_output.md"
        )
    
    @classmethod
    def create_analysis_task(
        cls,
        description: str,
        agent: Agent,
        input_data: Dict[str, Any]
    ) -> Task:
        return Task(
            description=f"{description}\n\nInput data: {input_data}",
            expected_output="Analysis report with recommendations and confidence score",
            agent=agent,
            context=[]
        )
    
    @classmethod
    def create_synthesis_task(
        cls,
        description: str,
        agent: Agent,
        debate_results: List[Task]
    ) -> Task:
        return Task(
            description=f"""{description}
            
            Synthesize the following perspectives into a coherent recommendation:
            {chr(10).join([f'- Perspective {i+1}' for i in range(len(debate_results))])}
            
            Identify:
            1. Points of consensus
            2. Remaining disagreements
            3. Risk-adjusted recommendation
            """,
            expected_output="Synthesized recommendation with rationale and risk warnings",
            agent=agent,
            context=debate_results
        )
```

#### 2.4 Create Debate Committees

```python
# src/crew/debate.py

from crewai import Crew, Process
from typing import List, Dict, Any

class DebateCommittee:
    """CrewAI-based debate committee for resolving conflicting analyses."""
    
    def __init__(self, bull_agents: List[Agent], bear_agents: List[Agent]):
        self.bull_agents = bull_agents
        self.bear_agents = bear_agents
        self.facilitator = self._create_facilitator()
    
    def _create_facilitator(self) -> Agent:
        return Agent(
            role="Debate Facilitator",
            goal="Ensure productive debate and identify consensus points",
            backstory="""You are an impartial moderator who ensures both 
            perspectives are heard and helps find common ground.""",
            verbose=True,
            allow_delegation=False
        )
    
    def run_debate(
        self,
        topic: str,
        bull_case: str,
        bear_case: str,
        rounds: int = 2
    ) -> Dict[str, Any]:
        """Run a structured debate between bull and bear perspectives."""
        
        debate_tasks = []
        
        # Round 1: Initial arguments
        bull_task = Task(
            description=f"Present the bullish case for: {topic}\nKey points: {bull_case}",
            expected_output="Bull argument with evidence",
            agent=self.bull_agents[0]
        )
        
        bear_task = Task(
            description=f"Present the bearish case for: {topic}\nKey points: {bear_case}",
            expected_output="Bear argument with evidence",
            agent=self.bear_agents[0]
        )
        
        # Round 2: Rebuttals (if multiple rounds)
        if rounds > 1:
            bull_rebuttal = Task(
                description=f"Rebut the bear case. Address: {bear_case}",
                expected_output="Rebuttal with counter-evidence",
                agent=self.bull_agents[0],
                context=[bear_task]
            )
            
            bear_rebuttal = Task(
                description=f"Rebut the bull case. Address: {bull_case}",
                expected_output="Rebuttal with counter-evidence",
                agent=self.bear_agents[0],
                context=[bull_task]
            )
            
            debate_tasks = [bull_task, bear_task, bull_rebuttal, bear_rebuttal]
        else:
            debate_tasks = [bull_task, bear_task]
        
        # Synthesis task
        synthesis_task = Task(
            description=f"""Based on the debate, synthesize:
            1. Points of consensus
            2. Key disagreements
            3. Risk-adjusted view
            4. Recommendation
            
            Debate topic: {topic}
            """,
            expected_output="Synthesized debate outcome",
            agent=self.facilitator,
            context=debate_tasks
        )
        
        crew = Crew(
            agents=self.bull_agents + self.bear_agents + [self.facilitator],
            tasks=debate_tasks + [synthesis_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            "debate_topic": topic,
            "bull_case": bull_case,
            "bear_case": bear_case,
            "synthesis": result,
            "consensus_points": self._extract_consensus(result),
            "disagreements": self._extract_disagreements(result)
        }
```

---

### Phase 3: Hybrid Orchestrator (Week 5-6)

#### 3.1 Create Hybrid Graph

```python
# src/core/hybrid_orchestrator.py

from langgraph.graph import StateGraph, START, END
from crewai import Crew, Process

class HybridOrchestrator:
    """Combines LangGraph orchestration with CrewAI delegation."""
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        self.planner = PlannerAgent(LLMClient())
        self.parallel_executor = ParallelExecutor()
        self.crew_factory = CrewAgentFactory()
        self.graph = self._build_hybrid_graph()
    
    def _node_intent_classify(self, state: HybridState) -> HybridState:
        """Classify intent to determine which agents to activate."""
        intent = self.intent_classifier.classify(
            query=state["query"],
            context=state.get("profile", {})
        )
        
        new = dict(state)
        new["intent"] = intent.intent
        new["agents_to_activate"] = intent.agents
        new["execution_mode"] = "parallel" if len(intent.agents) > 1 else "single"
        return new
    
    def _node_execute_crew(self, state: HybridState) -> HybridState:
        """Execute a CrewAI crew for complex multi-agent tasks."""
        crew_config = self._determine_crew_config(state)
        
        agents = [
            self.crew_factory.create_analyst_agent(
                role=cfg["role"],
                goal=cfg["goal"],
                backstory=cfg["backstory"],
                reasoning_strategy=cfg.get("reasoning_strategy", "step_by_step"),
                tools=cfg.get("tools", [])
            )
            for cfg in crew_config["agents"]
        ]
        
        tasks = [
            self.task_factory.create_analysis_task(
                description=cfg["description"],
                agent=agents[cfg["agent_index"]],
                input_data=state.get("level_outputs", {})
            )
            for cfg in crew_config["tasks"]
        ]
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.hierarchical,  # Manager coordinates
            verbose=True
        )
        
        result = crew.kickoff()
        
        new = dict(state)
        new["crew_result"] = result
        new.setdefault("scratchpad", []).append({
            "event": "crew_executed",
            "agents": [a.role for a in agents]
        })
        return new
    
    def _node_execute_debate(self, state: HybridState) -> HybridState:
        """Execute debate committee for conflicting analyses."""
        # Extract bull and bear cases from parallel execution
        analyses = state.get("level_outputs", {}).get(1, {})
        
        debate = DebateCommittee(
            bull_agents=[self._create_bull_agent()],
            bear_agents=[self._create_bear_agent()]
        )
        
        result = debate.run_debate(
            topic=state["query"],
            bull_case=analyses.get("bull_perspective", ""),
            bear_case=analyses.get("bear_perspective", "")
        )
        
        new = dict(state)
        new["debate_result"] = result
        return new
    
    def _node_synthesize(self, state: HybridState) -> HybridState:
        """Final synthesis of all results."""
        synthesis_prompt = self._build_synthesis_prompt(state)
        
        response = self.llm_client.get_chat_model(
            [{"role": "system", "content": synthesis_prompt}],
            temperature=0.5
        )
        
        new = dict(state)
        new["final_response"] = response
        return new
    
    def _should_run_crew(self, state: HybridState) -> str:
        """Determine if CrewAI delegation is needed."""
        intent = state.get("intent", "")
        complexity = state.get("complexity_score", 0)
        
        if intent in ["comprehensive_review", "portfolio_analysis"]:
            return "crew"
        elif complexity >= 4:
            return "crew"
        elif state.get("execution_mode") == "parallel":
            return "parallel"
        else:
            return "single"
    
    def _should_debate(self, state: HybridState) -> str:
        """Determine if debate is needed."""
        level_outputs = state.get("level_outputs", {})
        
        # Check for conflicting signals in analysis level
        if 1 in level_outputs:
            analyses = level_outputs[1].get("results", {})
            
            # Check if technical and fundamental disagree
            tech_signal = analyses.get("technical", {}).get("signal", "neutral")
            fund_signal = analyses.get("fundamental", {}).get("signal", "neutral")
            
            if tech_signal != fund_signal:
                return "debate"
        
        return "synthesize"
    
    def _build_hybrid_graph(self) -> StateGraph:
        g = StateGraph(HybridState)
        
        # Add nodes
        g.add_node("intent_classify", self._node_intent_classify)
        g.add_node("prepare_data", self._node_prepare_data)
        g.add_node("execute_parallel", self._node_execute_level)
        g.add_node("execute_crew", self._node_execute_crew)
        g.add_node("debate", self._node_execute_debate)
        g.add_node("synthesize", self._node_synthesize)
        
        # Add edges
        g.add_edge(START, "intent_classify")
        g.add_edge("intent_classify", "prepare_data")
        
        # Conditional routing after data prep
        g.add_conditional_edges(
            "prepare_data",
            self._should_run_crew,
            {
                "crew": "execute_crew",
                "parallel": "execute_parallel",
                "single": "synthesize"
            }
        )
        
        # After parallel execution, check for debate
        g.add_conditional_edges(
            "execute_parallel",
            self._should_debate,
            {
                "debate": "debate",
                "synthesize": "synthesize"
            }
        )
        
        g.add_edge("execute_crew", "synthesize")
        g.add_edge("debate", "synthesize")
        g.add_edge("synthesize", END)
        
        return g.compile()
```

---

### Phase 4: State Propagation and Belief System (Week 7-8)

#### 4.1 Structured State Schema

```python
# src/core/schemas.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class Level0Output(BaseModel):
    """Structured output from data gathering level."""
    portfolio: Optional[Dict[str, Any]]
    market_data: Dict[str, Any]
    news: List[Dict[str, Any]]
    data_quality_score: float = Field(ge=0, le=1)
    missing_data: List[str] = Field(default_factory=list)

class Level1Output(BaseModel):
    """Structured output from analysis level."""
    technical_analysis: Dict[str, Any]
    fundamental_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    cross_analysis_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    consensus_signal: str = Field(description="bullish, bearish, or neutral")
    confidence: float = Field(ge=0, le=1)

class Level2Output(BaseModel):
    """Structured output from debate level."""
    bull_case: Dict[str, Any]
    bear_case: Dict[str, Any]
    consensus_points: List[str]
    remaining_disagreements: List[str]
    risk_adjusted_view: str
    final_recommendation: str

class FinalResponse(BaseModel):
    """Final response to user."""
    advice: str
    action_items: List[str]
    rationale: str
    confidence: float
    sources: List[str]
    risk_warnings: List[str]
    disclaimers: List[str]

class Belief(BaseModel):
    """Learned belief from execution reflection."""
    key: str
    value: str
    confidence: float = Field(ge=0, le=1)
    source: str
    context: Dict[str, Any]
    created_at: datetime
    usage_count: int = 0
    success_rate: float = 0.5
```

#### 4.2 Belief Propagator

```python
# src/core/belief_propagator.py

from typing import Dict, List, Any
from datetime import datetime

class BeliefPropagator:
    """Propagates learned beliefs across execution levels."""
    
    def __init__(self):
        self.beliefs: Dict[str, Belief] = {}
        self.reflector = SharedReflector()
    
    def inject_beliefs(self, agent_prompt: str, agent_role: str) -> str:
        """Inject relevant beliefs into agent prompt."""
        relevant_beliefs = self._get_relevant_beliefs(agent_role)
        
        if not relevant_beliefs:
            return agent_prompt
        
        belief_context = "\n".join([
            f"- {b.key}: {b.value} (confidence: {b.confidence:.0%})"
            for b in relevant_beliefs
        ])
        
        return f"{agent_prompt}\n\nRelevant learned beliefs:\n{belief_context}"
    
    async def update_beliefs(
        self,
        execution_trace: List[Dict],
        outcomes: Dict[str, Any]
    ):
        """Update beliefs based on execution outcomes."""
        reflection = await self.reflector.reflect(
            trace=execution_trace,
            outcomes=outcomes
        )
        
        for learning in reflection.conceptual_learnings:
            key = self._extract_key(learning)
            
            if key in self.beliefs:
                # Update existing belief
                self.beliefs[key].merge(learning)
            else:
                # Create new belief
                self.beliefs[key] = Belief(
                    key=key,
                    value=learning.get("value"),
                    confidence=learning.get("confidence", 0.5),
                    source=learning.get("source"),
                    context=learning.get("context", {}),
                    created_at=datetime.now()
                )
    
    def _get_relevant_beliefs(self, agent_role: str) -> List[Belief]:
        """Get beliefs relevant to a specific agent role."""
        role_keywords = {
            "technical_analyst": ["technical", "price", "trend", "volume"],
            "fundamental_analyst": ["fundamental", "earnings", "valuation"],
            "sentiment_analyst": ["sentiment", "news", "social"],
            "risk_controller": ["risk", "volatility", "drawdown"],
            "portfolio_manager": ["allocation", "diversification", "rebalancing"]
        }
        
        keywords = role_keywords.get(agent_role, [])
        relevant = []
        
        for belief in self.beliefs.values():
            if any(kw in belief.key.lower() for kw in keywords):
                relevant.append(belief)
        
        return sorted(relevant, key=lambda b: b.confidence, reverse=True)[:5]
```

---

### Phase 5: Complexity-Based Scaling (Week 9-10)

#### 5.1 Complexity Scorer

```python
# src/core/complexity_scorer.py

import re
from typing import Dict, Any, List

class ComplexityScorer:
    """Scores query complexity to determine execution depth."""
    
    def __init__(self):
        self.complex_keywords = [
            "compare", "correlation", "scenario", "optimize",
            "risk-adjusted", "portfolio", "allocation", "diversify",
            "comprehensive", "holistic", "multiple"
        ]
        
        self.multi_asset_patterns = [
            r"\b[\d]+\s*(stocks?|funds?|assets?)\b",
            r"\b(portfolio|holdings)\b",
            r"\band\b.*\b(stocks?|funds?)\b"
        ]
    
    def score(self, query: str, context: Dict[str, Any]) -> ComplexityScore:
        """Calculate complexity score and determine execution level."""
        score = 0
        
        # Check for multiple assets
        symbols = self._extract_symbols(query)
        if len(symbols) > 3:
            score += 2
        elif len(symbols) > 1:
            score += 1
        
        # Check for complex keywords
        query_lower = query.lower()
        score += sum(1 for kw in self.complex_keywords if kw in query_lower)
        
        # Check portfolio value
        portfolio_value = context.get("portfolio_value", 0)
        if portfolio_value > 1_000_000:
            score += 1
        
        # Check time horizon
        long_term_keywords = ["long-term", "retirement", "5 year", "decade"]
        if any(kw in query_lower for kw in long_term_keywords):
            score += 1
        
        # Determine execution level
        if score >= 5:
            level = 4  # Full pipeline + debate + risk review
            agents = self._all_agents()
        elif score >= 3:
            level = 3  # Include debate
            agents = self._core_agents() + ["debate_committee"]
        else:
            level = 2  # Analysis only
            agents = self._core_agents()
        
        return ComplexityScore(
            raw_score=score,
            execution_level=level,
            agents_to_activate=agents,
            estimated_latency=self._estimate_latency(level, len(agents))
        )
    
    def _extract_symbols(self, query: str) -> List[str]:
        """Extract stock symbols from query."""
        # Common stock patterns
        patterns = [
            r"\b[A-Z]{2,5}\b",  # Stock symbols like AAPL, GOOGL
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"  # Company names
        ]
        
        symbols = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            symbols.extend(matches)
        
        return list(set(symbols))
    
    def _estimate_latency(self, level: int, agent_count: int) -> float:
        """Estimate execution latency in seconds."""
        base_latency = {
            2: 5,   # Analysis only
            3: 15,  # + Debate
            4: 25   # + Full pipeline
        }
        
        return base_latency.get(level, 10) * (agent_count / 5)
```

---

## Architecture Improvements Summary

### 1. Parallel Execution
- **Before**: Sequential agent execution
- **After**: Level-based parallel execution with barriers
- **Impact**: 2-4x latency reduction for multi-agent queries

### 2. Task Delegation via CrewAI
- **Before**: No agent-to-agent communication
- **After**: Agents can delegate sub-tasks via CrewAI
- **Impact**: Complex queries handled by specialized crews

### 3. Debate Mechanisms
- **Before**: No conflict resolution
- **After**: Structured debate committees with facilitation
- **Impact**: Better synthesis of conflicting analyses

### 4. Reasoning Strategy Assignment
- **Before**: All agents use default reasoning
- **After**: Each agent assigned unique reasoning approach
- **Impact**: Reduced herding, diverse perspectives

### 5. Belief Propagation
- **Before**: No learning across sessions
- **After**: Beliefs stored and injected into agent prompts
- **Impact**: Improved consistency and reduced hallucinations

### 6. Complexity-Based Scaling
- **Before**: All queries processed identically
- **After**: Execution depth based on query complexity
- **Impact**: Cost optimization for simple queries

---

## Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Parallel Execution | `ParallelExecutor`, updated `GraphState`, level-based nodes |
| 3-4 | CrewAI Integration | `CrewAgentFactory`, `TaskFactory`, `DebateCommittee` |
| 5-6 | Hybrid Orchestrator | `HybridOrchestrator`, conditional routing, crew/parallel selection |
| 7-8 | State & Beliefs | Structured schemas, `BeliefPropagator`, state persistence |
| 9-10 | Complexity Scaling | `ComplexityScorer`, dynamic agent activation, latency estimation |

---

## Testing Strategy

### Unit Tests
- `test_parallel_executor.py`: Verify concurrent execution and aggregation
- `test_crew_factory.py`: Validate agent creation with reasoning strategies
- `test_debate_committee.py`: Test debate flow and consensus extraction
- `test_complexity_scorer.py`: Verify scoring accuracy

### Integration Tests
- `test_hybrid_flow.py`: End-to-end flow with all components
- `test_belief_propagation.py`: Multi-session belief persistence

### Benchmarks
- Compare latency: Sequential vs Parallel vs Hybrid
- Compare quality: Single agent vs Crew vs Debate
- Compare cost: Full pipeline vs Complexity-scaled

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| CrewAI + LangGraph integration complexity | Start with simple crew delegation, expand gradually |
| Parallel execution race conditions | Use asyncio.Semaphore for rate limiting, test thoroughly |
| Debate committee token costs | Limit rounds to 2, use efficient summarization |
| Belief propagation memory growth | Implement TTL and confidence-based pruning |
| Complexity scoring accuracy | Start with conservative thresholds, tune based on feedback |

---

## Conclusion

This hybrid architecture combines the best of both frameworks:

1. **LangGraph** provides robust orchestration, state management, and conditional routing
2. **CrewAI** enables role-based collaboration, task delegation, and debate mechanisms

The result is a production-ready, scalable multi-agent financial advisory system that can:
- Execute agents in parallel for efficiency
- Delegate complex tasks to specialized crews
- Resolve conflicting analyses through structured debate
- Learn and improve through belief propagation
- Scale resources based on query complexity
