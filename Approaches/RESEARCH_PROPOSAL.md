# Research Proposal: Hierarchical Parallel Multi-Agent System for Financial Advisory (HP-MAS)

## Novel Architecture Combining MacNet DAG Scaling, TradingAgents Firm Hierarchy, and DMAD Diverse Reasoning

---

## Table of Contents

1. [Research Motivation](#research-motivation)
2. [Gap Analysis](#gap-analysis)
3. [Novel Contributions](#novel-contributions)
4. [Proposed Architecture: HP-MAS](#proposed-architecture-hp-mas)
5. [Research Questions](#research-questions)
6. [Methodology](#methodology)
7. [Evaluation Framework](#evaluation-framework)
8. [Implementation Plan](#implementation-plan)
9. [Expected Outcomes](#expected-outcomes)
10. [Timeline](#timeline)
11. [References](#references)

---

## Research Motivation

### The Financial Advisory Challenge

Financial advisory systems require:
1. **Multi-perspective analysis** - Fundamental, technical, sentiment, macro factors
2. **Real-time decision making** - Market conditions change rapidly
3. **Risk-aware synthesis** - Different risk profiles need different advice
4. **Explainability** - Regulatory requirements for audit trails
5. **Parallel execution** - Independent analyses should run concurrently

### Current State-of-the-Art Limitations

Based on the comprehensive review of A* conference papers (2024-2025):

| System | Strengths | Limitations |
|--------|-----------|-------------|
| **MacNet (ICLR-25)** | Scales to 1000+ agents, DAG-based, collaborative scaling law | No domain specialization, flat agent roles |
| **TradingAgents (AAAI-25)** | Firm hierarchy, structured reports, debate submodules | Sequential workflow, no parallel agent execution |
| **FinCon (NeurIPS-24)** | Risk management, conceptual verbal reinforcement | Manager bottleneck, synchronous execution |
| **DMAD (ICLR-25)** | Diverse reasoning strategies, faster convergence | Debate-only, no task execution |

### Key Insight

**No existing system combines:**
1. DAG-based parallel execution (MacNet)
2. Firm-inspired hierarchical roles (TradingAgents)
3. Diverse reasoning strategies (DMAD)
4. Structured communication with shared state

---

## Gap Analysis

### Gap 1: Parallel vs. Hierarchical Trade-off

**Problem:** 
- MacNet enables parallel execution but lacks domain-specific hierarchies
- TradingAgents has firm hierarchy but executes sequentially

**Opportunity:**
Hybrid architecture where agents within each organizational level execute in parallel, but levels themselves have hierarchical dependencies.

### Gap 2: Debate Integration

**Problem:**
- MAD systems debate globally (high token cost, herding issues)
- TradingAgents localizes debate but uses symmetric agents

**Opportunity:**
Inject DMAD's diverse reasoning strategies into localized debate committees within a hierarchical structure.

### Gap 3: Scaling with Quality

**Problem:**
- MacNet shows performance follows logistic growth with agent count
- But more agents ≠ better advice if they all reason similarly

**Opportunity:**
Scale agent count while enforcing reasoning diversity through explicit strategy assignment.

### Gap 4: Dynamic Agent Activation

**Problem:**
- All surveyed systems use static agent sets
- Financial queries vary in complexity - simple queries shouldn't activate full system

**Opportunity:**
Intent-based dynamic agent activation that scales resources based on query complexity.

---

## Novel Contributions

### Contribution 1: Hierarchical Parallel DAG (HP-DAG)

Novel graph structure that combines organizational hierarchy with parallel execution:

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    ORCHESTRATION LAYER                       │
                    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
                    │  │   Intent    │─▶│  Complexity │─▶│    Plan     │         │
                    │  │  Classifier │  │   Scorer    │  │  Generator  │         │
                    │  └─────────────┘  └─────────────┘  └──────┬──────┘         │
                    └───────────────────────────────────────────┼─────────────────┘
                                                              │
                    ┌─────────────────────────────────────────┼─────────────────┐
                    │                    LEVEL 0: DATA GATHERING               │
                    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
                    │  │ Upstox   │ │ WebScraper│ │ NewsAPI  │ │ FilingAPI│     │
                    │  │ (async)  │ │ (async)  │ │ (async)  │ │ (async)  │     │
                    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘     │
                    │       └────────────┴────────────┴────────────┘            │
                    │                           │                               │
                    └───────────────────────────┼───────────────────────────────┘
                                                │ BARRIER (wait for all)
                                                ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │                    LEVEL 1: SPECIALIZED ANALYSIS           │
                    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
                    │  │Technical │ │Fundamental│ │Sentiment │ │  Macro   │     │
                    │  │Analyst   │ │ Analyst  │ │ Analyst  │ │ Analyst  │     │
                    │  │[Reasoning│ │[Reasoning│ │[Reasoning│ │[Reasoning│     │
                    │  │  Style A]│ │  Style B]│ │  Style C]│ │  Style D]│     │
                    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘     │
                    │       └────────────┴────────────┴────────────┘            │
                    │                           │                               │
                    └───────────────────────────┼───────────────────────────────┘
                                                │ BARRIER
                                                ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │                    LEVEL 2: DEBATE COMMITTEES              │
                    │         ┌───────────────────┬───────────────────┐         │
                    │         │   BULL COMMITTEE  │   BEAR COMMITTEE  │         │
                    │         │ ┌───────┐ ┌─────┐ │ ┌───────┐ ┌─────┐ │         │
                    │         │ │Growth │ │Value│ │ │RiskOff│ │Short│ │         │
                    │         │ │Analyst│ │Strat│ │ │Strat  │ │Strat│ │         │
                    │         │ └───┬───┘ └──┬──┘ │ └───┬───┘ └──┬──┘ │         │
                    │         │     └────┬───┘    │     └────┬───┘    │         │
                    │         │          │Facilit.│          │Facilit.│         │
                    │         │          └───┬────┘          └───┬────┘         │
                    │         └──────────────┼───────────────────┼─────────────┘
                    │                        │                   │               │
                    └────────────────────────┼───────────────────┼───────────────┘
                                             └─────────┬─────────┘
                                                       │
                                                       ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │                    LEVEL 3: DECISION SYNTHESIS             │
                    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
                    │  │   Portfolio  │─▶│     Risk     │─▶│    Fund      │    │
                    │  │   Manager    │  │   Controller │  │   Manager    │    │
                    │  └──────────────┘  └──────────────┘  └──────────────┘    │
                    └───────────────────────────────────────────────────────────┘
```

**Key Innovation:** Each level executes all agents in parallel, but levels are sequential. This maximizes throughput while respecting dependencies.

### Contribution 2: Reasoning Strategy Assignment (RSA)

Each specialized analyst is assigned a unique reasoning strategy inspired by DMAD:

| Agent | Reasoning Strategy | Prompt Injection |
|-------|-------------------|------------------|
| Technical Analyst | **Backward Reasoning** | "Start from the target price, work backward to current levels" |
| Fundamental Analyst | **Step-by-Step CoT** | "Analyze each financial metric sequentially" |
| Sentiment Analyst | **Example-Based** | "Compare current sentiment to historical precedents" |
| Macro Analyst | **Symbolic/Quantitative** | "Use numerical relationships and economic formulas" |
| Growth Analyst | **Counterfactual** | "What if growth exceeds/exceeds expectations?" |
| Value Analyst | **First-Principles** | "Decompose to intrinsic value components" |

**Novelty:** Unlike DMAD which applies diverse reasoning to debate agents, we apply it to task-oriented specialists.

### Contribution 3: Structured State Propagation (SSP)

Instead of natural language chat between agents, use typed schemas:

```python
class Level0Output(BaseModel):
    """Structured output from data gathering level"""
    portfolio: Optional[PortfolioData]
    market_data: Dict[str, PriceData]
    news: List[NewsItem]
    filings: List[FilingData]
    data_quality_score: float
    missing_data: List[str]

class Level1Output(BaseModel):
    """Structured output from analysis level"""
    technical_analysis: TechnicalAnalysisSchema
    fundamental_analysis: FundamentalAnalysisSchema
    sentiment_analysis: SentimentAnalysisSchema
    macro_analysis: MacroAnalysisSchema
    cross_analysis_conflicts: List[ConflictItem]

class Level2Output(BaseModel):
    """Structured output from debate level"""
    bull_case: BullCaseSchema
    bear_case: BearCaseSchema
    consensus_points: List[str]
    remaining_disagreements: List[str]
    risk_adjusted_view: str
```

**Novelty:** Combines TradingAgents' structured reports with MacNet's DAG propagation.

### Contribution 4: Adaptive Complexity Scaling (ACS)

Dynamically adjust system depth based on query complexity:

```python
class ComplexityScorer:
    """
    Scores query complexity to determine execution depth.
    """
    
    def score(self, query: str, context: dict) -> ComplexityScore:
        score = 0
        
        # Multi-asset mentions
        if len(extract_symbols(query)) > 3:
            score += 2
        
        # Complex reasoning keywords
        complex_keywords = ["compare", "correlation", "scenario", "optimize", "risk-adjusted"]
        score += sum(1 for k in complex_keywords if k in query.lower())
        
        # Portfolio context
        if context.get("portfolio_value", 0) > 1_000_000:
            score += 1
        
        # Time horizon mentions
        if any(t in query.lower() for t in ["long-term", "retirement", "5 year"]):
            score += 1
        
        return ComplexityScore(
            raw_score=score,
            level=self._determine_level(score),
            agents_to_activate=self._select_agents(score)
        )
    
    def _determine_level(self, score: int) -> int:
        """
        Level 0-2: Data + Analysis (parallel)
        Level 0-3: + Debate (for complex queries)
        Level 0-4: + Risk Review (for high stakes)
        """
        if score >= 5:
            return 4  # Full pipeline including fund manager
        elif score >= 3:
            return 3  # Include debate
        else:
            return 2  # Analysis only
```

### Contribution 5: Belief Propagation with Reflection (BPR)

Inspired by FinCon's conceptual verbal reinforcement and COPPER's shared reflector:

```python
class BeliefPropagator:
    """
    Propagates learned beliefs across levels.
    Combines FinCon's verbal reinforcement with COPPER's shared reflection.
    """
    
    def __init__(self):
        self.beliefs: Dict[str, Belief] = {}
        self.reflector = SharedReflector()
    
    async def update_beliefs(
        self, 
        execution_trace: ExecutionTrace,
        outcomes: Dict[str, Any]
    ):
        # Generate reflection on execution
        reflection = await self.reflector.reflect(
            trace=execution_trace,
            outcomes=outcomes
        )
        
        # Extract conceptual beliefs
        new_beliefs = self._extract_concepts(reflection)
        
        # Update belief store
        for belief in new_beliefs:
            if belief.key in self.beliefs:
                self.beliefs[belief.key].merge(belief)
            else:
                self.beliefs[belief.key] = belief
    
    def inject_beliefs(self, agent_prompt: str, agent_role: str) -> str:
        """
        Inject relevant beliefs into agent prompt.
        Similar to FinCon's conceptual verbal reinforcement.
        """
        relevant_beliefs = self._get_relevant_beliefs(agent_role)
        if relevant_beliefs:
            belief_context = "\n".join([
                f"- {b.key}: {b.value} (confidence: {b.confidence})"
                for b in relevant_beliefs
            ])
            return f"{agent_prompt}\n\nRelevant learned beliefs:\n{belief_context}"
        return agent_prompt


class SharedReflector:
    """
    Single shared reflector that generates reflections for all agents.
    Inspired by COPPER's counterfactual PPO approach.
    """
    
    async def reflect(
        self, 
        trace: ExecutionTrace,
        outcomes: Dict[str, Any]
    ) -> Reflection:
        prompt = self._build_reflection_prompt(trace, outcomes)
        
        reflection = await self.llm.generate(
            prompt,
            response_schema=ReflectionSchema
        )
        
        return Reflection(
            success_factors=reflection.success_factors,
            improvement_areas=reflection.improvement_areas,
            conceptual_learnings=reflection.conceptual_learnings,
            counterfactuals=reflection.counterfactuals
        )
```

---

## Proposed Architecture: HP-MAS

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              HP-MAS ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         ENTRY LAYER                                      │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │   │
│  │  │   Query    │─▶│  Intent    │─▶│ Complexity │─▶│   Plan     │       │   │
│  │  │   Parser   │  │ Classifier │  │   Scorer   │  │  Generator │       │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └─────┬──────┘       │   │
│  └───────────────────────────────────────────────────────┼─────────────────┘   │
│                                                          │                     │
│  ┌───────────────────────────────────────────────────────┼─────────────────┐   │
│  │                    EXECUTION ENGINE                     │                 │   │
│  │  ┌─────────────────────────────────────────────────────┴───────┐       │   │
│  │  │                 DAG EXECUTOR                                  │       │   │
│  │  │                                                               │       │   │
│  │  │   LEVEL 0          LEVEL 1          LEVEL 2       LEVEL 3   │       │   │
│  │  │  ┌─────┐         ┌─────┐         ┌─────┐       ┌─────┐    │       │   │
│  │  │  │ D1  │──┐      │ A1  │──┐      │ B1  │──┐    │ PM  │    │       │   │
│  │  │  │ D2  │  ├─────▶│ A2  │  ├─────▶│ B2  │  ├───▶│ RC  │    │       │   │
│  │  │  │ D3  │  │      │ A3  │  │      │ B3  │  │    │ FM  │    │       │   │
│  │  │  │ D4  │──┘      │ A4  │──┘      └─────┘──┘    └─────┘    │       │   │
│  │  │  ───────         ───────         ───────       ───────    │       │   │
│  │  │  PARALLEL        PARALLEL        PARALLEL      SEQUENTIAL │       │   │
│  │  │                                                               │       │   │
│  │  │  Each agent has:                                             │       │   │
│  │  │  - Assigned reasoning strategy                               │       │   │
│  │  │  - Input schema from previous level                          │       │   │
│  │  │  - Output schema for next level                             │       │   │
│  │  │                                                               │       │   │
│  │  └───────────────────────────────────────────────────────────────┘       │   │
│  │                                                          │               │   │
│  │  ┌───────────────────────────────────────────────────────┼───────────────┐│   │
│  │  │                 BELIEF PROPAGATOR                       │              ││   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │              ││   │
│  │  │  │   Shared    │◀─│   Concept   │◀─│  Reflection │◀───┘              ││   │
│  │  │  │  Reflector  │  │  Extractor  │  │   Generator │                   ││   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘                    ││   │
│  │  │           │                                                       ││   │
│  │  │           ▼ (inject beliefs into agent prompts)                   ││   │
│  │  └───────────────────────────────────────────────────────────────────────┘│   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                         STATE MANAGEMENT                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Session   │  │  Execution  │  │   Belief    │  │    Audit    │     │   │
│  │  │   Store     │  │   Trace     │  │   Store     │  │    Log      │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Formal Definition

**Definition 1 (HP-DAG):** A Hierarchical Parallel DAG is a tuple $G = (V, E, L, \lambda)$ where:
- $V$ is a set of agents
- $E \subseteq V \times V$ is a set of directed edges representing data flow
- $L = \{l_0, l_1, ..., l_k\}$ is a set of levels
- $\lambda: V \rightarrow L$ assigns each agent to a level
- For all $(u, v) \in E$: $\lambda(u) < \lambda(v)$ (edges only go from lower to higher levels)
- Agents at the same level have no edges between them (parallelizable)

**Definition 2 (Reasoning Strategy Assignment):** A function $\rho: V \rightarrow \mathcal{R}$ where $\mathcal{R} = \{r_1, ..., r_m\}$ is a set of reasoning strategies, such that for any level $l$: $\forall u, v \in \lambda^{-1}(l): u \neq v \Rightarrow \rho(u) \neq \rho(v)$ (agents at same level have different strategies).

**Definition 3 (Structured State Propagation):** For each level $l_i$, the input is a typed schema $S_i$ and output is a typed schema $S_{i+1}$. The transition function $\tau_i: S_i \times V_{l_i} \rightarrow S_{i+1}$ processes all agents at level $l_i$ in parallel.

**Definition 4 (Belief Propagation):** A belief store $B$ is updated after each execution via reflection function $R: \text{Trace} \times \text{Outcome} \rightarrow \Delta B$, where $\Delta B$ represents belief updates. Beliefs are injected into agent prompts via injection function $\iota: \text{Prompt} \times B_{relevant} \rightarrow \text{Prompt}'$.

### Execution Algorithm

```python
async def execute_hp_mas(
    query: str, 
    context: dict,
    user_profile: UserProfile
) -> FinancialAdvice:
    
    # Step 1: Intent Classification
    intent = await intent_classifier.classify(query, context)
    
    # Step 2: Complexity Scoring
    complexity = await complexity_scorer.score(query, context)
    
    # Step 3: Plan Generation
    plan = await plan_generator.generate(
        intent=intent,
        complexity=complexity,
        available_agents=AGENT_REGISTRY
    )
    
    # Step 4: Initialize State
    state = GlobalState(
        query=query,
        context=context,
        user_profile=user_profile,
        level_outputs={}
    )
    
    # Step 5: Execute Levels
    for level in plan.levels:
        # Inject beliefs into agent prompts
        agents_at_level = plan.get_agents_at_level(level)
        for agent in agents_at_level:
            agent.prompt = belief_propagator.inject_beliefs(
                agent.prompt, 
                agent.role
            )
        
        # Execute agents in parallel
        level_output = await dag_executor.execute_level(
            level=level,
            agents=agents_at_level,
            input_schema=state.current_schema,
            barrier=True  # Wait for all agents at this level
        )
        
        # Update state
        state.level_outputs[level] = level_output
        
        # Update beliefs (COPPER-style reflection)
        if level > 0:  # Skip data gathering level
            await belief_propagator.update_beliefs(
                execution_trace=state.get_trace(),
                outcomes=level_output
            )
    
    # Step 6: Generate Final Response
    response = await response_generator.generate(
        level_outputs=state.level_outputs,
        user_profile=user_profile,
        beliefs=belief_propagator.get_relevant_beliefs("advisor")
    )
    
    return response
```

---

## Research Questions

### RQ1: Parallel Execution Efficiency

**Question:** How does parallel execution within hierarchical levels compare to sequential execution in terms of latency and quality?

**Hypothesis:** Parallel execution reduces latency by factor of $\frac{n}{\text{levels}}$ where $n$ is the number of agents, while maintaining or improving quality due to independent reasoning paths.

### RQ2: Reasoning Diversity Impact

**Question:** Does assigning diverse reasoning strategies to parallel agents improve synthesis quality compared to homogeneous agents?

**Hypothesis:** Diverse reasoning strategies reduce "herding" to consensus on incorrect answers (as observed in MAD) and provide richer inputs for synthesis.

### RQ3: Complexity-Based Scaling

**Question:** Does adaptive complexity scaling reduce computational cost without significantly impacting advice quality for simple queries?

**Hypothesis:** Simple queries can be processed at 40-60% lower cost with <5% quality degradation compared to full pipeline execution.

### RQ4: Belief Propagation Effectiveness

**Question:** Does belief propagation across sessions improve consistency and reduce hallucinations in multi-turn financial conversations?

**Hypothesis:** Belief propagation reduces contradiction rate by 30-50% in multi-turn conversations compared to stateless execution.

### RQ5: Scalability Limits

**Question:** What are the practical scalability limits of HP-MAS as the number of agents increases?

**Hypothesis:** Following MacNet's collaborative scaling law, performance follows logistic growth, but with hierarchical parallelization, the emergence phase occurs earlier due to level-based synchronization.

---

## Methodology

### Experimental Design

#### Experiment 1: Latency Comparison

**Setup:**
- Compare HP-MAS (parallel) vs. TradingAgents-style (sequential) execution
- Measure end-to-end latency for queries of varying complexity
- Control for token usage (same prompts, different execution order)

**Metrics:**
- Time to first token
- Total execution time
- Parallel efficiency: $\frac{T_{sequential}}{T_{parallel} \times n_{agents}}$

#### Experiment 2: Quality Evaluation

**Setup:**
- Financial advice benchmark (adapted from FinCon evaluation)
- Human expert evaluation on 100 test cases
- A/B comparison between homogeneous and diverse reasoning agents

**Metrics:**
- Advice quality score (1-5 Likert scale)
- Factual accuracy (checked against market data)
- Actionability rating
- Risk appropriateness for user profile

#### Experiment 3: Complexity Scaling

**Setup:**
- Route queries through different pipeline depths
- Measure cost (token usage, API calls) and quality
- Plot quality-cost trade-off curve

**Metrics:**
- Quality drop per level skipped
- Cost reduction per level skipped
- Optimal depth for query complexity

#### Experiment 4: Belief Propagation

**Setup:**
- Multi-turn conversation simulation
- Compare stateless vs. belief-propagating versions
- Measure consistency across turns

**Metrics:**
- Contradiction rate (percentage of responses contradicting previous)
- Advice consistency score
- Hallucination rate (claims not supported by data)

### Datasets

1. **Financial QA Dataset** - Curated from financial forums, expert Q&A
2. **Portfolio Analysis Cases** - Historical portfolio with known outcomes
3. **Market Event Reactions** - How advice changes with market events
4. **Multi-turn Conversations** - Simulated user conversations

### Baselines

1. **Single-Agent Baseline** - GPT-4 with all tools
2. **Sequential Pipeline** - TradingAgents-style without parallelization
3. **Flat MAD** - Multi-agent debate without hierarchy
4. **MacNet** - DAG execution without reasoning diversity

---

## Evaluation Framework

### Quantitative Metrics

```python
class EvaluationMetrics:
    """Comprehensive evaluation metrics for HP-MAS"""
    
    # Latency Metrics
    time_to_first_token: float
    total_execution_time: float
    parallel_efficiency: float  # T_sequential / (T_parallel * n_agents)
    
    # Quality Metrics
    advice_quality_score: float  # Expert rating 1-5
    factual_accuracy: float  # % of claims verifiable
    actionability_score: float  # % of advice actionable
    risk_appropriateness: float  # Match with user risk profile
    
    # Consistency Metrics
    contradiction_rate: float  # % responses contradicting previous
    belief_stability: float  # How stable are beliefs over time
    
    # Efficiency Metrics
    token_efficiency: float  # Quality / tokens used
    cost_per_quality_point: float
    agent_utilization: float  # % of agents contributing meaningfully
    
    # Scalability Metrics
    quality_per_agent: float
    marginal_benefit: float  # Quality gain per additional agent
```

### Qualitative Evaluation

**Expert Review Protocol:**
1. Present advice from HP-MAS and baselines (blind)
2. Expert rates each on multiple dimensions
3. Expert identifies which advice they would follow
4. Expert provides free-text feedback

**User Study Protocol:**
1. Real users interact with system
2. Measure satisfaction, trust, perceived usefulness
3. Post-interaction survey on advice quality

---

## Implementation Plan

### Phase 1: Core Framework (Weeks 1-4)

```python
# Core abstractions

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, Any, List
from pydantic import BaseModel
import asyncio

# Type variables for schemas
InputSchema = TypeVar('InputSchema', bound=BaseModel)
OutputSchema = TypeVar('OutputSchema', bound=BaseModel)

class Agent(ABC, Generic[InputSchema, OutputSchema]):
    """Base agent with assigned reasoning strategy"""
    
    def __init__(
        self,
        name: str,
        role: str,
        reasoning_strategy: ReasoningStrategy,
        tools: List[Tool]
    ):
        self.name = name
        self.role = role
        self.reasoning_strategy = reasoning_strategy
        self.tools = tools
        self.prompt_template = self._build_prompt()
    
    @abstractmethod
    def _build_prompt(self) -> str:
        """Build prompt with reasoning strategy injection"""
        pass
    
    @abstractmethod
    async def run(self, input: InputSchema) -> OutputSchema:
        """Execute agent logic"""
        pass
    
    def get_required_inputs(self) -> List[str]:
        return self._input_fields
    
    def get_outputs(self) -> List[str]:
        return self._output_fields


class Level(BaseModel):
    """A level in the HP-DAG"""
    
    id: int
    agents: List[Agent]
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    execution_mode: Literal["parallel", "debate", "sequential"]


class HPDAG:
    """Hierarchical Parallel DAG executor"""
    
    def __init__(self, levels: List[Level]):
        self.levels = levels
        self.belief_propagator = BeliefPropagator()
    
    async def execute(
        self, 
        initial_input: BaseModel,
        complexity_level: int
    ) -> Dict[int, BaseModel]:
        """Execute all levels up to complexity_level"""
        
        results = {}
        current_input = initial_input
        
        for level in self.levels[:complexity_level]:
            # Inject beliefs
            for agent in level.agents:
                agent.inject_beliefs(
                    self.belief_propagator.get_relevant_beliefs(agent.role)
                )
            
            # Execute level
            if level.execution_mode == "parallel":
                result = await self._execute_parallel(level, current_input)
            elif level.execution_mode == "debate":
                result = await self._execute_debate(level, current_input)
            else:
                result = await self._execute_sequential(level, current_input)
            
            results[level.id] = result
            current_input = result
            
            # Update beliefs
            await self.belief_propagator.update(
                trace=self._get_trace(results),
                outcome=result
            )
        
        return results
    
    async def _execute_parallel(
        self, 
        level: Level, 
        input: BaseModel
    ) -> BaseModel:
        """Execute all agents at level concurrently"""
        
        tasks = [
            agent.run(input) 
            for agent in level.agents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results into output schema
        return self._aggregate(results, level.output_schema)
```

### Phase 2: Agent Implementations (Weeks 5-8)

**Priority Order:**
1. Data Gathering Agents (Upstox, Market Data, News)
2. Analysis Agents (Technical, Fundamental, Sentiment)
3. Debate Agents (Bull/Bear committees)
4. Synthesis Agents (Portfolio Manager, Risk Controller)

### Phase 3: Integration (Weeks 9-10)

- Connect to real data sources
- Implement belief store persistence
- Add API layer

### Phase 4: Evaluation (Weeks 11-12)

- Run experiments
- Collect metrics
- Analyze results

---

## Expected Outcomes

### Primary Outcomes

1. **Novel Architecture Contribution**
   - HP-MAS: First system combining hierarchical parallelism with reasoning diversity
   - Published in top-tier AI/ML venue

2. **Empirical Findings**
   - Quantified benefits of parallel execution in hierarchical systems
   - Evidence for/against reasoning diversity hypothesis
   - Scaling characteristics of HP-DAG

3. **Open-Source Implementation**
   - Reference implementation of HP-MAS
   - Financial domain agents as case study
   - Evaluation benchmarks

### Secondary Outcomes

1. **Design Guidelines**
   - Best practices for multi-agent financial systems
   - Complexity-scaling recommendations
   - Belief propagation patterns

2. **Benchmark Contributions**
   - Financial advice quality benchmark
   - Multi-agent evaluation protocol
   - Consistency metrics framework

---

## Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1-2 | Framework Design | Architecture spec, core abstractions |
| 3-4 | DAG Executor | Parallel execution engine, level barriers |
| 5-6 | Data Agents | Upstox, Market Data, News integrations |
| 7-8 | Analysis Agents | Technical, Fundamental, Sentiment with RSA |
| 9-10 | Debate + Synthesis | Bull/Bear committees, Portfolio Manager |
| 11-12 | Belief Propagation | Reflector, belief store, injection |
| 13-14 | Integration | Full pipeline, API layer |
| 15-16 | Evaluation Setup | Benchmarks, human evaluation protocol |
| 17-18 | Experiments | Run all experiments, collect data |
| 19-20 | Analysis | Statistical analysis, visualization |
| 21-22 | Writing | Paper draft |
| 23-24 | Revision | Final paper, code release |

---

## References

### Core References

1. Qian et al. (ICLR 2025). "Scaling Large Language Model-based Multi-Agent Collaboration." - MacNet DAG architecture

2. [AAAI 2025]. "TradingAgents: Multi-Agents LLM Financial Trading Framework." - Firm hierarchy, structured communication

3. Liu et al. (ICLR 2025). "Breaking Mental Set to Improve Reasoning through Diverse Multi-Agent Debate." - DMAD reasoning strategies

4. [NeurIPS 2024]. "FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement." - Belief propagation

5. [NeurIPS 2024]. "Reflective Multi-Agent Collaboration based on Large Language Models." - COPPER shared reflector

6. Li et al. (EMNLP 2024). "Improving Multi-Agent Debate with Sparse Communication Topology." - Sparse communication benefits

7. Guo et al. (IJCAI 2024). "Large Language Model Based Multi-agents: A Survey." - Comprehensive taxonomy

8. Tran et al. (2025). "Multi-Agent Collaboration Mechanisms: A Survey." - Collaboration frameworks

### Supporting References

9. Smit et al. (ICML 2024). "Should we be going MAD?" - MAD benchmarking

10. [NeurIPS 2024]. "Multi-LLM Debate: Framework, Principals, and Interventions." - Theoretical analysis

11. [ACL 2024]. "LLMArena." - Agent evaluation benchmark

12. [EMNLP 2024]. "MAgIC." - Multi-agent benchmark

---

## Appendix A: Reasoning Strategy Prompt Templates

### Strategy A: Backward Reasoning
```
You are analyzing {asset} using BACKWARD REASONING.

Start by considering the target outcome (e.g., ideal price target, perfect entry point).
Work backward from this target to identify:
1. What conditions would make this target achievable?
2. What signals would indicate we're on track?
3. What are the key milestones along the way?
4. What could derail us from this path?

Current state: {current_state}
Provide your analysis working from target to present.
```

### Strategy B: Step-by-Step Chain of Thought
```
You are analyzing {asset} using STEP-BY-STEP CHAIN OF THOUGHT.

Analyze each factor sequentially, one at a time:
1. First, examine [factor 1] and derive conclusions
2. Then, examine [factor 2] building on factor 1
3. Continue step by step through all factors
4. Only then synthesize your final view

Available factors: {factors}
Show your complete reasoning chain.
```

### Strategy C: Example-Based Reasoning
```
You are analyzing {asset} using EXAMPLE-BASED REASONING.

Before concluding, identify historical precedents:
1. What similar situations occurred in the past?
2. What were the outcomes then?
3. What made those situations similar/different?
4. What can we learn from those examples?

Historical data: {historical_data}
Draw parallels and contrasts with current situation.
```

### Strategy D: Symbolic/Quantitative
```
You are analyzing {asset} using SYMBOLIC/QUANTITATIVE REASONING.

Express your analysis using numerical relationships:
1. Identify key quantitative variables
2. Define relationships between variables
3. Calculate implied values
4. Test sensitivity to assumptions

Quantitative framework:
- Variable 1: {var1_definition}
- Variable 2: {var2_definition}
- Relationship: {relationship_formula}

Provide numerical justification for your conclusions.
```

---

## Appendix B: Belief Schema

```python
class Belief(BaseModel):
    """A learned belief from execution reflection"""
    
    key: str  # e.g., "high_volatility_requires_wider_stops"
    value: str  # e.g., "When VIX > 30, use 2x normal stop distance"
    confidence: float  # 0-1
    source: str  # Which agent/experience generated this
    context: Dict[str, Any]  # Conditions where this applies
    created_at: datetime
    last_updated: datetime
    usage_count: int  # How often this belief has been applied
    success_rate: float  # How often applying this led to good outcomes


class Reflection(BaseModel):
    """Output of the shared reflector"""
    
    success_factors: List[str]
    improvement_areas: List[str]
    conceptual_learnings: List[str]
    counterfactuals: List[Counterfactual]
    confidence_adjustments: Dict[str, float]


class Counterfactual(BaseModel):
    """A counterfactual analysis"""
    
    what_changed: str
    what_would_happen: str
    confidence: float
```

---

*Document Version: 1.0*
*Created: 2026-03-01*
*Status: Research Proposal Draft*
