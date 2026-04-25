# FinAI Multi-Agent System Architecture

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture Analysis](#current-architecture-analysis)
3. [Proposed Architecture](#proposed-architecture)
4. [Agent Design Patterns](#agent-design-patterns)
5. [Communication Protocols](#communication-protocols)
6. [State Management](#state-management)
7. [Framework Comparison](#framework-comparison)
8. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

This document outlines a comprehensive architecture for building a production-grade multi-agent financial advisory system. The proposed architecture addresses the limitations of the current implementation while providing scalability, reliability, and intelligent agent coordination.

### Key Design Principles

1. **Separation of Concerns** - Each agent has a single, well-defined responsibility
2. **Loose Coupling** - Agents communicate through well-defined interfaces
3. **Parallel Execution** - Independent agents run concurrently
4. **Result Aggregation** - Multiple agent outputs are synthesized coherently
5. **Graceful Degradation** - System continues with reduced functionality on failures
6. **Observability** - Full tracing and logging for debugging

---

## Current Architecture Analysis

### Existing Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Entry Point                       │
│                         (src/app.py)                             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Orchestrator                              │
│                   (src/core/orchestrator.py)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Planner  │─▶│  Route   │─▶│ Execute  │─▶│   END    │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Individual Agents                             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │Upstox  │ │Research│ │USStock │ │INStock │ │Advisor │        │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
│       │        │          │          │          │               │
│       └────────┴──────────┴──────────┴──────────┘               │
│                         │                                        │
│                         ▼                                        │
│              "Placeholder Response"                              │
└─────────────────────────────────────────────────────────────────┘
```

### Problems with Current Architecture

| Problem | Impact | Severity |
|---------|--------|----------|
| Sequential execution only | High latency, no parallelism | Critical |
| Single agent per query | No multi-perspective analysis | Critical |
| No agent collaboration | Cannot synthesize insights | High |
| Placeholder implementations | No actual functionality | Critical |
| Router-Agent key mismatch | Runtime KeyError | Critical |
| No memory/state | No conversation context | High |
| No error recovery | Single point of failure | High |

---

## Proposed Architecture

### High-Level Architecture

```
                                    ┌─────────────────────┐
                                    │    API Gateway      │
                                    │   (FastAPI/REST)    │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │   Session Manager   │
                                    │  (Conversation IDs) │
                                    └──────────┬──────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATION LAYER                              │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐                │
│  │   Intent      │    │   Planning    │    │   Task        │                │
│  │   Classifier  │───▶│   Engine      │───▶│   Dispatcher  │                │
│  └───────────────┘    └───────────────┘    └───────┬───────┘                │
│                                                     │                         │
│  ┌──────────────────────────────────────────────────┼─────────────────────┐  │
│  │                    EXECUTION FABRIC              │                     │  │
│  │                                                  ▼                     │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    PARALLEL EXECUTOR                            │   │  │
│  │  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │   │  │
│  │  │   │ Agent 1 │  │ Agent 2 │  │ Agent 3 │  │ Agent N │          │   │  │
│  │  │   │ (async) │  │ (async) │  │ (async) │  │ (async) │          │   │  │
│  │  │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘          │   │  │
│  │  └────┼────────────┼────────────┼────────────┼───────────────────┘   │  │
│  │       │            │            │            │                        │  │
│  │       └────────────┴─────┬──────┴────────────┘                        │  │
│  │                          ▼                                            │  │
│  │               ┌─────────────────────┐                                │  │
│  │               │  Result Aggregator  │                                │  │
│  │               └──────────┬──────────┘                                │  │
│  └──────────────────────────┼───────────────────────────────────────────┘  │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      SYNTHESIS LAYER                                   │  │
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐         │  │
│  │  │   Conflict    │    │   Evidence    │    │   Response    │         │  │
│  │  │   Resolver    │───▶│   Scorer      │───▶│   Generator   │         │  │
│  │  └───────────────┘    └───────────────┘    └───────────────┘         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │   Memory Store      │
                                    │  (Redis/PostgreSQL) │
                                    └─────────────────────┘
```

### Component Descriptions

#### 1. API Gateway Layer

```python
class APIGateway:
    """
    Entry point for all requests.
    Handles authentication, rate limiting, request validation.
    """
    
    endpoints = {
        "/query": QueryEndpoint,
        "/chat": ChatEndpoint,
        "/analyze": AnalyzeEndpoint,
        "/portfolio": PortfolioEndpoint,
    }
    
    middleware = [
        AuthenticationMiddleware,
        RateLimitMiddleware,
        RequestLoggingMiddleware,
        ErrorHandlingMiddleware,
    ]
```

#### 2. Session Manager

```python
class SessionManager:
    """
    Manages conversation sessions and context.
    """
    
    def create_session(self, user_id: str) -> Session:
        session = Session(
            id=uuid4(),
            user_id=user_id,
            created_at=datetime.now(),
            context={},
            history=[]
        )
        self._store.set(f"session:{session.id}", session)
        return session
    
    def get_context(self, session_id: str) -> dict:
        return self._store.get(f"session:{session_id}").context
    
    def update_context(self, session_id: str, key: str, value: Any):
        session = self._store.get(f"session:{session_id}")
        session.context[key] = value
        session.history.append({"key": key, "value": value, "timestamp": datetime.now()})
        self._store.set(f"session:{session_id}", session)
```

#### 3. Intent Classifier

```python
class IntentClassifier:
    """
    Classifies user intent to determine required agents.
    """
    
    INTENTS = {
        "portfolio_analysis": {
            "agents": ["upstox", "risk_analyzer"],
            "confidence_threshold": 0.8
        },
        "stock_research": {
            "agents": ["deep_web_research", "us_stock", "indian_stock"],
            "confidence_threshold": 0.7
        },
        "financial_advice": {
            "agents": ["general_advisor", "digital_twin", "risk_analyzer"],
            "confidence_threshold": 0.75
        },
        "comprehensive_review": {
            "agents": ["upstox", "deep_web_research", "us_stock", "indian_stock", "general_advisor"],
            "confidence_threshold": 0.6
        }
    }
    
    def classify(self, query: str, context: dict) -> ClassifiedIntent:
        prompt = self._build_classification_prompt(query, context)
        response = self.llm.generate(prompt, response_schema=IntentSchema)
        return ClassifiedIntent(
            intent=response.intent,
            agents=response.agents,
            confidence=response.confidence,
            sub_intents=response.sub_intents
        )
```

#### 4. Planning Engine

```python
class PlanningEngine:
    """
    Creates execution plans based on classified intent.
    Determines agent execution order, dependencies, and data flow.
    """
    
    def create_plan(self, intent: ClassifiedIntent, context: dict) -> ExecutionPlan:
        # Determine dependencies between agents
        dependency_graph = self._build_dependency_graph(intent.agents)
        
        # Identify parallelizable agents
        execution_levels = self._topological_sort(dependency_graph)
        
        # Create execution plan
        plan = ExecutionPlan(
            levels=execution_levels,
            total_steps=len(intent.agents),
            estimated_latency=self._estimate_latency(execution_levels),
            data_flow=self._define_data_flow(intent.agents)
        )
        return plan
    
    def _build_dependency_graph(self, agents: List[str]) -> Dict[str, List[str]]:
        """
        Example:
        general_advisor depends on [upstox, deep_web_research]
        digital_twin depends on [general_advisor]
        """
        dependencies = {}
        for agent in agents:
            deps = self.AGENT_DEPENDENCIES.get(agent, [])
            dependencies[agent] = [d for d in deps if d in agents]
        return dependencies
```

#### 5. Parallel Executor

```python
class ParallelExecutor:
    """
    Executes agents in parallel where possible.
    Handles timeouts, retries, and error isolation.
    """
    
    def __init__(self, max_concurrent: int = 5, timeout: float = 30.0):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_level(
        self, 
        agents: List[AgentTask], 
        state: GraphState
    ) -> List[AgentResult]:
        """
        Execute all agents in a level concurrently.
        """
        tasks = [
            self._execute_with_semaphore(agent, state) 
            for agent in agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from failures
        successful = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        if failures:
            logger.warning(f"{len(failures)} agents failed in this level")
        
        return successful
    
    async def _execute_with_semaphore(
        self, 
        agent: AgentTask, 
        state: GraphState
    ) -> AgentResult:
        async with self.semaphore:
            try:
                return await asyncio.wait_for(
                    agent.run(state),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                raise AgentTimeoutError(agent.name, self.timeout)
```

#### 6. Result Aggregator

```python
class ResultAggregator:
    """
    Aggregates results from multiple agents.
    Handles conflicting information and data merging.
    """
    
    def aggregate(
        self, 
        results: List[AgentResult], 
        aggregation_strategy: str = "weighted"
    ) -> AggregatedResult:
        
        # Group results by type
        grouped = self._group_by_type(results)
        
        # Apply aggregation strategy
        if aggregation_strategy == "weighted":
            return self._weighted_aggregation(grouped)
        elif aggregation_strategy == "hierarchical":
            return self._hierarchical_aggregation(grouped)
        else:
            return self._simple_aggregation(grouped)
    
    def _weighted_aggregation(self, grouped: Dict[str, List]) -> AggregatedResult:
        """
        Weight results based on agent confidence and relevance.
        """
        weights = {
            "upstox": 1.0,  # Primary data source
            "deep_web_research": 0.8,
            "us_stock": 0.9,
            "indian_stock": 0.9,
            "digital_twin": 0.7,
            "general_advisor": 0.85
        }
        
        weighted_data = {}
        for agent_name, results in grouped.items():
            weight = weights.get(agent_name, 0.5)
            weighted_data[agent_name] = {
                "data": results,
                "weight": weight * self._calculate_confidence(results)
            }
        
        return AggregatedResult(
            data=weighted_data,
            sources=list(grouped.keys()),
            timestamp=datetime.now()
        )
```

#### 7. Synthesis Layer

```python
class ResponseSynthesizer:
    """
    Synthesizes final response from aggregated results.
    Generates coherent, actionable financial advice.
    """
    
    def synthesize(
        self, 
        aggregated: AggregatedResult, 
        query: str,
        user_context: dict
    ) -> SynthesizedResponse:
        
        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(aggregated, query, user_context)
        
        # Generate response
        response = self.llm.generate(
            prompt,
            response_schema=FinancialAdviceSchema
        )
        
        # Add citations and confidence
        synthesized = SynthesizedResponse(
            advice=response.advice,
            action_items=response.action_items,
            rationale=response.rationale,
            confidence=response.confidence,
            sources=self._format_sources(aggregated),
            disclaimers=self._generate_disclaimers(response)
        )
        
        return synthesized
    
    def _build_synthesis_prompt(
        self, 
        aggregated: AggregatedResult,
        query: str,
        user_context: dict
    ) -> str:
        return f"""
        You are a senior financial advisor synthesizing insights from multiple
        specialized analysis agents. Create a coherent, actionable response.
        
        USER QUERY: {query}
        
        USER CONTEXT:
        - Risk Profile: {user_context.get('risk_profile', 'moderate')}
        - Age: {user_context.get('age', 'unknown')}
        - Investment Horizon: {user_context.get('investment_horizon', 'unknown')}
        
        AGENT INSIGHTS:
        {self._format_agent_insights(aggregated)}
        
        Provide:
        1. Clear answer to the query
        2. Specific action items
        3. Rationale based on the analysis
        4. Risk warnings and disclaimers
        """

# Response Schema
class FinancialAdviceSchema(BaseModel):
    advice: str = Field(description="Main financial advice")
    action_items: List[str] = Field(description="Specific actions to take")
    rationale: str = Field(description="Explanation of the advice")
    confidence: float = Field(description="Confidence level 0-1", ge=0, le=1)
    risk_level: str = Field(description="Assessment of risk level")
    time_horizon: str = Field(description="Recommended time horizon")
```

---

## Agent Design Patterns

### Pattern 1: Specialized Agent

```python
class SpecializedAgent(ABC):
    """
    Base class for specialized financial agents.
    Each agent has a specific domain and tools.
    """
    
    def __init__(self, llm: LLMClient, tools: List[Tool]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.system_prompt = self._build_system_prompt()
    
    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Each agent defines its own system prompt."""
        pass
    
    @abstractmethod
    async def run(self, state: AgentState) -> AgentResult:
        """Execute agent's primary function."""
        pass
    
    def get_required_inputs(self) -> List[str]:
        """Define required inputs for this agent."""
        return []
    
    def get_outputs(self) -> List[str]:
        """Define outputs this agent provides."""
        return []


class USStockAgent(SpecializedAgent):
    """
    Analyzes US stock market data and trends.
    """
    
    def _build_system_prompt(self) -> str:
        return """You are a US stock market analyst specializing in:
        - Fundamental analysis of US equities
        - Technical analysis and chart patterns
        - Sector rotation and market cycles
        - Macro-economic impact on US markets
        
        Always provide:
        1. Data-backed analysis
        2. Risk assessment
        3. Time-based projections
        4. Alternative scenarios
        """
    
    async def run(self, state: AgentState) -> AgentResult:
        query = state.query
        profile = state.profile
        
        # Gather data
        market_data = await self.tools["market_data"].fetch(query.symbols)
        news_sentiment = await self.tools["news_analyzer"].analyze(query.symbols)
        technicals = await self.tools["technical_analyzer"].analyze(query.symbols)
        
        # Generate analysis
        analysis = await self.llm.generate(
            self._build_analysis_prompt(market_data, news_sentiment, technicals),
            response_schema=StockAnalysisSchema
        )
        
        return AgentResult(
            agent_name="us_stock",
            data=analysis.dict(),
            confidence=analysis.confidence,
            sources=["market_data", "news_sentiment", "technical_analysis"]
        )
    
    def get_required_inputs(self) -> List[str]:
        return ["symbols", "timeframe"]
    
    def get_outputs(self) -> List[str]:
        return ["stock_analysis", "recommendations", "risk_assessment"]
```

### Pattern 2: Tool-Augmented Agent

```python
class ToolAugmentedAgent:
    """
    Agent with access to external tools and APIs.
    Uses ReAct pattern for tool selection.
    """
    
    def __init__(self, llm: LLMClient, tools: List[BaseTool]):
        self.llm = llm
        self.tools = ToolCollection(tools)
        self.memory = AgentMemory()
    
    async def run(self, state: AgentState) -> AgentResult:
        thought_history = []
        
        while not self._should_complete(thought_history):
            # Generate thought
            thought = await self._generate_thought(state, thought_history)
            thought_history.append(thought)
            
            if thought.action == "COMPLETE":
                break
            
            # Execute tool if needed
            if thought.action == "USE_TOOL":
                tool_result = await self.tools.execute(
                    thought.tool_name, 
                    thought.tool_args
                )
                thought_history.append(Observation(data=tool_result))
        
        return AgentResult(
            agent_name=self.name,
            data=self._extract_result(thought_history),
            reasoning=[t.dict() for t in thought_history]
        )
    
    async def _generate_thought(
        self, 
        state: AgentState, 
        history: List[Thought]
    ) -> Thought:
        prompt = self._build_react_prompt(state, history)
        response = await self.llm.generate(
            prompt,
            response_schema=ThoughtSchema
        )
        return Thought(
            reasoning=response.reasoning,
            action=response.action,
            tool_name=response.tool_name,
            tool_args=response.tool_args
        )
```

### Pattern 3: Collaborative Agent

```python
class CollaborativeAgent:
    """
    Agent that can delegate sub-tasks to other agents.
    Implements a manager-worker pattern.
    """
    
    def __init__(
        self, 
        llm: LLMClient,
        available_agents: Dict[str, Agent],
        max_delegations: int = 3
    ):
        self.llm = llm
        self.agents = available_agents
        self.max_delegations = max_delegations
    
    async def run(self, state: AgentState) -> AgentResult:
        # Analyze query and determine if delegation needed
        plan = await self._create_delegation_plan(state)
        
        results = []
        for delegation in plan.delegations[:self.max_delegations]:
            agent = self.agents[delegation.agent_name]
            result = await agent.run(
                AgentState(
                    query=delegation.sub_query,
                    profile=state.profile,
                    context=state.context
                )
            )
            results.append(result)
        
        # Synthesize delegated results
        synthesized = await self._synthesize_results(results, state)
        
        return AgentResult(
            agent_name=self.name,
            data=synthesized.data,
            confidence=synthesized.confidence,
            sub_agent_results=[r.dict() for r in results]
        )
```

---

## Communication Protocols

### Inter-Agent Communication

```python
class AgentMessage(BaseModel):
    """Standard message format for agent communication."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    sender: str
    receiver: str
    message_type: Literal["request", "response", "broadcast", "error"]
    content: Dict[str, Any]
    correlation_id: Optional[str] = None  # For request-response correlation
    priority: Literal["low", "medium", "high"] = "medium"
    ttl: Optional[int] = None  # Time-to-live in seconds


class AgentBus:
    """
    Message bus for inter-agent communication.
    Supports pub/sub, request/response, and broadcast patterns.
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.response_handlers: Dict[str, asyncio.Future] = {}
        self.message_queue = asyncio.Queue()
    
    async def publish(self, topic: str, message: AgentMessage):
        """Publish message to all subscribers of a topic."""
        for handler in self.subscribers[topic]:
            await handler(message)
    
    async def subscribe(self, topic: str, handler: Callable):
        """Subscribe to a topic."""
        self.subscribers[topic].append(handler)
    
    async def request(
        self, 
        target: str, 
        message: AgentMessage,
        timeout: float = 30.0
    ) -> AgentMessage:
        """Send request and wait for response."""
        future = asyncio.Future()
        self.response_handlers[message.id] = future
        
        await self.send(target, message)
        
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            del self.response_handlers[message.id]
            raise AgentTimeoutError(target, timeout)
    
    async def respond(self, request_id: str, response: AgentMessage):
        """Respond to a request."""
        if request_id in self.response_handlers:
            self.response_handlers[request_id].set_result(response)
            del self.response_handlers[request_id]
```

### Agent Interface Contract

```python
class IAgent(Protocol):
    """Interface contract for all agents."""
    
    @property
    def name(self) -> str:
        """Unique agent identifier."""
        ...
    
    @property
    def capabilities(self) -> List[str]:
        """List of agent capabilities."""
        ...
    
    async def run(self, state: AgentState) -> AgentResult:
        """Execute agent's primary function."""
        ...
    
    async def health_check(self) -> HealthStatus:
        """Check agent health status."""
        ...
    
    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for agent inputs/outputs."""
        ...
```

---

## State Management

### Hierarchical State Structure

```python
class GlobalState(TypedDict):
    """Top-level state shared across all components."""
    
    session: SessionState
    execution: ExecutionState
    agents: Dict[str, AgentState]
    memory: MemoryState
    metrics: MetricsState


class SessionState(TypedDict):
    """Session-specific state."""
    
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    context: Dict[str, Any]


class ExecutionState(TypedDict):
    """Current execution state."""
    
    plan_id: str
    current_level: int
    total_levels: int
    started_at: datetime
    completed_agents: List[str]
    pending_agents: List[str]
    failed_agents: List[str]


class AgentState(TypedDict):
    """Per-agent state."""
    
    agent_name: str
    status: Literal["pending", "running", "completed", "failed"]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    error: Optional[str]


class MemoryState(TypedDict):
    """Conversation memory."""
    
    messages: List[Message]
    entities: Dict[str, Entity]
    summaries: List[Summary]
    last_summarized: datetime
```

### State Persistence

```python
class StateStore:
    """
    Persists state to durable storage.
    Supports Redis for fast access and PostgreSQL for durability.
    """
    
    def __init__(
        self, 
        redis: Redis,
        postgres: PostgresConnection,
        cache_ttl: int = 3600
    ):
        self.redis = redis
        self.postgres = postgres
        self.cache_ttl = cache_ttl
    
    async def get(self, key: str) -> Optional[GlobalState]:
        # Try Redis first
        cached = await self.redis.get(key)
        if cached:
            return GlobalState(**json.loads(cached))
        
        # Fall back to PostgreSQL
        record = await self.postgres.fetch_one(
            "SELECT state FROM states WHERE key = $1",
            key
        )
        if record:
            state = GlobalState(**record["state"])
            # Cache in Redis
            await self.redis.setex(
                key,
                self.cache_ttl,
                json.dumps(state)
            )
            return state
        
        return None
    
    async def set(self, key: str, state: GlobalState):
        # Write to PostgreSQL (durable)
        await self.postgres.execute("""
            INSERT INTO states (key, state, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (key) DO UPDATE
            SET state = $2, updated_at = NOW()
        """, key, json.dumps(state))
        
        # Update Redis cache
        await self.redis.setex(
            key,
            self.cache_ttl,
            json.dumps(state)
        )
    
    async def acquire_lock(self, key: str, ttl: int = 30) -> bool:
        """Distributed lock for state updates."""
        return await self.redis.set(
            f"lock:{key}",
            "1",
            nx=True,
            ex=ttl
        )
    
    async def release_lock(self, key: str):
        await self.redis.delete(f"lock:{key}")
```

---

## Framework Comparison

### Detailed Comparison Matrix

| Feature | LangGraph | CrewAI | AutoGen | Custom |
|---------|-----------|--------|---------|--------|
| **Parallel Execution** | Native (async nodes) | Via tasks | Via agents | Full control |
| **State Management** | Built-in TypedDict | Task context | Conversation | Custom |
| **Agent Communication** | State passing | Task delegation | Direct messaging | Custom |
| **Human-in-the-loop** | Interrupt/resume | Input tasks | Native | Custom |
| **Streaming** | Native | Limited | Native | Custom |
| **Debugging** | LangSmith | Limited | VS Code | Full control |
| **Learning Curve** | Medium | Low | Medium | High |
| **Customization** | High | Medium | High | Maximum |
| **Production Ready** | Yes | Emerging | Yes | Varies |
| **Error Handling** | Retry policies | Limited | Retry | Custom |
| **Persistence** | Checkpointer | Memory | Custom | Custom |

### LangGraph Strengths for Financial Agents

```python
# LangGraph excels at conditional flows with state

class FinancialGraph:
    def build(self):
        graph = StateGraph(FinancialState)
        
        # Add nodes
        graph.add_node("classify_intent", self.classify_intent)
        graph.add_node("fetch_portfolio", self.fetch_portfolio)
        graph.add_node("analyze_stocks", self.analyze_stocks)
        graph.add_node("research_market", self.research_market)
        graph.add_node("synthesize", self.synthesize)
        graph.add_node("validate", self.validate)
        
        # Conditional routing based on intent
        graph.add_conditional_edges(
            "classify_intent",
            self.route_by_intent,
            {
                "portfolio_only": "fetch_portfolio",
                "research_only": "research_market",
                "full_analysis": "fetch_portfolio"  # Then parallel
            }
        )
        
        # Parallel execution fan-out
        graph.add_edge("fetch_portfolio", "analyze_stocks")
        graph.add_edge("fetch_portfolio", "research_market")
        
        # Fan-in with barrier
        graph.add_edge("analyze_stocks", "synthesize")
        graph.add_edge("research_market", "synthesize")
        
        # Validation loop
        graph.add_conditional_edges(
            "validate",
            self.should_regenerate,
            {
                "regenerate": "synthesize",
                "complete": END
            }
        )
        
        return graph.compile(checkpointer=MemorySaver())
```

### CrewAI Strengths for Financial Agents

```python
# CrewAI excels at role-based collaboration

from crewai import Agent, Task, Crew

# Define agents with roles
portfolio_manager = Agent(
    role="Portfolio Manager",
    goal="Optimize portfolio allocation",
    backstory="Expert in asset allocation with 15 years experience",
    tools=[PortfolioTool(), RiskAnalyzerTool()],
    verbose=True
)

research_analyst = Agent(
    role="Research Analyst",
    goal="Analyze market trends and identify opportunities",
    backstory="Data-driven analyst specializing in equity research",
    tools=[WebSearchTool(), SentimentAnalyzerTool()],
    verbose=True
)

risk_advisor = Agent(
    role="Risk Advisor",
    goal="Assess and communicate risks",
    backstory="Risk management specialist with focus on downside protection",
    tools=[RiskCalculatorTool(), ScenarioAnalysisTool()],
    verbose=True
)

# Define tasks with dependencies
research_task = Task(
    description="Research {stocks} and identify trends",
    agent=research_analyst,
    expected_output="Detailed research report"
)

allocation_task = Task(
    description="Suggest optimal allocation based on research",
    agent=portfolio_manager,
    context=[research_task],  # Dependency
    expected_output="Allocation recommendation"
)

risk_assessment_task = Task(
    description="Assess risks of proposed allocation",
    agent=risk_advisor,
    context=[allocation_task],  # Dependency
    expected_output="Risk assessment report"
)

# Create crew
financial_crew = Crew(
    agents=[portfolio_manager, research_analyst, risk_advisor],
    tasks=[research_task, allocation_task, risk_assessment_task],
    process=Process.sequential,  # or hierarchical
    verbose=True
)

result = financial_crew.kickoff(inputs={"stocks": "AAPL, GOOGL, MSFT"})
```

### Recommendation Matrix

| Use Case | Recommended Framework |
|----------|----------------------|
| Complex multi-step workflows | **LangGraph** |
| Role-based collaboration | **CrewAI** |
| Conversation-heavy agents | **AutoGen** |
| High customization needs | **Custom** |
| Quick prototyping | **CrewAI** |
| Production at scale | **LangGraph + Custom** |

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Priority: Critical**

- [ ] Fix agent-router key mismatch
- [ ] Implement proper error handling
- [ ] Add input validation
- [ ] Create agent interface contracts
- [ ] Set up logging and tracing

```python
# Immediate fix for router mismatch
ROUTER_TO_AGENT_MAP = {
    "fin_score": "general_advisor",
    "credits_loans": "general_advisor",
    "investment_coach": "general_advisor",
    "insurance_analyzer": "general_advisor",
    "retirement_planner": "general_advisor",
    "tax_planner": "general_advisor",
    "fraud_shield": "general_advisor",
    "fin_advisor": "general_advisor",
    # Direct mappings
    "upstox": "upstox",
    "us_stock_analysis": "us_stock_analysis",
    "indian_stock_analysis": "indian_stock_analysis",
    "deep_web_research": "deep_web_research",
    "digital_twin_persona": "digital_twin_persona",
}
```

### Phase 2: Agent Implementation (Week 3-4)

**Priority: High**

- [ ] Implement USStock agent with real analysis
- [ ] Implement IndianStock agent
- [ ] Implement DeepWebResearch agent
- [ ] Implement Upstox integration
- [ ] Implement DigitalTwin persona
- [ ] Implement GeneralAdvisor with synthesis

### Phase 3: Parallel Execution (Week 5-6)

**Priority: High**

- [ ] Add async execution support
- [ ] Implement parallel executor
- [ ] Add result aggregation
- [ ] Implement conflict resolution
- [ ] Add timeout and retry logic

### Phase 4: Memory & State (Week 7-8)

**Priority: Medium**

- [ ] Implement session management
- [ ] Add conversation memory
- [ ] Implement state persistence
- [ ] Add context retrieval (RAG)

### Phase 5: Production Ready (Week 9-10)

**Priority: Medium**

- [ ] Add comprehensive logging
- [ ] Implement health checks
- [ ] Add rate limiting
- [ ] Set up monitoring
- [ ] Add unit and integration tests
- [ ] Documentation

### Phase 6: Optimization (Week 11-12)

**Priority: Low**

- [ ] Response caching
- [ ] Agent warm-up
- [ ] Latency optimization
- [ ] Cost optimization (LLM calls)

---

## Appendix A: Agent Specifications

### USStock Agent

```yaml
name: us_stock_analysis
description: Analyzes US stock market data and trends
inputs:
  - symbols: List[str]
  - timeframe: str
  - analysis_type: str
outputs:
  - stock_analysis: StockAnalysisSchema
  - recommendations: List[Recommendation]
  - risk_assessment: RiskAssessment
tools:
  - market_data_api
  - news_sentiment_analyzer
  - technical_analyzer
  - fundamental_analyzer
dependencies: []
estimated_latency: 5-10s
```

### IndianStock Agent

```yaml
name: indian_stock_analysis
description: Analyzes Indian stock market data and trends
inputs:
  - symbols: List[str]
  - timeframe: str
  - analysis_type: str
outputs:
  - stock_analysis: StockAnalysisSchema
  - recommendations: List[Recommendation]
  - risk_assessment: RiskAssessment
tools:
  - nse_data_api
  - bse_data_api
  - news_sentiment_analyzer
dependencies: []
estimated_latency: 5-10s
```

### DeepWebResearch Agent

```yaml
name: deep_web_research
description: Conducts deep web research on financial topics
inputs:
  - query: str
  - scope: str
  - depth: int
outputs:
  - research_findings: ResearchFindingsSchema
  - sources: List[Source]
  - summary: str
tools:
  - web_search
  - content_extractor
  - sentiment_analyzer
dependencies: []
estimated_latency: 10-30s
```

### Upstox Agent

```yaml
name: upstox
description: Portfolio management and brokerage operations
inputs:
  - action: str
  - params: dict
outputs:
  - portfolio: PortfolioSchema
  - positions: List[Position]
  - holdings: List[Holding]
tools:
  - upstox_api
  - portfolio_analyzer
dependencies: []
estimated_latency: 2-5s
```

### DigitalTwin Agent

```yaml
name: digital_twin_persona
description: Persona-based financial advice
inputs:
  - query: str
  - persona_type: str
outputs:
  - advice: str
  - perspective: str
  - confidence: float
tools:
  - persona_llm
  - knowledge_base
dependencies:
  - general_advisor (optional)
estimated_latency: 3-8s
```

### GeneralAdvisor Agent

```yaml
name: general_advisor
description: Synthesizes advice from multiple sources
inputs:
  - query: str
  - context: dict
outputs:
  - advice: FinancialAdviceSchema
  - action_items: List[ActionItem]
  - disclaimers: List[str]
tools:
  - llm_synthesizer
  - rule_engine
dependencies:
  - upstox
  - deep_web_research
  - us_stock_analysis
  - indian_stock_analysis
estimated_latency: 5-15s
```

---

## Appendix B: Error Handling Patterns

```python
class AgentErrorHandler:
    """Centralized error handling for agents."""
    
    ERROR_STRATEGIES = {
        AgentTimeoutError: "retry_with_backoff",
        AgentValidationError: "return_partial_result",
        AgentAPIError: "use_fallback",
        AgentCriticalError: "escalate_to_human",
    }
    
    async def handle(self, error: Exception, agent: str, state: AgentState):
        strategy = self.ERROR_STRATEGIES.get(type(error), "log_and_continue")
        
        if strategy == "retry_with_backoff":
            return await self._retry_with_backoff(agent, state)
        elif strategy == "return_partial_result":
            return self._create_partial_result(agent, state)
        elif strategy == "use_fallback":
            return await self._use_fallback_agent(agent, state)
        elif strategy == "escalate_to_human":
            return await self._escalate(agent, error, state)
```

---

## Appendix C: Monitoring & Observability

```python
class AgentMetrics:
    """Metrics collection for agent monitoring."""
    
    def __init__(self):
        self.metrics = {
            "agent_executions": Counter(),
            "agent_latency": Histogram(),
            "agent_errors": Counter(),
            "agent_retries": Counter(),
            "parallel_efficiency": Gauge(),
        }
    
    def record_execution(
        self,
        agent: str,
        latency: float,
        success: bool,
        error: Optional[str] = None
    ):
        self.metrics["agent_executions"].labels(agent=agent).inc()
        self.metrics["agent_latency"].labels(agent=agent).observe(latency)
        
        if not success:
            self.metrics["agent_errors"].labels(
                agent=agent, 
                error_type=error
            ).inc()
```

---

*Document Version: 1.0*
*Last Updated: 2026-03-01*
