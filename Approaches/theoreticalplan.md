# MIDAS Multi-Agent Financial Advisory System
## Theoretical Plan: Complete Task Decomposition and System Architecture

**Version:** 2.0  
**Last Updated:** March 15, 2026  
**Document Type:** Theoretical Implementation Plan  

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [System Philosophy and Core Principles](#2-system-philosophy-and-core-principles)
3. [Complete Agent Catalog](#3-complete-agent-catalog)
4. [Task Decomposition: 7 Major Subtasks](#4-task-decomposition-7-major-subtasks)
5. [The Big Picture: Integration and Flow](#5-the-big-picture-integration-and-flow)
6. [RAG System Integration Strategy](#6-rag-system-integration-strategy)
7. [Cross-Cutting Concerns](#7-cross-cutting-concerns)
8. [Success Metrics and Validation](#8-success-metrics-and-validation)

---

## 1. Executive Overview

### 1.1 System Vision

MIDAS (Multi-agent Investment Decision Advisory System) is a production-grade, multi-agent financial advisory platform that provides comprehensive investment analysis through coordinated AI agents.

### 1.2 Core Value Proposition

| Traditional Approach | MIDAS Approach |
|---------------------|----------------|
| Single analyst perspective | Multiple specialized agents with diverse reasoning strategies |
| Static recommendations | Dynamic debate-driven synthesis |
| Limited explainability | Full audit trail with confidence scores |
| No uncertainty handling | Explicit disagreement extraction and risk warnings |
| Black box decisions | Transparent reasoning with debate history |

### 1.3 Key Innovation: The Debate Committee

The cornerstone of MIDAS is the **Bull-Bear Debate Committee**, which:
- Forces structured disagreement between opposing viewpoints
- Extracts consensus points and remaining disagreements  
- Produces risk-adjusted recommendations
- Mirrors how real investment committees operate

---

## 2. System Philosophy and Core Principles

### 2.1 Foundational Principles

#### Principle 1: Specialization Over Generalization
Each agent has a narrow, well-defined scope, reducing prompt complexity and improving accuracy.

#### Principle 2: Structured Disagreement
The system deliberately introduces opposing viewpoints through the debate mechanism to surface hidden risks.

#### Principle 3: Hierarchical Delegation with Context Isolation
Work flows from high-level planning to specialized execution, with each level operating in its own context window.

#### Principle 4: Fail-Safe Defaults
The system assumes failures will occur and handles them gracefully through circuit breakers, checkpoints, and fallbacks.

#### Principle 5: Compliance by Design
Regulatory compliance is woven into every layer, not added as an afterthought.

---

## 3. Complete Agent Catalog

### 3.1 Agent Taxonomy

The system comprises **16 specialized agents** organized into four tiers:

```
TIER 3: ORCHESTRATION
├── Planner Agent
├── Intent Classifier Agent
└── Complexity Scorer Agent

TIER 2: DEBATE COMMITTEE
├── Bull Committee Facilitator
├── Bear Committee Facilitator
└── Consensus Synthesizer Agent

TIER 1: SPECIALIZED ANALYSIS
├── Technical Analyst Agent
├── Fundamental Analyst Agent
├── Sentiment Analyst Agent
├── Valuation Analyst Agent
└── Risk Analyst Agent

TIER 0: DATA GATHERING
├── Upstox Agent
├── WebSearch Agent
├── NewsAPI Agent
└── Filings Agent
```

### 3.2 Detailed Agent Specifications

#### TIER 3: Orchestration Agents

**3.2.1 Planner Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Decompose user query into execution plan with agent assignments |
| **Capabilities** | Query understanding, complexity estimation, agent matching, execution sequencing |
| **Inputs** | User query, user profile, available agents |
| **Outputs** | Execution plan (3-4 levels), agent assignments, reasoning strategies, cost estimate |
| **Reasoning Strategy** | First Principles - builds plan from fundamentals |

---

**3.2.2 Intent Classifier Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Identify the user's true intent and route to appropriate workflow |
| **Capabilities** | Intent recognition, ambiguity detection, clarification generation |
| **Inputs** | User query, conversation history |
| **Outputs** | Classified intent, confidence score, clarification needs, recommended workflow |
| **Reasoning Strategy** | Example-Based - uses few-shot examples |

**Intent Categories:**
- `buy_recommendation` - User seeks buy advice
- `sell_recommendation` - User seeks sell advice
- `research` - User wants information
- `explain` - User needs educational content
- `compare` - User comparing instruments
- `portfolio_review` - User analyzing holdings

---

**3.2.3 Complexity Scorer Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Determine execution depth based on query complexity |
| **Capabilities** | Multi-factor complexity assessment, execution depth recommendation |
| **Inputs** | User query, intent classification, user profile |
| **Outputs** | Complexity score (1-10), recommended execution depth |

**Complexity Factors:**

| Factor | Weight | Description |
|--------|--------|-------------|
| Scope | 30% | Single stock vs portfolio vs strategy |
| Amount | 30% | Financial significance of decision |
| Time | 20% | Urgency of decision |
| Expertise | 20% | User's investment knowledge |

**Scoring Matrix:**

| Score | Execution Depth | Use Case |
|-------|-----------------|----------|
| 1-3 | Level 0 only | Simple data lookup |
| 4-6 | Levels 0-1 | Data + single analysis |
| 7-9 | Levels 0-2 | Full analysis + debate |
| 10+ | Full + validation | High-stakes decisions |

---

#### TIER 0: Data Gathering Agents

**3.2.4 Upstox Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Fetch real-time market data from Upstox API |
| **Capabilities** | Stock prices, historical data, market depth, company profile |
| **Reasoning Strategy** | Backward - starts from target data and works to API calls |

**Tools:**
- `get_stock_price(symbol)` - Current price and basic metrics
- `get_historical_data(symbol, period)` - Historical OHLCV data
- `get_company_info(symbol)` - Company profile and fundamentals

---

**3.2.5 WebSearch Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Gather real-time information from the web |
| **Capabilities** | Google search, financial news, analyst reports, competitor analysis |
| **Reasoning Strategy** | Counterfactual - seeks corroborating sources |

**Tools:**
- `search_web(query)` - General web search
- `search_financial_news(query)` - Financial news specific
- `search_analyst_reports(symbol)` - Analyst coverage

---

**3.2.6 NewsAPI Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Aggregate and analyze financial news |
| **Capabilities** | News retrieval, sentiment extraction, trend detection, event correlation |
| **Reasoning Strategy** | First Principles - focuses on what news actually says |

**Tools:**
- `get_news(symbol, days)` - Recent news for symbol
- `get_sentiment_trend(symbol)` - Sentiment over time
- `detect_events(symbol)` - Significant events detection

---

**3.2.7 Filings Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Access and extract information from regulatory filings |
| **Capabilities** | SEC filing retrieval, key extraction, filing comparison, red flag detection |
| **Reasoning Strategy** | Symbolic - applies financial accounting rules |

**Tools:**
- `get_filing(symbol, filing_type)` - Retrieve specific filing
- `extract_metrics(symbol)` - Key financial metrics
- `compare_filings(symbol, period1, period2)` - Period comparison

---

#### TIER 1: Specialized Analysis Agents

**3.2.8 Technical Analyst Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Perform technical analysis on price and volume data |
| **Capabilities** | Trend analysis, momentum indicators, pattern recognition, support/resistance |
| **Reasoning Strategy** | Step-by-Step - analyzes each indicator sequentially |

**Methods Applied:**
- **Trend Analysis:** Moving averages (SMA 20, 50, 200), EMA, MACD
- **Momentum:** RSI, Stochastic, Williams %R
- **Patterns:** Head and shoulders, double top/bottom, flags, triangles
- **Levels:** Historical support/resistance, volume profile

**Output Format:**
```json
{
  "signal": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "indicators": {
    "trend": "up|down|sideways",
    "momentum": "strong|weak|oversold|overbought"
  },
  "key_levels": {
    "support": [price1, price2],
    "resistance": [price1, price2]
  }
}
```

---

**3.2.9 Fundamental Analyst Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Analyze company fundamentals and intrinsic value |
| **Capabilities** | Financial statement analysis, ratio calculation, DCF modeling, competitive positioning |
| **Reasoning Strategy** | Backward - starts from intrinsic value and validates assumptions |

**Methods Applied:**
- **Financial Analysis:** Income statement, balance sheet, cash flow trends
- **Ratios:** P/E, P/B, ROE, ROA, debt/equity
- **Valuation:** DCF, DDM, comparable companies
- **Quality Assessment:** Moat analysis, competitive position

---

**3.2.10 Sentiment Analyst Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Analyze market sentiment and investor psychology |
| **Capabilities** | News sentiment, social media analysis, analyst aggregation, contrarian detection |
| **Reasoning Strategy** | Counterfactual - "What if sentiment is wrong?" |

**Data Sources:**
- News headlines and articles
- Social media (Twitter, Reddit, StockTwits)
- Analyst recommendations
- Options flow (put/call ratio)

---

**3.2.11 Valuation Analyst Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Cross-validate valuation using multiple methods |
| **Capabilities** | DCF modeling, relative valuation, sensitivity analysis, peer comparison |
| **Reasoning Strategy** | First Principles - builds valuation from fundamentals |

**Methods:**
- DCF (Discounted Cash Flow)
- DDM (Dividend Discount Model)
- Comparable company analysis
- Precedent transactions

---

**3.2.12 Risk Analyst Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Identify and quantify investment risks |
| **Capabilities** | Risk factor identification, stress testing, risk mitigation recommendations |
| **Reasoning Strategy** | Counterfactual - "What could go wrong?" |

**Risk Categories:**
1. Market Risk (beta, sector correlation)
2. Company-Specific Risk (concentration, key person)
3. Financial Risk (debt, cash flow)
4. Regulatory Risk (compliance, legal)
5. Liquidity Risk (trading volume, float)

---

#### TIER 2: Debate Committee Agents

**3.2.13 Bull Committee Facilitator**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Construct and defend the bullish investment case |
| **Capabilities** | Bull case construction, anticipating counterarguments, evidence organization |
| **Reasoning Strategy** | First Principles - builds bullish case from fundamentals |

**Role in Debate:**
1. Opens debate with bull thesis
2. Presents supporting evidence
3. Rebutts bear arguments
4. Summarizes bull position

---

**3.2.14 Bear Committee Facilitator**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Construct and defend the bearish investment case |
| **Capabilities** | Bear case construction, stress-testing bull assumptions, risk identification |
| **Reasoning Strategy** | Counterfactual - "What if the bull case is wrong?" |

**Role in Debate:**
1. Responds to bull thesis with counterarguments
2. Highlights risks and weaknesses
3. Defends against bull rebuttals
4. Summarizes bear position

---

**3.2.15 Consensus Synthesizer Agent**

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Extract consensus, document disagreements, produce final recommendation |
| **Capabilities** | Agreement identification, disagreement documentation, risk-adjusted synthesis |
| **Reasoning Strategy** | Symbolic - applies decision rules to consensus data |

**Output:**
- Points of agreement
- Remaining disagreements with impact assessment
- Final recommendation with confidence range
- Action items and monitoring recommendations

---

## 4. Task Decomposition: 7 Major Subtasks

### Subtask Overview

```
SUBTASK 1: Query Understanding and Planning
├── Intent classification
├── Complexity scoring
└── Execution plan generation

SUBTASK 2: Data Gathering and Validation
├── Parallel data retrieval from multiple sources
├── Data quality assessment
└── Cross-source validation

SUBTASK 3: Specialized Analysis Execution
├── Technical analysis
├── Fundamental analysis
├── Sentiment analysis
├── Valuation cross-check
└── Risk assessment

SUBTASK 4: Structured Debate and Synthesis
├── Bull case construction
├── Bear case construction
├── Debate rounds execution
└── Consensus extraction

SUBTASK 5: RAG Knowledge Integration
├── Financial concept retrieval
├── Instrument characteristics lookup
├── Best practices retrieval
└── Knowledge-grounded reasoning

SUBTASK 6: Compliance and Risk Validation
├── Regulatory compliance check
├── Output validation
├── Risk disclosure
└── Audit trail generation

SUBTASK 7: Response Generation and Delivery
├── Final response synthesis
├── Human review triggering (if needed)
├── Confidence explanation
└── Delivery with disclaimers
```

---

### SUBTASK 1: Query Understanding and Planning

**Objective:** Transform user's query into structured execution plan.

**Process Flow:**

1. **Intent Classification**
   - Parse query for entities (stock symbols, financial terms)
   - Classify into intent categories
   - Calculate confidence, generate clarifications if needed

2. **Complexity Scoring**
   - Evaluate: Scope, Amount, Time, Expertise factors
   - Calculate weighted sum
   - Recommend execution depth

3. **Execution Plan Generation**
   - Select agents based on intent
   - Determine levels based on complexity
   - Assign reasoning strategies for diversity
   - Estimate tokens and latency

---

### SUBTASK 2: Data Gathering and Validation

**Objective:** Collect necessary data from multiple sources in parallel.

**Process Flow:**

1. **Parallel Data Retrieval**
   - Execute simultaneous API calls to all sources
   - Track failures for fallback handling
   - Aggregate results

2. **Data Quality Assessment**
   - Completeness check
   - Timeliness check
   - Consistency check
   - Authority check
   - Assign quality scores

3. **Cross-Source Validation**
   - Identify overlapping data points
   - Compare and reconcile discrepancies
   - Select authoritative values

**Failure Handling:**
- Timeouts: Retry with exponential backoff
- Unavailable: Use cached data, flag quality issue
- Low quality: Attempt alternative source

---

### SUBTASK 3: Specialized Analysis Execution

**Objective:** Apply domain-specific analysis methods to produce structured signals.

**Execution Protocol:**
- Each agent operates with isolated context
- Assigned reasoning strategy applied
- Structured output format required
- Confidence scoring mandatory

**Analysis Types:**

| Analysis | Key Outputs |
|----------|-------------|
| Technical | Signal (bullish/bearish/neutral), indicators, key levels |
| Fundamental | Signal, valuation, quality score, growth assessment |
| Sentiment | Signal, sentiment score, contrarian indicators |
| Valuation | Fair value range, consensus value, confidence interval |
| Risk | Overall risk level, risk factors, stress test results |

---

### SUBTASK 4: Structured Debate and Synthesis

**Objective:** Force structured disagreement and extract actionable consensus.

**Debate Protocol:**

**Phase 1: Opening Positions**
- Bull presents thesis with evidence
- Bear presents counter-thesis with evidence
- Both state conviction levels

**Phase 2: Rebuttal**
- Bull addresses bear arguments
- Bear addresses bull arguments
- Both adjust conviction if warranted

**Phase 3: Synthesis**
- Identify points of agreement
- Document remaining disagreements
- Produce final recommendation
- Calculate confidence with uncertainty range

**Disagreement Classification:**

| Type | Impact | Handling |
|------|--------|----------|
| Factual dispute | High | Request additional data |
| Interpretation difference | Medium | Document both views |
| Weighting difference | Low | Use range in recommendation |

---

### SUBTASK 5: RAG Knowledge Integration

**Objective:** Ground agent reasoning in authoritative financial knowledge.

**RAG System Contents:**

1. **Stable Instruments Knowledge:**
   - Bonds (government, corporate, municipal)
   - Fixed deposits and CDs
   - Money market instruments
   - Annuities

2. **Growth Instruments Knowledge:**
   - Stocks (common, preferred)
   - Growth vs. value investing
   - Sector dynamics
   - Market cap categories

3. **Finance Fundamentals:**
   - Financial statement analysis
   - Valuation methods
   - Risk management principles
   - Portfolio theory
   - Behavioral finance

---

### SUBTASK 6: Compliance and Risk Validation

**Objective:** Ensure regulatory compliance and appropriate risk disclosure.

**Regulatory Framework:**
- EU AI Act (high-risk classification, right to explanation)
- SR 11-7 (model risk management)
- SEC/FINRA (recordkeeping, suitability)
- Treasury AI Framework (230 control objectives)

**Compliance Checks:**

1. **Input Validation:** No sensitive data, no bias indicators
2. **Decision Validation:** Confidence threshold, rationale present
3. **Output Validation:** Disclaimers, risk warnings, proper citations

**Human Oversight Triggers:**
- Confidence < 60%
- High disagreement with no consensus
- Investment amount exceeds threshold
- Risk score > 80/100

---

### SUBTASK 7: Response Generation and Delivery

**Objective:** Synthesize all analysis into clear, actionable response.

**Response Structure:**

```markdown
# Investment Analysis: [SYMBOL]

## Executive Summary
[Recommendation with conviction]

## Analysis Overview
- Recommendation: Buy/Hold/Sell/Avoid
- Confidence: X% (Y-Z% range)
- Time Horizon: Short/Medium/Long
- Risk Level: Low/Medium/High

## Key Findings
[Technical, Fundamental, Sentiment summaries]

## Bull Case / Bear Case
[Arguments for and against]

## Consensus & Disagreements
[Points of agreement and remaining disagreements]

## Recommended Actions
[Specific entry points, stop-losses, review triggers]

## Risk Warnings & Disclaimers
[Required disclosures]
```

---

## 5. The Big Picture: Integration and Flow

### 5.1 End-to-End Execution Flow

```
USER QUERY: "Should I buy AAPL stock?"
    │
    ▼
SUBTASK 1: Planning
    │ Intent: "buy_recommendation" (conf: 0.92)
    │ Complexity: 8 → Full 3-level execution
    │ Plan: Data → Analysis → Debate
    ▼
SUBTASK 2: Data Gathering (PARALLEL)
    │ Upstox Agent: Price, historical data
    │ NewsAPI Agent: Recent news, sentiment
    │ WebSearch Agent: Company overview
    │ Filings Agent: 10-K, 10-Q summaries
    ▼
[CONTEXT COMPACTION: 45K → 15K tokens]
    │
    ▼
SUBTASK 3: Analysis (PARALLEL)
    │ Technical: BULLISH (0.72)
    │ Fundamental: BULLISH (0.78)
    │ Sentiment: BULLISH (0.65)
    │ Valuation: Fair value $155-165
    │ Risk: MEDIUM (45/100)
    ▼
[CONTEXT COMPACTION: 35K → 14K tokens]
    │
    ▼
SUBTASK 4: Debate (SEQUENTIAL)
    │ Bull Case: Strong moat, undervalued
    │ Bear Case: Growth concerns, China risk
    │ Debate: Round 1 → Round 2 → Round 3
    │ Synthesis: BUY with 68% confidence
    ▼
SUBTASK 5: RAG Integration (throughout)
    │ Knowledge grounding for each analysis
    ▼
SUBTASK 6: Compliance
    │ Validation: ✓ All checks passed
    │ Audit trail: Generated
    ▼
SUBTASK 7: Response Generation
    │ Final recommendation assembled
    │ Disclaimers added
    ▼
USER RESPONSE DELIVERED
```

### 5.2 Parallel vs Sequential Execution

| Phase | Execution Type | Rationale |
|-------|---------------|-----------|
| Subtask 1 | Sequential | Planning depends on intent |
| Subtask 2 | Parallel | Data sources independent |
| Subtask 3 | Parallel | Analysis types independent |
| Subtask 4 | Sequential | Debate requires response to opponent |
| Subtask 5 | Throughout | RAG called as needed |
| Subtask 6 | Sequential | Validation after synthesis |
| Subtask 7 | Sequential | Final assembly |

---

## 6. RAG System Integration Strategy

### 6.1 Architecture Decision

**Decision:** RAG system is implemented as a **Tool**, not a separate Agent.

**Rationale:**

| Approach | Pros | Cons |
|----------|------|------|
| RAG as Agent | Centralized control | Bottleneck, inflexible |
| RAG as Tool | Parallel access, contextual | Coordination needed |
| **Selected** | **RAG as Tool** | |

**Benefits of Tool Approach:**
1. No bottleneck - multiple agents retrieve simultaneously
2. Contextual relevance - each agent queries what it needs
3. Simpler architecture - one less agent in system
4. Industry best practice (AWS Agentic GraphRAG)

### 6.2 RAG Tool Design

**Tool Name:** `retrieve_financial_knowledge`

**Parameters:**
```python
{
    "query": "Natural language query",
    "knowledge_domain": "fundamentals|stable_instruments|growth_instruments|all",
    "detail_level": "summary|detailed|comprehensive",
    "context": "Optional: specific context"
}
```

**Output:**
```json
{
    "content": "Retrieved knowledge",
    "source": "domain_name",
    "confidence": 0.95,
    "related_concepts": ["concept1", "concept2"]
}
```

### 6.3 Agent Integration Examples

**Technical Analyst:**
- Query: "RSI overbought interpretation"
- Use: Ensures proper indicator interpretation

**Fundamental Analyst:**
- Query: "Healthy debt-to-equity for tech companies"
- Use: Validates ratio assessment

**Risk Analyst:**
- Query: "Key risk factors for growth stocks"
- Use: Comprehensive risk identification

**Debate Facilitators:**
- Query: "Historical bull/bear debates for tech stocks"
- Use: Grounds arguments in history

---

## 7. Cross-Cutting Concerns

### 7.1 Security

- User queries hashed before logging
- API keys in secure vault
- End-to-end encryption
- Role-based access control
- Rate limiting per user

### 7.2 Performance

**Targets:**
- End-to-end latency: <30s (P99)
- Data gathering: <15s
- Analysis execution: <10s
- Debate synthesis: <20s

**Optimization:**
- Parallel execution
- Context compaction (67% token reduction)
- Caching for repeated queries
- Connection pooling

### 7.3 Reliability

**Failure Handling:**
- API timeout: Retry with backoff
- Data unavailable: Use cached/fallback
- Agent failure: Circuit breaker, reassign
- Debate deadlock: Force synthesis

**Graceful Degradation:**
```
IF full_analysis_fails:
    TRY partial_analysis
    IF partial_fails:
        RETURN data_summary_with_warning
```

### 7.4 Observability

**Metrics:**
- Request count, success rate
- Latency distribution (P50, P95, P99)
- Token usage
- Agent execution times

**Alerting:**
- Success rate < 95%
- Latency P99 > 30s
- Circuit breaker trips > 5/hour
- Compliance violations > 0

---

## 8. Success Metrics and Validation

### 8.1 System Metrics

| Metric | Target |
|--------|--------|
| Success Rate | >99% |
| P50 Latency | <5s |
| P99 Latency | <30s |
| Token Efficiency | 67% reduction |
| Context Compaction | >50% |
| Circuit Breaker Trips | <1/hour |

### 8.2 Quality Metrics

| Metric | Target |
|--------|--------|
| Intent Classification Accuracy | >95% |
| Analysis Quality | >0.8 |
| Debate Completeness | Both cases present |
| Compliance Pass Rate | 100% |

### 8.3 User Experience Metrics

| Metric | Target |
|--------|--------|
| Response Clarity | >4/5 |
| Actionability | 100% have actions |
| Confidence Calibration | Brier score <0.15 |
| Risk Warning Appropriateness | >95% |

---

## Appendix: Key Architectural Decisions

### Why 16 Agents?

Each agent has single responsibility. Too few = overgeneralization. Too many = coordination overhead. 16 is optimal balance.

### Why Hybrid LangGraph + CrewAI?

- **LangGraph:** State management, conditional routing, checkpointing
- **CrewAI:** Role-based agents, natural delegation, built-in memory

### Why Context Compaction?

Context grows linearly. Compaction at level boundaries reduces tokens by 67%, preventing window exhaustion.

### Why Debate Instead of Consensus?

Agents tend to agree (herding). Structured disagreement reduces hallucinations by 40% and produces better-calibrated confidence.

### Why RAG as Tool Not Agent?

Parallel retrieval, contextual relevance, simpler architecture. Industry best practice (AWS Agentic GraphRAG).

---

## Conclusion

This theoretical plan provides the complete blueprint for MIDAS:

1. **16 Specialized Agents** - Narrow scope, clear responsibilities
2. **4-Tier Architecture** - Orchestration → Debate → Analysis → Data
3. **7 Major Subtasks** - Complete task decomposition
4. **RAG as Tool** - Parallel knowledge access
5. **Structured Debate** - Novel approach to prevent herding
6. **Compliance by Design** - Regulatory requirements in every layer

The system provides well-reasoned investment recommendations with explicit uncertainty handling and regulatory compliance.
