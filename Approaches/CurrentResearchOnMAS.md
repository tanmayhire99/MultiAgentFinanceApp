# Recent Multi-Agent LLM Systems in A* Conferences (Mar 2024–Feb 2026)

## Executive overview

Research on large-language-model (LLM) based multi-agent systems (MAS) has accelerated in 2024–2025, with a clear split between **collaborative architectures** (hierarchical pipelines, DAGs, shared-memory “societies”) and **debate-style test-time scaling**. Collaborative systems are increasingly used in realistic domains—including several finance-specific trading and portfolio frameworks—while debate-focused systems are mainly evaluated on math, multimodal reasoning, and alignment-labeling benchmarks.[^1][^2][^3][^4][^5][^6]

A* venues (NeurIPS, ICML, ICLR, ACL, EMNLP, IJCAI, AAAI) now contain: (1) general surveys and taxonomies of LLM multi-agents, (2) collaboration-scaling architectures, (3) multi-agent debate and communication-topology studies, and (4) domain systems for finance and socio‑economic simulation. The sections below synthesize these papers with a focus on architectures, agent orchestration/calling, debate vs. collaboration patterns, and frameworks.[^2][^7][^8][^9][^10][^11][^6][^1]

***

## Core surveys and taxonomies

### IJCAI‑24: LLM-based multi-agents survey

Guo et al. (IJCAI‑24) provide the first broad survey specifically on “Large Language Model Based Multi-agents (LLM‑MA).”[^7][^1]

- They decompose LLM‑MA systems into four architectural axes: **agents–environment interface** (sandbox, physical, none), **agent profiling** (pre-defined, model-generated, data-derived roles), **agent communication** (cooperative, debate, competitive with layered/centralized/decentralized/shared-pool topologies), and **agent capability acquisition** (feedback sources plus memory, self-evolution, dynamic agent generation).[^7]
- The survey catalogues problem-solving systems (software engineering, embodied robots, scientific workflows, science debate) and world simulation (society, gaming, psychology, recommender, economy/policy), and summarizes common **multi-agent frameworks** such as MetaGPT, CAMEL, and AutoGen.[^7]

### Multi-Agent Collaboration Mechanisms survey (2025)

Tran et al. (2025) focus specifically on **collaboration mechanisms** rather than general agent capabilities, offering a framework with dimensions: actors, collaboration type (cooperation/competition/coopetition), communication structure (centralized, decentralized, hierarchical), strategy (rule-/role-/model-based), and coordination architecture (static vs dynamic).[^5]

- They formalize a collaborative system as agents \(a_i\) interacting through collaboration channels \(c_j\), where each channel is a specific combination of type, structure, and strategy.[^5]
- The survey explicitly covers cooperative, competitive, coopetitive, and hybrid channels and highlights role-based systems (MetaGPT, AgentVerse), debate-style competition channels, and DAG-based coordination (e.g., code agents, Minecraft agents) as emerging patterns.[^5]

### Multi-agent capabilities benchmarks (LLMArena, MAgIC)

Two A* NLP benchmarks focus on evaluating LLM agents in multi-agent settings rather than proposing new architectures:

- **LLMArena (ACL‑24)** introduces seven dynamic game environments (e.g., spatial navigation, strategy, risk, opponent modeling, team collaboration) and uses TrueSkill to score abilities like spatial reasoning, strategic planning, numerical reasoning, risk assessment, communication, and team collaboration.[^8]
- **MAgIC (EMNLP‑24)** defines a competition-based benchmark over two social deduction games and three game-theory scenarios, and introduces seven metrics: judgment, reasoning, deception, self-awareness, cooperation, coordination, and rationality.[^12][^13]
  - It also proposes **PGM-aware agents**, integrating probabilistic graphical models with LLMs to reason over other agents’ beliefs and roles before deciding actions.[^13]

These benchmarks mainly adopt **turn-based, environment-driven orchestration** where agents are called each round by the environment, but they give useful design patterns for finance-like multi-agent simulations (social deduction, bargaining, cost-sharing, public-goods games).[^12][^13][^8]

***

## Collaboration-scaling architectures (MacNet, COPPER, etc.)

### ICLR‑25: MacNet – DAG-based multi-agent collaboration scaling

Qian et al. (ICLR‑25) study “Scaling Large Language Model-based Multi-Agent Collaboration,” introducing **MacNet**, a directed acyclic graph (DAG) of agents.[^2]

- **Architecture**: Agents are nodes in a DAG; edges represent information flow. Agent calls follow a topological order; each agent receives upstream agents’ outputs as context and tools, then emits its own contribution.[^2]
- **Topologies**: They compare regular vs irregular DAGs at scales up to \(>1000\) agents and find **irregular graphs outperform regular structures** for complex tasks.[^2]
- **Scaling law**: They empirically identify a **collaborative scaling law**—performance follows logistic growth as the number of agents increases, with an “emergence” phase occurring earlier than standard neural scaling.[^2]

**Agent calling/orchestration**: The orchestrator calls agents in topological order; within each node, calls are standard single-LLM prompts optionally augmented with tools. This maps naturally to frameworks like LangGraph/LangChain DAGs or Ray DAGs in production.

### NeurIPS‑24: COPPER – reflective multi-agent collaboration

COPPER (NeurIPS‑24 poster “Reflective Multi-Agent Collaboration based on Large Language Models”) focuses on **improving collaboration via shared reflection**.[^14][^15]

- Multiple “actor” agents collaborate on tasks like multi-hop QA, math, and chess; a **shared reflector model** generates reflections that adjust prompts or strategies for each actor.
- The reflector is fine-tuned using a **counterfactual PPO** scheme: counterfactual rewards estimate each agent’s reflection contribution, addressing multi-agent credit assignment.[^15]
- A single shared reflector conditions on agent roles to output personalized reflections, reducing compute vs per-agent reflectors and improving stability.[^15]

**Agent calling**: For each step, actors run, reflections are generated by the reflector, and actors are re-called with updated context. Orchestration is centralised (reflector as hub) but actual collaboration among actors is mediated via shared reflections rather than direct peer-to-peer communication.[^15]

### Finance-specific collaborative systems (FinCon, TradingAgents)

#### FinCon (NeurIPS‑24, arXiv 2407.06567)

FinCon proposes a **manager–analyst–risk hierarchy** for financial LLM agents.[^9][^3]

- **Architecture**: A **manager agent** coordinates multiple **analyst agents** (e.g., equity, macro, sentiment) plus a **risk-control component**.
  - Agents communicate using natural language but are organized in a firm-inspired hierarchy: analysts propose, manager aggregates, risk module occasionally triggers a self-critique “conceptual verbal reinforcement” phase.[^3]
- **Conceptual verbal reinforcement**: The system learns and stores conceptual “beliefs” (e.g., risk heuristics) as verbal summaries, which can be selectively propagated to relevant agents to update future decisions.[^3]
- **Tasks**: Single-stock trading and portfolio management with sequential decisions in volatile environments; FinCon shows improved performance vs baseline LLM traders on several financial tasks.[^3]

**Agent calling**: The orchestrator follows a fixed pipeline: analysts → manager → risk controller. Risk controller asynchronously inserts review rounds when performance or volatility triggers conditions, then beliefs are updated and used in later cycles.[^3]

#### TradingAgents (AAAI‑25 main track)

TradingAgents is a multi-agent LLM trading framework accepted at AAAI‑25, designed to emulate **a professional trading firm**.[^4]

- **Role hierarchy**:
  - **Analyst Team**: Fundamental, sentiment, news, and technical analysts each produce structured reports (not just free text).[^4]
  - **Researcher Team**: Bullish and bearish researchers debate based on analyst reports, supervised by a facilitator agent that records the prevailing perspective.[^4]
  - **Trader agents**: Aggregate research and market data to produce trade proposals; multiple trader profiles correspond to different risk appetites.[^4]
  - **Risk Management Team**: Aggressive, neutral, and conservative risk agents debate adjustments to the proposed trade; a fund manager then approves or vetoes.[^4]
- **Communication protocol**: The framework uses a **hybrid protocol**:
  - Primary communication is through **structured documents** in a shared global state (analyst reports, trader reports, debate summaries) to avoid “telephone effect” and context overflow.[^4]
  - Natural-language multi-round debates are used only inside the Researcher and Risk teams; a facilitator writes a structured summary back into the global state.[^4]
- **Reasoning style**: All agents follow a **ReAct-style prompting** scheme, interleaving reasoning with tool calls (e.g., price data APIs, news retrieval, technical indicator computation).[^4]
- **Backbone models**: Uses a mix of “quick-thinking” (e.g., GPT‑4o‑mini) and “deep-thinking” (e.g., OpenAI o1‑preview) models, assigned by role to balance latency vs reasoning depth.[^4]
- **Performance**: On Jan–Mar 2024 backtests for AAPL, GOOGL, AMZN, TradingAgents outperforms Buy&Hold and rule-based strategies (MACD, KDJ/RSI, SMA, ZMR) in cumulative and annualized return and Sharpe ratio while keeping maximum drawdown low.[^4]

**Agent calling**: Orchestration is a **sequential collaborative workflow**: analysts (parallel) → researchers (debate) → trader → risk team (debate) → fund manager. Each step results in structured artifacts; debates are localized to subteams rather than global all-to-all chat.[^4]

***

## Multi-agent debate and communication topologies

### ICML‑24: “Should we be going MAD?”

Smit et al. (ICML‑24) benchmark **multi-agent debate (MAD)** against other test-time scaling methods like self-consistency and multi-path ensembling.[^10]

- They evaluate a range of debate protocols (e.g., standard MAD, multi-persona) on reasoning/factual tasks.
- **Key finding**: Vanilla MAD **does not consistently outperform** strong single-agent baselines; performance is highly sensitive to hyperparameters like number of agents, debate rounds, and agreement thresholds.[^10]
- With careful hyperparameter tuning, specific debate variants (e.g., Multi-Persona) can surpass baselines, suggesting MAD is not fundamentally flawed but harder to optimize.[^10]

### Findings EMNLP‑24: Sparse communication in MAD

Li et al. (Findings of EMNLP‑24) show that **sparse communication topologies** can improve both efficiency and sometimes accuracy in MAD.[^6]

- Represent agents as nodes in a static graph; edges indicate whose previous-round answers each agent can see. They compare fully-connected vs sparser regular graphs with 6 agents and 5 rounds.[^6]
- On MATH and GSM8K, sparse MAD achieves accuracy similar to or higher than fully-connected MAD while reducing input-token cost by around 40–50%.[^6]
- For alignment-labeling tasks (Anthropic Helpful/Harmless), MAD improves over single-agent CoT/self-consistency, and sparse topologies match or slightly exceed fully-connected performance at half the cost.[^6]
- Analysis suggests two mechanisms: (1) fewer incorrect references for hard problems (reducing “herding” to wrong answers), and (2) more “effective debate rounds” before full convergence, allowing deeper deliberation.[^6]
- When mixing strong and weak models, placing strong models at **high-centrality nodes** improves overall accuracy by faster propagation of better reasoning.[^6]

### ICLR‑25: Diverse Multi-Agent Debate (DMAD)

Liu et al. (ICLR‑25 poster) propose **Diverse Multi-Agent Debate (DMAD)** to break “fixed mental sets.”[^11][^16]

- Each debating agent is explicitly instructed to use **different reasoning strategies** (e.g., step-by-step CoT, backward reasoning, example-based reasoning, symbolic style), and to learn from the others’ approaches during debate.[^16]
- On multiple reasoning benchmarks for both LLMs and multimodal LLMs, DMAD outperforms standard MAD, self-reflection, and other prompting methods, and converges in fewer rounds than MAD.[^11][^16]

### NeurIPS‑24: Multi-LLM Debate (theoretical analysis)

The NeurIPS‑24 paper “Multi-LLM Debate: Framework, Principals, and Interventions” offers a theoretical analysis of multi-agent debate.[^17]

- It shows that when agents share similar training data and biases, debate can converge to a **majority misconception**, with dynamics that simply reinforce the common error.[^17]
- They propose interventions (e.g., injecting diversity, modifying moderation rules) that can improve debate outcomes, supported by experiments on four benchmarks.[^17]

### ICLR‑25 blog + OpenReview: Multi‑LLM Agents Debate (MAD as test-time scaling)

Complementing the above, the ICLR‑25 blog-post “Multi‑LLM‑Agents Debate – Performance, Efficiency, and Scaling Challenges” and related work systematically evaluate five MAD frameworks across nine benchmarks.[^18][^19]

- They confirm that **MAD often fails to reliably beat carefully tuned single-agent strategies**, especially when counting inference-time cost.[^19][^18]
- Gains tend to appear on harder tasks or with weaker base models but diminish with strong base models.[^18]

**Overall picture for debate**: A* work in 2024–2025 suggests that MAD is *not* a free win; naïve all-to-all debate rarely justifies its cost. Architectures that add **structural biases** (diverse reasoning, sparse graphs, specialized roles, PGM-based reasoning) deliver more consistent gains.

***

## Comparative view: debate vs collaboration

### High-level trade-offs

The recent literature reveals systematic differences between collaboration-centric and debate-centric designs:

| Aspect | Collaboration architectures (MacNet, COPPER, FinCon, TradingAgents) | Debate architectures (MAD, sparse MAD, DMAD, Multi‑LLM Debate) |
|-------|---------------------------------------------------------------------|-----------------------------------------------------------------|
| Primary goal | Scale task-solving capacity via specialization and structure | Improve factuality/reasoning at test time via deliberation |
| Topology | DAGs, hierarchies, centralized or layered structures, shared message pools | Usually flat or regular graphs; some work on sparse / mixed graphs |
| Agent roles | Strongly role-based (analyst, manager, risk controller, coder, robot, etc.) | Often symmetric or light persona variations; DMAD enforces reasoning-strategy diversity |
| Calling pattern | Orchestrator schedules agents by graph order or workflow stage | Rounds of parallel calls; each agent reads others’ previous answers |
| Environment | Frequently explicit (finance market, game simulator, code interpreter) | Often “None” (pure textual questions) or evaluation datasets |
| Evaluation | End-to-end task performance, sample efficiency, robustness | Accuracy vs cost on QA/math/alignment benchmarks |

Collaboration systems are better matched to **multi-step workflows and tool use** (e.g., finance, software, robotics), whereas debate is more of a **test-time ensembling/verification technique**.

### When debate helps collaboration

Collaborative systems often embed *localized debate* rather than global MAD:

- TradingAgents uses debate among **bullish vs bearish researchers** and among **risk-seeking vs neutral vs conservative risk agents**, but confines debate within teams and writes outcomes back as structured state.[^4]
- FinCon includes a risk-control module that periodically initiates **self-critique/verbal reinforcement** episodes, which are debate-like but focused on updating shared beliefs rather than choosing among competing answers.[^3]
- In non-financial domains, several code and planning systems use **one or two critic/reviewer agents** rather than large debating swarms, effectively combining a collaborative pipeline with small-scale debate.[^7][^5]

Empirical evidence from sparse MAD and DMAD suggests that *if* debate is used, it should:

- Enforce **diverse reasoning styles** (as in DMAD) rather than just multiple similar agents.[^16]
- Use **sparser or structured communication** to reduce the impact of incorrect references and to extend effective debate rounds.[^6]
- Be combined with **strong aggregation rules** (facilitator/judge agents, PGM reasoning, or majority+consistency checks).[^13][^17][^6]

For finance agents, this points toward **small, structured debate modules inside a larger collaborative hierarchy** (e.g., bull vs bear researchers, risk profiles, or alpha factor committees) rather than whole‑system MAD.

***

## Finance-focused multi-agent LLM systems

### FinCon vs TradingAgents vs broader finance-agent landscape

Recent finance literature (plus an EMNLP‑25 survey on “LLM Agents in Finance”) identifies several multi-agent LLM frameworks:[^20][^21][^3][^4]

| System | Venue / Status | Architecture | Interaction style | Key tasks |
|--------|----------------|-------------|-------------------|-----------|
| **FinCon** | NeurIPS‑24 | Manager–analyst–risk hierarchy + conceptual verbal reinforcement | Mostly NL, with hierarchy and sporadic self-critique | Single-stock and portfolio trading with risk management[^3][^9] |
| **TradingAgents** | AAAI‑25 | Multi-team firm (analysts, researchers, traders, risk, manager) with hybrid structured+NL communication | Structured reports + localized ReAct debates | Multi-asset trading, back-tested across AAPL, GOOGL, AMZN etc.[^4][^20] |
| **Multi-Agents LLM Financial Trading (TradingAgents git)** | Preprint/implementation | Similar to AAAI paper; emphasizes debate-driven trading decisions and risk management | Structured global state + debates | Multi-asset backtests; strong returns and Sharpe ratio improvements[^20][^4] |
| **FinMEM / FinAgent / QuantAgent / AlphaAgents** | Mostly arXiv / finance venues | Single- or few-agent systems with memory, reflection, or small committees | Reflection, limited debate, RL-based fine-tuning | Alpha mining, portfolio construction, finetuned financial assistants[^22][^23][^24][^21] |

The EMNLP‑25 survey categorizes **multi-agent collaboration (MAC)** as one of the core patterns in finance agents, explicitly listing FinCon and TradingAgents as central MAC examples.[^21]

### Common architectural motifs in finance agents

Across these systems, several motifs recur:[^20][^21][^3][^4]

- **Firm-inspired hierarchies**: Manager/PM, analysts (fundamental, sentiment, technical, macro), traders, and risk controllers mimic real-world sell-side/buy-side organizations.[^20][^3][^4]
- **Multi-modal & multi-source tools**: Price/time-series, news, filings, social media, insider transactions, and technical indicators are ingested through tools that are called inside ReAct-style loops.[^21][^20][^3][^4]
- **Role-based debate**: Bull vs bear views, or risk-seeking vs conservative perspectives, are debated in constrained subgraphs, then summarized by facilitator/manager agents.[^20][^3][^4]
- **Memory and belief modules**: Systems such as FinCon and FinMEM use layered memories or conceptual belief stores to support longer-horizon reasoning, reduce hallucinations, and capture market regimes.[^21][^3]
- **Explainability requirements**: TradingAgents in particular logs natural-language reasoning and tool usage per decision, explicitly targeting auditability and debugging.[^4]

These motifs align naturally with role-based, hierarchical collaboration frameworks described in the IJCAI and collaboration-mechanisms surveys.[^5][^7]

***

## Implementation frameworks and libraries used in recent work

### General-purpose multi-agent frameworks

IJCAI‑24 and the collaboration-mechanisms survey both identify several widely used open-source frameworks for LLM multi-agents:[^7][^5]

- **MetaGPT**: Encodes human workflows as SOPs with role-based agents, plus a **shared message pool** where agents publish/subscribe to messages; primarily layered+shared-pool topology.[^5][^7]
- **CAMEL**: Role-playing framework where “user” and “assistant” agents both powered by LLMs collaborate; primarily used to generate conversational data and study role-based communication.[^7][^5]
- **AutoGen**: General framework for **conversable and programmable agents**, allowing agents configured in code or natural language and connected via conversational channels.[^5][^7]
- **AgentVerse / Consensus-LLM**: Support dynamic collaboration, consensus-seeking, and specialized roles (e.g., recruiter, decision-maker, evaluator) in a variety of domains.[^5]

Most A* papers referencing these frameworks still implement **custom orchestration logic** tuned to their domain; frameworks mainly provide abstractions for agent definitions, conversations, and tools.[^7][^5]

### Domain-specific and experimental frameworks

More specialized frameworks appear in particular subareas:

- **LLM-MA for software/code**: MetaGPT-style SOP pipelines, DAG-based orchestrators (similar to LangGraph) to manage code generation, testing, and repair agents.[^7][^5]
- **Game/society simulation**: Custom simulators for Chameleon/Undercover, Overcooked, diplomacy-style games, or town simulations like Generative Agents.[^8][^12][^7]
- **Finance**: TradingAgents open-source framework, FinCon codebase (manager–analyst–risk), plus alpha-mining frameworks like QuantAgent and AlphaAgents.[^21][^20][^3][^4]

Across these works, a common pattern is to treat orchestration as **graph or workflow programming** (DAGs, state machines, message buses) and to embed LLMs as **nodes or services** that are called with structured context and tool access.

***

## How agents are called and orchestrated in practice

### Orchestration patterns

From the surveyed A* literature, the dominant orchestration patterns are:[^2][^3][^5][^6][^7][^4]

1. **Static pipelines / workflows**
   - Example: FinCon and TradingAgents use fixed-stage pipelines closely mirroring human organizations: analysts → researchers → traders → risk managers → fund manager.[^3][^4]
   - Implementation: Orchestrator code or workflow graph that sequentially (sometimes with parallel fans) calls each agent, passing structured artifacts (reports, decisions) along.

2. **Graph-based scheduling (DAGs and communication graphs)**
   - MacNet treats agents as nodes in a DAG with topological orchestration (collaboration network).[^2]
   - Sparse MAD uses static undirected graphs to define whose answers each debating agent can see in each round.[^6]
   - Some Minecraft/code systems use DAGs where edges represent tool/agent handoffs.[^5]

3. **Centralized hub agents**
   - COPPER’s shared reflector acts as a hub that reads all actors’ trajectories and produces reflections that modify subsequent prompts.[^15]
   - Evaluator/judge agents in debate systems select final answers from competing agents (e.g., MAD with judges, PGM-aware decision agents).[^13][^17][^6]

4. **Environment-driven loops**
   - LLMArena and MAgIC treat the environment as the primary driver, calling agents each time a game tick or interaction step occurs (e.g., give clue, vote, invest, cooperate/defect).[^12][^8][^13]

5. **Dynamic agent generation / adaptation**
   - Surveys document systems that **spawn new agents on the fly** when tasks demand additional expertise (e.g., dynamic LLM-agent networks, learning-through-communication paradigms), though many of these are still in preprint stage rather than A* main tracks.[^7][^5]

### Communication modes and state management

Across most systems:[^20][^3][^5][^7][^4]

- **Text as primary medium**: All agents communicate via natural language messages, but many frameworks **structure those messages** (role tags, JSON-like fields, or report templates) for better parsing and retrieval.
- **Shared global state**: Finance systems and MetaGPT-like frameworks maintain explicit global state (message pool, knowledge base, environment) that agents query rather than reading full transcripts.[^7][^4]
- **Memory modules**: Reflection-based agents (e.g., FinMem, COPPER’s reflector) attach memory stores to agents or to a shared component that can summarize and retrieve prior experiences.[^15][^3][^7]

For finance MAS design, the **structured-report + global state pattern** used by TradingAgents is particularly attractive because it reduces context length, allows deterministic extraction (e.g., Pydantic schemas), and supports auditing.[^4]

***

## Design implications for a multi-agent finance analysis system

Although the primary goal of these papers is empirical benchmarking, they collectively suggest several concrete architectural choices for a multi-agent system oriented to financial analysis:

1. **Use a hierarchical, role-based collaboration backbone**
   - Adopt a firm-inspired structure: analysts (fundamental, sentiment, technical, macro), portfolio/trading planners, risk controllers, and a portfolio manager/orchestrator.[^21][^20][^3][^4]
   - Represent this as a DAG or workflow graph (MacNet-style) with clear stages and explicit artifacts (reports, recommendations, risk flags).[^2][^5][^4]

2. **Localize debate to small committees**
   - Use MAD/DMAD-type debate only inside submodules where trade-offs are truly ambiguous: bull vs bear research, alpha-factor selection, or risk adjustments.[^11][^16][^6][^4]
   - Force diverse reasoning styles or role personas in these debates (DMAD) rather than simply cloning the same agent; consider sparse connectivity or limited rounds.[^16][^6]

3. **Adopt structured communication with a shared state**
   - Follow TradingAgents’ hybrid protocol: structured reports as the main channel, short natural language debates logged into that state.[^4]
   - Use schemas to ensure downstream components (risk engine, compliance checker) can reliably parse decisions and rationales.

4. **Add reflective components for long-horizon adaptation**
   - Incorporate a COPPER- or FinCon-style reflector/belief module that periodically reviews performance, updates risk or strategy “concepts,” and re-injects them into agent prompts.[^15][^3]

5. **Anchor on established frameworks but customize orchestration**
   - Use AutoGen/MetaGPT/AgentVerse for agent definitions and basic conversation plumbing, but implement orchestration (DAG, committees, risk loops) explicitly in your own code for transparency and control.[^5][^7][^4]

These patterns are directly implied by the strongest A* work in 2024–2025 and align with both general MAS taxonomies and the specific requirements of financial workflows.

---

## References

1. [Large Language Model Based Multi-agents: A Survey of Progress ...](https://www.ijcai.org/proceedings/2024/890) - Electronic proceedings of IJCAI 2024

2. [Scaling Large Language Model-based Multi-Agent Collaboration](https://arxiv.org/abs/2406.07155) - Recent breakthroughs in large language model-driven autonomous agents have revealed that multi-agent...

3. [FinCon: A Synthesized LLM Multi-Agent System with Conceptual ...](https://arxiv.org/abs/2407.06567) - Large language models (LLMs) have demonstrated notable potential in conducting complex tasks and are...

4. [[PDF] TradingAgents: Multi-Agents LLM Financial Trading Framework](https://openreview.net/pdf/bf4d31f6b4162b5b1618ab5db04a32aec0bcbc25.pdf)

5. [Multi-Agent Collaboration Mechanisms: A Survey of LLMs](https://arxiv.org/html/2501.06322v1)

6. [[PDF] Improving Multi-Agent Debate with Sparse Communication Topology](https://aclanthology.org/2024.findings-emnlp.427.pdf)

7. [[PDF] Large Language Model Based Multi-agents - IJCAI](https://www.ijcai.org/proceedings/2024/0890.pdf)

8. [Assessing Capabilities of Large Language Models in Dynamic Multi ...](https://aclanthology.org/2024.acl-long.705/) - Junzhe Chen, Xuming Hu, Shuodi Liu, Shiyu Huang, Wei-Wei Tu, Zhaofeng He, Lijie Wen. Proceedings of ...

9. [FinCon: A Synthesized LLM Multi-Agent System with ...](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f7ae4fe91d96f50abc2211f09b6a7e49-Abstract-Conference.html)

10. [Should we be going MAD? A Look at Multi-Agent Debate Strategies ...](https://proceedings.mlr.press/v235/smit24a.html) - Recent advancements in large language models (LLMs) underscore their potential for responding to inq...

11. [Published as a conference paper at ICLR 2025](https://openreview.net/pdf/d67d70de207899a21f78262107dd3b5ec2d940b6.pdf)

12. [Investigation of Large Language Model Powered Multi-Agent in ...](https://aclanthology.org/2024.emnlp-main.416/) - MAgIC: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationa...

13. [[PDF] Investigation of Large Language Model Powered Multi-Agent in ...](https://aclanthology.org/2024.emnlp-main.416.pdf) - November 12-16, 2024 ©2024 Association for Computational Linguistics. MAgÏC: Investigation of Large ...

14. [NeurIPS Poster Reflective Multi-Agent Collaboration based on ...](https://neurips.cc/virtual/2024/poster/93147) - 2024 Poster. [ Paper] [ Slides] [ Poster] [ OpenReview] ... War and peace (waragent): Large language...

15. [NeurIPS Poster Reflective Multi-Agent Collaboration based ...](https://nips.cc/virtual/2024/poster/93147)

16. [Breaking Mental Set to Improve Reasoning through Diverse Multi ...](https://openreview.net/forum?id=t6QHYUOQL7) - The approach builds on the Multi-Agent Debate (MAD) method by requiring each agent—represented by an...

17. [Multi-LLM Debate: Framework, Principals, and Interventions](https://proceedings.neurips.cc/paper_files/paper/2024/hash/32e07a110c6c6acf1afbf2bf82b614ad-Abstract-Conference.html)

18. [Multi-LLM-Agents Debate - Performance, Efficiency, and Scaling ...](https://iclr.cc/virtual/2025/poster/31346) - Multi-Agent Debate (MAD) explores leveraging collaboration among multiple large language model (LLM)...

19. [Multi-LLM-Agents Debate - Performance, Efficiency, and Scaling Challenges](https://iclr-blogposts.github.io/2025/blog/mad/) - Multi-Agent Debate (MAD) explores leveraging collaboration among multiple large language model (LLM)...

20. [Multi-Agents LLM Financial Trading Framework](https://arxiv.org/html/2412.20138v3) - QuantAgent: Seeking Holy Grail in Trading by Self-Improving Large Language Model. ... BloombergGPT: ...

21. [PDF Large Language Model Agents in Finance: A Survey Bridging Research ...](https://aclanthology.org/anthology-files/anthology-files/pdf/findings/2025.findings-emnlp.972.pdf)

22. [Price-Driven Multi-Agent LLMs for High-Frequency Trading](https://ui.adsabs.harvard.edu/abs/2025arXiv250909995X/abstract) - Recent advances in Large Language Models (LLMs) have shown remarkable capabilities in financial reas...

23. [[PDF] Large Language Model based Multi-Agents for Equity Portfolio ...](https://arxiv.org/pdf/2508.11152.pdf)

24. [AlphaAgents: Large Language Model based Multi-Agents for Equity ...](https://arxiv.org/abs/2508.11152) - Abstract page for arXiv paper 2508.11152: AlphaAgents: Large Language Model based Multi-Agents for E...

