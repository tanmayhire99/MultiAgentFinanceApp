
# ==============================================
# File: src/core/orchestrator.py (UPDATED to integrate Crew path)
# ==============================================
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from src.tools.llm_client import LLMClient
try:
    from src.tools.data_loader import load_user_profile, load_user_transactions
except:
    print("[DEBUG from orchestrator.py]: Data loader not available, using stubs")
    def load_user_profile(user_id: str) -> Dict[str, Any]:
        return {"user_id": user_id, "name": "Demo User", "risk_profile": "moderate"}
    def load_user_transactions(user_id: str) -> List[Dict[str, Any]]:
        return [{"tx_id": "t1", "amount": 2500, "type": "expense", "category": "groceries"}]

from src.core.planner import PlannerAgent, Plan
from src.core.router import RouterAgent

# Import minimal agents for registry & resolution
from src.agents.__all_minimal__ import (
    Upstox, DigitalTwin, DeepWebResearch, USStock, IndianStock, Advisor
)


def _resolve_agent(agent_key: str):
    registry = {
        "upstox": Upstox,
        "digital_twin_persona": DigitalTwin,
        "deep_web_research": DeepWebResearch,
        "us_stock_analysis": USStock,
        "indian_stock_analysis": IndianStock,
        "general_advisor": Advisor,
    }
    if agent_key not in registry:
        raise KeyError(f"Unknown agent key: {agent_key}")
    return registry[agent_key]()


class GraphState(TypedDict, total=False):
    user_id: str
    query: str
    hinted_agent: Optional[str]
    intent: Optional[str]
    plan: Plan
    step_index: int
    profile: Dict[str, Any]
    transactions: List[Dict[str, Any]]
    selected_agent: Optional[str]
    scratchpad: List[Dict[str, Any]]
    result: Dict[str, Any]
    error: Optional[str]
    #mode: Optional[str] # [ main / agent ] (individual agent might need to use planner again)


@dataclass
class OrchestratorConfig:
    max_steps: int = 8


class Orchestrator:
    def __init__(self, config: Optional[OrchestratorConfig] = None, llm_client=LLMClient()):
        print("[DEBUG from orchestrator.py]: Initialized Core Orchestrator")
        self.config = config or OrchestratorConfig()
        self.planner = PlannerAgent(llm_client)
        self.router = RouterAgent()
        self.graph = self._build_graph()

    def _node_plan(self, state: GraphState) -> GraphState:
        '''
        START node
        Passes GraphState with 'query' and optional 'intent' to PlannerAgent.

        returns :
        Plan(
            goal="Answer the user's query",
            rationale="A tight, minimal sequence for reliability and speed.",
            steps=[
                PlanStep(
                    id=1,
                    description="Route to most suitable agent.",
                    agent="insurance_analyzer",
                    inputs={},
                    success_criteria="Selected agent matches the intent.",
                ), PlanStep(id=2, ...), ... ]
        )
        '''
        print("[DEBUG from orchestrator.py]: Planning step")
        plan = self.planner.plan(goal=state["query"], intent=state.get("intent"))
        new: GraphState = dict(state)
        new["plan"] = plan
        new["step_index"] = 0
        new.setdefault("scratchpad", []).append({"event": "planned", "steps": len(plan.steps)})
        return new

    def _node_prepare_data(self, state: GraphState) -> GraphState:
        print("[DEBUG from orchestrator.py]: Preparing user data")
        uid = state["user_id"]
        profile = load_user_profile(uid)
        print(f"[DEBUG from orchestrator.py]: Loaded profile for user_id={uid}")
        tx = load_user_transactions(uid)
        new = dict(state)
        new["profile"], new["transactions"] = profile, tx
        new.setdefault("scratchpad", []).append({"event": "data_loaded", "tx_count": len(tx)})
        #print(new)
        return new

    def _node_route(self, state: GraphState) -> GraphState:
        print("[DEBUG from orchestrator.py]: Routing to agent")
        hinted = state.get("fin_advisor")
        step = state["plan"].steps[state["step_index"]]
        agent_key = step.agent if step.agent != "router" else self.router.route(state["query"], hinted)
        new = dict(state)
        new["selected_agent"] = agent_key
        print(f"[DEBUG from orchestrator.py]: Routed to agent '{agent_key}'")
        new.setdefault("scratchpad", []).append({"event": "routed", "agent": agent_key})
        print("Printing the state from router",new)
        return new

    def _node_execute(self, state: GraphState) -> GraphState:
        print("[DEBUG from orchestrator.py]: Executing agent")
        agent_key = state.get("selected_agent")
        if not agent_key:
            raise RuntimeError("No agent selected to execute.")

        # # Crew-path: if plan asks for a composite outcome, run a small crew
        # if agent_key == "fin_advisor" and any(k in state["query"].lower() for k in ["overall", "holistic", "comprehensive"]):
        #     from src.core.crew import Crew, CrewTask
        #     crew = Crew([
        #         CrewTask("fin_score", lambda s: {"query": s["query"], "profile": s.get("profile"), "transactions": s.get("transactions")} ),
        #         CrewTask("investment_coach", lambda s: {"query": s["query"], "profile": s.get("profile"), "transactions": s.get("transactions")} ),
        #         CrewTask("tax_planner", lambda s: {"query": s["query"], "profile": s.get("profile"), "transactions": s.get("transactions")} ),
        #     ])
        #     payload = crew.execute({
        #         "query": state["query"],
        #         "profile": state.get("profile"),
        #         "transactions": state.get("transactions"),
        #     })
        # else:
        #     print(f"[DEBUG from orchestrator.py]: Resolving and running agent '{agent_key}'")
        #     agent = _resolve_agent(agent_key)
        #     payload = agent.run({
        #         "query": state["query"],
        #         "profile": state.get("profile"),
        #         "transactions": state.get("transactions"),
        #     })

        print(f"[DEBUG from orchestrator.py]: Resolving and running agent '{agent_key}'")
        agent = _resolve_agent(agent_key)
        payload = agent.run({
            "query": state["query"],
            "profile": state.get("profile"),
            "transactions": state.get("transactions"),
        })

        new = dict(state)
        new.setdefault("scratchpad", []).append({"event": "executed", "agent": agent_key})
        new["result"] = payload
        return new

    def _node_next_step(self, state: GraphState) -> GraphState:
        '''
        Move to next step in the plan and increment step_index.
        '''
        print("[DEBUG from orchestrator.py]: Moving to next step")
        new = dict(state)
        new["step_index"] = state.get("step_index", 0) + 1
        return new

    def _should_prepare_data(self, state: GraphState) -> str:
        print("[DEBUG from orchestrator.py]: Checking if data preparation is needed")
        if not state.get("profile") or not state.get("transactions"):
            return "prepare_data"
        return "route"

    def _should_continue(self, state: GraphState) -> str:
        print("[DEBUG from orchestrator.py]: Checking if should continue to next step")
        step_idx = state.get("step_index", 0)
        
        total = len(state["plan"].steps)
        if step_idx >= total or step_idx >= self.config.max_steps:
            print("[DEBUG from orchestrator.py]: Reached max steps or end of plan, ending orchestration")
            return "end"
        next_step_agent = state["plan"].steps[step_idx].agent
        if next_step_agent == "router":
            # if it is a routing step, END
            print("[DEBUG from orchestrator.py]: Next step is routing, ending orchestration")
            return "end"
        # print(state.get("plan"))
        return "execute"

    def _build_graph(self):
        g = StateGraph(GraphState)
        g.add_node("planner", self._node_plan)
        g.add_node("prepare_data", self._node_prepare_data)
        g.add_node("route", self._node_route)
        g.add_node("execute", self._node_execute)
        g.add_node("next", self._node_next_step)

        g.add_edge(START, "planner")
        g.add_conditional_edges("planner", self._should_prepare_data, {
            "prepare_data": "prepare_data",
            "route": "route",
        })
        g.add_edge("prepare_data", "route")
        g.add_conditional_edges("route", self._should_continue, {
            "execute": "execute",
            "route": "route",
            "end": END,
        })
        g.add_edge("execute", "next")
        g.add_conditional_edges("next", self._should_continue, {
            "route": "route",
            "execute": "execute",
            "end": END,
        })
        return g.compile()

    def run(self, user_id: str, query: str, hinted_agent: Optional[str] = None) -> Dict[str, Any]:
        '''
        takes user_id, query, optional hinted_agent as input
        returns final orchestration result with answer, trace, selected_agent, steps
        '''
        print("[DEBUG from orchestrator.py]: Running orchestration graph")
        state: GraphState = {"user_id": user_id, "query": query, "hinted_agent": hinted_agent}
        
        #invoke the graph
        final = self.graph.invoke(state)
        return {
            "answer": final.get("result"),
            #"trace": final.get("scratchpad", []),
            "agent": final.get("selected_agent"),
            #"steps": [s.__dict__ for s in final["plan"].steps],
        }
