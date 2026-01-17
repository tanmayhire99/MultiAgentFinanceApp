# ==============================================
# File: src/core/planner.py
# Description: Planner Agent for FinAI using LangGraph-compatible classes
# Updated to support structured output with guided_json
# ==============================================
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import os

load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable is missing")

from src.tools.llm_client import LLMClient


list_of_agents = {
    "upstox": {"upstox_login": "Log in to upstox account using credentials.", "get_portfolio": "Retrieve the user's investment portfolio from upstox."},
    "deep_web_research": {"research_stocks": "Conduct deep web research on specified stocks or financial topics."},
    "us_stock_analysis": {"analyze_us_stocks": "Analyze US stock market data and trends."},
    "indian_stock_analysis": {"analyze_indian_stocks": "Analyze Indian stock market data and trends."},
    "digital_twin_persona": {"ask_persona": "Interact with the digital twin of a financial expert to get personalized advice."},
    "general_advisor": {"advise": "Provide general financial advice based on user profile and transactions."},
}


class PlanStep(BaseModel):
    id: int = Field(description="Step number in sequence")
    description: str = Field(description="Clear description of what this step accomplishes")
    agent: str = Field(description="Name of the agent to handle this step")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters for the agent")
    success_criteria: str = Field(description="How to verify this step succeeded")


class Plan(BaseModel):
    goal: str = Field(description="The overall objective to achieve")
    rationale: str = Field(description="Why this plan structure was chosen")
    steps: List[PlanStep] = Field(description="Ordered list of steps to execute")


class PlannerAgent:
    """LLM-backed planner that turns a user goal into an executable multi-step plan.

    Outputs a Plan with structured steps that the Orchestrator can loop through.
    """
    def __init__(self, llm_client: LLMClient, system_prompt: Optional[str] = None):
        self.llm_client = llm_client
        self.system_prompt = system_prompt or (
            f'''You are the FinAI Planner. Decompose a user's financial query into a minimal,
            reliable plan. These are the list of agents available along with the tools available in them and their description : {list_of_agents}
            Choose the *single best* agent when in doubt. Only include steps that move the goal forward.'''
        )
        # Pre-compute the JSON schema for guided_json
        self.plan_schema = Plan.model_json_schema()

    def _parse_plan(self, text: str, intent: Optional[str], goal: str) -> Plan:
        """Parse the LLM's guaranteed JSON response from guided_json.
        
        Since NVIDIA NIM's guided_json ensures output matches the schema,
        this parser simply validates and constructs the Plan object.
        """
        try:
            # guided_json guarantees valid JSON matching the Plan schema
            text = text.strip()
            
            # Extract JSON if wrapped in any markdown or extra text
            if text.startswith("```"):
                # Handle markdown code blocks
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            
            # Parse JSON
            json_data = json.loads(text)
            
            # Validate and construct Plan using Pydantic
            plan = Plan(**json_data)
            
            print(f"[DEBUG from planner.py]: Parsed plan with {len(plan.steps)} steps") #, steps:\n{plan.steps}")
            print(f"[DEBUG from planner.py]: Full plan JSON:\n{json.dumps(json_data, indent=2)}")
            return plan
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[ERROR from planner.py]: Failed to parse plan: {e}")
            print(f"[DEBUG from planner.py]: Raw text: {text}")
            
            # Fallback: return a safe default plan
            fallback_plan = Plan(
                goal=goal,
                rationale="Fallback plan due to parsing error",
                steps=[
                    PlanStep(
                        id=1,
                        description="General advice based on the query.",
                        agent="general_advisor",
                        inputs={},
                        success_criteria="General advice provided without errors.",
                    )
                ]
            )
            print(f"[DEBUG from planner.py]: Using fallback plan with {len(fallback_plan.steps)} steps")
            return fallback_plan

    def plan(self, goal: str, intent: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Plan:
        """Create a structured plan for the given goal using guided_json."""
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Goal: {goal} \nKnown intent: {intent or 'unknown'}"},
        ]

        # Call LLM with guided_json for guaranteed structure
        resp = self.llm_client.get_chat_model(
            msgs,
            temperature=0.3,
            guided_json=self.plan_schema
        )
        
        text = resp.get("content", "") if isinstance(resp, dict) else str(resp)
        # print(f"[DEBUG from planner.py]: LLM response:\n{text}")
        
        return self._parse_plan(text, intent, goal)


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Initialize the planner
    llm_client = LLMClient()
    planner = PlannerAgent(llm_client)
    
    # Create a plan
    goal = "Analyze my portfolio and get recommendations for improving my investment strategy."
    intent = "investment_advice"
    
    print("="*60)
    print(f"Creating plan for goal: {goal}")
    print("="*60)
    
    plan = planner.plan(goal=goal, intent=intent)
    
    print("\n" + "="*60)
    print("Plan Generated Successfully!")
    print("="*60)
    print(f"Goal: {plan.goal}")
    print(f"Rationale: {plan.rationale}")
    print(f"\nSteps:")
    for step in plan.steps:
        print(f"  {step.id}. [{step.agent.upper()}] {step.description}")
        print(f"     Success Criteria: {step.success_criteria}")
        if step.inputs:
            print(f"     Inputs: {step.inputs}")
