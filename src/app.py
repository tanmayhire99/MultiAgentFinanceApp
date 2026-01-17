from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import re
import uvicorn
#from core import general_advisor
import os
from openai import OpenAI
#from core.finai_core import finai_agent
from .core.orchestrator import Orchestrator
from dotenv import load_dotenv

load_dotenv(".env")

app = FastAPI(title="FinAI MCP API", version="1.0.0")

# ---------------------------
# MCP Registry Definition
# ---------------------------
registry = {
    "name": "finai",
    "version": "1.0.0",
    "description": "FinAI MCP registry for 10 financial agents",
    "tools": [
        {"name": "upstox", "description": "Log in to upstox account using credentials."},
        {"name": "deep_web_research", "description": "Conduct deep web research on specified stocks or financial topics."},
        {"name": "us_stock", "description": "Analyze US stock market data and trends."},
        {"name": "indian_stock", "description": "Analyze Indian stock market data and trends."},
        {"name": "digital_twin", "description": "Interact with the digital twin of a financeial expert to get personalized advice."},
        {"name": "general_advisor", "description": "Provide general financial advice based on user profile and transactions."},
    ],
    "routing": {
        "strategy": "langgraph+rules",
        "rules": [
            {"pattern": r"\b(portfolio|login)\b", "route": "upstox"},
            {"pattern": r"\b(search|web)\b", "route": "deep_web_research"},
            {"pattern": r"\b(US|Stock)\b", "route": "us_stock"},
            {"pattern": r"\b(Indian|Stock|)\b", "route": "indian_stock"},
            {"pattern": r"\b(DigitalTwin|persona)\b", "route": "digital_twin"},
            {"pattern": r"\b(advice)\b", "route": "general_advisor"},
        ],
        "fallback": "general_advisor"
    }
}


# ---------------------------
# Models
# ---------------------------
class AgentRequest(BaseModel):
    query: str
    profile: dict
    transactions: list[dict] | None = None


# ---------------------------
# API Endpoints
# ---------------------------

@app.get("/registry")
def get_registry():
    """Return MCP registry JSON"""
    return registry


@app.get("/route")
def route_query(query: str = Query(..., description="User query")):
    """Route query to the correct agent"""
    for rule in registry["routing"]["rules"]:
        if re.search(rule["pattern"], query, re.IGNORECASE):
            return {"query": query, "agent": rule["route"]}
    return {"query": query, "agent": registry["routing"]["fallback"]}


@app.post("/agent/{agent_name}")
def run_agent(agent_name: str, request: AgentRequest):
    tools = {tool["name"]: tool for tool in registry["tools"]}
    if agent_name not in tools:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    # if agent_name == "general_advisor":

    #     result = src.agents.general_advisor.invoke({"query": request.query})
        
    #     return {"agent": agent_name, "result": result["response"]}

    return {
        "agent": agent_name,
        "input": request.dict(),
        "result": f"Simulated response from {agent_name}"
    }


@app.post("/query")
def query_entry(request: AgentRequest):
    print("[DEBUG from app.py]: Called Core Orchestrator")
    orch = Orchestrator()
    final = orch.run(
        user_id=request.profile.get("user_id"),# if request.profile else "anonymous",
        query=request.query,
    )
    print("[DEBUG from app.py]: Final result from Orchestrator:")
    return final

@app.post("/chat")
def chat_entry():
    print("Called Core Orchestrator")
    return True



# ---------------------------
# Run with: uvicorn app:app --reload #uvicorn src.app:app --reload
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# Test with:
''' 
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query": "Analyze my portfolio and recent transactions to provide financial advice", "profile": {"user_id": "MT24100"}}'
'''