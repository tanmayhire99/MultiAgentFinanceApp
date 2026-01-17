#!/usr/bin/env python3
"""
Sample program demonstrating guided_json structured output with NVIDIA NIM API.
Uses the OpenAI-compatible API endpoint for NVIDIA NIM models.

Usage:
    export NVIDIA_API_KEY="your_api_key_here"
    python nvidia_nim_guided_json_examples.py
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import field
from pydantic import BaseModel, Field
from openai import OpenAI

# ==========================================
# Configuration
# ==========================================
API_KEY = "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"

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

# Initialize OpenAI-compatible client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# ==========================================
# Pydantic Models (Define Your Schema)
# ==========================================
class FinancialAction(BaseModel):
    """Individual action item in a financial plan."""
    id: int = Field(description="Step number in sequence")
    description: str = Field(description="Clear description of what this action accomplishes")
    priority: str = Field(
        description="Priority level: high, medium, or low",
        pattern="^(high|medium|low)$"
    )
    estimated_time_hours: float = Field(
        description="Estimated time to complete in hours",
        ge=0
    )


class FinancialPlan(BaseModel):
    """Structured financial plan with multiple actions."""
    goal: str = Field(description="The overall financial objective")
    rationale: str = Field(description="Why this approach was chosen")
    actions: List[FinancialAction] = Field(description="Ordered list of action items")
    total_estimated_hours: float = Field(
        description="Total estimated time for all actions",
        ge=0
    )


# ==========================================
# Example 1: Simple Structured Output
# ==========================================
def example_1_simple_plan():
    """Generate a simple financial plan using guided_json."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Financial Plan with guided_json")
    print("="*60)
    
    schema = FinancialPlan.model_json_schema()
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Create a financial plan to save for retirement in 10 years with $5000 monthly savings."
            }
        ],
        temperature=0.3,
        max_tokens=1024,
        extra_body={"guided_json": schema}
    )
    
    response_text = completion.choices[0].message.content
    print(f"\nRaw LLM Response:\n{response_text}\n")
    
    # Parse the guaranteed JSON
    plan_data = json.loads(response_text)
    plan = FinancialPlan(**plan_data)
    
    print(f"\nParsed Plan:")
    print(f"  Goal: {plan.goal}")
    print(f"  Rationale: {plan.rationale}")
    print(f"  Total Estimated Hours: {plan.total_estimated_hours}")
    print(f"\n  Actions:")
    for action in plan.actions:
        print(f"    {action.id}. [{action.priority.upper()}] {action.description}")
        print(f"       Time: {action.estimated_time_hours} hours")
    
    return plan


# ==========================================
# Example 2: Investment Analysis Schema
# ==========================================
class Investment(BaseModel):
    """Investment opportunity analysis."""
    name: str = Field(description="Name of the investment")
    type: str = Field(
        description="Type of investment: stock, bond, etf, mutual_fund, crypto, real_estate",
        pattern="^(stock|bond|etf|mutual_fund|crypto|real_estate)$"
    )
    risk_level: str = Field(
        description="Risk level: low, medium, high",
        pattern="^(low|medium|high)$"
    )
    expected_return_percent: float = Field(
        description="Expected annual return as percentage",
        ge=-100,
        le=500
    )
    pros: List[str] = Field(description="List of advantages")
    cons: List[str] = Field(description="List of disadvantages")


class InvestmentPortfolio(BaseModel):
    """Portfolio recommendation with multiple investments."""
    portfolio_name: str = Field(description="Name of the portfolio strategy")
    target_allocation_percent: int = Field(
        description="Target allocation percentage",
        ge=0,
        le=100
    )
    investments: List[Investment] = Field(
        description="List of recommended investments",
        min_items=1,
        max_items=10
    )
    recommendation: str = Field(
        description="Overall recommendation summary"
    )


def example_2_investment_analysis():
    """Generate an investment portfolio recommendation using guided_json."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Investment Portfolio Analysis with guided_json")
    print("="*60)
    
    schema = InvestmentPortfolio.model_json_schema()
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": """Recommend an investment portfolio for a 35-year-old investor with 
                moderate risk tolerance and $100,000 to invest. Focus on long-term growth."""
            }
        ],
        temperature=0.3,
        max_tokens=2048,
        extra_body={"guided_json": schema}
    )
    
    response_text = completion.choices[0].message.content
    print(f"\nRaw LLM Response:\n{response_text}\n")
    
    # Parse the guaranteed JSON
    portfolio_data = json.loads(response_text)
    portfolio = InvestmentPortfolio(**portfolio_data)
    
    print(f"\nParsed Portfolio:")
    print(f"  Strategy: {portfolio.portfolio_name}")
    print(f"  Target Allocation: {portfolio.target_allocation_percent}%")
    print(f"\n  Investments:")
    for inv in portfolio.investments:
        print(f"    • {inv.name} ({inv.type.upper()})")
        print(f"      Risk: {inv.risk_level} | Expected Return: {inv.expected_return_percent}%")
        print(f"      Pros: {', '.join(inv.pros[:2])}")
        print(f"      Cons: {', '.join(inv.cons[:2])}")
    print(f"\n  Recommendation: {portfolio.recommendation}")
    
    return portfolio


# ==========================================
# Example 3: Budget Breakdown Schema
# ==========================================
class BudgetCategory(BaseModel):
    """Individual budget category."""
    category: str = Field(
        description="Budget category",
        pattern="^(Housing|Food|Transportation|Utilities|Healthcare|Entertainment|Savings|Debt|Other)$"
    )
    percentage_of_income: float = Field(
        description="Percentage of total income",
        ge=0,
        le=100
    )
    monthly_amount: float = Field(
        description="Monthly amount in dollars",
        ge=0
    )
    notes: str = Field(description="Additional notes for this category")


class BudgetPlan(BaseModel):
    """Complete monthly budget plan."""
    total_monthly_income: float = Field(
        description="Total monthly income in dollars",
        ge=0
    )
    categories: List[BudgetCategory] = Field(
        description="Budget categories and allocations",
        min_items=3,
        max_items=10
    )
    summary: str = Field(description="Summary and recommendations")


def example_3_budget_planning():
    """Generate a monthly budget plan using guided_json."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Monthly Budget Planning with guided_json")
    print("="*60)
    
    schema = BudgetPlan.model_json_schema()
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Create a monthly budget plan for someone earning $5000/month with realistic allocations."
            }
        ],
        temperature=0.3,
        max_tokens=1500,
        extra_body={"guided_json": schema}
    )
    
    response_text = completion.choices[0].message.content
    print(f"\nRaw LLM Response:\n{response_text}\n")
    
    # Parse the guaranteed JSON
    budget_data = json.loads(response_text)
    budget = BudgetPlan(**budget_data)
    
    print(f"\nParsed Budget Plan:")
    print(f"  Total Monthly Income: ${budget.total_monthly_income:.2f}")
    print(f"\n  Categories:")
    total_allocated = 0
    for cat in budget.categories:
        print(f"    {cat.category:20} {cat.percentage_of_income:5.1f}%  ${cat.monthly_amount:8.2f}")
        total_allocated += cat.monthly_amount
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':20} {(total_allocated/budget.total_monthly_income)*100:5.1f}%  ${total_allocated:8.2f}")
    print(f"\n  Summary: {budget.summary}")
    
    return budget


# ==========================================
# Example 4: For Your Planner Use Case
# ==========================================
class PlanStep(BaseModel):
    """Individual planning step."""
    id: int = Field(description="Step number in sequence")
    description: str = Field(description="Clear description of what this step accomplishes")
    agent: str = Field(description="Name of the agent to handle this step")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters for the agent")
    success_criteria: str = Field(description="How to verify this step succeeded")


class Plan(BaseModel):
    """Structured financial plan."""
    goal: str = Field(description="The overall objective to achieve")
    rationale: str = Field(description="Why this plan structure was chosen")
    steps: List[PlanStep] = Field(description="Ordered list of steps to execute")


def example_4_finai_planner():
    """Generate a FinAI planner output using guided_json."""
    print("\n" + "="*60)
    print("EXAMPLE 4: FinAI Planner with guided_json")
    print("="*60)
    
    list_of_agents = {
        "upstox": {"upstox_login": "Log in to upstox account", "get_portfolio": "Retrieve portfolio"},
        "us_stock_analysis": {"analyze_us_stocks": "Analyze US stock market data"},
        "indian_stock_analysis": {"analyze_indian_stocks": "Analyze Indian stock market data"},
        "general_advisor": {"advise": "Provide general financial advice"},
    }
    
    schema = Plan.model_json_schema()
    
    system_prompt = f"""You are the FinAI Planner. Decompose a user's financial query into a minimal,
reliable plan. Available agents: {list_of_agents}
Choose the *single best* agent when in doubt. Only include steps that move the goal forward."""
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "I want to analyze my portfolio and get recommendations for improving my investment strategy."
            }
        ],
        temperature=0.3,
        max_tokens=1024,
        extra_body={"guided_json": schema}
    )
    
    response_text = completion.choices[0].message.content
    print(f"\nRaw LLM Response:\n{response_text}\n")
    
    # Parse the guaranteed JSON
    plan_data = json.loads(response_text)
    plan = Plan(**plan_data)
    
    print(f"\nParsed Plan:")
    print(f"  Goal: {plan.goal}")
    print(f"  Rationale: {plan.rationale}")
    print(f"\n  Steps:")
    for step in plan.steps:
        print(f"    {step.id}. [{step.agent.upper()}] {step.description}")
        print(f"       Success Criteria: {step.success_criteria}")
        if step.inputs:
            print(f"       Inputs: {step.inputs}")
    
    return plan


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# NVIDIA NIM guided_json Structured Output Examples")
    print("#"*60)
    
    try:
        # Run examples
        # Uncomment the ones you want to test
        
        # plan = example_1_simple_plan()
        # portfolio = example_2_investment_analysis()
        # budget = example_3_budget_planning()
        finai_plan = example_4_finai_planner()
        
        print("\n" + "="*60)
        print("Examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
