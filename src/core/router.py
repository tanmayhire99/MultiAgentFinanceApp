# ==============================================
# File: src/core/router.py
# Description: Intent Router Agent
# ==============================================
from __future__ import annotations
from typing import Dict, Any, Optional
import re

try:
    from src.tools.llm_client import get_chat_model
except Exception:
    def get_chat_model(model: str = "gpt-4o-mini", **_: Any):
        class _Dummy:
            def invoke(self, messages):
                return {"content": "fin_advisor"}
        return _Dummy()

INTENT_MAP = {
    r"\b(score|cibil|credit score)\b": "fin_score",
    r"\b(loan|emi|credit|borrow|mortgage)\b": "credits_loans",
    r"\b(invest|portfolio|mutual fund|stocks?)\b": "investment_coach",
    r"\b(insurance|premium|coverage|term plan)\b": "insurance_analyzer",
    r"\b(retire|pension|401k|nps)\b": "retirement_planner",
    r"\b(tax|itr|deduction|exemption|section)\b": "tax_planner",
    r"\b(fraud|suspicious|chargeback|phishing)\b": "fraud_shield",
}

class RouterAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = get_chat_model(model=model)

    def route(self, query: str, hinted: Optional[str] = None) -> str:
        if hinted:
            return hinted
        for pattern, agent in INTENT_MAP.items():
            if re.search(pattern, query, re.I):
                #print(f"[DEBUG from router.py]: Routed by rule: pattern='{pattern}' to agent='{agent}'")
                return agent
        # fallback to LLM classification
        msgs = [
            {"role": "system", "content": "Return only the best agent key from: fin_score, credits_loans, investment_coach, insurance_analyzer, retirement_planner, tax_planner, fraud_shield, fin_advisor."},
            {"role": "user", "content": f"Query: {query}"},
        ]
        out = self.llm.invoke(msgs)
        text = (out.get("content") if isinstance(out, dict) else str(out)).strip()
        text = re.sub(r"[^a-zA-Z_]+", "", text)
        return text or "fin_advisor"
