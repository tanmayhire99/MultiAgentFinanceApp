# ==============================================
# File: src/agents/indian_stock.py (minimal example to make e2e work)
# ==============================================
from __future__ import annotations
from typing import Dict, Any


class IndianStock:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("[DEBUG from indian_stock.py]: Running IndianStock agent")
        profile = state.get("profile", {})
        tx = state.get("transactions", [])
        spend = sum(t.get("amount", 0) for t in tx if t.get("type") == "expense")
        return "This is a placeholder response from Indian Stock agent."
        # return {
        #     "summary": f"Hello {profile.get('name', 'user')}, your recent expenses total {spend}.",
        #     "next_best_actions": [
        #         "Set a monthly savings goal",
        #         "Automate 10% income to investments",
        #     ],
        # }
