# ==============================================
# File: src/agents/digital_twin.py
# ==============================================
from __future__ import annotations
from typing import Dict, Any


class DigitalTwin:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("[DEBUG from digital_twin.py]: Running DigitalTwin agent")
        profile = state.get("profile", {})
        tx = state.get("transactions", [])
        spend = sum(t.get("amount", 0) for t in tx if t.get("type") == "expense")
        return "This is a placeholder response from Digital Twin agent."
        # return {
        #     "summary": f"Hello from Digital Twin {profile.get('name', 'user')}, your recent expenses total {spend}.",
        #     "next_best_actions": [
        #         "Set a monthly savings goal",
        #         "Automate 10% income to investments",
        #     ],
        # }
