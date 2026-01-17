# ==============================================
# File: src/tools/data_loader.py
# Description: Load user profile & transactions from /data
# ==============================================
from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Dict, Any, List

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_user_profile(user_id: str) -> Dict[str, Any]:
    profile_path = DATA_DIR / "dummy_user.json"
    if profile_path.exists():
        with open(profile_path, "r", encoding="utf-8") as f:
            prof = json.load(f)
            # If file contains many, pick by id; else return same stub with override
            if isinstance(prof, dict) and prof.get("user_id"):
                print(f"[DEBUG from data_loader.py]: Loaded profile for user_id={user_id} from dummy_user.json")
                return prof
    # fallback stub
    return {"user_id": user_id, "name": "Demo User", "age": 35, "risk_profile": "moderate"}


def load_user_transactions(user_id: str) -> List[Dict[str, Any]]:
    tx_path = DATA_DIR / "dummy_transactions.csv"
    rows: List[Dict[str, Any]] = []
    if tx_path.exists():
        with open(tx_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    "tx_id": r.get("tx_id"),
                    "amount": float(r.get("amount", 0) or 0),
                    "type": r.get("type", "expense"),
                    "category": r.get("category", "misc"),
                    "date": r.get("date"),
                })
    if not rows:
        rows = [
            {"tx_id": "t1", "amount": 2500.0, "type": "expense", "category": "groceries", "date": "2025-08-01"},
            {"tx_id": "t2", "amount": 60000.0, "type": "income", "category": "salary", "date": "2025-08-01"},
        ]
    return rows

