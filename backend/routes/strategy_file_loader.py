from fastapi import APIRouter
from pathlib import Path
import json

router = APIRouter()

STRATEGY_DIR = Path("backend/backtester/strategies")

@router.get("/list")
def list_strategy_files():
    strategies = []
    for file in STRATEGY_DIR.glob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                strategies.append({
                    "strategy_id": file.stem,
                    "symbol": data.get("symbol", "UNKNOWN"),
                    "strategy_json": data
                })
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    return strategies
