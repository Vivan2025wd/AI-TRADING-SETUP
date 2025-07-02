from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.strategy_engine.json_strategy_parser import parse_strategy_json
from backend.strategy_registry import (
    save_strategy,
    register_strategy,
    get_registered_strategies,
    get_strategy_by_symbol,
    delete_strategy
)
from backend.strategy_engine.strategy_health import StrategyHealth

from pathlib import Path
import os
import json

router = APIRouter(tags=["Strategies"])
STRATEGY_DIR = Path("backend/backtester/strategies")
PERFORMANCE_DIR = Path("backend/storage/performance_logs")

class StrategyPayload(BaseModel):
    strategy_id: str
    symbol: str
    strategy_json: dict

# ✅ Save & register strategy
@router.post("/save")
def save_user_strategy(payload: StrategyPayload):
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

    # Construct full strategy dict to validate
    strategy_to_validate = {
        "symbol": payload.symbol,
        "indicators": payload.strategy_json
    }

    # Validate strategy format
    try:
        parse_strategy_json(json.dumps(strategy_to_validate))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save and register
    success = save_strategy(payload.symbol, payload.strategy_id, strategy_to_validate)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save strategy.")

    register_strategy(payload.symbol, payload.strategy_id)
    return {"message": "Strategy saved and registered successfully."}

# ✅ List all registered strategies
@router.get("/list")
def list_registered_strategies():
    return get_registered_strategies()

# ✅ List strategies for a specific symbol
@router.get("/{symbol}")
def list_strategies_by_symbol(symbol: str):
    strategies = get_strategy_by_symbol(symbol.upper())
    if not strategies:
        raise HTTPException(status_code=404, detail="No strategies found for this symbol.")
    return strategies

# ✅ Delete strategy file
@router.delete("/strategies/{symbol}/{strategy_id}")
def remove_strategy(symbol: str, strategy_id: str):
    success = delete_strategy(symbol.upper(), strategy_id)
    if not success:
        raise HTTPException(status_code=404, detail="Strategy not found or could not be deleted.")
    return {"message": f"Strategy {strategy_id} for {symbol.upper()} deleted successfully."}

# ✅ Load performance data for a given strategy
@router.get("/{symbol}/{strategy_id}/performance")
def get_strategy_performance(symbol: str, strategy_id: str):
    log_path = PERFORMANCE_DIR / f"{symbol}_strategy_{strategy_id}.json"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No performance log found.")

    with open(log_path, "r") as f:
        trades = json.load(f)

    stats = StrategyHealth(trades).summary()
    return stats
