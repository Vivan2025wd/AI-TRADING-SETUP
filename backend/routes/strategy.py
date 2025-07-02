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
    # The strategy_id from the path is the full file stem, e.g., "BTC_strategy_st2"
    # The delete_strategy function expects the symbol and the actual ID part, e.g., "BTC", "st2"
    # We need to extract the actual ID part if the full stem is passed.
    # Expected filename format: {symbol}_strategy_{actual_id}.json

    normalized_symbol = symbol.upper()
    actual_id_part = strategy_id

    # If strategy_id from path looks like "SYMBOL_strategy_IDPART"
    # and symbol from path matches SYMBOL, extract IDPART.
    prefix_to_check = f"{normalized_symbol}_strategy_"
    if strategy_id.startswith(prefix_to_check):
        actual_id_part = strategy_id[len(prefix_to_check):]

    success = delete_strategy(normalized_symbol, actual_id_part)
    if not success:
        raise HTTPException(status_code=404, detail=f"Strategy {actual_id_part} for {normalized_symbol} not found or could not be deleted.")
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
