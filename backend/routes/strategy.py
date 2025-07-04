from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import json
from pathlib import Path

from backend.strategy_engine.json_strategy_parser import parse_strategy_json
from backend.strategy_registry import (
    save_strategy,
    register_strategy,
    get_registered_strategies,
    get_strategy_by_symbol,
    delete_strategy,
)
from backend.strategy_engine.strategy_health import StrategyHealth

router = APIRouter(tags=["Strategies"])

STRATEGY_DIR = Path("backend/backtester/strategies")
PERFORMANCE_DIR = Path("backend/storage/performance_logs")

class StrategyPayload(BaseModel):
    strategy_id: str
    symbol: str
    strategy_json: dict


# Save & register strategy
@router.post("/save")
def save_user_strategy(payload: StrategyPayload):
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

    strategy_to_validate = {
        "symbol": payload.symbol,
        "indicators": payload.strategy_json,
    }

    try:
        parse_strategy_json(json.dumps(strategy_to_validate))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    success = save_strategy(payload.symbol, payload.strategy_id, strategy_to_validate)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save strategy.")

    register_strategy(payload.symbol, payload.strategy_id)
    return {"message": "Strategy saved and registered successfully."}


# List all registered strategies with pagination
@router.get("/list")
def list_registered_strategies(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
):
    all_strategies = get_registered_strategies()  # Dict[symbol, List[str]]
    # Flatten all strategy IDs into a list of dicts {symbol, strategy_id}
    flattened = []
    for symbol, strategy_ids in all_strategies.items():
        for sid in strategy_ids:
            flattened.append({"symbol": symbol, "strategy_id": sid})

    total = len(flattened)
    start = (page - 1) * limit
    end = start + limit
    paginated = flattened[start:end]

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "totalPages": (total + limit - 1) // limit,
        "data": paginated,
    }


# List strategies for a specific symbol
@router.get("/{symbol}")
def list_strategies_by_symbol(symbol: str):
    symbol = symbol.upper()
    strategies = get_strategy_by_symbol(symbol)
    if not strategies:
        raise HTTPException(status_code=404, detail="No strategies found for this symbol.")
    return strategies


# Delete strategy using combined strategy_key (symbol_strategy_strategyId)
@router.delete("/{strategy_key}")
async def delete_strategy_api(strategy_key: str):
    """
    Delete a strategy by strategy_key formatted as 'SYMBOL_strategy_strategyId'.
    """
    if "_strategy_" not in strategy_key:
        raise HTTPException(status_code=400, detail="Invalid strategy key format")

    symbol, strategy_id = strategy_key.split("_strategy_", 1)
    symbol = symbol.upper()

    success = delete_strategy(symbol, strategy_id)
    if not success:
        raise HTTPException(status_code=404, detail="Strategy not found or could not be deleted")

    return {"detail": f"Strategy {strategy_key} deleted successfully"}


# Load performance data for a given strategy
@router.get("/{symbol}/{strategy_id}/performance")
def get_strategy_performance(symbol: str, strategy_id: str):
    symbol = symbol.upper()
    log_path = PERFORMANCE_DIR / f"{symbol}_strategy_{strategy_id}.json"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No performance log found.")

    with open(log_path, "r") as f:
        trades = json.load(f)

    stats = StrategyHealth(trades).summary()
    return stats
