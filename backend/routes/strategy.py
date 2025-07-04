from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import json
from pathlib import Path

from backend.strategy_engine.json_strategy_parser import parse_strategy_json
from backend.strategy_engine.strategy_health import StrategyHealth

router = APIRouter(tags=["Strategies"])

STRATEGY_DIR = Path("backend/storage/strategies")
PERFORMANCE_DIR = Path("backend/storage/performance_logs")
STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

class StrategyPayload(BaseModel):
    strategy_id: str
    symbol: str
    strategy_json: dict


def get_strategy_file_path(symbol: str, strategy_id: str) -> Path:
    return STRATEGY_DIR / f"{symbol}_strategy_{strategy_id}.json"


# Save & register strategy
@router.post("/save")
def save_user_strategy(payload: StrategyPayload):
    symbol = payload.symbol.upper()
    strategy_to_validate = {
        "symbol": symbol,
        "indicators": payload.strategy_json,
    }

    # Validate the strategy format
    try:
        parse_strategy_json(json.dumps(strategy_to_validate))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save to disk
    path = get_strategy_file_path(symbol, payload.strategy_id)
    try:
        with open(path, "w") as f:
            json.dump(strategy_to_validate, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save: {e}")

    return {"message": f"Strategy {symbol}_{payload.strategy_id} saved successfully."}


# List all saved strategies with pagination
@router.get("/list")
def list_registered_strategies(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
):
    strategy_files = list(STRATEGY_DIR.glob("*.json"))

    parsed = []
    for file in strategy_files:
        name = file.stem
        if "_strategy_" in name:
            symbol, strategy_id = name.split("_strategy_", 1)
            parsed.append({"symbol": symbol, "strategy_id": strategy_id})

    total = len(parsed)
    start = (page - 1) * limit
    end = start + limit
    return {
        "page": page,
        "limit": limit,
        "total": total,
        "totalPages": (total + limit - 1) // limit,
        "data": parsed[start:end],
    }


# List all strategies for a specific symbol
@router.get("/{symbol}")
def list_strategies_by_symbol(symbol: str):
    symbol = symbol.upper()
    files = list(STRATEGY_DIR.glob(f"{symbol}_strategy_*.json"))
    if not files:
        raise HTTPException(status_code=404, detail="No strategies found for this symbol.")
    return [{"strategy_id": f.stem.split("_strategy_", 1)[1]} for f in files]


# Delete strategy
@router.delete("/{strategy_key}")
async def delete_strategy_api(strategy_key: str):
    if "_strategy_" not in strategy_key:
        raise HTTPException(status_code=400, detail="Invalid strategy key format")

    symbol, strategy_id = strategy_key.split("_strategy_", 1)
    path = get_strategy_file_path(symbol.upper(), strategy_id)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Strategy not found")

    try:
        path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")

    return {"detail": f"Strategy {strategy_key} deleted successfully"}


# Performance summary for a strategy
@router.get("/{symbol}/{strategy_id}/performance")
def get_strategy_performance(symbol: str, strategy_id: str):
    symbol = symbol.upper()
    log_path = PERFORMANCE_DIR / f"{symbol}_strategy_{strategy_id}.json"

    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No performance log found.")

    try:
        with open(log_path, "r") as f:
            trades = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load log: {e}")

    stats = StrategyHealth(trades).summary()
    return stats
