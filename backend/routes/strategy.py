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
import os
import json

router = APIRouter(prefix="/strategies", tags=["Strategies"])

STRATEGY_DIR = "backend/backtester/strategies"

class StrategyPayload(BaseModel):
    strategy_id: str
    symbol: str
    strategy_json: dict


@router.post("/save")
def save_user_strategy(payload: StrategyPayload):
    os.makedirs(STRATEGY_DIR, exist_ok=True)

    # Validate strategy format
    try:
        parse_strategy_json(json.dumps(payload.strategy_json))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save and register
    success = save_strategy(payload.symbol, payload.strategy_id, payload.strategy_json)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save strategy.")

    register_strategy(payload.symbol, payload.strategy_id)
    return {"message": "Strategy saved and registered successfully."}


@router.get("/list")
def list_registered_strategies():
    return get_registered_strategies()


@router.get("/{symbol}")
def list_strategies_by_symbol(symbol: str):
    strategies = get_strategy_by_symbol(symbol.upper())
    if not strategies:
        raise HTTPException(status_code=404, detail="No strategies found for this symbol.")
    return strategies


@router.delete("/{symbol}/{strategy_id}")
def remove_strategy(symbol: str, strategy_id: str):
    success = delete_strategy(symbol.upper(), strategy_id)
    if not success:
        raise HTTPException(status_code=404, detail="Strategy not found or could not be deleted.")
    return {"message": f"Strategy {strategy_id} for {symbol.upper()} deleted successfully."}
