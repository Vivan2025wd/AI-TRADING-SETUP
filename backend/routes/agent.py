from fastapi import APIRouter, HTTPException, Query
from typing import List
from backend.agents.generic_agent import GenericAgent
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.binance.fetch_live_ohlcv import fetch_ohlcv
from backend.strategy_engine.json_strategy_parser import load_strategy_for_symbol

import os
import inspect
import importlib.util
from pathlib import Path

router = APIRouter(tags=["Agent"])
AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"


def get_real_agents() -> List[str]:
    """
    Detects all real (non-generic) agent classes and extracts their symbol class attribute.
    """
    agent_symbols = []
    for file in AGENTS_DIR.glob("*.py"):
        if file.stem in {"__init__", "generic_agent"}:
            continue

        module_name = f"backend.agents.{file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file)
        if not spec or not spec.loader:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, GenericAgent) and cls is not GenericAgent:
                symbol = getattr(cls, "symbol", None)
                if symbol:
                    agent_symbols.append(symbol.upper())

    return sorted(set(agent_symbols))


@router.get("", response_model=List[str])
def list_available_agents():
    return get_real_agents()


@router.get("/{symbol}/predict")
def get_agent_prediction(symbol: str):
    try:
        symbol = symbol.upper()
        try:
            strategy_dict = load_strategy_for_symbol(symbol + "USDT")
        except FileNotFoundError:
            strategy_dict = load_strategy_for_symbol(symbol)

        strategy_runner = StrategyParser(strategy_dict)

        ohlcv_data = fetch_ohlcv(f"{symbol}/USDT")
        print(f"[Agent] Fetched OHLCV for {symbol}: {ohlcv_data.shape} rows")

        if ohlcv_data is None or ohlcv_data.empty:
            raise HTTPException(status_code=400, detail=f"No OHLCV data found for symbol: {symbol}")

        agent = GenericAgent(symbol=symbol, strategy_logic=strategy_runner)
        prediction_result = agent.predict(ohlcv_data)

        return {
            "symbol": symbol,
            "signal": prediction_result["action"].capitalize(),
            "confidence": round(prediction_result["confidence"] * 100, 2),
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/predictions")
def get_all_agent_predictions(
    page: int = Query(1, ge=1),
    limit: int = Query(5, ge=1)
):
    """
    Paginated predictions from all custom agents.
    """
    predictions = []
    all_agents = get_real_agents()
    total_agents = len(all_agents)
    start = (page - 1) * limit
    end = start + limit
    agents_slice = all_agents[start:end]

    for symbol in agents_slice:
        try:
            try:
                strategy_dict = load_strategy_for_symbol(symbol + "USDT")
            except FileNotFoundError:
                strategy_dict = load_strategy_for_symbol(symbol)

            strategy_runner = StrategyParser(strategy_dict)
            ohlcv_data = fetch_ohlcv(symbol)
            print(f"[Agent] Fetched OHLCV for {symbol}: {ohlcv_data.shape} rows")

            if ohlcv_data is None or ohlcv_data.empty:
                predictions.append({
                    "agentName": f"{symbol} Agent",
                    "prediction": "No Data",
                    "confidence": 0,
                    "tradeDetails": {
                        "symbol": symbol,
                        "entryPrice": 0,
                        "targetPrice": 0,
                        "stopLoss": 0,
                    },
                })
                continue

            agent = GenericAgent(symbol=symbol, strategy_logic=strategy_runner)
            prediction_result = agent.predict(ohlcv_data)
            action = prediction_result["action"]
            confidence = prediction_result["confidence"]
            entry_price = ohlcv_data["close"].iloc[-1]

            predictions.append({
                "agentName": f"{symbol} Agent",
                "prediction": action.capitalize(),
                "confidence": round(confidence * 100, 2),
                "tradeDetails": {
                    "symbol": symbol,
                    "entryPrice": entry_price,
                    "targetPrice": round(entry_price * 1.05, 2) if action == "buy" else round(entry_price * 0.95, 2),
                    "stopLoss": round(entry_price * 0.98, 2) if action == "buy" else round(entry_price * 1.02, 2),
                }
            })

        except Exception as e:
            predictions.append({
                "agentName": f"{symbol} Agent",
                "prediction": "Error",
                "confidence": 0,
                "tradeDetails": {
                    "symbol": symbol,
                    "entryPrice": 0,
                    "targetPrice": 0,
                    "stopLoss": 0,
                },
                "error": str(e)
            })

    return {
        "page": page,
        "limit": limit,
        "total": total_agents,
        "totalPages": (total_agents + limit - 1) // limit,
        "data": predictions
    }
