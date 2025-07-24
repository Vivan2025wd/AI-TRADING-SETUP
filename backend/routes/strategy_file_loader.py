from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict
from pathlib import Path
import json

# 🧠 Added for live confidence fetching
from backend.agents.generic_agent import GenericAgent
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.binance.fetch_live_ohlcv import fetch_ohlcv

router = APIRouter(tags=["StrategyLoader"])

STRATEGY_DIR = Path("backend/storage/strategies")
PERFORMANCE_DIR = Path("backend/storage/performance_logs")
TRADE_PROFITS_DIR = Path("backend/storage/trade_profits")


def load_all_strategies() -> List[Dict]:
    strategies = []
    for file in STRATEGY_DIR.glob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                symbol, strategy_id = file.stem.split("_strategy_", 1)
                strategies.append({
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "strategy_json": data.get("indicators", {}),
                })
        except Exception as e:
            print(f"[Error loading strategy] {file.name}: {e}")
    return strategies


def load_trade_profits_summary(symbol: str) -> Dict:
    try:
        summary_path = TRADE_PROFITS_DIR / f"{symbol.upper()}_summary.json"
        if not summary_path.exists():
            return {}

        with open(summary_path, "r") as f:
            raw_data = json.load(f)

        total_profit = float(raw_data.get("total_profit_dollars", 0.0))
        total_trades = int(raw_data.get("total_trades", 0))
        wins = int(raw_data.get("wins", 0))
        losses = int(raw_data.get("losses", 0))
        win_rate = float(raw_data.get("win_rate", 0.0))
        trades = raw_data.get("trades", [])

        formatted_trades = []
        for trade in trades:
            formatted_trade = {
                "entry_time": trade["entry_time"],
                "entry_price": round(float(trade["entry_price"]), 4),
                "exit_time": trade["exit_time"],
                "exit_price": round(float(trade["exit_price"]), 4),
                "qty": round(float(trade["qty"]), 6),
                "entry_value": round(float(trade.get("entry_value", trade["entry_price"] * trade["qty"])), 4),
                "exit_value": round(float(trade.get("exit_value", trade["exit_price"] * trade["qty"])), 4),
                "pnl_dollars": round(float(trade["pnl_dollars"]), 6),
                "pnl_percentage": round(float(trade["pnl_percentage"]), 4)
            }
            formatted_trades.append(formatted_trade)

        avg_profit = total_profit / total_trades if total_trades > 0 else 0.0

        return {
            "symbol": symbol.upper(),
            "total_profit_dollars": round(total_profit, 6),
            "avg_profit_per_trade": round(avg_profit, 6),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "trades": formatted_trades
        }

    except Exception as e:
        print(f"[Error loading trade profits for {symbol}]: {e}")
        return {}


@router.get("/list")
async def list_strategies(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1)
):
    all_strategies = load_all_strategies()
    total = len(all_strategies)
    start = (page - 1) * limit
    end = start + limit
    paginated_strategies = all_strategies[start:end]

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "totalPages": (total + limit - 1) // limit,
        "data": paginated_strategies
    }


@router.get("/{symbol}/rate-strategies")
async def rate_strategies_for_symbol(symbol: str):
    """Get rated strategies for a specific symbol with live confidence"""
    symbol = symbol.upper()

    all_strategies = load_all_strategies()
    symbol_strategies = [s for s in all_strategies if s["symbol"] == symbol]

    if not symbol_strategies:
        raise HTTPException(status_code=404, detail=f"No strategies found for {symbol}")

    trade_profits = load_trade_profits_summary(symbol)

    rated_strategies = []
    for strategy in symbol_strategies:
        strategy_id = strategy["strategy_id"]
        strategy_logic = StrategyParser({"symbol": symbol, "indicators": strategy["strategy_json"]})

        # 🧠 Fetch OHLCV and Predict Live Confidence
        try:
            ohlcv_data = fetch_ohlcv(f"{symbol}/USDT")
            if ohlcv_data is None or ohlcv_data.empty:
                avg_confidence = 0.0
            else:
                agent = GenericAgent(symbol=symbol, strategy_logic=strategy_logic)
                prediction_result = agent.predict(ohlcv_data)
                avg_confidence = prediction_result["confidence"]  # Already in 0-1 range
        except Exception as e:
            print(f"[Error predicting confidence for {symbol}]: {e}")
            avg_confidence = 0.0

        win_rate = trade_profits.get("win_rate", 0.0) / 100.0
        total_trades = trade_profits.get("total_trades", 0)
        total_profit = trade_profits.get("total_profit_dollars", 0.0)
        wins = trade_profits.get("wins", 0)
        losses = trade_profits.get("losses", 0)
        avg_profit = trade_profits.get("avg_profit_per_trade", 0.0)

        rated_strategy = {
            "strategy_id": strategy_id,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_confidence": avg_confidence,
            "total": total_trades,
            "wins": wins,
            "losses": losses,
            "total_profit": total_profit
        }

        rated_strategies.append(rated_strategy)

    return {
        "symbol": symbol,
        "strategies": rated_strategies
    }


@router.get("/{symbol}/{strategy_id}")
async def get_strategy_details(symbol: str, strategy_id: str):
    symbol = symbol.upper()
    strategy_path = STRATEGY_DIR / f"{symbol}_strategy_{strategy_id}.json"

    if not strategy_path.exists():
        raise HTTPException(status_code=404, detail="Strategy not found")

    try:
        with open(strategy_path, "r") as f:
            data = json.load(f)
            return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load strategy: {e}")


@router.get("/{symbol}/{strategy_id}/performance")
async def get_strategy_performance(symbol: str, strategy_id: str):
    symbol = symbol.upper()
    perf_path = PERFORMANCE_DIR / f"{symbol}_strategy_{strategy_id}.json"
    if perf_path.exists():
        try:
            with open(perf_path, "r") as f:
                data = json.load(f)
                wins = sum(1 for trade in data if trade["result"] == "win")
                total = len(data)
                win_rate = wins / total if total else 0
                return {"win_rate": round(win_rate, 4), "total_trades": total}
        except Exception as e:
            print(f"[Error loading performance log]: {e}")

    trade_profits = load_trade_profits_summary(symbol)
    if trade_profits:
        return {
            "win_rate": trade_profits.get("win_rate", 0.0) / 100.0,
            "total_trades": trade_profits.get("total_trades", 0),
            "total_profit": trade_profits.get("total_profit_dollars", 0.0),
            "wins": trade_profits.get("wins", 0),
            "losses": trade_profits.get("losses", 0)
        }

    raise HTTPException(status_code=404, detail="Performance data not found")


@router.delete("/{symbol}_strategy_{strategy_id}")
async def delete_strategy(symbol: str, strategy_id: str):
    symbol = symbol.upper()
    strategy_path = STRATEGY_DIR / f"{symbol}_strategy_{strategy_id}.json"
    perf_path = PERFORMANCE_DIR / f"{symbol}_strategy_{strategy_id}.json"

    if not strategy_path.exists():
        raise HTTPException(status_code=404, detail="Strategy file not found")

    try:
        strategy_path.unlink()
        if perf_path.exists():
            perf_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")

    return {"message": f"Strategy '{strategy_id}' for '{symbol}' deleted"}


@router.get("/trade-profits/{symbol}")
async def get_trade_profits_summary(symbol: str):
    symbol = symbol.upper()
    trade_profits = load_trade_profits_summary(symbol)

    if not trade_profits:
        print(f"[Info] No trade profits data found for {symbol}. Returning empty structure.")
        return {
            "symbol": symbol,
            "total_profit_dollars": 0.0,
            "avg_profit_per_trade": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "trades": []
        }

    return trade_profits
