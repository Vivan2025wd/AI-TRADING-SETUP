from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict
from pathlib import Path
import json

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
    """Load trade profits summary for a symbol"""
    try:
        summary_path = TRADE_PROFITS_DIR / f"{symbol.upper()}_summary.json"
        if not summary_path.exists():
            return {}
        
        with open(summary_path, "r") as f:
            return json.load(f)
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
    """Get rated strategies for a specific symbol with performance data"""
    symbol = symbol.upper()
    
    # Load all strategies for this symbol
    all_strategies = load_all_strategies()
    symbol_strategies = [s for s in all_strategies if s["symbol"] == symbol]
    
    if not symbol_strategies:
        raise HTTPException(status_code=404, detail=f"No strategies found for {symbol}")
    
    # Load trade profits summary for this symbol
    trade_profits = load_trade_profits_summary(symbol)
    
    # Create rated strategies with performance data
    rated_strategies = []
    for strategy in symbol_strategies:
        strategy_id = strategy["strategy_id"]
        
        # Calculate performance metrics from trade_profits data
        win_rate = trade_profits.get("win_rate", 0) / 100.0  # Convert percentage to decimal
        total_trades = trade_profits.get("total_trades", 0)
        total_profit = trade_profits.get("total_profit", 0)
        wins = trade_profits.get("wins", 0)
        losses = trade_profits.get("losses", 0)
        
        # Calculate average profit per trade
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        # For now, we'll use a mock confidence score (you can implement actual logic)
        avg_confidence = 0.75  # 75% confidence as default
        
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
    """Get detailed strategy configuration"""
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
    """Get performance data for a specific strategy"""
    symbol = symbol.upper()
    
    # First try to load from performance_logs (if you have strategy-specific data)
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
            pass
    
    # Fallback to symbol-level trade profits summary
    trade_profits = load_trade_profits_summary(symbol)
    if trade_profits:
        return {
            "win_rate": trade_profits.get("win_rate", 0) / 100.0,
            "total_trades": trade_profits.get("total_trades", 0),
            "total_profit": trade_profits.get("total_profit", 0),
            "wins": trade_profits.get("wins", 0),
            "losses": trade_profits.get("losses", 0)
        }
    
    raise HTTPException(status_code=404, detail="Performance data not found")


@router.delete("/{symbol}_strategy_{strategy_id}")
async def delete_strategy(symbol: str, strategy_id: str):
    """Delete a strategy and its performance data"""
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
    """Get trade profits summary for a symbol"""
    symbol = symbol.upper()
    trade_profits = load_trade_profits_summary(symbol)
    
    if not trade_profits:
        raise HTTPException(status_code=404, detail=f"No trade profits data found for {symbol}")
    
    return trade_profits