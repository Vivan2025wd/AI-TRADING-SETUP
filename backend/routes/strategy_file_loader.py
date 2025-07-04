from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict
from pathlib import Path
import json

router = APIRouter(tags=["StrategyLoader"])

STRATEGY_DIR = Path("backend/storage/strategies")
PERFORMANCE_DIR = Path("backend/storage/performance_logs")


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


@router.get("/{symbol}/{strategy_id}/performance")
async def get_strategy_performance(symbol: str, strategy_id: str):
    symbol = symbol.upper()
    path = PERFORMANCE_DIR / f"{symbol}_strategy_{strategy_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Performance data not found")
    try:
        with open(path, "r") as f:
            data = json.load(f)
            wins = sum(1 for trade in data if trade["result"] == "win")
            total = len(data)
            win_rate = wins / total if total else 0
            return {"win_rate": round(win_rate, 4), "total_trades": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read performance log: {e}")


@router.delete("/{symbol}/{strategy_id}")
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
