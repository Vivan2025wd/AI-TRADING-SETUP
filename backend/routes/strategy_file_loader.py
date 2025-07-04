from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict

router = APIRouter()

# In-memory stores (replace with DB in production)
strategies_store = {
    "BTC_strategy1": {
        "strategy_id": "strategy1",
        "symbol": "BTC",
        "strategy_json": {
            "indicators": {
                "rsi": {"buy_below": [30]},
                "macd": {"sell_above": [0]}
            }
        },
    },
    "ETH_strategy2": {
        "strategy_id": "strategy2",
        "symbol": "ETH",
        "strategy_json": {
            "indicators": {
                "sma": {"buy_above": [50]},
                "macd": {"sell_below": [-0.5]}
            }
        },
    },
    # Add more strategies here
}

performance_data = {
    "BTC_strategy1": {"win_rate": 0.75},
    "ETH_strategy2": {"win_rate": 0.6}
}

@router.get("/list")
async def list_strategies(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1)
):
    all_strategies = list(strategies_store.values())
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
    key = f"{symbol}_{strategy_id}"
    perf = performance_data.get(key)
    if perf is None:
        raise HTTPException(status_code=404, detail="Performance data not found")
    return perf

@router.delete("/{symbol}/{strategy_id}")
async def delete_strategy(symbol: str, strategy_id: str):
    key = f"{symbol}_{strategy_id}"
    if key not in strategies_store:
        raise HTTPException(status_code=404, detail="Strategy not found")
    strategies_store.pop(key)
    performance_data.pop(key, None)
    return {"message": f"Strategy '{strategy_id}' for '{symbol}' deleted"}
