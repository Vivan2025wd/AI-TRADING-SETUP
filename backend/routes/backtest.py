import json
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from backend.backtester.runner import run_backtest
from pathlib import Path

# Do NOT include prefix here if already prefixed in main.py
router = APIRouter(tags=["Backtesting"])  # ✅ Remove prefix if already set in main.py

# ------------------------------
# POST /api/backtest/ — Run Backtest
# ------------------------------
class BacktestPayload(BaseModel):
    symbol: str
    strategy_json: dict
    start_date: str
    end_date: str

@router.post("/")
def execute_backtest(payload: BacktestPayload):
    try:
        result = run_backtest(
            symbol=payload.symbol,
            strategy_json=json.dumps(payload.strategy_json),
            start_date=payload.start_date,
            end_date=payload.end_date
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------
# GET /api/backtest/results — Paginated Trade Logs
# -----------------------------------------------
@router.get("/results")
def get_recent_backtest_results(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1)
):
    logs_dir = Path("backend/storage/performance_logs")

    if not logs_dir.exists():
        raise HTTPException(status_code=404, detail="Performance log directory not found.")

    all_trades = []

    for log_file in logs_dir.glob("*_trades.json"):
        try:
            with open(log_file, "r") as f:
                trades = json.load(f)
                symbol = log_file.stem.replace("_trades", "")
                for trade in trades:
                    trade["symbol"] = symbol
                    all_trades.append(trade)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    sorted_trades = sorted(all_trades, key=lambda x: x.get("timestamp", ""), reverse=True)

    total = len(sorted_trades)
    start = (page - 1) * limit
    end = start + limit
    paginated = sorted_trades[start:end]

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "totalPages": (total + limit - 1) // limit,
        "data": paginated
    }
