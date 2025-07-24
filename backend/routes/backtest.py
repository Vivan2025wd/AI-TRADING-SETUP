import json
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
from backend.backtester.runner import run_backtest

router = APIRouter(tags=["Backtesting"])

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
# GET /api/backtest/results — Mother AI trade profits
# -----------------------------------------------
@router.get("/results")
def get_recent_backtest_results(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1)
):
    logs_dir = Path("backend/storage/trade_profits")

    if not logs_dir.exists():
        raise HTTPException(status_code=404, detail="Trade profits directory not found.")

    all_trades = []

    # Load and process each summary file
    for summary_file in logs_dir.glob("*_summary.json"):
        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)
                symbol = summary.get("symbol", summary_file.stem.replace("_summary", ""))
                trades = summary.get("trades", [])

                balance = 100.0  # Starting virtual balance
                for trade in trades:
                    pnl_dollars = trade.get("pnl_dollars", 0.0)
                    pnl_percentage = trade.get("pnl_percentage", 0.0)

                    balance += pnl_dollars

                    all_trades.append({
                        "type": "TRADE",
                        "timestamp": trade.get("exit_time"),
                        "price": trade.get("exit_price"),
                        "profit_percent": pnl_percentage,
                        "balance": balance,
                        "symbol": symbol
                    })
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")

    # Sort trades by timestamp (newest last)
    sorted_trades = sorted(all_trades, key=lambda x: x.get("timestamp", ""))

    # Pagination
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
