import json
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
from backend.backtester.runner import run_backtest
import os

router = APIRouter(tags=["Backtesting"])
RESULTS_DIR = Path("backend/storage/backtest_results")
RESULTS_FILE = RESULTS_DIR / "latest.json"

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

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
        # Save the result to a file
        with open(RESULTS_FILE, "w") as f:
            json.dump(result, f)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------
# GET /api/backtest/results — Fetch latest backtest result
# -----------------------------------------------
@router.get("/results")

def get_recent_backtest_results(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1)):
    logs_dir = Path("backend/storage/trade_profits")

    if not logs_dir.exists():
        raise HTTPException(status_code=404, detail="Trade profits directory not found.")

    all_trades_raw = []

    # Step 1: Load all trade summaries
    for summary_file in logs_dir.glob("*_summary.json"):
        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)
                symbol = summary.get("symbol", summary_file.stem.replace("_summary", ""))
                trades = summary.get("trades", [])

                for trade in trades:
                    all_trades_raw.append({
                        "symbol": symbol,
                        "exit_time": trade.get("exit_time"),
                        "exit_price": trade.get("exit_price"),
                        "pnl_dollars": trade.get("pnl_dollars", 0.0),
                        "pnl_percentage": trade.get("pnl_percentage", 0.0)
                    })
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")

    # Step 2: Sort trades chronologically
    sorted_trades = sorted(all_trades_raw, key=lambda x: x.get("exit_time", ""))

    # Step 3: Calculate cumulative balance across all trades
    balance = 100.0  # Starting Capital
    processed_trades = []
    for trade in sorted_trades:
        pnl_dollars = trade.get("pnl_dollars", 0.0)
        balance += pnl_dollars

        processed_trades.append({
            "type": "TRADE",
            "timestamp": trade.get("exit_time"),
            "price": trade.get("exit_price"),
            "profit_percent": trade.get("pnl_percentage"),
            "balance": balance,
            "symbol": trade.get("symbol")
        })

    # Step 4: Pagination after balance calculation
    total = len(processed_trades)
    start = (page - 1) * limit
    end = start + limit
    paginated = processed_trades[start:end]

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "totalPages": (total + limit - 1) // limit,
        "data": paginated
    }

def get_recent_backtest_results():
    if not RESULTS_FILE.exists():
        raise HTTPException(status_code=404, detail="No backtest results found.")

    try:
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading results file: {e}")
 main
