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
def get_full_capital_curve(
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1)  # default 100 for pagination, can increase or remove pagination
):
    logs_dir = Path("backend/storage/trade_profits")

    if not logs_dir.exists():
        raise HTTPException(status_code=404, detail="Trade profits directory not found.")

    all_trades = []

    # Load trades from all summary files
    for summary_file in logs_dir.glob("*_summary.json"):
        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)
                symbol = summary.get("symbol", summary_file.stem.replace("_summary", ""))
                trades = summary.get("trades", [])
                
                # Append trades with symbol info
                for trade in trades:
                    all_trades.append({
                        "symbol": symbol,
                        "entry_time": trade.get("entry_time"),
                        "exit_time": trade.get("exit_time"),
                        "exit_price": trade.get("exit_price"),
                        "pnl_dollars": trade.get("pnl_dollars", 0.0),
                        "pnl_percentage": trade.get("pnl_percentage", 0.0)
                    })
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")

    if not all_trades:
        raise HTTPException(status_code=404, detail="No trades found in summaries.")

    # Sort trades by exit_time ascending (oldest first)
    all_trades.sort(key=lambda x: x.get("exit_time") or "")

    # Calculate running capital curve starting from initial capital
    initial_capital = 10000.0
    running_balance = initial_capital
    capital_curve = []

    for trade in all_trades:
        pnl = trade["pnl_dollars"] or 0.0
        running_balance += pnl
        capital_curve.append({
            "timestamp": trade["exit_time"],
            "symbol": trade["symbol"],
            "balance": round(running_balance, 6),
            "pnl_dollars": pnl,
            "exit_price": trade["exit_price"]
        })

    # Pagination on capital_curve (optional)
    total = len(capital_curve)
    start = (page - 1) * limit
    end = start + limit
    paginated = capital_curve[start:end]

    return {
        "page": page,
        "limit": limit,
        "total_trades": total,
        "total_pages": (total + limit - 1) // limit,
        "initial_capital": initial_capital,
        "capital_curve": paginated
    }