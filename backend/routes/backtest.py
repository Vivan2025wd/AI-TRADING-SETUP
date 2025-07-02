import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.backtester.runner import run_backtest
from pathlib import Path

router = APIRouter(prefix="", tags=["Backtesting"])

# ------------------------------
# POST /backtest/ — Run Backtest
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


# ----------------------------------------
# GET /backtest/results — Recent Trade Logs
# ----------------------------------------
@router.get("/results")
def get_recent_backtest_results():
    logs_dir = Path("storage/Performance_logs")
    if not logs_dir.exists():
        raise HTTPException(status_code=404, detail="Performance log directory not found.")

    all_trades = []

    for log_file in logs_dir.glob("*_trades.json"):
        try:
            with open(log_file, "r") as f:
                trades = json.load(f)

                # Add symbol to each trade
                symbol = log_file.stem.replace("_trades", "")
                for trade in trades:
                    trade["symbol"] = symbol
                    all_trades.append(trade)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    # Sort by timestamp descending
    sorted_trades = sorted(
        all_trades,
        key=lambda x: x.get("timestamp", ""),
        reverse=True
    )

    # Return latest 10 trades
    return sorted_trades[:10]
