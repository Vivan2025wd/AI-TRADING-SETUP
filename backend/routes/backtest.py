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
def get_recent_backtest_results():
    if not RESULTS_FILE.exists():
        raise HTTPException(status_code=404, detail="No backtest results found.")

    try:
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading results file: {e}")
