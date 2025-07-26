from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from backend.mother_ai.mother_ai import MotherAI
from backend.mother_ai.profit_calculator import compute_trade_profits
from backend.utils.binance_api import place_live_order

import os
import json
import glob
from datetime import datetime
from functools import wraps

# Constants & Globals
TRADE_HISTORY_DIR = "backend/storage/performance_logs"
router = APIRouter(tags=["Mother AI"])
mother_ai_instance = MotherAI(agent_symbols=None)
latest_decision = None


# --- Utility Decorator for Logging ---
def log_endpoint(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"📥 {func.__name__} called")  # <-- Inline logging
        try:
            result = func(*args, **kwargs)
            print(f"✅ {func.__name__} completed successfully")
            return result
        except Exception as e:
            print(f"❌ {func.__name__} failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return wrapper


# --- Logging Helper ---
def log(message: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


# ======================
#      API ROUTES
# ======================

@router.get("/latest-decision")
@log_endpoint
def get_latest_decision():
    if latest_decision is None:
        return {"status": "waiting", "message": "No decision made yet.", "decision": {}}
    return latest_decision


@router.post("/trigger-decision")
@log_endpoint
def trigger_decision():
    global latest_decision
    latest_decision = mother_ai_instance.make_portfolio_decision(min_score=0.5)
    print(f"[DEBUG] Latest decision: {latest_decision}")
    return latest_decision


@router.get("/decision")
@log_endpoint
def get_mother_ai_decision(is_live: bool = Query(False)):
    # Get current trading mode - this overrides the is_live parameter
    from backend.utils.binance_api import get_trading_mode
    current_mode = get_trading_mode()
    
    # Log what mode we're actually in
    print(f"🔍 Current trading mode: {current_mode.upper()}")
    print(f"🔍 is_live parameter: {is_live}")
    
    result = mother_ai_instance.make_portfolio_decision()
    decision = result.get("decision", {})

    if not decision:
        return {
            "decision": [],
            "timestamp": mother_ai_instance.performance_tracker.current_time(),
            "trading_mode": current_mode
        }

    symbol = decision.get("symbol")
    signal = decision.get("signal", "").lower()

    # Note: The execute_trade method now automatically uses the correct trading mode
    # from binance_api.py, so we don't need to handle live orders separately here
    
    if signal == "sell" and symbol:
        compute_trade_profits(symbol)

    # Add trading mode info to response
    result["trading_mode"] = current_mode
    result["note"] = f"Executed in {current_mode.upper()} mode"
    
    return result

@router.get("/trades/{symbol}")
@log_endpoint
def get_symbol_trades(symbol: str):
    symbol = symbol.upper()
    log_file_path = os.path.join(TRADE_HISTORY_DIR, f"{symbol}_trades.json")

    if not os.path.exists(log_file_path):
        raise HTTPException(status_code=404, detail=f"No trade log for {symbol}")

    with open(log_file_path, "r") as f:
        data = json.load(f)

    return {"symbol": symbol, "data": data}


@router.get("/trades")
@log_endpoint
def get_all_trades():
    pattern = os.path.join(TRADE_HISTORY_DIR, "*_trades.json")
    all_files = glob.glob(pattern)

    if not all_files:
        return {"symbol": "ALL", "data": []}

    all_trades = []
    for file_path in all_files:
        symbol = os.path.basename(file_path).replace("_trades.json", "").upper()
        with open(file_path, "r") as f:
            trades = json.load(f)
            for trade in trades:
                trade["symbol"] = symbol
            all_trades.extend(trades)

    all_trades.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    return {"symbol": "ALL", "data": all_trades}


@router.get("/profits/{symbol}")
@log_endpoint
def get_profit_summary(symbol: str):
    symbol = symbol.upper()
    summary = compute_trade_profits(symbol)

    if summary is None:
        raise HTTPException(status_code=404, detail=f"No trades for {symbol}")

    return summary

