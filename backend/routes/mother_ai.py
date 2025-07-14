from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from backend.mother_ai.mother_ai import MotherAI
from backend.mother_ai.trade_executer import execute_mother_ai_decision
from backend.mother_ai.profit_calculator import compute_trade_profits

import os
import json
import glob
from datetime import datetime

router = APIRouter(tags=["Mother AI"])

# Global instances and constants
mother_ai_instance = MotherAI(agent_symbols=None)
TRADE_HISTORY_DIR = "backend/storage/trade_history"
latest_decision = None


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


@router.get("/latest-decision")
def get_latest_decision():
    log("📥 GET /latest-decision called")
    if latest_decision is None:
        log("⚠️ No decision made yet.")
        return {
            "status": "waiting",
            "message": "No decision made yet.",
            "decision": {}
        }
    log(f"✅ Returning latest decision: {latest_decision}")
    return latest_decision


@router.post("/trigger-decision")
def trigger_decision():
    global latest_decision
    log("🚨 POST /trigger-decision called")
    latest_decision = mother_ai_instance.make_portfolio_decision(min_score=0.5)
    log(f"✅ Decision triggered: {latest_decision}")
    return latest_decision


@router.post("/execute")
def execute_mother_decision():
    try:
        log("🧠 POST /execute called: Executing Mother AI trades...")
        result = mother_ai_instance.make_portfolio_decision()
        decision = result.get("decision", {})

        if not decision:
            log("⚠️ No valid decision to execute")
            return JSONResponse(
                status_code=200,
                content={"message": "No valid decision to execute.", "executed_trades": []}
            )

        executed = execute_mother_ai_decision(result)

        symbol = decision.get("symbol")
        signal = decision.get("signal", "").lower()
        if signal == "sell" and symbol:
            log(f"📊 SELL decision for {symbol}, updating profit summary...")
            compute_trade_profits(symbol)

        log(f"✅ Executed {len(executed)} trades.")
        return {
            "message": f"{len(executed)} trades executed.",
            "executed_trades": executed
        }

    except Exception as e:
        log(f"❌ Execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decision")
def get_mother_ai_decision():
    log("🧠 GET /decision called")
    try:
        result = mother_ai_instance.make_portfolio_decision()
        decision = result.get("decision", {})

        if not decision:
            log("⚠️ No valid decision found.")
            return {
                "decision": [],
                "timestamp": mother_ai_instance.performance_tracker.current_time()
            }

        symbol = decision.get("symbol")
        signal = decision.get("signal", "").lower()
        if signal == "sell" and symbol:
            log(f"📊 SELL decision for {symbol}, updating profit summary...")
            compute_trade_profits(symbol)

        log(f"✅ Returning decision: {decision}")
        return result

    except Exception as e:
        log(f"❌ Error during decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/{symbol}")
def get_symbol_trades(symbol: str):
    try:
        symbol = symbol.upper()
        log(f"📥 GET /trades/{symbol} called")
        log_file_path = f"{TRADE_HISTORY_DIR}/{symbol}_trades.json"

        if not os.path.exists(log_file_path):
            log(f"❌ No trade log for {symbol}")
            raise HTTPException(status_code=404, detail=f"No trade log for {symbol}")

        with open(log_file_path, "r") as f:
            data = json.load(f)

        log(f"✅ Loaded {len(data)} trades for {symbol}")
        return {
            "symbol": symbol,
            "data": data
        }

    except Exception as e:
        log(f"❌ Failed to load trade log for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades")
def get_all_trades():
    log("📥 GET /trades called to fetch all symbols")
    try:
        pattern = os.path.join(TRADE_HISTORY_DIR, "*_trades.json")
        all_files = glob.glob(pattern)
        if not all_files:
            log("⚠️ No trade files found.")
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
        log(f"✅ Total trades loaded: {len(all_trades)}")

        return {"symbol": "ALL", "data": all_trades}

    except Exception as e:
        log(f"❌ Failed to load all trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profits/{symbol}")
def get_profit_summary(symbol: str):
    symbol = symbol.upper()
    log(f"📈 GET /profits/{symbol} called")
    try:
        summary = compute_trade_profits(symbol)
        if summary is None:
            log(f"⚠️ No trades found for {symbol}")
            raise HTTPException(status_code=404, detail=f"No trades for {symbol}")
        log(f"✅ Profit summary for {symbol}: {summary}")
        return summary

    except Exception as e:
        log(f"❌ Profit summary error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
