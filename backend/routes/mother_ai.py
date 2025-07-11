from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from backend.mother_ai.mother_ai import MotherAI
from backend.mother_ai.trade_executer import execute_mother_ai_decision
from backend.mother_ai.profit_calculator import compute_trade_profits

import os
import json
import glob
from apscheduler.schedulers.background import BackgroundScheduler

router = APIRouter(tags=["Mother AI"])

# Global MotherAI instance
mother_ai_instance = MotherAI(agent_symbols=None)

TRADE_HISTORY_DIR = "backend/storage/performance_logs"

# To store the latest decision
latest_decision = None

# Scheduler job to run every 2 hours
def scheduled_trade_decision():
    global latest_decision
    print("⏰ Running scheduled MotherAI decision...")
    decision = mother_ai_instance.make_portfolio_decision(min_score=0.5)
    latest_decision = decision
    print(f"Decision made at {decision.get('timestamp')} for symbol: "
          f"{decision['decision'].get('symbol') if decision.get('decision') else 'None'}")

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_trade_decision, "interval", hours=2)
scheduler.start()


@router.get("/latest-decision")
def get_latest_decision():
    if latest_decision is None:
        return {"message": "No decision made yet. Waiting for scheduled run."}
    return latest_decision


@router.post("/trigger-decision")
def trigger_decision():
    global latest_decision
    latest_decision = mother_ai_instance.make_portfolio_decision(min_score=0.5)
    return latest_decision


@router.get("/decision")
def get_mother_ai_decision():
    try:
        print("🧠 Mother AI: Starting portfolio decision...")
        result = mother_ai_instance.make_portfolio_decision()

        decision = result.get("decision", {})
        symbol = decision.get("symbol")
        signal = decision.get("signal", "").lower()

        # ✅ Trigger profit summary update after SELL signal
        if signal == "sell" and symbol:
            print(f"📊 SELL decision for {symbol}, updating profit summary...")
            compute_trade_profits(symbol)

        if not decision:
            print("⚠️ No valid decision found.")
            return {
                "decision": [],
                "timestamp": mother_ai_instance.performance_tracker.current_time()
            }

        print("✅ Decision found!")
        return result

    except Exception as e:
        print("❌ Mother AI Decision Error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
def execute_mother_decision():
    try:
        print("🧠 Executing Mother AI trades...")
        result = mother_ai_instance.make_portfolio_decision()
        decision = result.get("decision", {})

        if not decision:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "No valid decision to execute.",
                    "executed_trades": []
                }
            )

        executed = execute_mother_ai_decision(result)

        # ✅ Trigger profit summary update after SELL signal
        symbol = decision.get("symbol")
        signal = decision.get("signal", "").lower()
        if signal == "sell" and symbol:
            print(f"📊 SELL decision for {symbol}, updating profit summary...")
            compute_trade_profits(symbol)

        return {
            "message": f"{len(executed)} trades executed.",
            "executed_trades": executed
        }

    except Exception as e:
        print("❌ Execution failed:", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/{symbol}")
def get_symbol_trades(symbol: str):
    try:
        symbol = symbol.upper()
        log_file_path = f"{TRADE_HISTORY_DIR}/{symbol}_trades.json"

        if not os.path.exists(log_file_path):
            print(f"⚠️ No trade log found for {symbol}: {log_file_path}")
            raise HTTPException(status_code=404, detail=f"No trade log found for {symbol}")

        with open(log_file_path, "r") as f:
            data = json.load(f)

        return {
            "symbol": symbol,
            "data": data
        }

    except Exception as e:
        print("❌ Failed to load trade log:", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades")
def get_all_trades():
    try:
        pattern = os.path.join(TRADE_HISTORY_DIR, "*_trades.json")
        all_files = glob.glob(pattern)
        if not all_files:
            print("⚠️ No trade logs found.")
            return {
                "symbol": "ALL",
                "data": []
            }

        all_trades = []
        for file_path in all_files:
            symbol = os.path.basename(file_path).replace("_trades.json", "").upper()
            with open(file_path, "r") as f:
                trades = json.load(f)

                for trade in trades:
                    trade["symbol"] = symbol  # Add for frontend usage

                all_trades.extend(trades)

        all_trades.sort(key=lambda x: x.get("timestamp") or "", reverse=True)

        return {
            "symbol": "ALL",
            "data": all_trades
        }

    except Exception as e:
        print("❌ Failed to load all trade logs:", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profits/{symbol}")
def get_profit_summary(symbol: str):
    try:
        symbol = symbol.upper()
        summary = compute_trade_profits(symbol)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"No trades found for {symbol}")
        return summary
    except Exception as e:
        print("❌ Profit summary generation failed:", e)
        raise HTTPException(status_code=500, detail=str(e))
