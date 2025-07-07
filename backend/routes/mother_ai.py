from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from backend.mother_ai.mother_ai import MotherAI
from backend.mother_ai.trade_executer import execute_mother_ai_decision

import os
import json
import glob

router = APIRouter(tags=["Mother AI"])

# Global MotherAI instance (optional usage depending on your app)
mother_ai_instance = MotherAI(agent_symbols=None)

@router.get("/decision")
def get_mother_ai_decision():
    try:
        print("🧠 Mother AI: Starting portfolio decision...")
        result = mother_ai_instance.make_portfolio_decision()

        if not result or not result.get("decision"):
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
        decision = mother_ai_instance.make_portfolio_decision()

        if not decision or not decision.get("decision"):
            return JSONResponse(
                status_code=200,
                content={
                    "message": "No valid decision to execute.",
                    "executed_trades": []
                }
            )

        executed = execute_mother_ai_decision(decision)

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
        log_file_path = f"backend/storage/performance_logs/{symbol}_trades.json"

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
        print("❌ Failed to load performance log:", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades")
def get_all_trades():
    try:
        base_dir = "backend/storage/performance_logs"
        pattern = os.path.join(base_dir, "*_trades.json")
        all_files = glob.glob(pattern)

        all_trades = []
        for file_path in all_files:
            symbol = os.path.basename(file_path).replace("_trades.json", "").upper()
            with open(file_path, "r") as f:
                trades = json.load(f)

                # Add symbol to each trade for frontend identification
                for trade in trades:
                    trade["symbol"] = symbol

                all_trades.extend(trades)

        # Optional: sort all trades by timestamp ascending
        all_trades.sort(key=lambda x: x.get("timestamp") or "")

        return {
            "symbol": "ALL",
            "data": all_trades
        }

    except Exception as e:
        print("❌ Failed to load all performance logs:", e)
        raise HTTPException(status_code=500, detail=str(e))
