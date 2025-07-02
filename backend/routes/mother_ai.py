from fastapi import APIRouter, HTTPException
from backend.mother_ai.mother_ai import MotherAI

router = APIRouter(tags=["Mother AI"])

# Create a MotherAI instance with example agent symbols
mother_ai_instance = MotherAI(agent_symbols=["AAPL", "TSLA", "GOOG"])

@router.get("/decision")
def get_mother_ai_decision():
    try:
        result = mother_ai_instance.make_portfolio_decision()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trades")
def get_trades():
    try:
        all_trades = []
        for symbol in mother_ai_instance.agent_symbols:
            trades = mother_ai_instance.performance_tracker.get_agent_log(symbol)
            if trades:
                all_trades.extend(trades)
        # Sort trades by timestamp descending
        sorted_trades = sorted(all_trades, key=lambda x: x.get("timestamp", ""), reverse=True)
        return sorted_trades
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
