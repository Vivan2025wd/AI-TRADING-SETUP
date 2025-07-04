from fastapi import APIRouter, HTTPException, Query
from backend.mother_ai.mother_ai import MotherAI
import os

router = APIRouter(tags=["Mother AI"])

# Global MotherAI instance (load all agents)
mother_ai_instance = MotherAI(agent_symbols=None)


@router.get("/decision")
def get_mother_ai_decision():
    try:
        print("🧠 Mother AI: Starting portfolio decision...")

        result = mother_ai_instance.make_portfolio_decision()

        if not result or not result.get("decision"):
            print("⚠️ No valid decision found.")
            # Still return consistent empty structure to avoid frontend crashing
            return {
                "decision": [],
                "timestamp": mother_ai_instance.performance_tracker.current_time()
            }

        print("✅ Decision found!")
        return result

    except Exception as e:
        print("❌ Mother AI Decision Error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades")
def get_trades(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1)
):
    try:
        all_trades = []

        logs_dir = mother_ai_instance.performance_tracker.log_dir

        if not os.path.exists(logs_dir):
            print(f"⚠️ Trade log directory not found: {logs_dir}")
            return {
                "page": page,
                "limit": limit,
                "total": 0,
                "totalPages": 0,
                "data": []
            }

        for filename in os.listdir(logs_dir):
            if filename.endswith("_log.json"):
                symbol = filename.replace("_log.json", "")
                trades = mother_ai_instance.performance_tracker.get_agent_log(symbol)
                if trades:
                    all_trades.extend(trades)

        # Sort by most recent
        sorted_trades = sorted(all_trades, key=lambda x: x.get("timestamp", ""), reverse=True)

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

    except Exception as e:
        print("❌ Trade history fetch failed:", e)
        raise HTTPException(status_code=500, detail=str(e))
