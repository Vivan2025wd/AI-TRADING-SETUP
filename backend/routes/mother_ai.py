from fastapi import APIRouter
from backend.mother_ai.mother_ai import MotherAI

router = APIRouter(prefix="/mother-ai", tags=["Mother AI"])

# Create a MotherAI instance with agent symbols (example symbols)
mother_ai_instance = MotherAI(agent_symbols=["AAPL", "TSLA", "GOOG"])

@router.get("/decision")
def get_mother_ai_decision():
    # Call the instance method
    result = mother_ai_instance.make_portfolio_decision()
    return result
