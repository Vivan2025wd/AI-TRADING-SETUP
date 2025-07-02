from fastapi import APIRouter

router = APIRouter()

@router.get("/agents")
async def list_agents():
    # Return a list of agent names
    return ["BTC", "ETH", "SOL"]
