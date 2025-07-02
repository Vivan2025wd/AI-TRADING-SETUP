# backend/routes/binance.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.utils.binance_api import connect_user_api

# Removed prefix to avoid redundancy — route will be /api/binance/connect after inclusion in main.py
router = APIRouter(tags=["BinanceAPI"])

class APIKeys(BaseModel):
    apiKey: str
    secretKey: str

@router.post("/connect")
def connect_to_binance(keys: APIKeys):
    try:
        result = connect_user_api(keys.apiKey, keys.secretKey)
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("message", "Invalid API keys"))
        return {"message": result.get("message", "Connected to Binance")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
