# backend/routes/binance.py
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from backend.utils.binance_api import connect_user_api, user_binance_client
from backend.binance.fetch_live_ohlcv import fetch_ohlcv  # import your OHLCV fetch function

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

@router.get("/price")
def get_latest_price(symbol: str = Query(..., description="Trading symbol like ADAUSDT")):
    try:
        symbol = symbol.upper()
        if user_binance_client is None:
            raise HTTPException(status_code=503, detail="Binance client not connected")
        ticker = user_binance_client.get_symbol_ticker(symbol=symbol)
        price = float(ticker.get("price", 0))
        return {"symbol": symbol, "price": price}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching price for {symbol}: {e}")

@router.get("/fetch_ohlcv")
def get_ohlcv(
    symbol: str = Query(..., description="Trading symbol like BTCUSDT"),
    interval: str = Query("1h", description="Time interval, e.g. 1m, 5m, 1h, 1d"),
    limit: int = Query(100, description="Number of candles to fetch"),
):
    df = fetch_ohlcv(symbol, interval=interval, limit=limit)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No OHLCV data found for symbol {symbol}")
    data = df.reset_index().to_dict(orient="records")
    return {"symbol": symbol, "interval": interval, "limit": limit, "ohlcv": data}

@router.get("/test")
def test_route():
    return {"message": "Binance route is working!"}
