from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from backend.utils.binance_api import connect_user_api, get_safe_binance_client,disconnect_client
from backend.binance.fetch_live_ohlcv import fetch_ohlcv

router = APIRouter(tags=["BinanceAPI"])

class APIKeys(BaseModel):
    apiKey: str
    secretKey: str
    tradingMode: str = "mock"  # default mode is "mock"

@router.post("/connect")
def connect_to_binance(keys: APIKeys):
    """
    Connect to Binance API using provided keys and set trading mode.
    """
    try:
        result = connect_user_api(keys.apiKey, keys.secretKey, keys.tradingMode)
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("message", "Invalid API keys or connection failed"))
        return {"message": result.get("message", "Connected to Binance")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/price")
def get_latest_price(symbol: str = Query(..., description="Trading symbol like ADAUSDT")):
    """
    Get the latest ticker price for a symbol.
    """
    try:
        client = get_safe_binance_client()
        symbol = symbol.upper()
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = float(ticker.get("price", 0))
        return {"symbol": symbol, "price": price}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching price for {symbol}: {e}")

@router.get("/fetch_ohlcv")
def get_ohlcv(
    symbol: str = Query(..., description="Trading symbol like BTCUSDT"),
    interval: str = Query("1h", description="Interval like 1m, 5m, 1h, 1d"),
    limit: int = Query(100, description="Number of candles to fetch"),
):
    """
    Fetch OHLCV candles data.
    """
    df = fetch_ohlcv(symbol.upper(), interval=interval, limit=limit)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No OHLCV data found for {symbol}")
    data = df.reset_index().to_dict(orient="records")
    return {"symbol": symbol.upper(), "interval": interval, "limit": limit, "ohlcv": data}

@router.get("/account/balance")
def get_account_balance():
    """
    Get spot account balances with non-zero amounts.
    Requires valid API keys connected.
    """
    try:
        client = get_safe_binance_client()
        account_info = client.get_account()
        balances = account_info.get("balances", [])
        non_zero_balances = [
            b for b in balances if float(b.get("free", 0)) > 0 or float(b.get("locked", 0)) > 0
        ]
        return {"balances": non_zero_balances}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching account balance: {str(e)}")

@router.get("/test")
def test_route():
    """
    Simple health check endpoint.
    """
    return {"message": "Binance route is working!"}

@router.post("/disconnect")
def disconnect_binance_client():
    """
    Disconnect the current Binance API session.
    """
    try:
        disconnect_client()
        return {"message": "Disconnected from Binance successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disconnect: {str(e)}")
    

@router.get("/status")
def get_connection_status():
    """
    Check if Binance API client is connected.
    """
    from backend.utils.binance_api import user_binance_client, get_trading_mode
    is_connected = user_binance_client is not None
    mode = get_trading_mode()
    return {"connected": is_connected, "mode": mode}
