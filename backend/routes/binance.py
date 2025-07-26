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

@router.get("/test-permissions")
def test_api_permissions():
    """
    Test Binance API permissions and diagnose issues
    """
    try:
        from backend.binance.binance_trader import test_api_permissions as test_perms
        result = test_perms()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Permission test failed: {str(e)}")

@router.get("/test-connection")
def test_binance_connection():
    """
    Enhanced connection test with detailed diagnostics
    """
    try:
        from backend.utils.binance_api import get_trading_mode, get_binance_client
        
        client = get_binance_client()
        mode = get_trading_mode()
        
        # Test basic connectivity
        server_time = client.get_server_time()
        account_info = client.get_account()
        
        # Check permissions
        can_trade = account_info.get("canTrade", False)
        permissions = account_info.get("permissions", [])
        
        return {
            "success": True,
            "trading_mode": mode,
            "server_time": server_time,
            "account_type": account_info.get("accountType"),
            "can_trade": can_trade,
            "permissions": permissions,
            "api_url": getattr(client, 'API_URL', 'https://api.binance.com'),
            "balances": [
                {"asset": b["asset"], "free": float(b["free"]), "locked": float(b["locked"])}
                for b in account_info.get("balances", [])
                if float(b["free"]) > 0 or float(b["locked"]) > 0
            ][:10]  # Show top 10 non-zero balances
        }
    except Exception as e:
        error_msg = str(e)
        suggestions = []
        
        if "Invalid API-key" in error_msg:
            suggestions = [
                "Check API key permissions in Binance dashboard",
                "Ensure 'Spot & Margin Trading' is enabled",
                "Verify IP restrictions",
                "Confirm using correct API keys (mainnet vs testnet)"
            ]
        
        return {
            "success": False,
            "error": error_msg,
            "suggestions": suggestions
        }
    

    # Add this to your binance routes (routes/binance.py)

@router.get("/balance-summary")
def get_balance_summary():
    """
    Get a summary of account balances and trading readiness
    """
    try:
        client = get_safe_binance_client()
        account_info = client.get_account()
        balances = account_info.get("balances", [])
        
        # Get non-zero balances
        non_zero_balances = [
            {
                "asset": b["asset"],
                "free": float(b["free"]),
                "locked": float(b["locked"]),
                "total": float(b["free"]) + float(b["locked"])
            }
            for b in balances 
            if float(b.get("free", 0)) > 0 or float(b.get("locked", 0)) > 0
        ]
        
        # Check USDT balance for trading
        usdt_balance = next((b for b in non_zero_balances if b["asset"] == "USDT"), None)
        usdt_free = usdt_balance["free"] if usdt_balance else 0
        
        # Trading readiness check
        can_trade_buy = usdt_free >= 10  # Minimum $10 USDT for buy orders
        
        # Estimate number of possible trades
        max_trades = int(usdt_free / 10) if usdt_free > 0 else 0
        
        return {
            "total_balances": len(non_zero_balances),
            "usdt_available": usdt_free,
            "can_trade_buy": can_trade_buy,
            "estimated_max_trades": max_trades,
            "balances": non_zero_balances,
            "recommendations": [
                "Add USDT to your account for buy orders" if not can_trade_buy else "Ready for trading!",
                f"Current balance allows ~{max_trades} trades at $10 each" if max_trades > 0 else "Insufficient balance for trading"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching balance summary: {str(e)}")