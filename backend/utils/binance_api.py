import os
import sys
from binance.client import Client
from dotenv import load_dotenv
from typing import Optional

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import logger after path setup
try:
    from backend.utils.logger import log
except ImportError:
    # Fallback to basic logging if custom logger isn't available
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

load_dotenv()

# Global (single-user) Binance client instance
user_binance_client: Optional[Client] = None

# Optional fallback from .env
DEFAULT_API_KEY = os.getenv("BINANCE_API_KEY", "")
DEFAULT_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Trading Mode
# Load from .env, default to "False" if not set or invalid
REAL_TRADING_MODE_STR = os.getenv("REAL_TRADING_MODE", "False")
REAL_TRADING_MODE = REAL_TRADING_MODE_STR.lower() == "true"

def is_real_trading_mode() -> bool:
    """Returns True if real trading mode is enabled, False otherwise."""
    return REAL_TRADING_MODE

def connect_user_api(api_key: Optional[str] = None, secret_key: Optional[str] = None) -> dict:
    """
    Initializes the Binance client using provided or default API keys.
    Returns a dict with 'success': True/False and a message.
    """
    global user_binance_client

    key = api_key or DEFAULT_API_KEY
    secret = secret_key or DEFAULT_API_SECRET

    if not key or not secret:
        log.error("[Binance Init Error] Missing API key or secret.")
        return {"success": False, "message": "Missing API key or secret."}

    try:
        # Create Binance client instance
        client = Client(api_key=key, api_secret=secret)
        
        # For testnet, you need to set the testnet flag
        # If you want to use testnet, uncomment the line below:
        # client = Client(api_key=key, api_secret=secret, testnet=True)
        
        # Test credentials with a simple API call
        account_info = client.get_account()
        
        # Store the client globally
        user_binance_client = client
        
        log.info("[Binance] API connection successful.")
        if is_real_trading_mode():
            log.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            log.warning("!!! REAL TRADING MODE IS ACTIVE. LIVE ORDERS WILL BE PLACED !!!")
            log.warning("!!! Ensure you are using TESTNET keys if testing.            !!!")
            log.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            log.info("[Binance] Connection is in MOCK TRADING MODE.")

        return {"success": True, "message": "Binance API connected successfully."}

    except Exception as e:
        log.error(f"[Binance Init Error] {e}")
        return {"success": False, "message": f"Connection failed: {str(e)}"}

def get_binance_client() -> Client:
    """
    Returns the connected Binance client, or raises error if not connected.
    """
    if user_binance_client is None:
        raise Exception("Binance API client not initialized. Call connect_user_api() first.")
    return user_binance_client

def fetch_ohlcv(symbol: str, interval: str = Client.KLINE_INTERVAL_1MINUTE, limit: int = 100) -> list:
    """
    Fetches OHLCV candle data for a given symbol and interval.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval (use Client.KLINE_INTERVAL_* constants)
        limit: Number of klines to fetch (max 1000)
        
    Returns:
        List of OHLCV dictionaries
    """
    try:
        client = get_binance_client()
        klines = client.get_klines(symbol=symbol.upper(), interval=interval, limit=limit)
        
        return [
            {
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            }
            for k in klines
        ]
    except Exception as e:
        log.error(f"[Binance OHLCV Error] {e}")
        return []

def get_symbol_price(symbol: str) -> float:
    """
    Fetches the current ticker price for the given symbol from Binance.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        
    Returns:
        Current price as float
        
    Raises:
        Exception: If price fetch fails
    """
    try:
        client = get_binance_client()
        ticker = client.get_symbol_ticker(symbol=symbol.upper())
        return float(ticker['price'])
    except Exception as e:
        log.error(f"[Binance Price Fetch Error] {e}")
        raise e

def get_account_info() -> dict:
    """
    Get account information including balances.
    
    Returns:
        Account information dictionary
    """
    try:
        client = get_binance_client()
        return client.get_account()
    except Exception as e:
        log.error(f"[Binance Account Info Error] {e}")
        raise e

def get_exchange_info(symbol: Optional[str] = None) -> dict:
    """
    Get exchange information for symbols.
    
    Args:
        symbol: Optional specific symbol to get info for
        
    Returns:
        Exchange information dictionary
        
    Raises:
        Exception: If symbol not found or API call fails
    """
    try:
        client = get_binance_client()
        if symbol:
            result = client.get_symbol_info(symbol.upper())
            if result is None:
                raise Exception(f"Symbol '{symbol}' not found")
            return result
        else:
            return client.get_exchange_info()
    except Exception as e:
        log.error(f"[Binance Exchange Info Error] {e}")
        raise e

def disconnect_client():
    """
    Disconnect the current Binance client.
    """
    global user_binance_client
    user_binance_client = None
    log.info("[Binance] Client disconnected.")

# Test function to verify connection
def test_connection() -> dict:
    """
    Test the current Binance connection.
    
    Returns:
        Test result dictionary with success status and details
    """
    try:
        client = get_binance_client()
        
        # Test server connectivity
        server_time = client.get_server_time()
        
        # Test account access
        account_info = client.get_account()
        
        return {
            "success": True,
            "server_time": server_time,
            "account_type": account_info.get("accountType"),
            "can_trade": account_info.get("canTrade"),
            "permissions": account_info.get("permissions", [])
        }
    except Exception as e:
        log.error(f"[Binance Connection Test Error] {e}")
        return {
            "success": False,
            "error": str(e)
        }