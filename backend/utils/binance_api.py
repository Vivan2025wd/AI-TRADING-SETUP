import os
import sys
from binance.client import Client
from dotenv import load_dotenv
from typing import Optional

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# --- Logger Setup ---
try:
    from backend.utils.logger import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)

# Load .env variables
load_dotenv()

# --- Global Variables ---
CURRENT_TRADING_MODE = "mock"  # Default Mode: mock
user_binance_client: Optional[Client] = None  # Global Binance Client Instance

# Optional fallback from .env (used if API keys not provided in function call)
DEFAULT_API_KEY = os.getenv("BINANCE_API_KEY", "")
DEFAULT_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# --- Trading Mode Functions ---
def set_trading_mode(mode: str):
    global CURRENT_TRADING_MODE
    CURRENT_TRADING_MODE = mode.lower()

def get_trading_mode() -> str:
    return CURRENT_TRADING_MODE

# --- API Connection Functions ---
def connect_user_api(api_key: Optional[str] = None, secret_key: Optional[str] = None, mode: str = "mock") -> dict:
    """
    Connect to Binance API with provided or default keys.
    Supports 'mock' mode for Testnet and 'live' mode for real trading.
    """
    global user_binance_client
    set_trading_mode(mode)

    key = api_key or DEFAULT_API_KEY
    secret = secret_key or DEFAULT_API_SECRET

    if not key or not secret:
        log.error("[Binance Init Error] Missing API key or secret.")
        return {"success": False, "message": "Missing API key or secret."}

    try:
        client = Client(api_key=key, api_secret=secret)
        
        if mode.lower() == "mock":
            client.API_URL = 'https://testnet.binance.vision/api'
        
        # Test API connectivity
        client.get_account()
        user_binance_client = client
        
        log.info(f"[Binance] API connected in {mode.upper()} MODE.")
        return {"success": True, "message": f"Connected in {mode.upper()} MODE."}

    except Exception as e:
        log.error(f"[Binance Init Error] {e}")
        return {"success": False, "message": f"Connection failed: {str(e)}"}

def get_binance_client() -> Client:
    """
    Returns the connected Binance client or raises an error if not connected.
    """
    if user_binance_client is None:
        raise Exception("Binance API client not initialized. Call connect_user_api() first.")
    return user_binance_client

def disconnect_client():
    """
    Disconnect the current Binance client.
    """
    global user_binance_client
    user_binance_client = None
    log.info("[Binance] Client disconnected.")

# --- Binance API Utility Functions ---
def fetch_ohlcv(symbol: str, interval: str = Client.KLINE_INTERVAL_1MINUTE, limit: int = 100) -> list:
    """
    Fetch OHLCV candle data for a given symbol and interval.
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
    Fetch the current ticker price for the given symbol.
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
    """
    try:
        client = get_binance_client()
        return client.get_account()
    except Exception as e:
        log.error(f"[Binance Account Info Error] {e}")
        raise e

def get_exchange_info(symbol: Optional[str] = None) -> dict:
    """
    Get exchange information for all symbols or a specific symbol.
    """
    try:
        client = get_binance_client()
        if symbol:
            info = client.get_symbol_info(symbol.upper())
            if not info:
                raise Exception(f"Symbol '{symbol}' not found.")
            return info
        return client.get_exchange_info()
    except Exception as e:
        log.error(f"[Binance Exchange Info Error] {e}")
        raise e

def test_connection() -> dict:
    """
    Test Binance API connection (server time + account check).
    """
    try:
        client = get_binance_client()
        server_time = client.get_server_time()
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

def get_safe_binance_client() -> Client:
    if user_binance_client is None:
        raise Exception("Binance client not connected. Please connect API keys first.")
    return user_binance_client
