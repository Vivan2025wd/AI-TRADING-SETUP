import os
from binance.client import Client
from dotenv import load_dotenv
from backend.utils.logger import log
from typing import Optional
import sys

load_dotenv()

# Global (single-user) Binance client instance
user_binance_client: Optional[Client] = None

# Optional fallback from .env
DEFAULT_API_KEY = os.getenv("BINANCE_API_KEY", "")
DEFAULT_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

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
        client = Client(api_key=key, api_secret=secret)
        client.get_account()  # Test credentials

        user_binance_client = client
        log.info("[Binance] API connection successful.")
        return {"success": True, "message": "Binance API connected successfully."}

    except Exception as e:
        log.error(f"[Binance Init Error] {e}")
        return {"success": False, "message": f"Connection failed: {str(e)}"}


def get_binance_client() -> Client:
    """
    Returns the connected Binance client, or raises error if not connected.
    """
    if user_binance_client is None:
        raise Exception("Binance API client not initialized.")
    return user_binance_client

def fetch_ohlcv(symbol: str, interval: str = Client.KLINE_INTERVAL_1MINUTE, limit: int = 100):
    """
    Fetches OHLCV candle data for a given symbol and interval.
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def get_symbol_price(symbol: str) -> float:
    """
    Fetches the current ticker price for the given symbol from Binance.
    """
    try:
        client = get_binance_client()
        ticker = client.get_symbol_ticker(symbol=symbol.upper())
        return float(ticker['price'])
    except Exception as e:
        log.error(f"[Binance Price Fetch Error] {e}")
        raise e
