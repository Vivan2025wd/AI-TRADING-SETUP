# utils/binance_api.py

import os
from binance.client import Client
from dotenv import load_dotenv
from utils.logger import log

load_dotenv()

# Optional fallback from .env (can be overridden by user injection)
DEFAULT_API_KEY = os.getenv("BINANCE_API_KEY", "")
DEFAULT_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Store active client per user/session (dict for simplicity)
client_sessions = {}


from typing import Optional

def init_binance_client(user_id: str, api_key: Optional[str] = None, api_secret: Optional[str] = None):
    """
    Initializes or updates a Binance client instance for a user.
    """
    try:
        key = api_key or DEFAULT_API_KEY
        secret = api_secret or DEFAULT_API_SECRET
        client_sessions[user_id] = Client(api_key=key, api_secret=secret)
        log.info(f"[Binance] Client initialized for user: {user_id}")
    except Exception as e:
        log.error(f"[Binance Init Error] {e}")
        raise


def get_client(user_id: str) -> Client:
    """
    Returns an existing Binance client instance for a given user.
    """
    if user_id not in client_sessions:
        raise ValueError("Client not initialized. Call init_binance_client() first.")
    return client_sessions[user_id]


def fetch_ohlcv(user_id: str, symbol: str, interval: str = Client.KLINE_INTERVAL_1MINUTE, limit: int = 100):
    try:
        client = get_client(user_id)
        klines = client.get_klines(symbol=symbol.upper(), interval=interval, limit=limit)
        ohlcv = [
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
        return ohlcv
    except Exception as e:
        log.error(f"[Binance Fetch Error] {e}")
        return []
