# backend/binance/fetch_live_ohlcv.py

import ccxt
import pandas as pd
from datetime import datetime

# Initialize Binance exchange client (no auth needed for public data)
binance = ccxt.binance()

def fetch_ohlcv(symbol: str, interval="1h", limit=100) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance using ccxt.

    symbol: 'BTC/USDT', 'ETH/USDT', etc.
    interval: '1m', '5m', '1h', '1d', etc.
    limit: number of candles to fetch (max 1000 for most intervals)
    """
    try:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe=interval, limit=limit)

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    except Exception as e:
        print(f"[fetch_ohlcv] Error fetching data for {symbol}: {e}")
        return pd.DataFrame()
