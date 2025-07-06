import ccxt
import pandas as pd

# Initialize Binance exchange client (no auth needed for public data)
binance = ccxt.binance()

def format_symbol(symbol: str) -> str:
    """
    Converts symbols like 'BTCUSDT' to 'BTC/USDT' for ccxt compatibility.
    """
    symbol = symbol.upper()
    if '/' in symbol:
        return symbol  # already formatted
    base = symbol[:-4]
    quote = symbol[-4:]
    return f"{base}/{quote}"

def fetch_ohlcv(symbol: str, interval="1h", limit=100) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance using ccxt.

    Args:
        symbol: e.g. 'BTCUSDT' or 'BTC/USDT'
        interval: e.g. '1m', '5m', '1h', '1d', etc.
        limit: number of candles to fetch (max 1000 for most intervals)

    Returns:
        pd.DataFrame with columns: timestamp (datetime index), open, high, low, close, volume
    """
    try:
        formatted_symbol = format_symbol(symbol)
        print(f"📡 [fetch_ohlcv] Fetching: {formatted_symbol} | Interval: {interval} | Limit: {limit}")  # ✅ LOG

        ohlcv = binance.fetch_ohlcv(formatted_symbol, timeframe=interval, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        print(f"✅ [fetch_ohlcv] Fetched {len(df)} rows for {formatted_symbol}")  # ✅ LOG
        return df
    except Exception as e:
        print(f"❌ [fetch_ohlcv] Error fetching data for {symbol}: {e}")
        return pd.DataFrame()
