import ccxt
import pandas as pd

binance = ccxt.binance()

def format_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    if '/' in symbol:
        return symbol
    base = symbol[:-4]
    quote = symbol[-4:]
    return f"{base}/{quote}"

def fetch_ohlcv(symbol: str, interval="1m", limit=100) -> pd.DataFrame:
    try:
        formatted_symbol = format_symbol(symbol)
        print(f"📡 [Live OHLCV] Fetching: {formatted_symbol} | Interval: {interval} | Limit: {limit}")

        # 1. Fetch candles (limit + 1 for buffer)
        ohlcv = binance.fetch_ohlcv(formatted_symbol, timeframe=interval, limit=limit)

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # 2. Fetch live ticker price
        ticker = binance.fetch_ticker(formatted_symbol)
        live_price = ticker["last"]

        # 3. Modify the last candle's close/high/low
        if not df.empty:
            last_candle = df.iloc[-1]

            new_close = live_price
            high = last_candle["high"] if pd.notnull(last_candle["high"]) else live_price
            low = last_candle["low"] if pd.notnull(last_candle["low"]) else live_price
            safe_high = high if high is not None else live_price
            safe_low = low if low is not None else live_price
            # Ensure both values are not None for max/min
            safe_high = safe_high if safe_high is not None else 0
            safe_low = safe_low if safe_low is not None else 0
            safe_live_price = live_price if live_price is not None else 0
            new_high = max(safe_high, safe_live_price)
            new_low = min(safe_low, safe_live_price)

            df.at[df.index[-1], "close"] = new_close
            df.at[df.index[-1], "high"] = new_high
            df.at[df.index[-1], "low"] = new_low

        print(f"✅ [Live OHLCV] Returning {len(df)} candles with live-updated close")

        return df

    except Exception as e:
        print(f"❌ [Live OHLCV] Error fetching live data for {symbol}: {e}")
        return pd.DataFrame()
