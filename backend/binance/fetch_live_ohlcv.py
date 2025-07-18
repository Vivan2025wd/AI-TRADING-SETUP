import ccxt
import pandas as pd
import time
from datetime import datetime
from typing import Optional, Union, Dict, Any, List

# Initialize with better error handling and type annotations
binance: Optional[ccxt.binance] = None

try:
    binance = ccxt.binance({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
        }
    })
except Exception as e:
    print(f"❌ Failed to initialize Binance: {e}")
    binance = None

def format_symbol(symbol: str) -> str:
    """Enhanced symbol formatting"""
    symbol = symbol.upper()
    if '/' in symbol:
        return symbol
    
    # Handle USDT pairs
    if symbol.endswith('USDT'):
        base = symbol[:-4]
        return f"{base}/USDT"
    
    # Fallback - assume USDT
    return f"{symbol}/USDT"

def get_current_candle_start_time(interval: str) -> int:
    """Get the start time of the current candle"""
    now = int(time.time() * 1000)
    
    # Convert interval to milliseconds
    interval_ms = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000
    }.get(interval, 60 * 1000)
    
    # Calculate current candle start time
    return (now // interval_ms) * interval_ms

def fetch_ohlcv(symbol: str, interval: str = "1m", limit: int = 100) -> pd.DataFrame:
    """Fetch OHLCV with live current candle data"""
    if binance is None:
        print("❌ Binance not initialized")
        return pd.DataFrame()
    
    try:
        formatted_symbol = format_symbol(symbol)
        print(f"📡 [Live OHLCV] Fetching: {formatted_symbol} | Interval: {interval} | Limit: {limit}")

        # Step 1: Fetch OHLCV data (get extra candles to ensure we have current one)
        ohlcv_data: List[List[Union[int, float]]] = binance.fetch_ohlcv(
            formatted_symbol, timeframe=interval, limit=limit + 1
        )
        
        if not ohlcv_data:
            print(f"❌ [Live OHLCV] No OHLCV data received")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Step 2: Get live ticker for most recent price
        ticker = binance.fetch_ticker(formatted_symbol)
        # Handle both dict-like and object-like ticker responses
        if hasattr(ticker, 'get'):
            last_price = ticker.get("last")
        else:
            last_price = getattr(ticker, 'last', None)
        
        if last_price is None:
            print(f"❌ [Live OHLCV] No last price available in ticker")
            return df
            
        live_price = float(last_price)
        
        print(f"💰 [Live Price] Current: ${live_price:.6f}")

        # Step 3: Update the last (current) candle with live data
        if not df.empty:
            last_idx = df.index[-1]
            current_candle_start = get_current_candle_start_time(interval)
            last_candle_time = int(df.index[-1].timestamp() * 1000)
            
            # Check if the last candle is the current incomplete candle
            if abs(last_candle_time - current_candle_start) < 60000:  # Within 1 minute tolerance
                # Update current candle
                current_high = float(df.at[last_idx, "high"])
                current_low = float(df.at[last_idx, "low"])
                
                # Update with live price
                df.at[last_idx, "close"] = live_price
                df.at[last_idx, "high"] = max(current_high, live_price)
                df.at[last_idx, "low"] = min(current_low, live_price)
                
                print(f"🔄 [Live OHLCV] Updated current candle with live price")
            else:
                # Add a new current candle
                prev_close = float(df.iloc[-1]["close"])
                current_candle = pd.DataFrame({
                    "open": [prev_close],
                    "high": [live_price],
                    "low": [live_price],
                    "close": [live_price],
                    "volume": [0.0]
                }, index=[pd.to_datetime(current_candle_start, unit='ms')])
                
                df = pd.concat([df, current_candle])
                print(f"📊 [Live OHLCV] Added new current candle")

        # Step 4: Ensure we return exactly the requested number of candles
        if len(df) > limit:
            df = df.iloc[-limit:]

        # Step 5: Add metadata
        df["is_live"] = False
        if not df.empty:
            df.at[df.index[-1], "is_live"] = True

        # Step 6: Calculate and display price change
        if len(df) >= 2:
            prev_close = float(df.iloc[-2]["close"])
            price_change = live_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            print(f"📈 [Price Change] {price_change:+.6f} ({price_change_pct:+.2f}%)")

        print(f"✅ [Live OHLCV] Returning {len(df)} candles with live-updated close")
        return df

    except ccxt.NetworkError as e:
        print(f"❌ [Live OHLCV] Network error for {symbol}: {e}")
        return pd.DataFrame()
    except ccxt.ExchangeError as e:
        print(f"❌ [Live OHLCV] Exchange error for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ [Live OHLCV] Unexpected error for {symbol}: {e}")
        return pd.DataFrame()

def get_live_price(symbol: str) -> float:
    """Get just the current live price"""
    if binance is None:
        print("❌ [Live Price] Binance not initialized")
        return 0.0
        
    try:
        formatted_symbol = format_symbol(symbol)
        ticker = binance.fetch_ticker(formatted_symbol)
        # Handle both dict-like and object-like ticker responses
        if hasattr(ticker, 'get'):
            last_price = ticker.get("last")
        else:
            last_price = getattr(ticker, 'last', None)
        
        if last_price is None:
            print(f"❌ [Live Price] No last price available for {symbol}")
            return 0.0
            
        return float(last_price)
    except Exception as e:
        print(f"❌ [Live Price] Error getting price for {symbol}: {e}")
        return 0.0

def test_live_updates(symbol: str = "BTCUSDT", interval: str = "1m") -> None:
    """Test live price updates"""
    print(f"🧪 Testing live updates for {symbol}")
    
    for i in range(3):
        df = fetch_ohlcv(symbol, interval, 5)
        if not df.empty:
            live_price = float(df.iloc[-1]["close"])
            is_live = bool(df.iloc[-1].get("is_live", False))
            print(f"   Test {i+1}: ${live_price:.6f} (live: {is_live})")
        
        if i < 2:  # Don't sleep after last iteration
            time.sleep(2)

if __name__ == "__main__":
    # Test the function
    test_live_updates("BTCUSDT", "1m")