import os
from backend.binance.fetch_live_ohlcv import fetch_ohlcv

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "LTCUSDT", 
    "BCHUSDT",
    "AVAXUSDT",
]

OUTPUT_DIR = "data/ohlcv"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for symbol in SYMBOLS:
        print(f"Fetching 30m data for {symbol}...")
        # Changed from 1h to 30m interval and increased limit for more historical data
        df = fetch_ohlcv(symbol, interval="30m", limit=2000)  # 2000 * 30m = ~41 days of data
        if df.empty:
            print(f"❌ No data fetched for {symbol}")
            continue

        # Updated filename to reflect 30m interval
        file_path = os.path.join(OUTPUT_DIR, f"{symbol}_30m.csv")
        df.to_csv(file_path)
        print(f"✅ Saved {symbol} 30m OHLCV to {file_path}")

if __name__ == "__main__":
    main()