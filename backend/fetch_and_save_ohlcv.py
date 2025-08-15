import os
from backend.binance.fetch_live_ohlcv import fetch_ohlcv

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "ATOMUSDT",
    "DOTUSDT",
    "LTCUSDT", 
    "BCHUSDT",
    "AVAXUSDT",
]

OUTPUT_DIR = "data/ohlcv"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for symbol in SYMBOLS:
        print(f"Fetching 1h data for {symbol}...")
        # Fetch 1h interval data with sufficient history
        df = fetch_ohlcv(symbol, interval="1h", limit=2000)  # 2000 * 1h = ~83 days of data
        if df.empty:
            print(f"❌ No data fetched for {symbol}")
            continue
        
        # Save with 1h filename for clarity
        file_path = os.path.join(OUTPUT_DIR, f"{symbol}_1h.csv")
        df.to_csv(file_path)
        print(f"✅ Saved {symbol} 1h OHLCV to {file_path}")

if __name__ == "__main__":
    main()