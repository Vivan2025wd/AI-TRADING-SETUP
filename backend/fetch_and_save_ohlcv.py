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
        print(f"Fetching data for {symbol}...")
        df = fetch_ohlcv(symbol, interval="1h", limit=1000)
        if df.empty:
            print(f"❌ No data fetched for {symbol}")
            continue

        file_path = os.path.join(OUTPUT_DIR, f"{symbol}_1h.csv")
        df.to_csv(file_path)
        print(f"✅ Saved {symbol} OHLCV to {file_path}")

if __name__ == "__main__":
    main()
