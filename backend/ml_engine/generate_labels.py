import os
import pandas as pd
from datetime import datetime

OHLCV_DIR = "data/ohlcv"
TRADES_DIR = "backend/storage/trade_history"
LABELS_DIR = "data/labels"

os.makedirs(LABELS_DIR, exist_ok=True)

def load_ohlcv(symbol: str) -> pd.DataFrame:
    path = os.path.join(OHLCV_DIR, f"{symbol}_1h.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def load_trades(symbol: str) -> pd.DataFrame:
    path = os.path.join(TRADES_DIR, f"{symbol}_predictions.json")
    if not os.path.exists(path):
        print(f"⚠️ Trades file not found for {symbol} at {path}")
        return pd.DataFrame(columns=["timestamp", "signal"])

    trades = pd.read_json(path)
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades = trades.sort_values("timestamp")
    return trades

def generate_labels(symbol: str, window: int = 3):
    print(f"\n🔧 Generating labels for {symbol}...")

    ohlcv = load_ohlcv(symbol)
    trades = load_trades(symbol)

    # Default all to "hold"
    labels = pd.Series("hold", index=ohlcv.index)

    for _, trade in trades.iterrows():
        trade_time = trade.get("timestamp")
        signal = trade.get("signal", "hold").lower()

        if pd.isnull(trade_time) or signal not in ["buy", "sell", "hold"]:
            continue

        # Find index closest to trade_time
        idx = ohlcv.index.get_indexer([trade_time], method='nearest')[0]

        # Mark a window around that index
        start = max(0, idx - window)
        end = min(len(ohlcv) - 1, idx + window)

        labels.iloc[start:end + 1] = signal

    # Save labels
    label_df = pd.DataFrame({"action": labels})
    out_path = os.path.join(LABELS_DIR, f"{symbol}_labels.csv")
    label_df.to_csv(out_path)
    print(f"✅ Saved labels to {out_path}")

    # Print class distribution for debugging
    distribution = label_df["action"].value_counts(normalize=True).round(3) * 100
    print(f"📊 Label distribution for {symbol}:")
    for label, pct in distribution.items():
        print(f"   - {label}: {pct:.1f}%")

def main():
    symbols = ["DOGEUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT", "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"]
    for symbol in symbols:
        try:
            generate_labels(symbol)
        except Exception as e:
            print(f"❌ Failed to generate labels for {symbol}: {e}")

if __name__ == "__main__":
    main()
