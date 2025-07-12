import os
import pandas as pd
from datetime import datetime

OHLCV_DIR = "data/ohlcv"
TRADES_DIR = "backend/storage/performance_logs"
LABELS_DIR = "data/labels"

os.makedirs(LABELS_DIR, exist_ok=True)

def load_ohlcv(symbol: str) -> pd.DataFrame:
    path = os.path.join(OHLCV_DIR, f"{symbol}_1h.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def load_trades(symbol: str) -> pd.DataFrame:
    path = os.path.join(TRADES_DIR, f"{symbol}_trades.json")
    if not os.path.exists(path):
        print(f"⚠️ Trades file not found for {symbol} at {path}")
        return pd.DataFrame(columns=["timestamp", "signal"])

    trades = pd.read_json(path)
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades = trades.sort_values("timestamp")
    return trades

def generate_labels(symbol: str):
    print(f"Generating labels for {symbol}...")

    ohlcv = load_ohlcv(symbol)
    trades = load_trades(symbol)

    # Prepare label series, default to "hold"
    labels = pd.Series("hold", index=ohlcv.index)

    # For each trade signal, assign label to closest OHLCV timestamp <= trade time
    for _, trade in trades.iterrows():
        trade_time = trade["timestamp"]
        signal = trade.get("signal", "hold").lower()
        # Find the last OHLCV timestamp <= trade_time
        possible_times = ohlcv.index[ohlcv.index <= trade_time]
        if not possible_times.empty:
            closest_time = possible_times[-1]
            labels.loc[closest_time] = signal

    # Save labels to CSV
    label_df = pd.DataFrame({"action": labels})
    out_path = os.path.join(LABELS_DIR, f"{symbol}_labels.csv")
    label_df.to_csv(out_path)
    print(f"Saved labels to {out_path}")

def main():
    symbols = ["DOGEUSDT","SOLUSDT","XRPUSDT","DOTUSDT", "LTCUSDT", "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"]
    for symbol in symbols:
        try:
            generate_labels(symbol)
        except Exception as e:
            print(f"Failed to generate labels for {symbol}: {e}")

if __name__ == "__main__":
    main()
