import os
import pandas as pd
from typing import Dict
from backend.agents.agent_training import prepare_training_data, train_agent_model
from joblib import load  # <-- added to load saved model file

# Configuration: Paths where your OHLCV and label data reside
OHLCV_DATA_DIR = "data/ohlcv"
LABELS_DATA_DIR = "data/labels"
MODEL_SAVE_DIR = "backend/agents/models"

# List your symbols here, e.g. ["BTCUSDT", "ETHUSDT", "AVAXUSDT"]
SYMBOLS = ["DOGEUSDT","SOLUSDT","XRPUSDT","DOTUSDT", "LTCUSDT", "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"]

def load_ohlcv(symbol: str) -> pd.DataFrame:
    filepath = os.path.join(OHLCV_DATA_DIR, f"{symbol}_1h.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"OHLCV file missing for {symbol}: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df

def load_labels(symbol: str) -> pd.Series:
    filepath = os.path.join(LABELS_DATA_DIR, f"{symbol}_labels.csv")
    if not os.path.exists(filepath):
        print(f"⚠️ Labels file missing for {symbol}, using default 'hold' labels")
        # Return a Series of "hold" labels for the OHLCV index (fallback)
        ohlcv = load_ohlcv(symbol)
        return pd.Series(["hold"] * len(ohlcv), index=ohlcv.index)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if "action" not in df.columns:
        raise ValueError(f"Labels CSV for {symbol} must have an 'action' column")
    return df["action"]

def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    for symbol in SYMBOLS:
        try:
            print(f"\n=== Training agent model for {symbol} ===")
            ohlcv = load_ohlcv(symbol)
            labels = load_labels(symbol)
            training_data = prepare_training_data(ohlcv, labels)
            
            # Train the model (train_agent_model might return None or path, so ignore return)
            train_agent_model(symbol, training_data, model_dir=MODEL_SAVE_DIR)
            
            # Explicitly load the saved model to access attributes like classes_
            model_path = os.path.join(MODEL_SAVE_DIR, f"{symbol.lower()}_model.pkl")
            model = load(model_path)
            
            print(f"Trained model classes for {symbol}: {model.classes_}")
        except Exception as e:
            print(f"❌ Failed to train model for {symbol}: {e}")

if __name__ == "__main__":
    main()
