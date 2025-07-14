import os
import sys
import pandas as pd
from typing import Dict
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# 🔧 Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.ml_engine.feature_extractor import extract_features
from backend.agents.agent_training import train_agent_model

# Directories
OHLCV_DATA_DIR = "data/ohlcv"
LABELS_DATA_DIR = "data/labels"
MODEL_SAVE_DIR = "backend/agents/models"

# Tradable symbols
SYMBOLS = [
    "DOGEUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT",
    "LTCUSDT", "ADAUSDT", "BCHUSDT", "BTCUSDT",
    "ETHUSDT", "AVAXUSDT"
]

def load_ohlcv(symbol: str) -> pd.DataFrame:
    filepath = os.path.join(OHLCV_DATA_DIR, f"{symbol}_1h.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ OHLCV file missing for {symbol}: {filepath}")
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def load_labels(symbol: str) -> pd.Series:
    filepath = os.path.join(LABELS_DATA_DIR, f"{symbol}_labels.csv")
    if not os.path.exists(filepath):
        print(f"⚠️ Labels file missing for {symbol}. Defaulting to 'hold' only.")
        ohlcv = load_ohlcv(symbol)
        return pd.Series(["hold"] * len(ohlcv), index=ohlcv.index)
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if "action" not in df.columns:
        raise ValueError(f"❌ Labels file for {symbol} must contain 'action' column")
    return df["action"]

def print_label_distribution(labels: pd.Series, symbol: str):
    counts = labels.value_counts(normalize=True).round(4) * 100
    print(f"📊 Label distribution for {symbol}:")
    for action, pct in counts.items():
        print(f"   - {action}: {pct:.2f}%")

def prepare_training_data_with_resampling(ohlcv: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    features = extract_features(ohlcv)
    if features.empty:
        raise ValueError("No features extracted")

    # Align labels by feature index (timestamps)
    labels_aligned = labels.reindex(features.index)

    # Drop rows where labels or features have NaNs
    valid_mask = labels_aligned.notna() & features.notna().all(axis=1)
    features = features.loc[valid_mask]
    labels_aligned = labels_aligned.loc[valid_mask]

    if len(features) != len(labels_aligned):
        raise ValueError("Mismatch between features and labels length after alignment")

    features["action"] = labels_aligned.values

    X = features.drop(columns=["action"])
    y = features["action"]

    print(f"📦 Original class counts:\n{y.value_counts()}")

    # Oversample minority classes
    ros = RandomOverSampler(random_state=42)
    resample_result = ros.fit_resample(X, y)
    X_resampled, y_resampled = resample_result[0], resample_result[1]

    # Convert y_resampled to plain list for compatibility with pandas.Series
    y_resampled_series = pd.Series(list(y_resampled))

    print(f"✅ Resampled class counts:\n{y_resampled_series.value_counts()}")

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled["action"] = y_resampled_series.values

    return df_resampled

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    preds = model.predict(X_test)
    print("\n📈 Classification Report:")
    print(classification_report(y_test, preds))
    
    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    for symbol in SYMBOLS:
        try:
            print(f"\n=== 🧠 Training agent model for {symbol} ===")
            ohlcv = load_ohlcv(symbol)
            labels = load_labels(symbol)
            print_label_distribution(labels, symbol)

            df_balanced = prepare_training_data_with_resampling(ohlcv, labels)

            # Train/test split for evaluation
            from sklearn.model_selection import train_test_split
            X = df_balanced.drop(columns=["action"])
            y = df_balanced["action"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            # Train model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            model.fit(X_train, y_train)

            # Save model
            from joblib import dump
            model_path = os.path.join(MODEL_SAVE_DIR, f"{symbol.lower()}_model.pkl")
            dump(model, model_path)

            print(f"✅ Saved model to {model_path}")
            print(f"✅ Model classes: {model.classes_}")

            # Evaluate model
            evaluate_model(model, X_test, y_test)

        except Exception as e:
            print(f"❌ Failed to train model for {symbol}: {e}")

if __name__ == "__main__":
    main()
