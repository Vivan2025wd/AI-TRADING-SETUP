import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging

LABELS_DIR = "data/labels"
MODEL_DIR = "backend/agents/models"
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOLS = [
    "DOGEUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT",
    "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"
]

def load_feature_data(symbol: str, method: str = "outcome") -> pd.DataFrame:
    path = os.path.join(LABELS_DIR, f"{symbol}_{method}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found for {symbol}")
    df = pd.read_csv(path, index_col=0)
    df = df.dropna()
    if "label" not in df.columns:
        raise ValueError(f"No 'label' column in {path}")
    return df

def train_and_save_model(symbol: str, df: pd.DataFrame):
    X = df.drop(columns=["label"])
    y = df["label"]

    # Convert labels to integers
    y_encoded = y.astype("category").cat.codes
    label_mapping = dict(enumerate(y.astype("category").cat.categories))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=list(label_mapping.values()), zero_division=0)

    logger.info(f"📊 {symbol} Accuracy: {acc:.3f}")
    logger.info(f"🔎 Classification report:\n{report}")

    # Save model and label mapping
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    joblib.dump({"model": clf, "label_mapping": label_mapping}, model_path)
    logger.info(f"✅ Saved model for {symbol} to {model_path}")
