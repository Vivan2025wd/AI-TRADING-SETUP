import os
import joblib
import pandas as pd
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from backend.ml_engine.feature_extractor import extract_features


def load_or_train_model(
    symbol: str,
    raw_ohlcv: pd.DataFrame,
    label_column: str = "label",
    model_path: 'Optional[str]' = None,
    force_retrain: bool = False
) -> RandomForestClassifier:
    """
    Load a trained ML model from disk or train a new one using OHLCV data.

    Args:
        symbol (str): The trading symbol (e.g., 'BTCUSDT').
        raw_ohlcv (pd.DataFrame): Historical OHLCV data.
        label_column (str): The column to use as target (e.g., 'label').
        model_path (str): Optional path to save/load the model.
        force_retrain (bool): If True, retrain even if a model exists.

    Returns:
        RandomForestClassifier: A trained model instance.
    """
    if not model_path:
        model_path = f"backend/agents/models/{symbol.lower()}_model.pkl"

    # Check if model exists
    if os.path.exists(model_path) and not force_retrain:
        print(f"✅ Loaded existing model from: {model_path}")
        return joblib.load(model_path)

    print(f"⚙️ Training new model for {symbol}...")

    # Extract features
    df = extract_features(raw_ohlcv)
    if label_column not in df.columns:
        raise ValueError(f"Expected label column '{label_column}' in features")

    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("🧪 Model evaluation:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, model_path)
    print(f"💾 Saved model to: {model_path}")

    return model
