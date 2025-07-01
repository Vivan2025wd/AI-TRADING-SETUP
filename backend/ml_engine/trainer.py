# backend/ml_engine/trainer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model(csv_file_path: str, model_output_path: str):
    # Load OHLCV data
    df = pd.read_csv(csv_file_path)

    # Example feature engineering
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['label'] = (df['price_change'].shift(-1) > 0).astype(int)  # 1 if price goes up next candle
    df.dropna(inplace=True)

    features = ['open', 'high', 'low', 'close', 'volume', 'price_change']
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc:.2f}")

    joblib.dump(clf, model_output_path)
    print(f"Model saved to {model_output_path}")


