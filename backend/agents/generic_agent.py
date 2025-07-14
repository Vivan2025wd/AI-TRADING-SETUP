import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.ml_engine.feature_extractor import extract_features
from backend.mother_ai.performance_tracker import PerformanceTracker  # Add tracker
from typing import Optional

class GenericAgent:
    def __init__(self, symbol: str, strategy_logic: StrategyParser, model_path: Optional[str] = None):
        self.symbol = symbol
        self.strategy_logic = strategy_logic
        self.model_path = model_path or f"backend/agents/models/{symbol.lower()}_model.pkl"
        self.model = self._load_model()

        # Initialize PerformanceTracker for saving predictions
        self.tracker = PerformanceTracker(log_dir_type="trade_history")

    def _load_model(self):
        if os.path.exists(self.model_path):
            print(f"✅ Loading ML model from: {self.model_path}")
            return joblib.load(self.model_path)
        else:
            print(f"⚠️ No ML model found at {self.model_path}. Using rule-based logic only.")
            return None

    def train_model(self, labeled_data: pd.DataFrame):
        if "action" not in labeled_data.columns:
            raise ValueError("Training data must have an 'action' column")

        features = labeled_data.drop(columns=["action"])
        labels = labeled_data["action"]

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight="balanced"  # Handle label imbalance
        )

        model.fit(features, labels)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        self.model = model
        print(f"✅ Trained and saved ML model to {self.model_path}")

    def _predict_with_model(self, features: pd.DataFrame) -> tuple[str, float]:
        if self.model is None or features.empty:
            return "hold", 0.0

        try:
            latest_features = features.iloc[[-1]]  # Predict on last row
            proba = self.model.predict_proba(latest_features)[0]
            pred_class = self.model.classes_[np.argmax(proba)]
            confidence = float(np.max(proba))
            print(f"🔍 ML predicted: {pred_class} with confidence: {confidence:.4f}")
            return pred_class, confidence
        except Exception as e:
            print(f"❌ Model prediction failed: {e}")
            return "hold", 0.0

    def evaluate(self, ohlcv_data: pd.DataFrame) -> dict:
        if ohlcv_data.empty:
            raise ValueError(f"⚠️ OHLCV data for {self.symbol} is empty")

        ohlcv_data = ohlcv_data.sort_index()

        try:
            features = extract_features(ohlcv_data)
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            features = pd.DataFrame()

        action_ml, confidence_ml = self._predict_with_model(features)

        confidence_threshold = 0.6  # You can tune this

        if action_ml == "hold" or confidence_ml < confidence_threshold:
            action_rb = self.strategy_logic.evaluate(ohlcv_data)
            action = action_rb
            confidence = 0.3  # Distinct low confidence for fallback
            print(f"⚙️ Rule-based fallback: {action_rb} (ML: {action_ml}, conf: {confidence_ml:.2f})")
        else:
            action = action_ml
            confidence = confidence_ml

        timestamp = pd.to_datetime(ohlcv_data.index[-1]).isoformat()

        prediction = {
            "symbol": self.symbol,
            "action": action,
            "confidence": round(confidence, 4),
            "timestamp": timestamp
        }

        # ✅ Log the prediction to trade_history
        self.tracker.log_trade(self.symbol, {
            "timestamp": timestamp,
            "symbol": self.symbol,
            "signal": action,
            "confidence": confidence,
            "source": "GenericAgent"
        })

        return prediction

    def predict(self, ohlcv_data: pd.DataFrame) -> dict:
        return self.evaluate(ohlcv_data)
