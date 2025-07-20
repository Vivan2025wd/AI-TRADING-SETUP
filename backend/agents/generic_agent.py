import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.ml_engine.feature_extractor import extract_features
from backend.mother_ai.performance_tracker import PerformanceTracker
from typing import Optional, Dict


class GenericAgent:
    def __init__(self, symbol: str, strategy_logic: StrategyParser, model_path: Optional[str] = None):
        self.symbol = symbol
        self.strategy_logic = strategy_logic
        self.model_path = model_path or f"backend/agents/models/{symbol.lower()}_model.pkl"
        self.model = self._load_model()
        self.tracker = PerformanceTracker(log_dir_type="trade_history")
        self.position_state = None  # 'long' if in position, None if flat

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

        labeled_data = labeled_data[labeled_data["action"].isin(["buy", "sell"])]
        features = labeled_data.drop(columns=["action"])
        labels = labeled_data["action"]

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(features, labels)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        self.model = model
        print(f"✅ Trained and saved ML model to {self.model_path}")

    def _predict_with_model(self, features: pd.DataFrame) -> tuple[str, float]:
        if self.model is None or features.empty:
            return np.random.choice(["buy", "sell"]), 0.5

        try:
            latest_features = features.iloc[[-1]]
            proba = self.model.predict_proba(latest_features)[0]
            pred_class = self.model.classes_[np.argmax(proba)]
            confidence = float(np.max(proba))
            print(f"🔍 ML predicted: {pred_class} with confidence: {confidence:.4f}")
            return pred_class, confidence
        except Exception as e:
            print(f"❌ Model prediction failed: {e}")
            return np.random.choice(["buy", "sell"]), 0.5

    def _load_last_trade_signal(self) -> Optional[str]:
        path = f"backend/storage/performance_logs/{self.symbol}_trades.json"
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                trades = json.load(f)
                if trades:
                    return trades[-1].get("signal", None)
        except Exception as e:
            print(f"⚠️ Failed to load last trade signal: {e}")
        return None

    def evaluate(self, ohlcv_data: pd.DataFrame) -> Dict:
        if ohlcv_data.empty:
            raise ValueError(f"⚠️ OHLCV data for {self.symbol} is empty")

        ohlcv_data = ohlcv_data.sort_index()

        try:
            features = extract_features(ohlcv_data)
            action_ml, confidence_ml = self._predict_with_model(features)
        except Exception as e:
            print(f"❌ Feature extraction or ML failed: {e}")
            features = pd.DataFrame()
            action_ml, confidence_ml = "searching", 0.0

        try:
            rule_result = self.strategy_logic.evaluate(ohlcv_data)

            if isinstance(rule_result, str):
                rule_result = json.loads(rule_result)

            action_rule = rule_result.get("action", "searching")
            confidence_rule = rule_result.get("confidence", 0.0)
            print(f"🧠 Rule-based result: {rule_result}")
        except Exception as e: 
            print(f"❌ Strategy evaluation failed: {e}")
            action_rule, confidence_rule = "searching", 0.0

        # Decision Fusion
        if confidence_rule >= 0.9:
            final_action = action_rule
            final_confidence = confidence_rule
            source = "rule_based"
        else:
            final_action = action_ml
            final_confidence = confidence_ml
            source = "ml"

        # Enforce Position Rules
        last_signal = self._load_last_trade_signal()
        if last_signal == "buy" and final_action == "buy":
            print(f"⛔ Preventing consecutive buy for {self.symbol}")
            final_action = "hold"

        if self.position_state == "long":
            if final_action == "buy":
                final_action = "hold"
            elif final_action == "sell":
                self.position_state = None
        elif self.position_state is None:
            if final_action == "sell":
                final_action = "searching"
            elif final_action == "buy":
                self.position_state = "long"
            else:
                if final_confidence > 0.6:
                    final_action = "buy_soon"
                else:
                    final_action = "searching"

        timestamp = pd.to_datetime(ohlcv_data.index[-1]).isoformat()

        result = {
            "symbol": self.symbol,
            "action": final_action,
            "confidence": round(final_confidence, 4), 
            "timestamp": timestamp,
            "source": source,
            "ml": {
                "action": action_ml,
                "confidence": round(confidence_ml, 4)
            },
            "rule_based": {
                "action": action_rule,
                "confidence": round(confidence_rule, 4)
            }
        }

        self.tracker.log_trade(self.symbol, {
            "timestamp": timestamp,
            "symbol": self.symbol,
            "signal": final_action,
            "confidence": final_confidence,
            "source": f"GenericAgent/{source}"
        })

        return result


    def predict(self, ohlcv_data: pd.DataFrame) -> Dict:
        return self.evaluate(ohlcv_data)
