import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.ml_engine.feature_extractor import extract_features
from backend.mother_ai.performance_tracker import PerformanceTracker
from typing import Optional


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
            random_action = np.random.choice(["buy", "sell"])
            return random_action, 0.5

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

    def _combine_signals(self, ml_action: str, ml_confidence: float, rule_signal: str) -> tuple[str, float]:
        """
        Combine ML and rule-based signals with confidence weighting.
        """
        # Strong agreement cases
        if ml_action == rule_signal:
            if ml_action in ["buy", "sell"]:
                # Both agree on buy/sell - boost confidence
                boosted_confidence = min(ml_confidence + 0.2, 1.0)
                print(f"🤝 Strong agreement: ML={ml_action}, Rules={rule_signal}, Confidence boosted to {boosted_confidence:.4f}")
                return ml_action, boosted_confidence
            else:
                # Both neutral - maintain original confidence
                return ml_action, ml_confidence
        
        # Conflict resolution
        if ml_action in ["buy", "sell"] and rule_signal in ["buy", "sell"]:
            # Both have strong opinions but disagree - be cautious
            if ml_confidence > 0.7:
                # High ML confidence wins but reduce confidence
                final_confidence = ml_confidence * 0.8
                print(f"⚠️ Signal conflict: ML={ml_action}({ml_confidence:.4f}) vs Rules={rule_signal}, ML wins with reduced confidence")
                return ml_action, final_confidence
            else:
                # Low ML confidence, default to hold/search
                print(f"⚠️ Signal conflict with low ML confidence: ML={ml_action}({ml_confidence:.4f}) vs Rules={rule_signal}, defaulting to hold")
                return "hold", 0.3
        
        # One signal is neutral, other is actionable
        if rule_signal in ["buy", "sell"] and ml_action == "hold":
            # Rules want action, ML is neutral - use rules but lower confidence
            final_confidence = min(ml_confidence + 0.1, 0.6)
            print(f"📋 Rules signal: {rule_signal}, ML neutral, using rules with confidence {final_confidence:.4f}")
            return rule_signal, final_confidence
        
        if ml_action in ["buy", "sell"] and rule_signal == "hold":
            # ML wants action, rules are neutral - use ML
            print(f"🤖 ML signal: {ml_action}, Rules neutral, using ML")
            return ml_action, ml_confidence
        
        # Default case - both neutral or unknown combination
        return ml_action, ml_confidence

    def _apply_position_logic(self, action: str, confidence: float) -> str:
        """
        Apply position state management logic.
        """
        original_action = action
        
        if self.position_state == "long":
            if action == "buy":
                action = "hold"
            elif action == "sell":
                self.position_state = None
                print(f"📤 Closing long position: {original_action} -> {action}")
        
        elif self.position_state is None:
            if action == "sell":
                action = "searching"
                print(f"🔍 No position to sell: {original_action} -> {action}")
            elif action == "buy":
                self.position_state = "long"
                print(f"📥 Opening long position: {original_action} -> {action}")
            elif action == "hold":
                if confidence > 0.6:
                    action = "buy_soon"
                    print(f"⏳ High confidence but no clear signal: {original_action} -> {action}")
                else:
                    action = "searching"
                    print(f"🔍 Low confidence: {original_action} -> {action}")
        
        return action

    def evaluate(self, ohlcv_data: pd.DataFrame) -> dict:
        if ohlcv_data.empty:
            raise ValueError(f"⚠️ OHLCV data for {self.symbol} is empty")

        ohlcv_data = ohlcv_data.sort_index()

        try:
            features = extract_features(ohlcv_data)
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            features = pd.DataFrame()

        # Get ML prediction
        action_ml, confidence_ml = self._predict_with_model(features)
        
        # Get rule-based signal from strategy parser
        rule_signal = self.strategy_logic.evaluate(ohlcv_data)
        
        # Combine ML and rule-based signals
        final_action, final_confidence = self._combine_signals(action_ml, confidence_ml, rule_signal)
        
        # Apply position state management
        final_action = self._apply_position_logic(final_action, final_confidence)

        timestamp = pd.to_datetime(ohlcv_data.index[-1]).isoformat()

        prediction = {
            "symbol": self.symbol,
            "action": final_action,
            "confidence": round(final_confidence, 4),
            "timestamp": timestamp,
            "ml_signal": action_ml,
            "rule_signal": rule_signal,
            "ml_confidence": round(confidence_ml, 4)
        }

        self.tracker.log_trade(self.symbol, {
            "timestamp": timestamp,
            "symbol": self.symbol,
            "signal": final_action,
            "confidence": final_confidence,
            "ml_signal": action_ml,
            "rule_signal": rule_signal,
            "source": "GenericAgent"
        })

        return prediction

    def predict(self, ohlcv_data: pd.DataFrame) -> dict:
        return self.evaluate(ohlcv_data)