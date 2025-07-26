import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.ml_engine.feature_extractor import extract_features
from backend.mother_ai.performance_tracker import PerformanceTracker
from typing import Optional, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenericAgent:
    def __init__(self, symbol: str, strategy_logic: StrategyParser, model_path: Optional[str] = None):
        self.symbol = symbol
        self.strategy_logic = strategy_logic
        self.model_path = model_path or f"backend/agents/models/{symbol.lower()}_model.pkl"
        self.model = self._load_model()
        self.tracker = PerformanceTracker(log_dir_type="trade_history")
        self.position_state = self._load_position_state()
        self.model_metadata = self._load_model_metadata()
        
        # ML prediction settings
        self.ml_confidence_threshold = 0.6  # Minimum confidence for ML predictions
        self.feature_cache = None  # Cache for feature computation
        
    def _load_model(self):
        """Load ML model with proper error handling"""
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                logger.info(f"✅ Loading ML model from: {self.model_path}")
                
                # Validate model
                if hasattr(model, 'classes_') and hasattr(model, 'predict_proba'):
                    valid_classes = set(model.classes_)
                    expected_classes = {'buy', 'sell'}
                    
                    if not expected_classes.issubset(valid_classes):
                        logger.warning(f"⚠️ Model classes {valid_classes} don't match expected {expected_classes}")
                        
                    logger.info(f"🧠 Model loaded with classes: {list(model.classes_)}")
                    return model
                else:
                    logger.error(f"❌ Invalid model format at {self.model_path}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Failed to load model from {self.model_path}: {e}")
                return None
        else:
            logger.info(f"⚠️ No ML model found at {self.model_path}. Using rule-based logic only.")
            return None

    def _load_model_metadata(self) -> Dict:
        """Load model training metadata if available"""
        metadata_path = os.path.join(os.path.dirname(self.model_path), "training_summary.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    all_metadata = json.load(f)
                    return all_metadata.get(self.symbol, {})
            except Exception as e:
                logger.warning(f"Failed to load model metadata: {e}")
        
        return {}

    def train_model(self, labeled_data: pd.DataFrame):
        """Train model with improved validation and error handling"""
        if "action" not in labeled_data.columns:
            raise ValueError("Training data must have an 'action' column")

        # Filter to only buy/sell labels
        labeled_data = labeled_data[labeled_data["action"].isin(["buy", "sell"])]
        
        if len(labeled_data) == 0:
            raise ValueError("No valid buy/sell labels found in training data")
        
        # Check class distribution
        class_counts = labeled_data["action"].value_counts()
        logger.info(f"Training data distribution: {class_counts.to_dict()}")
        
        if len(class_counts) < 2:
            raise ValueError("Need at least 2 classes (buy and sell) for training")
        
        # Prepare features and labels
        features = labeled_data.drop(columns=["action"])
        labels = labeled_data["action"]
        
        # Validate features
        if features.empty or features.isnull().all().all():
            raise ValueError("Features are empty or all null")
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        
        try:
            model.fit(features, labels)
            
            # Validate trained model
            train_score = model.score(features, labels)
            logger.info(f"Training accuracy: {train_score:.4f}")
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(model, self.model_path)
            self.model = model
            
            logger.info(f"✅ Trained and saved ML model to {self.model_path}")
            logger.info(f"🧠 Model classes: {model.classes_}")
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise

    def _extract_features_safely(self, ohlcv_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Safely extract features with error handling and caching"""
        try:
            # Check if we can use cached features
            if (self.feature_cache is not None and 
                len(self.feature_cache) > 0 and
                self.feature_cache.index[-1] == ohlcv_data.index[-1]):
                return self.feature_cache
            
            # Extract new features
            features = extract_features(ohlcv_data)
            
            if features.empty:
                logger.warning(f"Feature extraction returned empty DataFrame for {self.symbol}")
                return None
            
            # Validate features
            if features.isnull().all().all():
                logger.warning(f"All features are null for {self.symbol}")
                return None
            
            # Handle missing values
            features = features.fillna(features.mean())
            
            # Cache features
            self.feature_cache = features
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed for {self.symbol}: {e}")
            return None

    def _predict_with_model(self, features: Optional[pd.DataFrame]) -> Tuple[str, float]:
        """Enhanced ML prediction with proper error handling"""
        if self.model is None:
            logger.debug(f"No ML model available for {self.symbol}")
            return "searching", 0.0
            
        if features is None or features.empty:
            logger.warning(f"No features available for ML prediction for {self.symbol}")
            return "searching", 0.0

        try:
            # Use the latest feature row
            latest_features = features.iloc[[-1]]
            
            # Check for missing or infinite values
            if latest_features.isnull().any().any() or np.isinf(latest_features).any().any():
                logger.warning(f"Invalid feature values detected for {self.symbol}")
                latest_features = latest_features.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Make prediction
            probabilities = self.model.predict_proba(latest_features)[0]
            predicted_class = self.model.classes_[np.argmax(probabilities)]
            confidence = float(np.max(probabilities))
            
            # Apply confidence threshold
            if confidence < self.ml_confidence_threshold:
                logger.debug(f"ML confidence {confidence:.3f} below threshold {self.ml_confidence_threshold}")
                return "searching", confidence
            
            logger.info(f"🔍 ML predicted: {predicted_class} with confidence: {confidence:.4f}")
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"❌ Model prediction failed for {self.symbol}: {e}")
            return "searching", 0.0

    def _get_rule_based_signal(self, ohlcv_data: pd.DataFrame) -> Tuple[str, float]:
        """Get rule-based signal with proper error handling and position validation"""
        try:
            rule_result = self.strategy_logic.evaluate(ohlcv_data)
            
            # Handle string result
            if isinstance(rule_result, str):
                rule_result = json.loads(rule_result)
            
            raw_action = rule_result.get("action", "searching")
            raw_confidence = float(rule_result.get("confidence", 0.0))
            
            # VALIDATION: Fix invalid rule signals based on position state
            validated_action, validated_confidence = self._validate_rule_signal(
                raw_action, raw_confidence
            )
            
            logger.info(f"🧠 Rule-based result: action={validated_action}, confidence={validated_confidence:.3f}")
            if validated_action != raw_action:
                logger.info(f"   ⚠️ Corrected from: {raw_action} (invalid for current position state)")
            
            return validated_action, validated_confidence
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse rule result JSON: {e}")
            return "searching", 0.0
        except Exception as e:
            logger.error(f"❌ Strategy evaluation failed for {self.symbol}: {e}")
            return "searching", 0.0

    def _validate_rule_signal(self, action: str, confidence: float) -> Tuple[str, float]:
        """Validate rule-based signals against current position state"""
        # If trying to sell without a position, convert to searching with low confidence
        if action == "sell" and self.position_state is None:
            logger.warning(f"Rule wants to SELL {self.symbol} but no position exists - converting to searching")
            return "searching", 0.0
        
        # If trying to buy when already long, convert to hold
        if action == "buy" and self.position_state == "long":
            logger.warning(f"Rule wants to BUY {self.symbol} but already long - converting to hold")
            return "hold", confidence * 0.5  # Reduce confidence for converted signal
        
        # Valid signals pass through unchanged
        return action, confidence

    def _fuse_decisions(self, ml_action: str, ml_confidence: float, 
                        rule_action: str, rule_confidence: float) -> Tuple[str, float, str]:
        """FIXED: Intelligent decision fusion with better conflict resolution"""
        
        # Case 1: If rule signal is invalid (searching/hold), prioritize ML
        if rule_action in ["searching", "hold"] and ml_action in ["buy", "sell"]:
            if ml_confidence >= self.ml_confidence_threshold:
                logger.info(f"Using ML signal over inactive rule: {ml_action} ({ml_confidence:.3f})")
                return ml_action, ml_confidence, "ml_primary"
        
        # Case 2: High confidence rule-based takes priority (but only for valid signals)
        if rule_confidence >= 0.9 and rule_action in ["buy", "sell"]:
            # Check for high-confidence conflicts
            if (ml_confidence >= 0.9 and ml_action in ["buy", "sell"] and 
                ml_action != rule_action):
                logger.warning(f"⚠️ HIGH CONFIDENCE CONFLICT: ML={ml_action}({ml_confidence:.3f}) vs Rule={rule_action}({rule_confidence:.3f})")
                
                # In high-confidence conflicts, prefer ML if it's more confident
                if ml_confidence > rule_confidence:
                    logger.info(f"Resolving conflict: Using ML {ml_action} ({ml_confidence:.3f} > {rule_confidence:.3f})")
                    return ml_action, ml_confidence, "ml_conflict_resolution"
                else:
                    logger.info(f"Resolving conflict: Using Rule {rule_action} ({rule_confidence:.3f} >= {ml_confidence:.3f})")
                    return rule_action, rule_confidence, "rule_conflict_resolution"
            else:
                logger.info(f"Using high-confidence rule signal: {rule_action} ({rule_confidence:.3f})")
                return rule_action, rule_confidence, "rule_based"
        
        # Case 3: If ML is available and confident
        if self.model is not None and ml_confidence >= self.ml_confidence_threshold:
            # If both agree, use higher confidence
            if ml_action == rule_action:
                final_confidence = max(ml_confidence, rule_confidence)
                logger.info(f"ML and rules agree: {ml_action} (confidence: {final_confidence:.3f})")
                return ml_action, final_confidence, "consensus"
            
            # If they disagree, use ML if significantly more confident
            elif ml_confidence > rule_confidence + 0.2:
                logger.info(f"Using confident ML signal over rule: {ml_action} ({ml_confidence:.3f})")
                return ml_action, ml_confidence, "ml"
            else:
                logger.info(f"Using rule signal in disagreement: {rule_action} ({rule_confidence:.3f})")
                return rule_action, rule_confidence, "rule_based"
        
        # Case 4: Fallback to rule-based (but validate it's actionable)
        if rule_action in ["buy", "sell"]:
            logger.info(f"Falling back to rule signal: {rule_action} ({rule_confidence:.3f})")
            return rule_action, rule_confidence, "rule_based"
        else:
            # If rule is not actionable and ML is not confident enough, default to searching
            logger.info(f"No confident signals available - defaulting to searching")
            return "searching", 0.0, "no_signal"

    def _load_last_trade_signal(self) -> Optional[str]:
        """Load the last trade signal from logs"""
        path = f"backend/storage/performance_logs/{self.symbol}_trades.json"
        if not os.path.exists(path):
            return None
            
        try:
            with open(path, "r") as f:
                trades = json.load(f)
                if trades and len(trades) > 0:
                    return trades[-1].get("signal", None)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load last trade signal for {self.symbol}: {e}")
        
        return None

    def _load_position_state(self) -> Optional[str]:
        """Load current position state from trade history"""
        path = f"backend/storage/performance_logs/{self.symbol}_trades.json"
        if not os.path.exists(path):
            return None
            
        try:
            with open(path, "r") as f:
                trades = json.load(f)
                if trades and len(trades) > 0:
                    last_signal = trades[-1].get("signal", None)
                    if last_signal == "buy":
                        logger.info(f"📦 Loaded position state for {self.symbol}: long")
                        return "long"
                    elif last_signal == "sell":
                        logger.info(f"📦 Loaded position state for {self.symbol}: flat")
                        return None
            
            logger.info(f"📦 No previous signals found for {self.symbol}")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to determine position state for {self.symbol}: {e}")
            return None

    def evaluate(self, ohlcv_data: pd.DataFrame) -> Dict:
        """Main evaluation method with improved error handling and logic"""
        if ohlcv_data.empty:
            raise ValueError(f"⚠️ OHLCV data for {self.symbol} is empty")

        # Ensure data is sorted chronologically
        ohlcv_data = ohlcv_data.sort_index()
        
        # Extract features for ML
        features = self._extract_features_safely(ohlcv_data)
        
        # Get ML prediction
        ml_action, ml_confidence = self._predict_with_model(features)
        
        # Get rule-based prediction (now with validation)
        rule_action, rule_confidence = self._get_rule_based_signal(ohlcv_data)
        
        # Fuse decisions (now with better conflict resolution)
        fused_action, fused_confidence, source = self._fuse_decisions(
            ml_action, ml_confidence, rule_action, rule_confidence
        )
        
        # Apply position management
        final_action = self._apply_position_management(fused_action)
        
        # Adjust confidence if action was modified by position management
        final_confidence = fused_confidence if final_action == fused_action else fused_confidence * 0.8
        
        # Create result
        timestamp = pd.to_datetime(ohlcv_data.index[-1]).isoformat()
        
        result = {
            "symbol": self.symbol,
            "action": final_action,
            "confidence": round(final_confidence, 4), 
            "timestamp": timestamp,
            "source": source,
            "position_state": self.position_state,
            "ml": {
                "action": ml_action,
                "confidence": round(ml_confidence, 4),
                "available": self.model is not None
            },
            "rule_based": {
                "action": rule_action,
                "confidence": round(rule_confidence, 4)
            },
            "metadata": {
                "features_extracted": features is not None,
                "model_classes": list(self.model.classes_) if self.model else None,
                "decision_flow": f"{ml_action}({ml_confidence:.2f}) + {rule_action}({rule_confidence:.2f}) -> {fused_action}({fused_confidence:.2f}) -> {final_action}"
            }
        }

        # Log the trade
        self.tracker.log_trade(self.symbol, {
            "timestamp": timestamp,
            "symbol": self.symbol,
            "signal": final_action,
            "confidence": final_confidence,
            "source": f"GenericAgent/{source}",
            "ml_available": self.model is not None,
            "position_state": self.position_state
        })

        return result

    def predict(self, ohlcv_data: pd.DataFrame) -> Dict:
        """Alias for evaluate method"""
        return self.evaluate(ohlcv_data)

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        info = {
            "symbol": self.symbol,
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "position_state": self.position_state
        }
        
        if self.model is not None:
            info.update({
                "model_type": type(self.model).__name__,
                "classes": list(self.model.classes_),
                "n_features": getattr(self.model, 'n_features_in_', None)
            })
        
        if self.model_metadata:
            info["training_metadata"] = self.model_metadata
            
        return info

    def reset_position_state(self):
        """Reset position state (useful for testing or recovery)"""
        logger.info(f"🔄 Resetting position state for {self.symbol}")
        self.position_state = None

    def set_ml_confidence_threshold(self, threshold: float):
        """Set the minimum confidence threshold for ML predictions"""
        if 0.0 <= threshold <= 1.0:
            self.ml_confidence_threshold = threshold
            logger.info(f"Updated ML confidence threshold for {self.symbol}: {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        

    def _apply_position_management(self, action: str) -> str:
        """Enhanced position management with better logic"""
    
        # REMOVED: Overly restrictive consecutive buy prevention
        # The original logic was preventing valid new positions
    
        # Position state management with better validation
        if self.position_state == "long":
            if action == "buy":
                # Instead of always converting to hold, check if this is really a consecutive buy
                last_signal = self._load_last_trade_signal()
                if last_signal == "buy":
                    # Only prevent if we JUST bought (same evaluation cycle)
                    last_trade_time = self._get_last_trade_time()
                    if last_trade_time and self._is_recent_trade(last_trade_time, minutes=5):
                        logger.info(f"⛔ Preventing consecutive buy within 5 minutes for {self.symbol}")
                        return "hold"
                    else:
                        logger.info(f"📥 Allowing buy signal - previous buy was not recent for {self.symbol}")
                        return action
                else:
                    logger.info(f"📥 Opening long position for {self.symbol}")
                    return action
            elif action == "sell":
                logger.info(f"📤 Closing long position for {self.symbol}")
                self.position_state = None
                return action
            
        elif self.position_state is None:
            if action == "sell":
                logger.info(f"⛔ No position to sell, converting to searching for {self.symbol}")
                return "searching"
            elif action == "buy":
                logger.info(f"📥 Opening long position for {self.symbol}")
                self.position_state = "long"
                return action
    
        return action

    def _get_last_trade_time(self) -> Optional[str]:
        """Get timestamp of last trade"""
        path = f"backend/storage/performance_logs/{self.symbol}_trades.json"
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "r") as f:
                trades = json.load(f)
                if trades and len(trades) > 0:
                    return trades[-1].get("timestamp", None)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load last trade time for {self.symbol}: {e}")
    
        return None

    def _is_recent_trade(self, timestamp_str: str, minutes: int = 5) -> bool:
        """Check if trade timestamp is within the last N minutes"""
        try:
            from datetime import datetime, timedelta
            import dateutil.parser
        
            trade_time = dateutil.parser.parse(timestamp_str)
            now = datetime.now(trade_time.tzinfo) if trade_time.tzinfo else datetime.now()
        
            return (now - trade_time) < timedelta(minutes=minutes)
        except Exception as e:
            logger.warning(f"Failed to parse trade time: {e}")
            return False

# Also add this method to force position state reset if needed
    def reset_position_state_if_invalid(self):
        """Reset position state if it doesn't match trade history"""
        try:
            path = f"backend/storage/performance_logs/{self.symbol}_trades.json"
            if not os.path.exists(path):
                self.position_state = None
                return
            
            with open(path, "r") as f:
                trades = json.load(f)
            
            if not trades:
                self.position_state = None
                return
            
            last_signal = trades[-1].get("signal", "")
            correct_state = "long" if last_signal == "buy" else None
        
            if self.position_state != correct_state:
                logger.info(f"🔧 Correcting {self.symbol} position state: {self.position_state} → {correct_state}")
                self.position_state = correct_state
            
        except Exception as e:
            logger.error(f"Failed to validate position state for {self.symbol}: {e}")