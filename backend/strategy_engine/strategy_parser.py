import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from backend.ml_engine.indicators import (
    calculate_rsi,
    calculate_ema,
    calculate_sma,
    calculate_macd
)

class StrategyParser:
    def __init__(self, strategy_json: dict):
        self.strategy = strategy_json
        self.symbol = strategy_json.get("symbol", "")
        self.indicators = strategy_json.get("indicators", {})
        self.logic = strategy_json.get("logic", "any")  # "any", "all", "weighted", "custom"
        self.weights = strategy_json.get("weights", {})  # For weighted logic
        self.min_confidence = strategy_json.get("min_confidence", 0.6)
        
    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced indicator application with validation"""
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
            
        # Ensure sufficient data for indicators
        min_periods = self._get_min_required_periods()
        if len(df) < min_periods:
            raise ValueError(f"Insufficient data: need at least {min_periods} periods")
        
        # Apply indicators (same as original but with validation)
        if "rsi" in self.indicators:
            period = self.indicators["rsi"].get("period", 14)
            df["rsi"] = calculate_rsi(df["close"], period)

        if "ema" in self.indicators:
            period = self.indicators["ema"].get("period", 20)
            df["ema"] = calculate_ema(df["close"], period)
            
            compare_period = self.indicators["ema"].get("compare_period")
            if compare_period and compare_period != period:
                df["ema_compare"] = calculate_ema(df["close"], compare_period)

        if "sma" in self.indicators:
            period = self.indicators["sma"].get("period", 20)
            df["sma"] = calculate_sma(df["close"], period)
            
            compare_period = self.indicators["sma"].get("compare_period")
            if compare_period and compare_period != period:
                df["sma_compare"] = calculate_sma(df["close"], compare_period)

        if "macd" in self.indicators:
            fast = self.indicators["macd"].get("fast_period", 12)
            slow = self.indicators["macd"].get("slow_period", 26)
            signal = self.indicators["macd"].get("signal_period", 9)
            macd_line, signal_line, histogram = calculate_macd(df["close"], fast, slow, signal)
            df["macd"] = macd_line
            df["macd_signal"] = signal_line
            df["macd_histogram"] = histogram

        return df
    
    def _get_min_required_periods(self) -> int:
        """Calculate minimum periods needed for all indicators"""
        min_periods = 1
        
        if "rsi" in self.indicators:
            min_periods = max(min_periods, self.indicators["rsi"].get("period", 14) + 1)
            
        if "ema" in self.indicators:
            period = self.indicators["ema"].get("period", 20)
            compare_period = self.indicators["ema"].get("compare_period", 0)
            min_periods = max(min_periods, max(period, compare_period) + 1)
            
        if "sma" in self.indicators:
            period = self.indicators["sma"].get("period", 20)
            compare_period = self.indicators["sma"].get("compare_period", 0)
            min_periods = max(min_periods, max(period, compare_period) + 1)
            
        if "macd" in self.indicators:
            slow = self.indicators["macd"].get("slow_period", 26)
            signal = self.indicators["macd"].get("signal_period", 9)
            min_periods = max(min_periods, slow + signal + 1)
            
        return min_periods

    def evaluate_conditions_with_confidence(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Enhanced evaluation with confidence scoring"""
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            indicator_signals = {}
            
            # Evaluate each indicator and store results with confidence
            if "rsi" in self.indicators:
                buy, sell, conf = self._evaluate_rsi_with_confidence(row, prev_row, self.indicators["rsi"])
                indicator_signals["rsi"] = {"buy": buy, "sell": sell, "confidence": conf}

            if "ema" in self.indicators:
                buy, sell, conf = self._evaluate_ema_with_confidence(row, prev_row, self.indicators["ema"])
                indicator_signals["ema"] = {"buy": buy, "sell": sell, "confidence": conf}

            if "sma" in self.indicators:
                buy, sell, conf = self._evaluate_sma_with_confidence(row, prev_row, self.indicators["sma"])
                indicator_signals["sma"] = {"buy": buy, "sell": sell, "confidence": conf}

            if "macd" in self.indicators:
                buy, sell, conf = self._evaluate_macd_with_confidence(row, prev_row, self.indicators["macd"])
                indicator_signals["macd"] = {"buy": buy, "sell": sell, "confidence": conf}

            # Combine signals based on strategy logic
            final_signal = self._combine_signals(indicator_signals)
            signals.append(final_signal)

        # Add initial hold signal
        signals.insert(0, {"action": "hold", "confidence": 0.0, "indicators": {}})
        return signals

    def _evaluate_rsi_with_confidence(self, row, prev_row, rsi_config) -> Tuple[bool, bool, float]:
        """RSI evaluation with confidence scoring"""
        rsi_val = row.get("rsi")
        if rsi_val is None:
            return False, False, 0.0
            
        buy_signals = []
        sell_signals = []
        confidence = 0.0
        
        for condition, value in rsi_config.items():
            if condition == "period":
                continue
                
            if condition == "buy_below":
                signal = rsi_val < value
                buy_signals.append(signal)
                if signal:
                    # Higher confidence when RSI is further from threshold
                    confidence = max(confidence, min(1.0, (value - rsi_val) / 10))
                    
            elif condition == "sell_above":
                signal = rsi_val > value
                sell_signals.append(signal)
                if signal:
                    confidence = max(confidence, min(1.0, (rsi_val - value) / 10))
                    
        return any(buy_signals), any(sell_signals), confidence

    def _evaluate_ema_with_confidence(self, row, prev_row, ema_config) -> Tuple[bool, bool, float]:
        """EMA evaluation with confidence scoring"""
        price_now = row.get("close")
        price_prev = prev_row.get("close")
        ema_now = row.get("ema")
        ema_prev = prev_row.get("ema")
        
        buy_signals = []
        sell_signals = []
        confidence = 0.0
        
        for condition, value in ema_config.items():
            if condition in ["period", "compare_period"]:
                continue
                
            if condition == "price_crosses_above" and value:
                if all([price_now, price_prev, ema_now, ema_prev]):
                    signal = price_prev <= ema_prev and price_now > ema_now
                    buy_signals.append(signal)
                    if signal:
                        # Confidence based on strength of crossover
                        crossover_strength = (price_now - ema_now) / ema_now
                        confidence = max(confidence, min(1.0, abs(crossover_strength) * 100))
                        
        return any(buy_signals), any(sell_signals), confidence

    def _evaluate_sma_with_confidence(self, row, prev_row, sma_config) -> Tuple[bool, bool, float]:
        """SMA evaluation with confidence scoring"""
        # Similar to EMA but for SMA - implementation would follow same pattern
        return False, False, 0.0  # Placeholder
    
    def _evaluate_macd_with_confidence(self, row, prev_row, macd_config) -> Tuple[bool, bool, float]:
        """MACD evaluation with confidence scoring"""
        macd_now = row.get("macd")
        signal_now = row.get("macd_signal")
        histogram_now = row.get("macd_histogram")
        
        buy_signals = []
        sell_signals = []
        confidence = 0.0
        
        for condition, value in macd_config.items():
            if condition in ["fast_period", "slow_period", "signal_period"]:
                continue
                
            if condition == "histogram_positive" and value:
                if histogram_now is not None:
                    signal = histogram_now > 0
                    buy_signals.append(signal)
                    if signal:
                        # Confidence based on histogram strength
                        confidence = max(confidence, min(1.0, abs(histogram_now) * 10))
                        
        return any(buy_signals), any(sell_signals), confidence

    def _combine_signals(self, indicator_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine indicator signals based on strategy logic"""
        if not indicator_signals:
            return {"action": "hold", "confidence": 0.0, "indicators": {}}
        
        if self.logic == "any":
            return self._combine_any_logic(indicator_signals)
        elif self.logic == "all":
            return self._combine_all_logic(indicator_signals)
        elif self.logic == "weighted":
            return self._combine_weighted_logic(indicator_signals)
        else:
            return self._combine_any_logic(indicator_signals)  # Default fallback
    
    def _combine_any_logic(self, indicator_signals: Dict) -> Dict[str, Any]:
        """Original 'any' logic with confidence"""
        buy_signals = [sig["buy"] for sig in indicator_signals.values()]
        sell_signals = [sig["sell"] for sig in indicator_signals.values()]
        
        if any(buy_signals):
            # Find highest confidence among buy signals
            buy_confidences = [sig["confidence"] for sig in indicator_signals.values() if sig["buy"]]
            confidence = max(buy_confidences) if buy_confidences else 0.0
            return {"action": "buy", "confidence": confidence, "indicators": indicator_signals}
        elif any(sell_signals):
            sell_confidences = [sig["confidence"] for sig in indicator_signals.values() if sig["sell"]]
            confidence = max(sell_confidences) if sell_confidences else 0.0
            return {"action": "sell", "confidence": confidence, "indicators": indicator_signals}
        else:
            return {"action": "hold", "confidence": 0.0, "indicators": indicator_signals}
    
    def _combine_all_logic(self, indicator_signals: Dict) -> Dict[str, Any]:
        """All indicators must agree"""
        buy_signals = [sig["buy"] for sig in indicator_signals.values()]
        sell_signals = [sig["sell"] for sig in indicator_signals.values()]
        
        if buy_signals and all(buy_signals):
            # Average confidence when all agree
            confidences = [sig["confidence"] for sig in indicator_signals.values()]
            confidence = np.mean(confidences)
            return {"action": "buy", "confidence": confidence, "indicators": indicator_signals}
        elif sell_signals and all(sell_signals):
            confidences = [sig["confidence"] for sig in indicator_signals.values()]
            confidence = np.mean(confidences)
            return {"action": "sell", "confidence": confidence, "indicators": indicator_signals}
        else:
            return {"action": "hold", "confidence": 0.0, "indicators": indicator_signals}
    
    def _combine_weighted_logic(self, indicator_signals: Dict) -> Dict[str, Any]:
        """Weighted combination of signals"""
        total_buy_weight = 0.0
        total_sell_weight = 0.0
        total_weight = 0.0
        
        for indicator, signals in indicator_signals.items():
            weight = self.weights.get(indicator, 1.0)
            total_weight += weight
            
            if signals["buy"]:
                total_buy_weight += weight * signals["confidence"]
            elif signals["sell"]:
                total_sell_weight += weight * signals["confidence"]
        
        if total_weight == 0:
            return {"action": "hold", "confidence": 0.0, "indicators": indicator_signals}
        
        buy_score = total_buy_weight / total_weight
        sell_score = total_sell_weight / total_weight
        
        if buy_score > self.min_confidence and buy_score > sell_score:
            return {"action": "buy", "confidence": buy_score, "indicators": indicator_signals}
        elif sell_score > self.min_confidence and sell_score > buy_score:
            return {"action": "sell", "confidence": sell_score, "indicators": indicator_signals}
        else:
            return {"action": "hold", "confidence": 0.0, "indicators": indicator_signals}

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced evaluation with detailed confidence and reasoning"""
        df = self.apply_indicators(df)
        signals = self.evaluate_conditions_with_confidence(df)
        latest_signal = signals[-1] if signals else {"action": "hold", "confidence": 0.0}
        
        return {
            "action": latest_signal["action"],
            "confidence": latest_signal["confidence"],
            "indicators": latest_signal.get("indicators", {}),
            "strategy_logic": self.logic,
            "min_confidence": self.min_confidence
        }