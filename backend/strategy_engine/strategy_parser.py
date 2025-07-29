import pandas as pd
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

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds indicator columns (RSI, EMA, SMA, MACD) to the OHLCV dataframe based on strategy.
        Assumes df has 'close' price column.
        """
        if "rsi" in self.indicators:
            period = self.indicators["rsi"].get("period", 14)
            df["rsi"] = calculate_rsi(df["close"], period)

        if "ema" in self.indicators:
            period = self.indicators["ema"].get("period", 20)
            df["ema"] = calculate_ema(df["close"], period)
            
            # Handle compare_period for EMA crossovers
            compare_period = self.indicators["ema"].get("compare_period")
            if compare_period and compare_period != period:
                df["ema_compare"] = calculate_ema(df["close"], compare_period)

        if "sma" in self.indicators:
            period = self.indicators["sma"].get("period", 20)
            df["sma"] = calculate_sma(df["close"], period)
            
            # Handle compare_period for SMA crossovers
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

    def evaluate_rsi_conditions(self, row, prev_row, rsi_config):
        """Evaluate all RSI conditions"""
        rsi_val = row.get("rsi")
        if rsi_val is None:
            return False, False
            
        buy_signals = []
        sell_signals = []
        
        # Check all possible RSI conditions
        for condition, value in rsi_config.items():
            if condition == "period":
                continue
                
            if condition == "buy_below":
                buy_signals.append(rsi_val < value)
            elif condition == "buy_equals":
                buy_signals.append(abs(rsi_val - value) < 0.01)
            elif condition == "sell_above":
                sell_signals.append(rsi_val > value)
            elif condition == "sell_equals":
                sell_signals.append(abs(rsi_val - value) < 0.01)
                
        return any(buy_signals), any(sell_signals)

    def evaluate_ema_conditions(self, row, prev_row, ema_config):
        """Evaluate all EMA conditions"""
        price_now = row.get("close")
        price_prev = prev_row.get("close")
        ema_now = row.get("ema")
        ema_prev = prev_row.get("ema")
        ema_compare_now = row.get("ema_compare")
        ema_compare_prev = prev_row.get("ema_compare")
        
        buy_signals = []
        sell_signals = []
        
        for condition, value in ema_config.items():
            if condition in ["period", "compare_period"]:
                continue
                
            if condition == "price_crosses_above" and value:
                if all([price_now, price_prev, ema_now, ema_prev]):
                    buy_signals.append(price_prev <= ema_prev and price_now > ema_now)
                    
            elif condition == "price_crosses_below" and value:
                if all([price_now, price_prev, ema_now, ema_prev]):
                    sell_signals.append(price_prev >= ema_prev and price_now < ema_now)
                    
            elif condition == "price_above" and value:
                if price_now and ema_now:
                    buy_signals.append(price_now > ema_now)
                    
            elif condition == "price_below" and value:
                if price_now and ema_now:
                    sell_signals.append(price_now < ema_now)
                    
            elif condition == "ema_crosses_above_ema" and value:
                if all([ema_now, ema_prev, ema_compare_now, ema_compare_prev]):
                    buy_signals.append(ema_prev <= ema_compare_prev and ema_now > ema_compare_now)
                    
            elif condition == "ema_crosses_below_ema" and value:
                if all([ema_now, ema_prev, ema_compare_now, ema_compare_prev]):
                    sell_signals.append(ema_prev >= ema_compare_prev and ema_now < ema_compare_now)
                    
            elif condition == "ema_above_value":
                if ema_now:
                    buy_signals.append(ema_now > value)
                    
            elif condition == "ema_below_value":
                if ema_now:
                    sell_signals.append(ema_now < value)
                    
        return any(buy_signals), any(sell_signals)

    def evaluate_sma_conditions(self, row, prev_row, sma_config):
        """Evaluate all SMA conditions"""
        price_now = row.get("close")
        price_prev = prev_row.get("close")
        sma_now = row.get("sma")
        sma_prev = prev_row.get("sma")
        sma_compare_now = row.get("sma_compare")
        sma_compare_prev = prev_row.get("sma_compare")
        
        buy_signals = []
        sell_signals = []
        
        for condition, value in sma_config.items():
            if condition in ["period", "compare_period"]:
                continue
                
            if condition == "price_crosses_above" and value:
                if all([price_now, price_prev, sma_now, sma_prev]):
                    buy_signals.append(price_prev <= sma_prev and price_now > sma_now)
                    
            elif condition == "price_crosses_below" and value:
                if all([price_now, price_prev, sma_now, sma_prev]):
                    sell_signals.append(price_prev >= sma_prev and price_now < sma_now)
                    
            elif condition == "price_above" and value:
                if price_now and sma_now:
                    buy_signals.append(price_now > sma_now)
                    
            elif condition == "price_below" and value:
                if price_now and sma_now:
                    sell_signals.append(price_now < sma_now)
                    
            elif condition == "sma_crosses_above_sma" and value:
                if all([sma_now, sma_prev, sma_compare_now, sma_compare_prev]):
                    buy_signals.append(sma_prev <= sma_compare_prev and sma_now > sma_compare_now)
                    
            elif condition == "sma_crosses_below_sma" and value:
                if all([sma_now, sma_prev, sma_compare_now, sma_compare_prev]):
                    sell_signals.append(sma_prev >= sma_compare_prev and sma_now < sma_compare_now)
                    
            elif condition == "sma_above_value":
                if sma_now:
                    buy_signals.append(sma_now > value)
                    
            elif condition == "sma_below_value":
                if sma_now:
                    sell_signals.append(sma_now < value)
                    
        return any(buy_signals), any(sell_signals)

    def evaluate_macd_conditions(self, row, prev_row, macd_config):
        """Evaluate all MACD conditions"""
        macd_now = row.get("macd")
        macd_prev = prev_row.get("macd")
        signal_now = row.get("macd_signal")
        signal_prev = prev_row.get("macd_signal")
        histogram_now = row.get("macd_histogram")
        
        buy_signals = []
        sell_signals = []
        
        for condition, value in macd_config.items():
            if condition in ["fast_period", "slow_period", "signal_period"]:
                continue
                
            if condition == "macd_crosses_above_signal" and value:
                if all([macd_now, macd_prev, signal_now, signal_prev]):
                    buy_signals.append(macd_prev <= signal_prev and macd_now > signal_now)
                    
            elif condition == "macd_crosses_below_signal" and value:
                if all([macd_now, macd_prev, signal_now, signal_prev]):
                    sell_signals.append(macd_prev >= signal_prev and macd_now < signal_now)
                    
            elif condition == "macd_above_zero" and value:
                if macd_now is not None:
                    buy_signals.append(macd_now > 0)
                    
            elif condition == "macd_below_zero" and value:
                if macd_now is not None:
                    sell_signals.append(macd_now < 0)
                    
            elif condition == "histogram_positive" and value:
                if histogram_now is not None:
                    buy_signals.append(histogram_now > 0)
                    
            elif condition == "histogram_negative" and value:
                if histogram_now is not None:
                    sell_signals.append(histogram_now < 0)
                    
            elif condition == "macd_above_value":
                if macd_now is not None:
                    buy_signals.append(macd_now > value)
                    
            elif condition == "macd_below_value":
                if macd_now is not None:
                    sell_signals.append(macd_now < value)
                    
        return any(buy_signals), any(sell_signals)

    def evaluate_conditions(self, df: pd.DataFrame) -> list[str]:
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            buy_signals = []
            sell_signals = []

            # Evaluate each indicator's conditions
            if "rsi" in self.indicators:
                rsi_buy, rsi_sell = self.evaluate_rsi_conditions(row, prev_row, self.indicators["rsi"])
                buy_signals.append(rsi_buy)
                sell_signals.append(rsi_sell)

            if "ema" in self.indicators:
                ema_buy, ema_sell = self.evaluate_ema_conditions(row, prev_row, self.indicators["ema"])
                buy_signals.append(ema_buy)
                sell_signals.append(ema_sell)

            if "sma" in self.indicators:
                sma_buy, sma_sell = self.evaluate_sma_conditions(row, prev_row, self.indicators["sma"])
                buy_signals.append(sma_buy)
                sell_signals.append(sma_sell)

            if "macd" in self.indicators:
                macd_buy, macd_sell = self.evaluate_macd_conditions(row, prev_row, self.indicators["macd"])
                buy_signals.append(macd_buy)
                sell_signals.append(macd_sell)

            # Combine signals: buy if any indicator signals buy, sell if any sell, else hold
            if any(buy_signals):
                signals.append("buy")
            elif any(sell_signals):
                signals.append("sell")
            else:
                signals.append("hold")

        # pad first row with "hold" since we skip index 0
        signals.insert(0, "hold")
        return signals

    @staticmethod
    def parse(strategy_json: dict) -> "StrategyParser":
        """
        Static factory method to create an instance from a strategy dict.
        """
        return StrategyParser(strategy_json)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluates the latest row and returns a dict:
        {
            "action": "buy" | "sell" | "hold",
            "confidence": float (currently defaulted to 1.0)
        }
        """
        df = self.apply_indicators(df)
        signals = self.evaluate_conditions(df)
        action = signals[-1] if signals else "hold"

        # Optional: in future, use more complex logic to assign real confidence values
        confidence = 1.0 if action in ["buy", "sell"] else 0.0

        return {
            "action": action,
            "confidence": confidence
        }