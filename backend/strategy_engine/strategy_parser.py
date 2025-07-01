# strategy_engine/strategy_parser.py

import pandas as pd
from backend.ml_engine.indicators import calculate_rsi, calculate_ema

class StrategyParser:
    def __init__(self, strategy_json):
        self.strategy = strategy_json
        self.symbol = strategy_json.get("symbol", "")
        self.indicators = strategy_json.get("indicators", {})

    def apply_indicators(self, df):
        """
        Adds indicator columns (e.g., RSI, EMA) to the OHLCV dataframe based on strategy
        """
        if "rsi" in self.indicators:
            period = self.indicators["rsi"].get("period", 14)
            df["rsi"] = calculate_rsi(df["close"], period)

        if "ema" in self.indicators:
            period = self.indicators["ema"].get("period", 20)
            df["ema"] = calculate_ema(df["close"], period)

        return df

    def evaluate_conditions(self, df):
        """
        Evaluate buy/sell/hold logic row by row.
        Returns a list of signals: ["buy", "hold", "sell", ...]
        """
        signals = []

        for _, row in df.iterrows():
            buy_signal = False
            sell_signal = False

            # RSI logic
            if "rsi" in self.indicators:
                rsi_val = row.get("rsi", None)
                rsi_conf = self.indicators["rsi"]
                if rsi_val:
                    if rsi_val < rsi_conf.get("buy_below", 30):
                        buy_signal = True
                    elif rsi_val > rsi_conf.get("sell_above", 70):
                        sell_signal = True

            # EMA logic
            if "ema" in self.indicators:
                price = row.get("close", None)
                ema_val = row.get("ema", None)
                ema_conf = self.indicators["ema"]
                if price and ema_val:
                    if price > ema_conf.get("buy_crosses_above", ema_val):
                        buy_signal = True
                    elif price < ema_conf.get("sell_crosses_below", ema_val):
                        sell_signal = True

            # Combine conditions
            if buy_signal and not sell_signal:
                signals.append("buy")
            elif sell_signal and not buy_signal:
                signals.append("sell")
            else:
                signals.append("hold")

    @staticmethod
    def parse(strategy_json: dict) -> "StrategyParser":
        """
        Static method to initialize the parser from a strategy dict.
        """
        return StrategyParser(strategy_json)

    def evaluate(self, df: pd.DataFrame) -> str:
        """
        Evaluates the most recent row and returns one of: 'buy', 'sell', 'hold'
        """
        df = self.apply_indicators(df)
        signals = self.evaluate_conditions(df)
        return signals[-1] if signals else "hold"


        return signals
