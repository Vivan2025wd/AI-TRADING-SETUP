import pandas as pd
from backend.ml_engine.indicators import calculate_rsi, calculate_ema

class StrategyParser:
    def __init__(self, strategy_json: dict):
        self.strategy = strategy_json
        self.symbol = strategy_json.get("symbol", "")
        self.indicators = strategy_json.get("indicators", {})

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds indicator columns (e.g., RSI, EMA) to the OHLCV dataframe based on strategy.
        Assumes df has 'close' price column.
        """
        if "rsi" in self.indicators:
            period = self.indicators["rsi"].get("period", 14)
            df["rsi"] = calculate_rsi(df["close"], period)

        if "ema" in self.indicators:
            period = self.indicators["ema"].get("period", 20)
            df["ema"] = calculate_ema(df["close"], period)

        return df

    def evaluate_conditions(self, df: pd.DataFrame) -> list[str]:
        """
        Evaluate buy/sell/hold logic for each row in the dataframe.
        Returns a list of signals (strings): ["buy", "hold", "sell", ...]
        """
        signals = []

        for _, row in df.iterrows():
            buy_signal = False
            sell_signal = False

            # RSI logic
            if "rsi" in self.indicators:
                rsi_val = row.get("rsi", None)
                rsi_conf = self.indicators["rsi"]
                if rsi_val is not None:
                    buy_below = rsi_conf.get("buy_below", 30)
                    sell_above = rsi_conf.get("sell_above", 70)
                    if rsi_val < buy_below:
                        buy_signal = True
                    elif rsi_val > sell_above:
                        sell_signal = True

            # EMA logic
            if "ema" in self.indicators:
                price = row.get("close", None)
                ema_val = row.get("ema", None)
                ema_conf = self.indicators["ema"]
                if price is not None and ema_val is not None:
                    # Buy when price crosses above EMA if buy_crosses_above is True
                    if ema_conf.get("buy_crosses_above", False) and price > ema_val:
                        buy_signal = True
                    # Sell when price crosses below EMA if sell_crosses_below is True
                    elif ema_conf.get("sell_crosses_below", False) and price < ema_val:
                        sell_signal = True

            # Combine conditions to determine final signal
            if buy_signal and not sell_signal:
                signals.append("buy")
            elif sell_signal and not buy_signal:
                signals.append("sell")
            else:
                signals.append("hold")

        return signals

    @staticmethod
    def parse(strategy_json: dict) -> "StrategyParser":
        """
        Static factory method to create an instance from a strategy dict.
        """
        return StrategyParser(strategy_json)

    def evaluate(self, df: pd.DataFrame) -> str:
        """
        Evaluates the latest row and returns one of: 'buy', 'sell', 'hold'.
        """
        df = self.apply_indicators(df)
        signals = self.evaluate_conditions(df)
        return signals[-1] if signals else "hold"
