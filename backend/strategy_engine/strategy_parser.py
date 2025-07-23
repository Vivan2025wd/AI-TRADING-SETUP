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

        if "sma" in self.indicators:
            period = self.indicators["sma"].get("period", 20)
            df["sma"] = calculate_sma(df["close"], period)

        if "macd" in self.indicators:
            fast = self.indicators["macd"].get("fast_period", 12)
            slow = self.indicators["macd"].get("slow_period", 26)
            signal = self.indicators["macd"].get("signal_period", 9)
            macd_line, signal_line, _ = calculate_macd(df["close"], fast, slow, signal)
            df["macd"] = macd_line
            df["macd_signal"] = signal_line

        return df

    def evaluate_conditions(self, df: pd.DataFrame) -> list[str]:
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # RSI logic
            rsi_buy = rsi_sell = False
            if "rsi" in self.indicators:
                rsi_val = row.get("rsi")
                rsi_conf = self.indicators["rsi"]
                if rsi_val is not None:
                    rsi_buy = rsi_val < rsi_conf.get("buy_below", 30)
                    rsi_sell = rsi_val > rsi_conf.get("sell_above", 70)

            # EMA crossover logic
            ema_buy = ema_sell = False
            if "ema" in self.indicators:
                price_now = row.get("close")
                price_prev = prev_row.get("close")
                ema_now = row.get("ema")
                ema_prev = prev_row.get("ema")
                if price_now and ema_now and price_prev and ema_prev:
                    ema_buy = (
                        self.indicators["ema"].get("buy_crosses_above", False)
                        and price_prev < ema_prev and price_now > ema_now
                    )
                    ema_sell = (
                        self.indicators["ema"].get("sell_crosses_below", False)
                        and price_prev > ema_prev and price_now < ema_now
                    )

            # SMA crossover logic
            sma_buy = sma_sell = False
            if "sma" in self.indicators:
                price_now = row.get("close")
                price_prev = prev_row.get("close")
                sma_now = row.get("sma")
                sma_prev = prev_row.get("sma")
                if price_now and sma_now and price_prev and sma_prev:
                    sma_buy = price_prev < sma_prev and price_now > sma_now
                    sma_sell = price_prev > sma_prev and price_now < sma_now

            # MACD crossover logic
            macd_buy = macd_sell = False
            if "macd" in self.indicators:
                macd_now = row.get("macd")
                macd_prev = prev_row.get("macd")
                signal_now = row.get("macd_signal")
                signal_prev = prev_row.get("macd_signal")
                if macd_now is not None and macd_prev is not None and signal_now is not None and signal_prev is not None:
                    macd_buy = macd_prev < signal_prev and macd_now > signal_now
                    macd_sell = macd_prev > signal_prev and macd_now < signal_now

            # Combine signals: simple logic — buy if any indicator signals buy, sell if any sell, else hold
            if any([rsi_buy, ema_buy, sma_buy, macd_buy]):
                signals.append("buy")
            elif any([rsi_sell, ema_sell, sma_sell, macd_sell]):
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
