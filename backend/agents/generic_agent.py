import pandas as pd
from datetime import datetime
from backend.strategy_engine.strategy_parser import StrategyParser

class GenericAgent:
    def __init__(self, symbol: str, strategy_logic: StrategyParser):
        """
        Initialize the agent with a symbol and its parsed strategy logic.

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            strategy_logic (StrategyParser): An instance of StrategyParser containing the strategy logic.
        """
        self.symbol = symbol
        self.strategy_logic = strategy_logic

    def evaluate(self, ohlcv_data: pd.DataFrame) -> dict:
        """
        Evaluate the current OHLCV data using the strategy logic.

        Args:
            ohlcv_data (pd.DataFrame): DataFrame containing OHLCV data with a datetime index.

        Returns:
            dict: A dictionary containing symbol, action, confidence, and timestamp.
                Action is one of 'buy', 'sell', or 'hold'.
        """
        if ohlcv_data.empty:
            raise ValueError(f"OHLCV data for {self.symbol} is empty")

        # Make sure the DataFrame is sorted by datetime ascending
        ohlcv_data = ohlcv_data.sort_index()

        # Evaluate strategy
        action = self.strategy_logic.evaluate(ohlcv_data)
        if action not in {"buy", "sell", "hold"}:
            action = "hold"  # fallback to 'hold' on invalid action

        # Confidence is a placeholder; customize if your StrategyParser can return confidence
        confidence = 0.75

        # Use the timestamp of the latest candle in ISO format
        timestamp = pd.to_datetime(ohlcv_data.index[-1]).isoformat()

        return {
            "symbol": self.symbol,
            "action": action,
            "confidence": confidence,
            "timestamp": timestamp
        }

    def predict(self, ohlcv_data: pd.DataFrame) -> dict:
        """
        Alias for evaluate; useful for clarity when the agent predicts the next move.

        Args:
            ohlcv_data (pd.DataFrame): OHLCV data.

        Returns:
            dict: Same as evaluate.
        """
        return self.evaluate(ohlcv_data)
