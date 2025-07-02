import pandas as pd
from datetime import datetime
from backend.strategy_engine.strategy_parser import StrategyParser # Added import

class GenericAgent:
    def __init__(self, symbol: str, strategy_logic: StrategyParser):
        self.symbol = symbol
        self.strategy_logic = strategy_logic # This is now an instance of StrategyParser

    def evaluate(self, ohlcv_data: pd.DataFrame):
        # self.strategy_logic is an instance of StrategyParser
        action = self.strategy_logic.evaluate(ohlcv_data) # Returns 'buy', 'sell', or 'hold'

        # Default confidence (0.0 to 1.0)
        # TODO: Future enhancement could be to have StrategyParser provide confidence
        confidence = 0.75

        timestamp = None
        if not ohlcv_data.empty and not ohlcv_data.index.empty:
            timestamp = pd.to_datetime(ohlcv_data.index[-1]).isoformat()
        else:
            timestamp = datetime.now().isoformat()

        return {
            "symbol": self.symbol,
            "action": action, # 'buy', 'sell', or 'hold'
            "confidence": confidence, # Range 0.0 - 1.0
            "timestamp": timestamp
        }

    def predict(self, ohlcv_data: pd.DataFrame):
        # Alias for evaluate
        return self.evaluate(ohlcv_data)
