class GenericAgent:
    def __init__(self, symbol, strategy_logic):
        self.symbol = symbol
        self.strategy_logic = strategy_logic  # Should be a parsed strategy object with run()

    def evaluate(self, ohlcv_data):
        # Run strategy's run() method on OHLCV data to get decision
        result = self.strategy_logic.run(ohlcv_data)
        return {
            "symbol": self.symbol,
            "action": result.get("action", "hold"),  # 'buy' / 'sell' / 'hold'
            "confidence": result.get("confidence", 0.5),
            "timestamp": result.get("timestamp")
        }

    def predict(self, ohlcv_data):
        # Alias for evaluate for easier route integration
        return self.evaluate(ohlcv_data)
