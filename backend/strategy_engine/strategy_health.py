# backend/strategy_engine/strategy_health.py

import numpy as np
from typing import List, Dict, Any

class StrategyHealth:
    def __init__(self, performance_log: List[Dict[str, Any]]):
        """
        Initializes with a list of trade dictionaries.
        Each trade dict should include keys like 'result' ('win'/'loss') and 'roi' (float).
        """
        self.performance_log = performance_log

    def win_rate(self, lookback: int = 20) -> float:
        """
        Calculate the win rate over the last `lookback` trades.
        Win rate = number of wins / total trades.
        """
        trades = self.performance_log[-lookback:]
        if not trades:
            return 0.0
        wins = sum(1 for trade in trades if trade.get("result") == "win")
        return wins / len(trades)

    def avg_profit(self, lookback: int = 20) -> float:
        """
        Calculate average ROI (return on investment) over the last `lookback` trades.
        """
        trades = self.performance_log[-lookback:]
        if not trades:
            return 0.0
        profits = [trade.get("roi", 0.0) for trade in trades if "roi" in trade]
        return float(np.mean(profits)) if profits else 0.0

    def recent_drawdown(self, lookback: int = 20) -> float:
        """
        Calculate the maximum drawdown (largest absolute loss) in the last `lookback` trades.
        """
        trades = self.performance_log[-lookback:]
        losses = [abs(trade.get("roi", 0.0)) for trade in trades if trade.get("result") == "loss"]
        return max(losses) if losses else 0.0

    def summary(self, lookback: int = 20) -> Dict[str, float]:
        """
        Returns a summary dict with win_rate, avg_profit, max_drawdown,
        and number of trades analyzed.
        """
        recent_trades = self.performance_log[-lookback:]
        return {
            "win_rate": round(self.win_rate(lookback), 3),
            "avg_profit": round(self.avg_profit(lookback), 3),
            "max_drawdown": round(self.recent_drawdown(lookback), 3),
            "trades_analyzed": len(recent_trades)
        }
