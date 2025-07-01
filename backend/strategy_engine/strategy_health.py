# backend/strategy_engine/strategy_health.py

import numpy as np
from datetime import datetime, timedelta

class StrategyHealth:
    def __init__(self, performance_log):
        self.performance_log = performance_log  # List of trade dicts

    def win_rate(self, lookback=20):
        trades = self.performance_log[-lookback:]
        if not trades:
            return 0.0
        wins = sum(1 for trade in trades if trade["result"] == "win")
        return wins / len(trades)

    def avg_profit(self, lookback=20):
        trades = self.performance_log[-lookback:]
        if not trades:
            return 0.0
        profits = [trade["roi"] for trade in trades if "roi" in trade]
        return np.mean(profits) if profits else 0.0

    def recent_drawdown(self, lookback=20):
        trades = self.performance_log[-lookback:]
        losses = [abs(trade["roi"]) for trade in trades if trade["result"] == "loss"]
        return np.max(losses) if losses else 0.0

    def summary(self, lookback=20):
        return {
            "win_rate": round(self.win_rate(lookback), 3),
            "avg_profit": round(self.avg_profit(lookback), 3),
            "max_drawdown": round(self.recent_drawdown(lookback), 3),
            "trades_analyzed": len(self.performance_log[-lookback:])
        }
