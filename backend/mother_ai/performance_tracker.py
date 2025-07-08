# backend/mother_ai/performance_tracker.py

import os
import json
from datetime import datetime
from typing import Dict, List


class PerformanceTracker:
    def __init__(self, log_dir_type="trade_history"):
        base_dir = "backend/storage"
        if log_dir_type == "trade_history":
            self.log_dir = os.path.join(base_dir, "trade_history")
            self.file_suffix = "_predictions.json"
        elif log_dir_type == "performance_logs":
            self.log_dir = os.path.join(base_dir, "performance_logs")
            self.file_suffix = "_trades.json"
        else:
            raise ValueError("log_dir_type must be 'trade_history' or 'performance_logs'")

        os.makedirs(self.log_dir, exist_ok=True)

    def get_log_path(self, symbol: str) -> str:
        return os.path.join(self.log_dir, f"{symbol}{self.file_suffix}")

    def log_trade(self, symbol: str, trade_data: Dict):
        path = self.get_log_path(symbol)
        logs = self._load_logs(path)
        logs.append(trade_data)
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)

    def get_agent_log(self, symbol: str, limit: int = 100) -> List[Dict]:
        path = self.get_log_path(symbol)
        logs = self._load_logs(path)
        return logs[-limit:] if len(logs) > limit else logs

    def clear_log(self, symbol: str):
        path = self.get_log_path(symbol)
        if os.path.exists(path):
            os.remove(path)

    def current_time(self) -> str:
        return datetime.utcnow().isoformat()

    def _load_logs(self, path: str) -> List[Dict]:
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def get_strategy_health(self, symbol: str, limit: int = 100) -> Dict:
        logs = self.get_agent_log(symbol, limit=limit)
        if not logs:
            return {
                "win_rate": 0.0,
                "loss_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_profit": 0.0,
                "total": 0
            }

        wins = 0
        losses = 0
        confidence_sum = 0.0
        profit_sum = 0.0
        count = 0

        for log in logs:
            confidence = log.get("confidence", 0.0)
            confidence_sum += confidence

            if "profit_percent" in log:
                # This is a trade log
                profit_sum += log.get("profit_percent", 0.0)
                if log["profit_percent"] > 0:
                    wins += 1
                else:
                    losses += 1
            elif "signal" in log:
                # This is a prediction log, use score (if win/loss defined)
                if log.get("result") == "win":
                    wins += 1
                elif log.get("result") == "loss":
                    losses += 1

            count += 1

        win_rate = round(wins / count, 2) if count else 0.0
        loss_rate = round(losses / count, 2) if count else 0.0
        avg_confidence = round(confidence_sum / count, 2) if count else 0.0
        avg_profit = round(profit_sum / count, 2) if count else 0.0

        return {
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "avg_confidence": avg_confidence,
            "avg_profit": avg_profit,
            "total": count
        }
