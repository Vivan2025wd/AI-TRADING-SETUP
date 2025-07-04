# backend/mother_ai/performance_tracker.py

import os
import json
from datetime import datetime

class PerformanceTracker:
    def __init__(self, log_dir="backend/storage/trade_history"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def get_log_path(self, symbol):
        return os.path.join(self.log_dir, f"{symbol}_log.json")

    def log_trade(self, symbol, trade_data):
        path = self.get_log_path(symbol)
        logs = self._load_logs(path)
        logs.append(trade_data)
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)

    def get_agent_log(self, symbol, limit=100):
        path = self.get_log_path(symbol)
        logs = self._load_logs(path)
        return logs[-limit:] if len(logs) > limit else logs

    def _load_logs(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def current_time(self):
        return datetime.utcnow().isoformat()

    def clear_log(self, symbol):
        path = self.get_log_path(symbol)
        if os.path.exists(path):
            os.remove(path)

    def log_summary(self, symbol):
        logs = self.get_agent_log(symbol)
        if not logs:
            return {"win_rate": 0, "loss_rate": 0, "total": 0}

        wins = [log for log in logs if log.get("result") == "win"]
        losses = [log for log in logs if log.get("result") == "loss"]
        total = len(logs)

        win_rate = round(len(wins) / total, 2)
        loss_rate = round(len(losses) / total, 2)

        return {
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "total": total
        }
