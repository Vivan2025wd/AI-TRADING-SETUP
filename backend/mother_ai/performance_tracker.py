import os
import json
import glob
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

        self.strategy_dir = os.path.join(base_dir, "strategies")
        os.makedirs(self.log_dir, exist_ok=True)

    def get_log_path(self, symbol: str) -> str:
        return os.path.join(self.log_dir, f"{symbol}{self.file_suffix}")

    def log_trade(self, symbol: str, trade_data: Dict):
        path = self.get_log_path(symbol)
        logs = self._load_logs(path)
        logs.append(trade_data)
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)

    def log_prediction(self, symbol: str, prediction_data: Dict):
        """
        Logs an agent prediction to {symbol}_predictions.json in the trade_history directory.
        """
        path = self.get_log_path(symbol)
        logs = self._load_logs(path)
        logs.append(prediction_data)
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)

    def get_agent_log(self, symbol: str, limit: int = 100) -> List[Dict]:
        path = self.get_log_path(symbol)
        logs = self._load_logs(path)
        # Return last N logs, newest last
        return logs[-limit:] if len(logs) > limit else logs

    def clear_log(self, symbol: str):
        path = self.get_log_path(symbol)
        if os.path.exists(path):
            os.remove(path)

    def current_time(self) -> str:
        from datetime import datetime
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

            profit_percent = log.get("profit_percent")
            if profit_percent is not None:
                profit_sum += profit_percent
                if profit_percent > 0:
                    wins += 1
                else:
                    losses += 1
            else:
                result = log.get("result", "").lower()
                if result == "win":
                    wins += 1
                elif result == "loss":
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

    def list_strategies(self, symbol: str) -> List[Dict]:
        pattern = os.path.join(self.strategy_dir, f"{symbol}_strategy_*.json")
        files = glob.glob(pattern)
        strategies = []

        for filepath in files:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                strategy_id = os.path.basename(filepath).replace(f"{symbol}_strategy_", "").replace(".json", "")
                strategies.append({
                    "strategy_id": strategy_id,
                    "filename": os.path.basename(filepath),
                    "metadata": data.get("metadata", {}),
                    "raw_data": data,
                })
            except Exception as e:
                print(f"⚠️ Failed to load strategy file {filepath}: {e}")

        return strategies

    def rate_strategies(self, symbol: str, limit: int = 100) -> List[Dict]:
        strategies = self.list_strategies(symbol)
        health = self.get_strategy_health(symbol, limit=limit)

        rated = []
        for s in strategies:
            rated.append({
                "strategy_id": s["strategy_id"],
                "win_rate": health.get("win_rate", 0.0),
                "avg_profit": health.get("avg_profit", 0.0),
                "avg_confidence": health.get("avg_confidence", 0.0),
                "total_predictions": health.get("total", 0),
                "metadata": s.get("metadata", {}),
            })

        rated.sort(key=lambda x: (x["win_rate"], x["avg_profit"]), reverse=True)
        return rated
