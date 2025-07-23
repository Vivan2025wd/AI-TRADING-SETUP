import os
import json
from pathlib import Path

# Constants
TRADE_LOG_DIR = Path("storage/performance_logs")
PREDICTION_LOG_DIR = Path("storage/trade_history")
TRADE_LOG_LIMIT = 1000
PREDICTION_LOG_LIMIT = 1000

def cleanup_file(file_path: Path, max_entries: int):
    if not file_path.exists():
        return
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > max_entries:
            trimmed_data = data[-max_entries:]
            with open(file_path, "w") as f:
                json.dump(trimmed_data, f, indent=2)
    except Exception as e:
        print(f"Error cleaning {file_path.name}: {e}")

def auto_cleanup_logs(log_type=None, symbol=None):
    if log_type == "trade_logs" and symbol:
        file_path = TRADE_LOG_DIR / f"{symbol}_trades.json"
        cleanup_file(file_path, TRADE_LOG_LIMIT)
    elif log_type == "prediction_logs" and symbol:
        file_path = PREDICTION_LOG_DIR / f"{symbol}_predictions.json"
        cleanup_file(file_path, PREDICTION_LOG_LIMIT)
    else:
        # Cleanup all if no specific args given
        for file in TRADE_LOG_DIR.glob("*_trades.json"):
            cleanup_file(file, TRADE_LOG_LIMIT)
        for file in PREDICTION_LOG_DIR.glob("*_predictions.json"):
            cleanup_file(file, PREDICTION_LOG_LIMIT)

if __name__ == "__main__":
    auto_cleanup_logs()
