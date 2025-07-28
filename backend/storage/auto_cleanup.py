import os
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil

# --- Configuration ---
TRADE_LOG_DIR = Path("storage/performance_logs")
PREDICTION_LOG_DIR = Path("storage/trade_history")
TRADE_LOG_LIMIT = 1000
PREDICTION_LOG_LIMIT = 1000

# --- Setup Logger ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LogCleaner")

# --- Core Cleanup Function ---
def cleanup_file(file_path: Path, max_entries: int):
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error(f"Invalid data format in {file_path.name}: Expected a list.")
            return

        if len(data) <= max_entries:
            return  # No trimming needed
        
        trimmed_data = data[-max_entries:]

        # Atomic write to temp file, then replace original
        with NamedTemporaryFile("w", delete=False, dir=file_path.parent, suffix=".tmp") as tmp_file:
            json.dump(trimmed_data, tmp_file, indent=2)
            temp_path = Path(tmp_file.name)

        shutil.move(str(temp_path), file_path)
        logger.info(f"Trimmed {file_path.name} to last {max_entries} entries.")

    except Exception as e:
        logger.error(f"Error cleaning {file_path.name}: {e}")

# --- Cleanup Dispatcher ---
def auto_cleanup_logs(log_type=None, symbol=None):
    if log_type == "trade_logs" and symbol:
        file_path = TRADE_LOG_DIR / f"{symbol}_trades.json"
        cleanup_file(file_path, TRADE_LOG_LIMIT)
    elif log_type == "prediction_logs" and symbol:
        file_path = PREDICTION_LOG_DIR / f"{symbol}_predictions.json"
        cleanup_file(file_path, PREDICTION_LOG_LIMIT)
    else:
        logger.info("Starting full logs cleanup...")

        for file in TRADE_LOG_DIR.glob("*_trades.json"):
            cleanup_file(file, TRADE_LOG_LIMIT)

        for file in PREDICTION_LOG_DIR.glob("*_predictions.json"):
            cleanup_file(file, PREDICTION_LOG_LIMIT)

        logger.info("Log cleanup completed.")

# --- Entry Point ---
if __name__ == "__main__":
    auto_cleanup_logs()
