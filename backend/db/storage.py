import os
import json
from typing import Any, Dict, List
from pathlib import Path
from storage.auto_cleanup import auto_cleanup_logs

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent / "storage"
TRADE_LOGS_DIR = BASE_DIR / "trade_logs"
STRATEGIES_DIR = BASE_DIR / "strategies"

# Ensure directories exist
TRADE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

# ========== TRADE LOGS ==========

def save_trade_log(symbol: str, trade_data: Dict[str, Any]):
    """
    Appends a trade to the symbol's trade log JSON file.
    Cleans up if file exceeds limits.
    """ 
    auto_cleanup_logs(log_type="trade_logs", symbol=symbol)  # no arguments, runs cleanup for all logs

    file_path = TRADE_LOGS_DIR / f"{symbol}_trades.json"
    trades = load_trade_logs(symbol)
    trades.append(trade_data)
    with open(file_path, "w") as f:
        json.dump(trades, f, indent=4)


def load_trade_logs(symbol: str) -> List[Dict[str, Any]]:
    """
    Loads trade history for a symbol from local storage.
    """
    file_path = TRADE_LOGS_DIR / f"{symbol}_trades.json"
    if not file_path.exists():
        return []
    with open(file_path, "r") as f:
        return json.load(f)


# ========== STRATEGIES (Flat Format) ==========

def save_strategy(symbol: str, strategy_id: str, strategy_data: Dict[str, Any]):
    """
    Saves a strategy to the strategies folder (flat structure).
    """
    filename = f"{symbol}_strategy_{strategy_id}.json"
    file_path = STRATEGIES_DIR / filename
    with open(file_path, "w") as f:
        json.dump(strategy_data, f, indent=4)

def load_strategy(symbol: str, strategy_id: str) -> Dict[str, Any]:
    """
    Loads a specific strategy from disk.
    """
    file_path = STRATEGIES_DIR / f"{symbol}_strategy_{strategy_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy '{strategy_id}' for {symbol} not found.")
    with open(file_path, "r") as f:
        return json.load(f)

def list_strategies(symbol: str) -> List[str]:
    """
    Lists all strategy IDs for a given symbol.
    """
    strategy_files = STRATEGIES_DIR.glob(f"{symbol}_strategy_*.json")
    return [
        file.stem.replace(f"{symbol}_strategy_", "")
        for file in strategy_files
    ]

