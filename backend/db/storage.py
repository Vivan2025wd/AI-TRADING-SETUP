import os
import json
from typing import Any, Dict, List

# Base local directory for storage (relative to root of project)
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'storage')
TRADE_LOGS_DIR = os.path.join(BASE_DIR, 'trade_logs')
STRATEGIES_DIR = os.path.join(BASE_DIR, 'strategies')

# Ensure directories exist
os.makedirs(TRADE_LOGS_DIR, exist_ok=True)
os.makedirs(STRATEGIES_DIR, exist_ok=True)

# ========== TRADE LOGS ==========

def save_trade_log(symbol: str, trade_data: Dict[str, Any]):
    file_path = os.path.join(TRADE_LOGS_DIR, f'{symbol}_trades.json')
    trades = load_trade_logs(symbol)
    trades.append(trade_data)
    with open(file_path, 'w') as f:
        json.dump(trades, f, indent=4)

def load_trade_logs(symbol: str) -> List[Dict[str, Any]]:
    file_path = os.path.join(TRADE_LOGS_DIR, f'{symbol}_trades.json')
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return json.load(f)

# ========== STRATEGIES ==========

def save_strategy(symbol: str, strategy_name: str, strategy_data: Dict[str, Any]):
    symbol_dir = os.path.join(STRATEGIES_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    file_path = os.path.join(symbol_dir, f'{strategy_name}.json')
    with open(file_path, 'w') as f:
        json.dump(strategy_data, f, indent=4)

def load_strategy(symbol: str, strategy_name: str) -> Dict[str, Any]:
    file_path = os.path.join(STRATEGIES_DIR, symbol, f'{strategy_name}.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Strategy "{strategy_name}" for {symbol} not found.')
    with open(file_path, 'r') as f:
        return json.load(f)

def list_strategies(symbol: str) -> List[str]:
    symbol_dir = os.path.join(STRATEGIES_DIR, symbol)
    if not os.path.exists(symbol_dir):
        return []
    return [f.replace('.json', '') for f in os.listdir(symbol_dir) if f.endswith('.json')]
