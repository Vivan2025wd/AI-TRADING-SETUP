import os
import json
from datetime import datetime, timedelta
from typing import Dict, List

from backend.utils.logger import logger

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
STORAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'storage'))

MOCK_BALANCE_FILE = os.path.join(STORAGE_DIR, 'mock_balance.json')
# Removed TRADE_HISTORY_DIR since we no longer use trade_history logs
PERFORMANCE_LOG_DIR = os.path.join(STORAGE_DIR, 'performance_logs')
TRADE_PROFITS_DIR = os.path.join(STORAGE_DIR, 'trade_profits')  # new folder
LAST_EXECUTION_FILE = os.path.join(STORAGE_DIR, 'last_execution_time.json')
MOCK_PROFIT_FILE = os.path.join(TRADE_PROFITS_DIR, 'mock_profit.json')  # moved here

# --- Ensure directories exist ---
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)
os.makedirs(TRADE_PROFITS_DIR, exist_ok=True)  # ensure it exists

# --- Defaults ---
DEFAULT_BALANCE = {
    "USD": 10000.0,
    "holdings": {}
}

DEFAULT_PROFIT_TRACKER = {
    "total_profit_usd": 0.0,
    "total_trades": 0
}

# --- Cooldown Logic ---
def get_last_execution_time():
    if os.path.exists(LAST_EXECUTION_FILE):
        with open(LAST_EXECUTION_FILE, 'r') as f:
            try:
                data = json.load(f)
                return datetime.fromisoformat(data.get("last_executed"))
            except Exception:
                return None
    return None

def set_last_execution_time():
    with open(LAST_EXECUTION_FILE, 'w') as f:
        json.dump({"last_executed": datetime.utcnow().isoformat()}, f, indent=4)

# --- Helpers ---
def load_mock_balance() -> Dict:
    if not os.path.exists(MOCK_BALANCE_FILE):
        save_mock_balance(DEFAULT_BALANCE)
    with open(MOCK_BALANCE_FILE, 'r') as f:
        return json.load(f)

def save_mock_balance(data: Dict):
    with open(MOCK_BALANCE_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def append_json_log(file_path: str, record: Dict):
    existing = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.append(record)
    with open(file_path, 'w') as f:
        json.dump(existing, f, indent=4)

def load_profit_tracker() -> Dict:
    if not os.path.exists(MOCK_PROFIT_FILE):
        save_profit_tracker(DEFAULT_PROFIT_TRACKER)
    with open(MOCK_PROFIT_FILE, 'r') as f:
        return json.load(f)

def save_profit_tracker(data: Dict):
    with open(MOCK_PROFIT_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- Trade Execution ---
def execute_mock_trade(symbol: str, action: str, price: float, confidence: float) -> Dict:
    balance = load_mock_balance()
    profit_tracker = load_profit_tracker()

    amount = 0.0
    result = {}
    timestamp = datetime.utcnow().isoformat()
    usd_balance = balance.get("USD", 0)

    symbol = symbol.upper()
    perf_path = os.path.join(PERFORMANCE_LOG_DIR, f"{symbol}_trades.json")  # unified log path

    if action.lower() == "buy":
        amount = round((usd_balance * 0.1) / price, 6)
        cost = amount * price
        if cost <= usd_balance and amount > 0:
            balance["USD"] -= cost
            holding = balance["holdings"].get(symbol, {"amount": 0, "entry_price": 0})
            total_amount = holding["amount"] + amount
            new_entry_price = ((holding["entry_price"] * holding["amount"]) + (price * amount)) / total_amount if holding["amount"] > 0 else price
            balance["holdings"][symbol] = {
                "amount": round(total_amount, 6),
                "entry_price": round(new_entry_price, 2)
            }
            result = {
                "timestamp": timestamp,
                "balance": round(balance["USD"], 2),
                "price": price,
                "type": "BUY",
                "profit_percent": 0.0,
                "symbol": symbol
            }
            logger.info(f"BUY {amount} {symbol} at {price}")
        else:
            result = {"status": "INSUFFICIENT_FUNDS"}
            logger.warning(f"Not enough USD to execute buy trade for {symbol}.")
            return result

    elif action.lower() == "sell":
        holding = balance["holdings"].get(symbol)
        if holding and holding["amount"] > 0:
            amount = holding["amount"]
            entry_price = holding["entry_price"]
            revenue = amount * price
            profit_usd = revenue - (amount * entry_price)
            profit_percent = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

            balance["USD"] += revenue
            del balance["holdings"][symbol]

            # Update profit tracker
            profit_tracker["total_profit_usd"] += round(profit_usd, 2)
            profit_tracker["total_trades"] += 1
            save_profit_tracker(profit_tracker)

            result = {
                "timestamp": timestamp,
                "balance": round(balance["USD"], 2),
                "price": price,
                "type": "SELL",
                "profit_percent": round(profit_percent, 2),
                "profit_usd": round(profit_usd, 2),
                "symbol": symbol,
                "cumulative_profit": profit_tracker["total_profit_usd"]
            }
            logger.info(f"SELL {amount} {symbol} at {price}, profit: {profit_usd:.2f} USD")
        else:
            result = {"status": "NO_HOLDINGS"}
            logger.warning(f"No holdings for {symbol} to sell.")
            return result

    else:
        result = {"status": "INVALID_ACTION"}
        logger.error(f"Invalid trade action: {action}")
        return result

    save_mock_balance(balance)
    append_json_log(perf_path, result)

    # Removed the separate trade_history log append (hist_path and snapshot)

    return result

# --- Execute Mother AI Trades (every 15 min) ---
def execute_mother_ai_decision(decision_data: Dict) -> List[Dict]:
    last_exec = get_last_execution_time()
    now = datetime.utcnow()
    if last_exec and (now - last_exec) < timedelta(minutes=15):
        logger.info("⏳ Skipping execution: Last trade was within 15 minutes.")
        return [{
            "status": "SKIPPED",
            "reason": "Cooldown active (15 min)",
            "last_executed": last_exec.isoformat()
        }]

    decisions = decision_data.get("decision", [])
    if not decisions:
        logger.warning("Mother AI returned no decision.")
        return []

    logger.info(f"🧠 Executing {len(decisions)} Mother AI trades...")
    trade_results = []

    for d in decisions:
        symbol = d.get("symbol")
        if symbol and symbol.endswith("USDT"):
            symbol = symbol[:-4]
        action = d.get("signal") or d.get("action")
        price = d.get("last_price") or d.get("price")
        confidence = d.get("confidence", 0.0)

        if symbol and action and price:
            res = execute_mock_trade(symbol, action, price, confidence)
            trade_results.append(res)
        else:
            logger.warning(f"Invalid trade decision skipped: {d}")

    set_last_execution_time()
    return trade_results

# --- Get Portfolio State ---
def get_mock_portfolio():
    return load_mock_balance()
