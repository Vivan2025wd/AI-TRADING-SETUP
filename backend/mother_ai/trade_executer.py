import os
import json
from datetime import datetime
from typing import Dict, List

from backend.utils.logger import logger

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
STORAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'storage'))

MOCK_BALANCE_FILE = os.path.join(STORAGE_DIR, 'mock_balance.json')
TRADE_HISTORY_DIR = os.path.join(STORAGE_DIR, 'trade_history')
PERFORMANCE_LOG_DIR = os.path.join(STORAGE_DIR, 'performance_logs')  # fixed folder name to match usage

# Ensure directories exist
os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)

# --- Default Balance ---
DEFAULT_BALANCE = {
    "USD": 10000.0,
    "holdings": {}  # e.g., {"BTC": {"amount": 0.01, "entry_price": 29200}}
}

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

# --- Trade Execution ---
def execute_mock_trade(symbol: str, action: str, price: float, confidence: float) -> Dict:
    balance = load_mock_balance()
    amount = 0.0
    result = {}
    timestamp = datetime.utcnow().isoformat()
    usd_balance = balance.get("USD", 0)

    symbol = symbol.upper()
    perf_path = os.path.join(PERFORMANCE_LOG_DIR, f"{symbol}_trades.json")
    hist_path = os.path.join(TRADE_HISTORY_DIR, f"{symbol}_log.json")

    if action.lower() == "buy":
        amount = round((usd_balance * 0.1) / price, 6)  # Use 10% of USD balance
        cost = amount * price
        if cost <= usd_balance and amount > 0:
            balance["USD"] -= cost
            holding = balance["holdings"].get(symbol, {"amount": 0, "entry_price": 0})
            total_amount = holding["amount"] + amount
            # Weighted average entry price for new buy
            if holding["amount"] > 0:
                new_entry_price = ((holding["entry_price"] * holding["amount"]) + (price * amount)) / total_amount
            else:
                new_entry_price = price
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
            profit_percent = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            balance["USD"] += revenue
            del balance["holdings"][symbol]
            result = {
                "timestamp": timestamp,
                "balance": round(balance["USD"], 2),
                "price": price,
                "type": "SELL",
                "profit_percent": round(profit_percent, 2),
                "symbol": symbol
            }
            logger.info(f"SELL {amount} {symbol} at {price}")
        else:
            result = {"status": "NO_HOLDINGS"}
            logger.warning(f"No holdings for {symbol} to sell.")
            return result

    else:
        result = {"status": "INVALID_ACTION"}
        logger.error(f"Invalid trade action: {action}")
        return result

    # Save balance
    save_mock_balance(balance)

    # Log performance trade
    append_json_log(perf_path, result)

    # Log full portfolio snapshot for debugging/history
    snapshot = {
        "timestamp": timestamp,
        "symbol": symbol,
        "price": price,
        "action": action.upper(),
        "amount": amount,
        "usd_balance": round(balance.get("USD", 0), 2),
        "holdings": balance.get("holdings", {}),
        "confidence": confidence
    }
    append_json_log(hist_path, snapshot)

    return result

# --- Execute trades from Mother AI's decision ---
def execute_mother_ai_decision(decision_data: Dict) -> List[Dict]:
    """
    decision_data = {
        "decision": [
            {"symbol": "BTCUSDT", "signal": "BUY", "price": 29200, "confidence": 0.82},
            ...
        ]
    }
    """
    decisions = decision_data.get("decision", [])
    if not decisions:
        logger.warning("Mother AI returned no decision.")
        return []

    logger.info(f"🧠 Executing {len(decisions)} Mother AI trades...")
    trade_results = []

    for d in decisions:
        symbol = d.get("symbol")
        # Normalize symbol to just coin part (e.g. BTCUSDT -> BTC)
        if symbol and symbol.endswith("USDT"):
            symbol = symbol[:-4]
        action = d.get("signal") or d.get("action")  # Accept both keys if possible
        price = d.get("last_price") or d.get("price")
        confidence = d.get("confidence", 0.0)

        if symbol and action and price:
            res = execute_mock_trade(symbol, action, price, confidence)
            trade_results.append(res)
        else:
            logger.warning(f"Invalid trade decision skipped: {d}")

    return trade_results

# --- Get Portfolio State ---
def get_mock_portfolio():
    return load_mock_balance()
