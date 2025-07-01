import os
import json
from datetime import datetime
from typing import Dict

from utils.logger import logger

# Local storage file for mock balance
MOCK_BALANCE_FILE = os.path.join(os.path.dirname(__file__), '..', 'storage', 'mock_balance.json')

# Default starting balance if file doesn't exist
DEFAULT_BALANCE = {
    "USD": 10000.0,
    "holdings": {}  # e.g. {"BTCUSDT": {"amount": 0.01, "entry_price": 29200}}
}


def load_mock_balance() -> Dict:
    if not os.path.exists(MOCK_BALANCE_FILE):
        save_mock_balance(DEFAULT_BALANCE)
    with open(MOCK_BALANCE_FILE, 'r') as f:
        return json.load(f)


def save_mock_balance(data: Dict):
    with open(MOCK_BALANCE_FILE, 'w') as f:
        json.dump(data, f, indent=4)


def execute_mock_trade(symbol: str, action: str, price: float, confidence: float) -> Dict:
    balance = load_mock_balance()
    amount = 0.0
    result = {}

    timestamp = datetime.utcnow().isoformat()
    usd_balance = balance.get("USD", 0)

    if action.lower() == "buy":
        amount = round(usd_balance / price * 0.1, 6)  # Use 10% of USD balance
        cost = amount * price
        if cost <= usd_balance:
            balance["USD"] -= cost
            balance["holdings"][symbol] = {
                "amount": amount,
                "entry_price": price
            }
            result = {
                "status": "BUY_EXECUTED",
                "symbol": symbol,
                "price": price,
                "amount": amount,
                "confidence": confidence,
                "timestamp": timestamp
            }
            logger.info(f"BUY {amount} {symbol} at {price}")
        else:
            result = {"status": "INSUFFICIENT_FUNDS"}
            logger.warning("Not enough USD to execute buy trade.")

    elif action.lower() == "sell":
        holding = balance["holdings"].get(symbol)
        if holding:
            amount = holding["amount"]
            revenue = amount * price
            balance["USD"] += revenue
            del balance["holdings"][symbol]
            result = {
                "status": "SELL_EXECUTED",
                "symbol": symbol,
                "price": price,
                "amount": amount,
                "confidence": confidence,
                "timestamp": timestamp
            }
            logger.info(f"SELL {amount} {symbol} at {price}")
        else:
            result = {"status": "NO_HOLDINGS"}
            logger.warning(f"No holdings for {symbol} to sell.")

    else:
        result = {"status": "INVALID_ACTION"}
        logger.error("Invalid trade action.")

    # Save updated balance
    save_mock_balance(balance)
    return result


def get_mock_portfolio():
    return load_mock_balance()
