import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from backend.utils.logger import logger
from backend.utils.binance_api import get_trading_mode, get_binance_client

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
STORAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'storage'))
MOCK_BALANCE_FILE = os.path.join(STORAGE_DIR, 'mock_balance.json')
PERFORMANCE_LOG_DIR = os.path.join(STORAGE_DIR, 'performance_logs')
TRADE_PROFITS_DIR = os.path.join(STORAGE_DIR, 'trade_profits')
SYMBOL_EXECUTION_FILE = os.path.join(STORAGE_DIR, 'symbol_last_execution.json')
MOCK_PROFIT_FILE = os.path.join(TRADE_PROFITS_DIR, 'mock_profit.json')

# --- Ensure directories exist ---
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)
os.makedirs(TRADE_PROFITS_DIR, exist_ok=True)

# --- Defaults ---
DEFAULT_BALANCE = {
    "USD": 10000.0,
    "holdings": {}
}

DEFAULT_PROFIT_TRACKER = {
    "total_profit_usd": 0.0,
    "total_trades": 0
}

# --- Symbol Cooldown Logic ---
def load_symbol_execution_times() -> Dict[str, str]:
    if os.path.exists(SYMBOL_EXECUTION_FILE):
        with open(SYMBOL_EXECUTION_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_symbol_execution_times(data: Dict[str, str]):
    with open(SYMBOL_EXECUTION_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_symbol_last_execution_time(symbol: str) -> Optional[datetime]:
    data = load_symbol_execution_times()
    ts = data.get(symbol.upper())
    return datetime.fromisoformat(ts) if ts else None

def set_symbol_last_execution_time(symbol: str):
    data = load_symbol_execution_times()
    data[symbol.upper()] = datetime.utcnow().isoformat()
    save_symbol_execution_times(data)

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
def calculate_trade_amount(symbol: str, price: float, usd_balance: float, config: Dict) -> float:
    """
    Calculate trade amount dynamically based on risk settings, asset price, and balance.
    """

    # Fetch risk percentage per trade from config (default 10%)
    risk_percent = config.get("risk_per_trade", 0.10)
    min_position_usd = config.get("min_position_usd", 10)  # Ensure minimum $10 position
    max_position_usd = config.get("max_position_usd", usd_balance * risk_percent)

    # Calculate position size based on risk %
    allocated_usd = usd_balance * risk_percent

    # Enforce min/max bounds
    if allocated_usd < min_position_usd:
        allocated_usd = min_position_usd
    elif allocated_usd > max_position_usd:
        allocated_usd = max_position_usd

    # Final amount in asset units
    amount = round(allocated_usd / price, 6)

    # Asset specific rounding rules (BTC needs smaller decimals, DOGE larger)
    if price > 1000:
        amount = round(amount, 6)  # BTC, ETH, etc.
    elif price > 1:
        amount = round(amount, 2)  # Medium priced coins like ADA, SOL
    else:
        amount = round(amount, 0)  # Penny coins like DOGE, SHIBA

    return amount



def execute_mock_trade(symbol: str, action: str, price: float, confidence: float) -> Dict:
    balance = load_mock_balance()
    profit_tracker = load_profit_tracker()

    result = {}
    timestamp = datetime.utcnow().isoformat()
    usd_balance = balance.get("USD", 0)

    symbol = symbol.upper()
    perf_path = os.path.join(PERFORMANCE_LOG_DIR, f"{symbol}_trades.json")

    RISK_CONFIG = {
        "risk_per_trade": 0.10,
        "min_position_usd": 10,
        "max_position_usd": usd_balance * 0.5
    }

    if action.lower() == "buy":
        amount = calculate_trade_amount(symbol, price, usd_balance, RISK_CONFIG)
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
                "amount": amount,
                "profit_percent": 0.0,
                "symbol": symbol
            }
            logger.info(f"BUY {amount} {symbol} at {price} (Cost: {cost})")
        else:
            result = {"status": "INSUFFICIENT_FUNDS"}
            logger.warning(f"Not enough USD to execute buy trade for {symbol}. Needed: {cost}, Available: {usd_balance}")
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

            profit_tracker["total_profit_usd"] += round(profit_usd, 2)
            profit_tracker["total_trades"] += 1
            save_profit_tracker(profit_tracker)

            result = {
                "timestamp": timestamp,
                "balance": round(balance["USD"], 2),
                "price": price,
                "type": "SELL",
                "amount": amount,
                "profit_percent": round(profit_percent, 2),
                "profit_usd": round(profit_usd, 2),
                "symbol": symbol,
                "cumulative_profit": profit_tracker["total_profit_usd"]
            }
            logger.info(f"SELL {amount} {symbol} at {price}, Profit: {profit_usd:.2f} USD")
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
    return result

def execute_live_trade(symbol: str, action: str, price: float, confidence: float) -> dict:
    client = get_binance_client()
    result = {}
    try:
        quantity = round((100 / price), 3)  # Example: Buy $100 worth (adjust sizing logic)
        if action.lower() == "buy":
            order = client.order_market_buy(symbol=f"{symbol}USDT", quantity=quantity)
        elif action.lower() == "sell":
            order = client.order_market_sell(symbol=f"{symbol}USDT", quantity=quantity)
        else:
            return {"status": "INVALID_ACTION"}
        
        result = {
            "status": "EXECUTED",
            "symbol": symbol,
            "action": action.upper(),
            "executed_qty": quantity,
            "order_id": order['orderId']
        }
        logger.info(f"[LIVE TRADE] {action.upper()} {quantity} {symbol} at market price.")
    except Exception as e:
        logger.error(f"[LIVE TRADE ERROR] {e}")
        result = {"status": "ERROR", "detail": str(e)}
    return result

def execute_trade(symbol: str, action: str, price: float, confidence: float) -> Dict:
    mode = get_trading_mode()
    if mode == "live":
        return execute_live_trade(symbol, action, price, confidence)
    else:
        return execute_mock_trade(symbol, action, price, confidence)

# --- Execute Mother AI Trades (Symbol-specific Cooldown) ---
def execute_mother_ai_decision(decision_data: Dict) -> List[Dict]:
    decisions = decision_data.get("decision", [])
    if not decisions:
        logger.warning("Mother AI returned no decision.")
        return []

    logger.info(f"🧠 Executing {len(decisions)} Mother AI trades... (Mode: {get_trading_mode().upper()})")
    trade_results = []

    for d in decisions:
        symbol = d.get("symbol")
        if symbol and symbol.endswith("USDT"):
            symbol = symbol[:-4]
        action = d.get("signal") or d.get("action")
        price = d.get("last_price") or d.get("price")
        confidence = d.get("confidence", 0.0)

        if not (symbol and action and price):
            logger.warning(f"Invalid trade decision skipped: {d}")
            continue

        # --- Symbol Cooldown Check (60 minutes) ---
        last_exec = get_symbol_last_execution_time(symbol)
        now = datetime.utcnow()
        if last_exec and (now - last_exec) < timedelta(minutes=60):
            logger.info(f"⏳ Skipping {symbol}: Cooldown active (60 min). Last executed at {last_exec.isoformat()}")
            trade_results.append({
                "status": "SKIPPED",
                "symbol": symbol,
                "reason": "Symbol cooldown active (60 min)",
                "last_executed": last_exec.isoformat()
            })
            continue

        # --- Execute Trade ---
        res = execute_trade(symbol, action, price, confidence)
        trade_results.append(res)
        set_symbol_last_execution_time(symbol)

    return trade_results



# --- Get Portfolio State ---
def get_mock_portfolio():
    return load_mock_balance()
