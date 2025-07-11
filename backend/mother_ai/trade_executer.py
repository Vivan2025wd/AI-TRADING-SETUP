import os
import json
from datetime import datetime, timedelta
from typing import Dict, List

from backend.utils.logger import logger
from backend.utils.binance_api import is_real_trading_mode, get_binance_client
from binance.exceptions import BinanceAPIException, BinanceOrderException

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
STORAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'storage'))

MOCK_BALANCE_FILE = os.path.join(STORAGE_DIR, 'mock_balance.json')
TRADE_HISTORY_DIR = os.path.join(STORAGE_DIR, 'trade_history')
PERFORMANCE_LOG_DIR = os.path.join(STORAGE_DIR, 'performance_logs')
TRADE_PROFITS_DIR = os.path.join(STORAGE_DIR, 'trade_profits')  # ✅ new folder
LAST_EXECUTION_FILE = os.path.join(STORAGE_DIR, 'last_execution_time.json')
MOCK_PROFIT_FILE = os.path.join(TRADE_PROFITS_DIR, 'mock_profit.json')  # ✅ moved here

# --- Ensure directories exist ---
os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)
os.makedirs(TRADE_PROFITS_DIR, exist_ok=True)  # ✅ make sure it exists
REAL_TRADES_LOG_FILE = os.path.join(STORAGE_DIR, 'real_trades_log.json')


# --- Configuration ---
# Percentage of available balance to use for a single trade in real trading mode
REAL_TRADE_RISK_PERCENTAGE = 0.1  # e.g., 10%

# --- Defaults ---
DEFAULT_BALANCE = {
    "USD": 10000.0,  # Assuming USDT or a USD equivalent for mock balance
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

# --- Mock Trade Execution ---
def _execute_mock_order(symbol: str, action: str, price: float, confidence: float) -> Dict:
    # This function contains the original logic of execute_mock_trade
    balance = load_mock_balance()
    profit_tracker = load_profit_tracker()

    amount = 0.0
    result = {}
    timestamp = datetime.utcnow().isoformat()
    usd_balance = balance.get("USD", 0)

    symbol = symbol.upper()
    perf_path = os.path.join(PERFORMANCE_LOG_DIR, f"{symbol}_trades.json")
    hist_path = os.path.join(TRADE_HISTORY_DIR, f"{symbol}_log.json")

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

            # ✅ Update profit tracker
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

# --- Real Trade Execution ---
def _execute_real_order(client, symbol: str, action: str, current_price: float, confidence: float) -> Dict:
    timestamp = datetime.utcnow().isoformat()
    trade_details = {
        "timestamp": timestamp,
        "symbol": symbol,
        "action": action.upper(),
        "price": current_price, # This is the price used for decision, actual fill price may vary for market orders
        "status": "PENDING",
        "confidence": confidence,
        "order_id": None,
        "message": ""
    }

    try:
        # Symbol for Binance API usually needs to be like BTCUSDT
        trade_symbol = symbol.replace('/', '') + "USDT" # Assuming all trades are against USDT for now. This needs to be more robust.

        quote_asset = "USDT" # Assuming quote asset is USDT
        base_asset = symbol.replace('/', '')

        logger.info(f"Attempting REAL {action.upper()} for {trade_symbol} at (approx) {current_price}")

        if action.lower() == "buy":
            # Check USDT balance
            usdt_balance_info = client.get_asset_balance(asset=quote_asset)
            available_usdt = float(usdt_balance_info['free'])
            logger.info(f"Available {quote_asset} for buying {base_asset}: {available_usdt}")

            if available_usdt <= 0:
                trade_details["status"] = "FAILED"
                trade_details["message"] = f"Insufficient {quote_asset} balance to buy {base_asset}. Available: {available_usdt}"
                logger.warning(trade_details["message"])
                return trade_details

            # Calculate quantity to buy based on risk percentage of USDT balance
            # Amount of USDT to spend
            usdt_to_spend = available_usdt * REAL_TRADE_RISK_PERCENTAGE

            if usdt_to_spend < 10: # Binance minimum order size is often around 10 USDT
                 trade_details["status"] = "FAILED"
                 trade_details["message"] = f"Calculated USDT to spend ({usdt_to_spend}) is below exchange minimum (e.g., 10 USDT)."
                 logger.warning(trade_details["message"])
                 return trade_details

            # Quantity is specified in the base asset for market buys using quoteOrderQty
            # For simplicity, we use quoteOrderQty for market buys specifying how much quote currency to spend
            logger.info(f"Placing MARKET BUY for {trade_symbol} spending approx {usdt_to_spend:.2f} {quote_asset}")
            order = client.order_market_buy(symbol=trade_symbol, quoteOrderQty=round(usdt_to_spend, 2))
            # For market orders, price is not specified. quantity is for base asset.
            # Example: client.order_market_buy(symbol=trade_symbol, quantity=quantity_to_buy)
            # where quantity_to_buy = usdt_to_spend / current_price (approx)

        elif action.lower() == "sell":
            # Check base asset balance (e.g., BTC)
            base_asset_balance_info = client.get_asset_balance(asset=base_asset)
            available_base = float(base_asset_balance_info['free'])
            logger.info(f"Available {base_asset} to sell: {available_base}")

            if available_base <= 0:
                trade_details["status"] = "FAILED"
                trade_details["message"] = f"No {base_asset} holdings to sell."
                logger.warning(trade_details["message"])
                return trade_details

            # Calculate quantity to sell based on risk percentage of available base asset
            quantity_to_sell = available_base * REAL_TRADE_RISK_PERCENTAGE

            # Ensure quantity meets minimum tradeable unit for the symbol (tricky without exchange info)
            # For now, we assume it's fine if it's > 0. Min notional value also applies.
            # Smallest tradable unit might be an issue for some coins.
            # A common check is if quantity_to_sell * current_price > 10 (approx min notional USDT)
            if quantity_to_sell * current_price < 10: # Rough check for min notional
                trade_details["status"] = "FAILED"
                trade_details["message"] = f"Calculated sell amount for {base_asset} (qty: {quantity_to_sell}, value: {quantity_to_sell * current_price:.2f} USDT) is likely below exchange minimum notional value."
                logger.warning(trade_details["message"])
                return trade_details

            logger.info(f"Placing MARKET SELL for {quantity_to_sell:.8f} {base_asset} ({trade_symbol})")
            order = client.order_market_sell(symbol=trade_symbol, quantity=round(quantity_to_sell, 8)) # Precision needs to be right for the asset

        else:
            trade_details["status"] = "FAILED"
            trade_details["message"] = f"Invalid trade action: {action}"
            logger.error(trade_details["message"])
            return trade_details

        trade_details.update({
            "status": order.get("status", "UNKNOWN_FROM_API"),
            "order_id": order.get("orderId"),
            "client_order_id": order.get("clientOrderId"),
            "fills": order.get("fills", []), # Market orders fill immediately
            "message": f"Order placed successfully: {action.upper()} {symbol}"
        })
        logger.info(f"Real trade executed: {trade_details}")

    except BinanceAPIException as e:
        logger.error(f"Binance API Exception during real trade for {symbol}: {e}")
        trade_details["status"] = "ERROR_API"
        trade_details["message"] = str(e)
    except BinanceOrderException as e:
        logger.error(f"Binance Order Exception during real trade for {symbol}: {e}")
        trade_details["status"] = "ERROR_ORDER"
        trade_details["message"] = str(e)
    except Exception as e:
        logger.error(f"Generic Exception during real trade for {symbol}: {e}")
        trade_details["status"] = "ERROR_UNKNOWN"
        trade_details["message"] = str(e)

    append_json_log(REAL_TRADES_LOG_FILE, trade_details)
    return trade_details

# --- Unified Trade Processing ---
def process_trade_decision(symbol: str, action: str, price: float, confidence: float) -> Dict:
    """
    Processes a trade decision, either executing a mock trade or a real trade
    based on the REAL_TRADING_MODE configuration.
    """
    if is_real_trading_mode():
        logger.info(f"REAL TRADING MODE: Processing trade for {symbol}, Action: {action}, Price: {price}")
        try:
            client = get_binance_client() # This will raise error if not connected
            # Note: 'price' here is the price at the time of decision, actual fill for market order will vary.
            return _execute_real_order(client, symbol, action, price, confidence)
        except Exception as e:
            logger.error(f"Failed to get Binance client or other critical error in real trade pre-check: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "action": action.upper(),
                "price": price,
                "status": "ERROR_CLIENT_INIT",
                "message": str(e),
                "confidence": confidence
            }
    else:
        logger.info(f"MOCK TRADING MODE: Processing trade for {symbol}, Action: {action}, Price: {price}")
        return _execute_mock_order(symbol, action, price, confidence)

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
            # Ensure symbol is just the base asset part, e.g., BTC, ETH for internal consistency
            # The _execute_real_order will append USDT or format as needed for Binance.
            # MotherAI might send "BTCUSDT", agent might use "BTC". Standardize here.
            if symbol.endswith("USDT"):
                processed_symbol = symbol[:-4] # e.g., BTC from BTCUSDT
            elif symbol.endswith("USD"): # Or other quote currencies if logic expands
                 processed_symbol = symbol[:-3]
            else:
                processed_symbol = symbol # Assume it's already base, e.g. BTC

            res = process_trade_decision(processed_symbol, action, price, confidence)
            trade_results.append(res)
        else:
            logger.warning(f"Invalid trade decision skipped: {d}")

    set_last_execution_time()
    return trade_results

# --- Get Portfolio State ---
def get_mock_portfolio():
    return load_mock_balance()
