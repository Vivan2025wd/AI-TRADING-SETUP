import os
import json
import time
import shutil
import numpy as np
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import pandas as pd

from backend.binance.binance_trader import place_market_order
from backend.mother_ai.cooldown import CooldownManager
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

# --- Defaults ---
DEFAULT_BALANCE = {
    "USD": 1000.0,
    "holdings": {}
}

DEFAULT_PROFIT_TRACKER = {
    "total_profit_usd": 0.0,
    "total_trades": 0
}


class TradeExecutor:
    """Handles trade execution, logging, and cooldown management"""
    
    def __init__(self, data_interval="1h", trade_history_dir="backend/storage/trade_history",
                 performance_log_dir="backend/storage/performance_logs"):
        """
        Initialize TradeExecutor
        
        Args:
            data_interval: Data interval being used
            trade_history_dir: Directory for trade history storage
            performance_log_dir: Directory for performance logs
        """
        self.data_interval = data_interval
        self.trade_history_dir = trade_history_dir
        self.performance_log_dir = performance_log_dir
        self.cooldown_manager = CooldownManager()
        
        # Ensure directories exist
        os.makedirs(trade_history_dir, exist_ok=True)
        os.makedirs(performance_log_dir, exist_ok=True)
        os.makedirs(STORAGE_DIR, exist_ok=True)
        os.makedirs(TRADE_PROFITS_DIR, exist_ok=True)
        
        print(f"🚀 TradeExecutor initialized:")
        print(f"   Data interval: {data_interval}")
        print(f"   Trade history dir: {trade_history_dir}")
        print(f"   Performance log dir: {performance_log_dir}")

    def execute_trade(self, symbol: str, signal: str, price: float, confidence: float, 
                     risk_manager=None, current_positions=None, live=False) -> Optional[float]:
        """
        Execute a trade with comprehensive risk management
        
        Args:
            symbol: Trading symbol
            signal: "buy" or "sell"
            price: Current price
            confidence: Trade confidence
            risk_manager: Risk manager instance for validation
            current_positions: Current portfolio positions
            live: Whether to execute live trades
            
        Returns:
            Executed quantity or None if failed
        """
        if signal not in ("buy", "sell") or not price:
            print(f"❌ Invalid trade parameters: signal={signal}, price={price}")
            return None

        print(f"🚀 Executing {signal.upper()} order for {symbol} at ${price:.4f}")
        print(f"   Confidence: {confidence:.3f}, Data interval: {self.data_interval}")

        # Check cooldown
        if self.cooldown_manager.is_in_cooldown(symbol):
            cooldown_remaining = self.cooldown_manager.get_cooldown_remaining(symbol)
            print(f"❌ {symbol} is in cooldown for {cooldown_remaining:.0f}s")
            return None

        try:
            # Get trading balance
            balance = self._get_trading_balance()
            
            # Risk management validation if available
            if risk_manager and current_positions is not None:
                should_trade, reason, trade_params = self._validate_with_risk_manager(
                    symbol, signal, price, balance, risk_manager, current_positions
                )
                
                if not should_trade:
                    print(f"❌ Risk Manager blocked trade: {reason}")
                    return None
                
                print(f"✅ Risk Manager approved: {reason}")
                qty = trade_params.get("recommended_quantity", 0)
            else:
                # Fallback quantity calculation
                qty = self._calculate_fallback_quantity(symbol, signal, price, balance)

            if qty <= 0:
                print(f"❌ Invalid quantity calculated: {qty}")
                return None

            # Execute the order
            binance_symbol = symbol if "/" in symbol else symbol.replace("USDT", "/USDT")
            executed_qty = self._execute_order(binance_symbol, signal, qty, price)

            if executed_qty:
                # Set cooldown after successful execution
                self.cooldown_manager.set_cooldown(symbol)
                print(f"✅ {signal.upper()} order executed for {symbol}, qty: {executed_qty:.6f}")
                return executed_qty
            else:
                print(f"❌ {signal.upper()} order failed for {symbol}")
                return None

        except Exception as e:
            print(f"❌ Trade execution error for {symbol}: {e}")
            return None

    def execute_mock_trade(self, symbol: str, action: str, price: float, confidence: float) -> Dict:
        """Execute mock trade with portfolio tracking"""
        balance = self.load_mock_balance()
        profit_tracker = self.load_profit_tracker()

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
            amount = self.calculate_trade_amount(symbol, price, usd_balance, RISK_CONFIG)
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
                    "symbol": symbol,
                    "status": "mock"
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
                self.save_profit_tracker(profit_tracker)

                result = {
                    "timestamp": timestamp,
                    "balance": round(balance["USD"], 2),
                    "price": price,
                    "type": "SELL",
                    "amount": amount,
                    "profit_percent": round(profit_percent, 2),
                    "profit_usd": round(profit_usd, 2),
                    "symbol": symbol,
                    "cumulative_profit": profit_tracker["total_profit_usd"],
                    "status": "mock"
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

        self.save_mock_balance(balance)
        self.append_json_log(perf_path, result)
        return result

    def execute_live_trade(self, symbol: str, action: str, price: float, confidence: float) -> dict:
        """Execute live trade via Binance API"""
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

    def execute_mother_ai_decision(self, decision_data: Dict) -> List[Dict]:
        """Execute Mother AI trades with symbol-specific cooldown"""
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
            last_exec = self.get_symbol_last_execution_time(symbol)
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
            mode = get_trading_mode()
            if mode == "live":
                res = self.execute_live_trade(symbol, action, price, confidence)
            else:
                res = self.execute_mock_trade(symbol, action, price, confidence)
            
            trade_results.append(res)
            self.set_symbol_last_execution_time(symbol)

        return trade_results

    def calculate_trade_amount(self, symbol: str, price: float, usd_balance: float, config: Dict) -> float:
        """Calculate trade amount dynamically based on risk settings, asset price, and balance."""
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

    def _get_trading_balance(self) -> float:
        """Get current trading balance"""
        try:
            current_mode = get_trading_mode()
            
            if current_mode == "live":
                try:
                    client = get_binance_client()
                    account_info = client.get_account()
                    balances = {b['asset']: float(b['free']) for b in account_info['balances']}
                    usdt_balance = balances.get('USDT', 0)
                    if usdt_balance > 0:
                        print(f"💰 Using live USDT balance: ${usdt_balance:.2f}")
                        return usdt_balance
                except Exception as e:
                    print(f"⚠️ Could not get live balance: {e}")
        except ImportError:
            pass
        
        # Default balance
        mock_balance = self.load_mock_balance()
        balance = mock_balance.get("USD", 1000.0)
        print(f"💰 Using mock balance: ${balance:.2f}")
        return balance

    def _validate_with_risk_manager(self, symbol: str, signal: str, price: float, 
                                  balance: float, risk_manager, current_positions: Dict):
        """Validate trade with risk manager"""
        try:
            # Get market data for risk calculations
            from backend.binance.fetch_live_ohlcv import fetch_ohlcv
            symbol_formatted = symbol if symbol.endswith("/USDT") else symbol.replace("USDT", "") + "/USDT"
            df = fetch_ohlcv(symbol_formatted, self.data_interval, 100)
            
            if df is None or df.empty:
                return False, "no_market_data", {}
            
            is_long = (signal == "buy")
            return risk_manager.should_enter_trade(symbol, price, balance, df, current_positions, is_long)
            
        except Exception as e:
            print(f"⚠️ Risk manager validation failed: {e}")
            return True, "risk_manager_unavailable", {}  # Allow trade if risk manager fails

    def _calculate_fallback_quantity(self, symbol: str, signal: str, price: float, balance: float) -> float:
        """Calculate fallback quantity when risk manager is unavailable"""
        if signal == "buy":
            # Use 10% of balance for buy orders
            position_value = balance * 0.1
            return position_value / price
        elif signal == "sell":
            # For sell orders, try to get actual holdings
            try:
                current_mode = get_trading_mode()
                
                if current_mode == "live":
                    client = get_binance_client()
                    account_info = client.get_account()
                    balances = {b['asset']: float(b['free']) for b in account_info['balances']}
                    base_asset = symbol.replace("USDT", "")
                    return balances.get(base_asset, 0)
                else:
                    # Mock quantity for paper trading - get from mock balance
                    mock_balance = self.load_mock_balance()
                    holdings = mock_balance.get("holdings", {})
                    holding = holdings.get(symbol.upper(), {})
                    return holding.get("amount", 0)
            except:
                return balance * 0.1 / price
        
        return 0

    def _execute_order(self, binance_symbol: str, signal: str, qty: float, price: float) -> Optional[float]:
        """Execute the actual order"""
        print(f"📊 Order details: {signal.upper()} {qty:.6f} {binance_symbol} at ${price:.4f}")
        
        order = place_market_order(binance_symbol, signal, qty)
        
        if order:
            if order.get("status") == "mock":
                print(f"✅ MOCK {signal.upper()} order logged")
                return qty
            else:
                executed_qty = order.get("executedQty", qty)
                print(f"✅ LIVE {signal.upper()} order executed")
                return executed_qty
        
        return None

    # --- Symbol Cooldown Logic ---
    def load_symbol_execution_times(self) -> Dict[str, str]:
        if os.path.exists(SYMBOL_EXECUTION_FILE):
            with open(SYMBOL_EXECUTION_FILE, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def save_symbol_execution_times(self, data: Dict[str, str]):
        with open(SYMBOL_EXECUTION_FILE, 'w') as f:
            json.dump(data, f, indent=4)

    def get_symbol_last_execution_time(self, symbol: str) -> Optional[datetime]:
        data = self.load_symbol_execution_times()
        ts = data.get(symbol.upper())
        return datetime.fromisoformat(ts) if ts else None

    def set_symbol_last_execution_time(self, symbol: str):
        data = self.load_symbol_execution_times()
        data[symbol.upper()] = datetime.utcnow().isoformat()
        self.save_symbol_execution_times(data)

    # --- Helper Methods ---
    def load_mock_balance(self) -> Dict:
        if not os.path.exists(MOCK_BALANCE_FILE):
            self.save_mock_balance(DEFAULT_BALANCE)
        with open(MOCK_BALANCE_FILE, 'r') as f:
            return json.load(f)

    def save_mock_balance(self, data: Dict):
        with open(MOCK_BALANCE_FILE, 'w') as f:
            json.dump(data, f, indent=4)

    def append_json_log(self, file_path: str, record: Dict):
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

    def load_profit_tracker(self) -> Dict:
        if not os.path.exists(MOCK_PROFIT_FILE):
            self.save_profit_tracker(DEFAULT_PROFIT_TRACKER)
        with open(MOCK_PROFIT_FILE, 'r') as f:
            return json.load(f)

    def save_profit_tracker(self, data: Dict):
        with open(MOCK_PROFIT_FILE, 'w') as f:
            json.dump(data, f, indent=4)

    def get_mock_portfolio(self):
        return self.load_mock_balance()

    def log_trade_execution(self, symbol: str, signal: str, price: float, confidence: float,
                          score: float, timestamp: str, qty: Optional[float] = None,
                          source: str = "trade_executor", risk_info: Optional[Dict] = None):
        """
        Log trade execution with comprehensive data
        
        Args:
            symbol: Trading symbol
            signal: "buy" or "sell"
            price: Execution price
            confidence: Trade confidence
            score: Trade score
            timestamp: Execution timestamp
            qty: Executed quantity
            source: Source of the trade decision
            risk_info: Additional risk information
        """
        path = os.path.join(self.performance_log_dir, f"{symbol}_trades.json")
        
        # Sanitize all data for JSON serialization
        entry = {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(float(confidence), 4),
            "score": round(float(score), 4),
            "last_price": round(float(price), 4),
            "price": round(float(price), 4),
            "qty": round(float(qty), 6) if qty is not None else None,
            "timestamp": timestamp,
            "source": source,
            "data_interval": self.data_interval,
            "risk_info": self._sanitize_for_json(risk_info or {})
        }

        # Load existing history with robust error handling
        history = self._load_trade_history(path)
        
        # Add new entry
        history.append(entry)

        # Write with atomic operation to prevent corruption
        self._save_trade_history(path, history)
        
        print(f"✅ Trade logged successfully for {symbol}")

    def _sanitize_for_json(self, data: Any) -> Any:
        """Recursively sanitize data for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_for_json(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.bool_, bool)):
            return bool(data)
        elif hasattr(data, 'item'):  # numpy scalar types
            return data.item()
        elif isinstance(data, (datetime, pd.Timestamp)):
            return data.isoformat()
        elif data is None:
            return None
        elif isinstance(data, (str, int, float)):
            return data
        else:
            # For any other type, try to convert to appropriate JSON type
            try:
                return float(data)  # Try numeric conversion first
            except (ValueError, TypeError):
                return str(data)  # Fall back to string

    def _load_trade_history(self, path: str) -> list:
        """Load trade history with robust error handling"""
        history = []
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        history = json.loads(content)
                        if not isinstance(history, list):
                            print(f"⚠️ Trade history file corrupted, resetting to empty list")
                            history = []
                    else:
                        print(f"⚠️ Empty trade history file, starting fresh")
                        history = []
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode error in trade history: {e}")
                self._backup_corrupted_file(path)
                history = []
            except Exception as e:
                print(f"⚠️ Unexpected error loading trade history: {e}")
                history = []
        
        return history

    def _save_trade_history(self, path: str, history: list):
        """Save trade history with atomic write to prevent corruption"""
        temp_path = path + ".tmp"
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Write to temporary file first
            with open(temp_path, "w", encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            # Atomically move temp file to final location
            shutil.move(temp_path, path)
            
        except Exception as e:
            print(f"❌ Failed to write trade log: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def _json_serializer(self, obj):
        """Handle non-JSON serializable objects"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'item'):  # numpy scalar types
            return obj.item()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        else:
            return str(obj)

    def _backup_corrupted_file(self, path: str):
        """Backup corrupted file"""
        backup_path = path + f".backup_{int(time.time())}"
        try:
            shutil.copy2(path, backup_path)
            print(f"⚠️ Corrupted file backed up to: {backup_path}")
        except Exception as backup_error:
            print(f"⚠️ Could not backup corrupted file: {backup_error}")

    def get_cooldown_status(self) -> Dict:
        """Get cooldown status for all symbols"""
        return self.cooldown_manager.get_cooldown_status()

    def is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown"""
        return self.cooldown_manager.is_in_cooldown(symbol)

    def get_cooldown_remaining(self, symbol: str) -> float:
        """Get remaining cooldown time for symbol"""
        return self.cooldown_manager.get_cooldown_remaining(symbol)

    def clear_cooldown(self, symbol: str):
        """Clear cooldown for a symbol"""
        self.cooldown_manager.clear_cooldown(symbol)

    def get_trade_history(self, symbol: str, limit: Optional[int] = None) -> list:
        """Get trade history for a symbol"""
        path = os.path.join(self.performance_log_dir, f"{symbol}_trades.json")
        history = self._load_trade_history(path)
        
        if limit:
            return history[-limit:]
        return history

    def get_last_trade(self, symbol: str) -> Optional[Dict]:
        """Get the last trade for a symbol"""
        history = self.get_trade_history(symbol, limit=1)
        return history[0] if history else None

    def validate_agent_position(self, symbol: str, claimed_state: Optional[str]) -> Optional[str]:
        """Validate agent position state against trade history"""
        try:
            last_trade = self.get_last_trade(symbol)
            if not last_trade:
                return None
            
            last_signal = last_trade.get("signal", "")
            last_timestamp = last_trade.get("timestamp", "")
            
            if last_signal == "buy":
                correct_state = "long"
            elif last_signal == "sell":
                correct_state = None
            else:
                correct_state = None
            
            # Check if position is stale (older than 48 hours)
            if correct_state == "long" and self._is_position_stale(last_timestamp, 48):
                print(f"⚠️ {symbol} position appears stale, resetting to flat")
                correct_state = None
            
            return correct_state
            
        except Exception as e:
            print(f"⚠️ Failed to validate position for {symbol}: {e}")
            return None

    def _is_position_stale(self, timestamp_str: str, max_hours: int = 48) -> bool:
        """Check if a position is older than max_hours"""
        try:
            import dateutil.parser
            
            trade_time = dateutil.parser.parse(timestamp_str)
            now = datetime.now(trade_time.tzinfo) if trade_time.tzinfo else datetime.now()
            return (now - trade_time) > timedelta(hours=max_hours)
        except:
            return False

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old trade logs"""
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        for filename in os.listdir(self.performance_log_dir):
            if filename.endswith("_trades.json"):
                filepath = os.path.join(self.performance_log_dir, filename)
                try:
                    if os.path.getmtime(filepath) < cutoff_time:
                        backup_path = filepath + f".archived_{int(time.time())}"
                        shutil.move(filepath, backup_path)
                        print(f"📦 Archived old log: {filename}")
                except Exception as e:
                    print(f"⚠️ Could not archive {filename}: {e}")

    def get_execution_summary(self) -> Dict:
        """Get summary of trade execution statistics"""
        summary = {
            "data_interval": self.data_interval,
            "cooldown_manager": self.cooldown_manager.get_cooldown_status(),
            "recent_trades": {},
            "total_trades_today": 0
        }
        
        # Count recent trades
        today = datetime.now().date()
        
        for filename in os.listdir(self.performance_log_dir):
            if filename.endswith("_trades.json"):
                symbol = filename.replace("_trades.json", "")
                history = self.get_trade_history(symbol)
                
                today_trades = []
                for trade in history:
                    try:
                        trade_date = datetime.fromisoformat(trade.get("timestamp", "")).date()
                        if trade_date == today:
                            today_trades.append(trade)
                    except:
                        continue
                
                summary["recent_trades"][symbol] = {
                    "total_trades": len(history),
                    "today_trades": len(today_trades),
                    "last_trade": history[-1] if history else None
                }
                
                summary["total_trades_today"] += len(today_trades)
        
        return summary


# --- Standalone Functions for Backward Compatibility ---
def execute_trade(symbol: str, action: str, price: float, confidence: float) -> Dict:
    """Standalone function for backward compatibility"""
    mode = get_trading_mode()
    executor = TradeExecutor()
    
    if mode == "live":
        return executor.execute_live_trade(symbol, action, price, confidence)
    else:
        return executor.execute_mock_trade(symbol, action, price, confidence)

def execute_mother_ai_decision(decision_data: Dict) -> List[Dict]:
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.execute_mother_ai_decision(decision_data)

def get_mock_portfolio():
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.get_mock_portfolio()

def calculate_trade_amount(symbol: str, price: float, usd_balance: float, config: Dict) -> float:
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.calculate_trade_amount(symbol, price, usd_balance, config)

def load_symbol_execution_times() -> Dict[str, str]:
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.load_symbol_execution_times()

def save_symbol_execution_times(data: Dict[str, str]):
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.save_symbol_execution_times(data)

def get_symbol_last_execution_time(symbol: str) -> Optional[datetime]:
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.get_symbol_last_execution_time(symbol)

def set_symbol_last_execution_time(symbol: str):
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.set_symbol_last_execution_time(symbol)

def load_mock_balance() -> Dict:
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.load_mock_balance()

def save_mock_balance(data: Dict):
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.save_mock_balance(data)

def append_json_log(file_path: str, record: Dict):
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.append_json_log(file_path, record)

def load_profit_tracker() -> Dict:
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.load_profit_tracker()

def save_profit_tracker(data: Dict):
    """Standalone function for backward compatibility"""
    executor = TradeExecutor()
    return executor.save_profit_tracker(data)