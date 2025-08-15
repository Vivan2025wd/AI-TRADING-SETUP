import os
import json
import threading
from typing import Dict, Optional, Tuple
from datetime import datetime
from backend.utils.logger import logger
from backend.utils.binance_api import get_trading_mode, get_binance_client

class SharedBalanceManager:
    """
    Centralized balance management system shared across all trading components.
    Handles both mock and live trading balance tracking with thread safety.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
            
        self.storage_dir = "backend/storage"
        self.balance_file = os.path.join(self.storage_dir, "shared_balance.json")
        self.balance_history_file = os.path.join(self.storage_dir, "balance_history.json")
        
        # Ensure storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Default configuration
        self.default_config = {
            "initial_balance_usd": 1000.0,
            "last_updated": None,
            "balance_usd": 1000.0,
            "holdings": {},  # {symbol: {"amount": float, "entry_price": float, "entry_time": str}}
            "total_invested": 0.0,
            "total_realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "portfolio_value": 1000.0,
            "max_historical_balance": 1000.0,
            "trading_mode": "mock"
        }
        
        # Load or initialize balance
        self.balance_data = self._load_balance()
        self.initialized = True
        
        logger.info(f"🏦 SharedBalanceManager initialized - Balance: ${self.balance_data['balance_usd']:.2f}")
    
    def _load_balance(self) -> Dict:
        """Load balance from file or create default"""
        try:
            if os.path.exists(self.balance_file):
                with open(self.balance_file, 'r') as f:
                    data = json.load(f)
                
                # Merge with defaults for any missing keys
                for key, value in self.default_config.items():
                    if key not in data:
                        data[key] = value
                
                return data
            else:
                self._save_balance(self.default_config)
                return self.default_config.copy()
                
        except Exception as e:
            logger.error(f"Error loading balance: {e}, using defaults")
            return self.default_config.copy()
    
    def _save_balance(self, data: Dict):
        """Save balance to file with thread safety"""
        try:
            with self._lock:
                data["last_updated"] = datetime.now().isoformat()
                with open(self.balance_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                # Save to history for tracking
                history_entry = {
                    "timestamp": data["last_updated"],
                    "balance_usd": data["balance_usd"],
                    "portfolio_value": data["portfolio_value"],
                    "unrealized_pnl": data["unrealized_pnl"],
                    "total_realized_pnl": data["total_realized_pnl"],
                    "holdings_count": len(data["holdings"])
                }
                self._append_balance_history(history_entry)
                
        except Exception as e:
            logger.error(f"Error saving balance: {e}")
    
    def _append_balance_history(self, entry: Dict):
        """Append balance snapshot to history"""
        try:
            history = []
            if os.path.exists(self.balance_history_file):
                with open(self.balance_history_file, 'r') as f:
                    history = json.load(f)
            
            history.append(entry)
            
            # Keep only last 1000 entries
            if len(history) > 1000:
                history = history[-1000:]
            
            with open(self.balance_history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving balance history: {e}")
    
    def get_balance(self) -> Dict:
        """Get current balance information"""
        with self._lock:
            # Update unrealized PnL for current holdings
            self._update_unrealized_pnl()
            return self.balance_data.copy()
    
    def get_usd_balance(self) -> float:
        """Get available USD balance"""
        return self.balance_data["balance_usd"]
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value (USD + holdings value)"""
        self._update_unrealized_pnl()
        return self.balance_data["portfolio_value"]
    
    def get_holdings(self) -> Dict:
        """Get current holdings"""
        return self.balance_data["holdings"].copy()
    
    def get_holding(self, symbol: str) -> Optional[Dict]:
        """Get holding information for specific symbol"""
        return self.balance_data["holdings"].get(symbol.upper())
    
    def has_sufficient_balance(self, required_usd: float) -> bool:
        """Check if sufficient USD balance is available"""
        return self.balance_data["balance_usd"] >= required_usd
    
    def has_holding(self, symbol: str, required_amount: float = 0) -> bool:
        """Check if sufficient holding amount is available"""
        holding = self.get_holding(symbol)
        if not holding:
            return False
        return holding["amount"] >= required_amount
    
    def execute_buy(self, symbol: str, amount: float, price: float, 
                   trade_data: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        Execute buy order and update balance
        Returns: (success, result_data)
        """
        symbol = symbol.upper()
        cost = amount * price
        
        with self._lock:
            # Check sufficient balance
            if not self.has_sufficient_balance(cost):
                return False, {
                    "error": "insufficient_balance",
                    "required": cost,
                    "available": self.balance_data["balance_usd"]
                }
            
            # Update balance
            self.balance_data["balance_usd"] -= cost
            self.balance_data["total_invested"] += cost
            
            # Update holdings
            current_holding = self.balance_data["holdings"].get(symbol, {
                "amount": 0,
                "entry_price": 0,
                "entry_time": datetime.now().isoformat()
            })
            
            # Calculate new average entry price
            total_amount = current_holding["amount"] + amount
            if current_holding["amount"] > 0:
                total_cost = (current_holding["amount"] * current_holding["entry_price"]) + cost
                new_entry_price = total_cost / total_amount
            else:
                new_entry_price = price
            
            self.balance_data["holdings"][symbol] = {
                "amount": round(total_amount, 8),
                "entry_price": round(new_entry_price, 6),
                "entry_time": current_holding["entry_time"]
            }
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            self._save_balance(self.balance_data)
            
            result = {
                "symbol": symbol,
                "action": "buy",
                "amount": amount,
                "price": price,
                "cost": cost,
                "new_balance": self.balance_data["balance_usd"],
                "portfolio_value": self.balance_data["portfolio_value"],
                "holding": self.balance_data["holdings"][symbol].copy()
            }
            
            logger.info(f"💰 BUY executed: {amount} {symbol} at ${price:.6f} (Cost: ${cost:.2f})")
            return True, result
    
    def execute_sell(self, symbol: str, amount: float, price: float,
                    trade_data: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        Execute sell order and update balance
        Returns: (success, result_data)
        """
        symbol = symbol.upper()
        
        with self._lock:
            # Check if we have the holding
            if not self.has_holding(symbol, amount):
                current_amount = self.get_holding(symbol)
                current_amount = current_amount["amount"] if current_amount else 0
                return False, {
                    "error": "insufficient_holding",
                    "required": amount,
                    "available": current_amount
                }
            
            holding = self.balance_data["holdings"][symbol]
            revenue = amount * price
            cost_basis = amount * holding["entry_price"]
            realized_pnl = revenue - cost_basis
            
            # Update balance
            self.balance_data["balance_usd"] += revenue
            self.balance_data["total_realized_pnl"] += realized_pnl
            
            # Update holding
            remaining_amount = holding["amount"] - amount
            if remaining_amount <= 0.00000001:  # Practically zero
                del self.balance_data["holdings"][symbol]
            else:
                self.balance_data["holdings"][symbol]["amount"] = round(remaining_amount, 8)
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            self._save_balance(self.balance_data)
            
            result = {
                "symbol": symbol,
                "action": "sell",
                "amount": amount,
                "price": price,
                "revenue": revenue,
                "cost_basis": cost_basis,
                "realized_pnl": realized_pnl,
                "pnl_percentage": (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0,
                "new_balance": self.balance_data["balance_usd"],
                "portfolio_value": self.balance_data["portfolio_value"],
                "remaining_holding": remaining_amount
            }
            
            logger.info(f"💰 SELL executed: {amount} {symbol} at ${price:.6f} "
                       f"(Revenue: ${revenue:.2f}, P&L: ${realized_pnl:.2f})")
            return True, result
    
    def _update_unrealized_pnl(self):
        """Update unrealized P&L for current holdings"""
        # For mock trading, we'd need current prices
        # For live trading, we can fetch from Binance
        total_unrealized = 0.0
        
        try:
            if get_trading_mode() == "live" and self.balance_data["holdings"]:
                client = get_binance_client()
                for symbol, holding in self.balance_data["holdings"].items():
                    try:
                        ticker = client.get_symbol_ticker(symbol=f"{symbol}USDT")
                        current_price = float(ticker['price'])
                        current_value = holding["amount"] * current_price
                        cost_basis = holding["amount"] * holding["entry_price"]
                        unrealized = current_value - cost_basis
                        total_unrealized += unrealized
                    except:
                        continue
            
            self.balance_data["unrealized_pnl"] = total_unrealized
            
        except Exception as e:
            logger.warning(f"Could not update unrealized PnL: {e}")
            self.balance_data["unrealized_pnl"] = 0.0
    
    def _update_portfolio_metrics(self):
        """Update portfolio value and other metrics"""
        holdings_value = 0.0
        
        # For holdings value, use entry price in mock mode
        for holding in self.balance_data["holdings"].values():
            holdings_value += holding["amount"] * holding["entry_price"]
        
        self.balance_data["portfolio_value"] = self.balance_data["balance_usd"] + holdings_value
        self.balance_data["max_historical_balance"] = max(
            self.balance_data["max_historical_balance"],
            self.balance_data["portfolio_value"]
        )
    
    def reset_balance(self, new_balance: Optional[float] = None):
        """Reset balance to initial or specified amount"""
        if new_balance is None:
            new_balance = self.default_config["initial_balance_usd"]
        
        with self._lock:
            self.balance_data = self.default_config.copy()
            self.balance_data["balance_usd"] = new_balance
            self.balance_data["portfolio_value"] = new_balance
            self.balance_data["max_historical_balance"] = new_balance
            self._save_balance(self.balance_data)
            
        logger.info(f"🏦 Balance reset to ${new_balance:.2f}")
    
    def get_balance_history(self, days: int = 7) -> list:
        """Get balance history for specified days"""
        try:
            if os.path.exists(self.balance_history_file):
                with open(self.balance_history_file, 'r') as f:
                    history = json.load(f)
                
                # Filter by days if needed
                if days > 0:
                    cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
                    
                    filtered_history = []
                    for entry in history:
                        entry_date = datetime.fromisoformat(entry["timestamp"])
                        if entry_date >= cutoff_date:
                            filtered_history.append(entry)
                    
                    return filtered_history
                
                return history
            else:
                return []
        except:
            return []
    
    def get_risk_metrics(self) -> Dict:
        """Get risk metrics based on current balance state"""
        current_balance = self.get_portfolio_value()
        max_balance = self.balance_data["max_historical_balance"]
        
        drawdown = (max_balance - current_balance) / max_balance if max_balance > 0 else 0
        
        return {
            "portfolio_value": current_balance,
            "max_historical_value": max_balance,
            "current_drawdown": drawdown,
            "total_realized_pnl": self.balance_data["total_realized_pnl"],
            "unrealized_pnl": self.balance_data["unrealized_pnl"],
            "balance_usd": self.balance_data["balance_usd"],
            "total_invested": self.balance_data["total_invested"],
            "active_positions": len(self.balance_data["holdings"])
        }
    
    def export_balance_report(self) -> Dict:
        """Export comprehensive balance report"""
        self._update_unrealized_pnl()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "balance_summary": self.get_balance(),
            "risk_metrics": self.get_risk_metrics(),
            "recent_history": self.get_balance_history(7),
            "holdings_detail": [
                {
                    "symbol": symbol,
                    "amount": holding["amount"],
                    "entry_price": holding["entry_price"],
                    "current_value": holding["amount"] * holding["entry_price"],
                    "entry_time": holding["entry_time"]
                }
                for symbol, holding in self.balance_data["holdings"].items()
            ]
        }


# Singleton instance for global access
balance_manager = SharedBalanceManager()

# Convenience functions for easy access
def get_shared_balance() -> float:
    """Get current USD balance"""
    return balance_manager.get_usd_balance()

def get_portfolio_value() -> float:
    """Get total portfolio value"""
    return balance_manager.get_portfolio_value()

def execute_shared_buy(symbol: str, amount: float, price: float) -> Tuple[bool, Dict]:
    """Execute buy through shared balance manager"""
    return balance_manager.execute_buy(symbol, amount, price)

def execute_shared_sell(symbol: str, amount: float, price: float) -> Tuple[bool, Dict]:
    """Execute sell through shared balance manager"""
    return balance_manager.execute_sell(symbol, amount, price)

def get_shared_holding(symbol: str) -> Optional[Dict]:
    """Get holding information for symbol"""
    return balance_manager.get_holding(symbol)

def get_balance_report() -> Dict:
    """Get comprehensive balance report"""
    return balance_manager.export_balance_report()