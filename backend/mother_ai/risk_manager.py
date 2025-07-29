import os
import json
import time
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd

RISK_CONFIG_FILE = "backend/storage/risk_config.json"

# Enhanced Risk Configuration
DEFAULT_RISK_CONFIG = {
    "trade_cooldown_seconds": 15,        # Faster trading (was 60s)
    "max_hold_seconds": 43200,           # 12 hours hold (was 6h)
    "risk_per_trade": 10,              # 10% per trade (was 5%)
    "default_balance_usd": 100,          
    "tp_ratio": 15,                     # Bigger profit target (was 1.25x)
    "sl_percent": 10,                  # Wider Stop Loss (was 6%)
    "max_portfolio_exposure": 90,      # Allow 90% of balance in positions (was 60%)
    "max_daily_loss": 30,              # Can lose up to 30% daily (was 15%)
    "max_drawdown": 50,                # Tolerate 50% drawdown (was 35%)
    "max_concurrent_positions": 25,      # Can hold up to 25 positions at once (was 15)
    "max_correlation_exposure": 80,    # Loosen sector correlation limits (was 50%)
    "volatility_lookback": 20,           
    "volatility_multiplier": 2.5,        # Accept more volatile moves (was 2.0)
    "min_position_size": 0.01,           # Increase minimum position size (was 0.005)
    "max_position_size": 0.50,           # Allow up to 50% of balance in a single trade (was 25%)
    "emergency_stop_loss": 0.30,         # Hard stop at 30% loss per position (was 20%)
    "max_trades_per_hour": 50,           # Allow up to 50 trades per hour (was 20)
    "market_volatility_threshold": 0.20, # Allow trading in higher vol markets (was 0.12)
    "symbol_overrides": {}
}


class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config_file=RISK_CONFIG_FILE):
        self.config_file = config_file
        self.config = self._load_risk_config()
        self.daily_pnl = 0.0
        self.portfolio_value = self.config["default_balance_usd"]
        self.max_historical_value = self.portfolio_value
        self.hourly_trade_count = {}
        self.correlation_matrix = {}
        self.volatility_cache = {}
        
    def _load_risk_config(self):
        """Load risk configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in DEFAULT_RISK_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                self._save_risk_config(DEFAULT_RISK_CONFIG)
                return DEFAULT_RISK_CONFIG.copy()
        except Exception as e:
            print(f"⚠️ Error loading risk config: {e}, using defaults")
            return DEFAULT_RISK_CONFIG.copy()
    
    def _save_risk_config(self, config):
        """Save risk configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving risk config: {e}")
    
    def get_symbol_config(self, symbol: str) -> Dict:
        """Get risk configuration for specific symbol"""
        base_config = self.config.copy()
        symbol_overrides = self.config.get("symbol_overrides", {}).get(symbol, {})
        base_config.update(symbol_overrides)
        return base_config
    
    def calculate_dynamic_position_size(self, symbol: str, price: float, df: Optional[pd.DataFrame] = None) -> float:
        """Calculate position size based on volatility and risk parameters"""
        config = self.get_symbol_config(symbol)
        base_risk = config["risk_per_trade"]
        
        if df is not None and len(df) >= config["volatility_lookback"]:
            # Calculate volatility-adjusted position size
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(config["volatility_lookback"]).std().iloc[-1]
            
            if not np.isnan(volatility) and volatility > 0:
                # Inverse relationship: higher volatility = smaller position
                vol_adjustment = 1.0 / (1.0 + volatility * config["volatility_multiplier"])
                adjusted_risk = base_risk * vol_adjustment
                
                # Apply min/max constraints
                adjusted_risk = max(config["min_position_size"], 
                                  min(config["max_position_size"], adjusted_risk))
                
                print(f"📊 {symbol} volatility adjustment: {volatility:.4f} -> risk {base_risk:.3f} -> {adjusted_risk:.3f}")
                return adjusted_risk
        
        return base_risk
    
    def check_portfolio_limits(self, new_trade_symbol: str, new_trade_size: float, current_positions: Dict) -> Tuple[bool, str]:
        """Check if new trade violates portfolio-level limits"""
        
        # 1. Check maximum concurrent positions
        active_positions = len([p for p in current_positions.values() if p.get('position_state') == 'long'])
        if active_positions >= self.config["max_concurrent_positions"]:
            return False, f"max concurrent positions reached ({active_positions}/{self.config['max_concurrent_positions']})"
        
        # 2. Check total portfolio exposure
        current_exposure = sum(p.get('exposure', 0) for p in current_positions.values())
        total_exposure = current_exposure + new_trade_size
        
        if total_exposure > self.config["max_portfolio_exposure"]:
            return False, f"portfolio exposure limit ({total_exposure:.3f} > {self.config['max_portfolio_exposure']:.3f})"
        
        # 3. Check daily loss limit
        if self.daily_pnl < -self.config["max_daily_loss"] * self.portfolio_value:
            return False, f"daily loss limit exceeded ({self.daily_pnl:.2f})"
        
        # 4. Check maximum drawdown
        current_drawdown = (self.max_historical_value - self.portfolio_value) / self.max_historical_value
        if current_drawdown > self.config["max_drawdown"]:
            return False, f"maximum drawdown exceeded ({current_drawdown:.3f})"
        
        # 5. Check hourly trade limit
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        hourly_trades = self.hourly_trade_count.get(current_hour, 0)
        if hourly_trades >= self.config["max_trades_per_hour"]:
            return False, f"hourly trade limit reached ({hourly_trades}/{self.config['max_trades_per_hour']})"
        
        return True, "approved"
    
    def check_market_conditions(self, symbol: str, df) -> Tuple[bool, str]:
        """Check if market conditions are suitable for trading"""
        if df is None or len(df) < 2:
            return False, "insufficient market data"
        
        # Check market volatility
        recent_returns = df['close'].pct_change().dropna().tail(6)  # Last 6 hours
        if len(recent_returns) > 0:
            hourly_volatility = recent_returns.std()
            if hourly_volatility > self.config["market_volatility_threshold"]:
                return False, f"market too volatile ({hourly_volatility:.4f})"
        
        # Check for extreme price movements
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        if abs(price_change) > self.config["emergency_stop_loss"]:
            return False, f"extreme price movement detected ({price_change:.4f})"
        
        return True, "market conditions acceptable"
    
    def update_portfolio_metrics(self, current_positions: Dict):
        """Update portfolio-level metrics"""
        # Update portfolio value
        total_value = self.config["default_balance_usd"]
        for position in current_positions.values():
            if position.get('position_state') == 'long':
                total_value += position.get('unrealized_pnl', 0)
        
        self.portfolio_value = total_value
        self.max_historical_value = max(self.max_historical_value, self.portfolio_value)
        
        # Clean up old hourly trade counts
        current_time = datetime.now()
        expired_hours = [h for h in self.hourly_trade_count.keys() 
                        if datetime.strptime(h, "%Y-%m-%d-%H") < current_time - timedelta(hours=24)]
        for hour in expired_hours:
            del self.hourly_trade_count[hour]
    
    def log_trade_attempt(self):
        """Log trade attempt for rate limiting"""
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        self.hourly_trade_count[current_hour] = self.hourly_trade_count.get(current_hour, 0) + 1
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics summary"""
        current_drawdown = (self.max_historical_value - self.portfolio_value) / self.max_historical_value
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        
        return {
            "portfolio_value": self.portfolio_value,
            "max_historical_value": self.max_historical_value,
            "current_drawdown": current_drawdown,
            "daily_pnl": self.daily_pnl,
            "hourly_trades": self.hourly_trade_count.get(current_hour, 0),
            "drawdown_limit": self.config["max_drawdown"],
            "daily_loss_limit": self.config["max_daily_loss"] * self.portfolio_value,
            "max_portfolio_exposure": self.config["max_portfolio_exposure"]
        }
    
    def update_config(self, new_config: Dict):
        """Update risk management configuration"""
        self.config.update(new_config)
        self._save_risk_config(self.config)
    
    def get_risk_report(self, current_positions: Dict) -> Dict:
        """Generate comprehensive risk report"""
        self.update_portfolio_metrics(current_positions)
        
        # Calculate additional metrics
        active_positions = [p for p in current_positions.values() if p['position_state'] == 'long']
        total_exposure = sum(p['exposure'] for p in active_positions)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_metrics": self.get_risk_metrics(),
            "position_summary": {
                "total_positions": len(active_positions),
                "total_exposure": total_exposure,
                "max_allowed_exposure": self.config["max_portfolio_exposure"],
                "positions": current_positions
            },
            "risk_limits": {
                "max_drawdown": self.config["max_drawdown"],
                "max_daily_loss": self.config["max_daily_loss"],
                "max_concurrent_positions": self.config["max_concurrent_positions"],
                "max_trades_per_hour": self.config["max_trades_per_hour"]
            },
            "warnings": []
        }
        
        # Add warnings
        risk_metrics = report["portfolio_metrics"]
        if risk_metrics["current_drawdown"] > 0.15:
            report["warnings"].append("High drawdown detected")
        if risk_metrics["daily_pnl"] < -risk_metrics["daily_loss_limit"] * 0.8:
            report["warnings"].append("Approaching daily loss limit")
        if total_exposure > self.config["max_portfolio_exposure"] * 0.8:
            report["warnings"].append("High portfolio exposure")
        
        return report