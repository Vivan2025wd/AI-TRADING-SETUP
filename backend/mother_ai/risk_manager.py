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
  "trade_cooldown_seconds": 60,
  "max_hold_seconds": 7200,
  "risk_per_trade": 0.02,
  "default_balance_usd": 10000,
  "tp_ratio": 1.8,
  "sl_percent": 1.5,
  "max_portfolio_exposure": 30,
  "max_daily_loss": 2,
  "max_drawdown": 5,
  "max_concurrent_positions": 5,
  "max_correlation_exposure": 30,
  "volatility_lookback": 14,
  "volatility_multiplier": 1.2,
  "min_position_size": 0.5,
  "max_position_size": 3.0,
  "emergency_stop_loss": 1.0,
  "max_trades_per_hour": 15,
  "market_volatility_threshold": 0.05,
  "exit_volatility_threshold": 0.06,
  "symbol_overrides": {
    "BTCUSDT": {
      "risk_per_trade": 0.1,
      "tp_ratio": 1.5,
      "sl_percent": 0.5,
      "max_hold_seconds": 5400
    },
    "ETHUSDT": {
      "risk_per_trade": 0.1,
      "tp_ratio": 2.2,
      "sl_percent": 1.4,
      "max_hold_seconds": 7200
    },
    "ADAUSDT": {
      "risk_per_trade": 0.1,
      "tp_ratio": 2.5,
      "sl_percent": 1.6,
      "max_hold_seconds": 9000
    }
  }
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
    
    def calculate_stop_loss_and_take_profit(self, entry_price: float, position_size_percent: float, is_long: bool = True) -> Tuple[float, float]:
        """Calculate proper stop loss and take profit levels"""
        config = self.config
        
        # Calculate stop loss distance as percentage of entry price
        sl_distance_percent = config["sl_percent"] / 100.0  # Convert to decimal
        tp_distance_percent = sl_distance_percent * config["tp_ratio"]  # TP is ratio of SL distance
        
        if is_long:
            # For long positions
            stop_loss = entry_price * (1 - sl_distance_percent)
            take_profit = entry_price * (1 + tp_distance_percent)
        else:
            # For short positions
            stop_loss = entry_price * (1 + sl_distance_percent)
            take_profit = entry_price * (1 - tp_distance_percent)
        
        return stop_loss, take_profit
    
    def calculate_position_quantity_from_risk(self, entry_price: float, stop_loss: float, risk_amount: float) -> float:
        """Calculate position quantity based on risk amount and stop loss distance (Kelly/Fixed Risk method)"""
        # Risk per share = difference between entry and stop loss
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0.0
        
        # Quantity = Total risk amount / Risk per share
        quantity = risk_amount / risk_per_share
        return quantity
    
    def calculate_position_quantity_from_percent(self, entry_price: float, position_size_percent: float, balance: float) -> float:
        """Calculate position quantity based on percentage of balance (Fixed Fractional method)"""
        # Total position value = percentage of balance
        position_value = (position_size_percent / 100.0) * balance
        
        # Quantity = Position value / Entry price
        quantity = position_value / entry_price
        return quantity
    
    def calculate_trade_parameters(self, symbol: str, entry_price: float, balance: float, df: Optional[pd.DataFrame] = None) -> Dict:
        """Calculate all trade parameters: position size, quantity, SL, TP"""
        
        # Get dynamic position size percentage
        position_size_percent = self.calculate_dynamic_position_size(symbol, entry_price, df)
        
        # Calculate stop loss and take profit prices
        stop_loss, take_profit = self.calculate_stop_loss_and_take_profit(entry_price, position_size_percent)
        
        # Method 1: Fixed Fractional (what your system currently uses)
        # Position size is percentage of total balance
        quantity_fractional = self.calculate_position_quantity_from_percent(entry_price, position_size_percent, balance)
        position_value_fractional = quantity_fractional * entry_price
        
        # Method 2: Kelly/Fixed Risk (more sophisticated)
        # Risk amount is percentage of balance, position size varies based on SL distance
        risk_amount = (position_size_percent / 100.0) * balance
        quantity_kelly = self.calculate_position_quantity_from_risk(entry_price, stop_loss, risk_amount)
        position_value_kelly = quantity_kelly * entry_price
        
        # Calculate actual risk for each method
        risk_per_share = abs(entry_price - stop_loss)
        actual_risk_fractional = quantity_fractional * risk_per_share
        actual_risk_kelly = quantity_kelly * risk_per_share
        
        return {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size_percent": position_size_percent,
            
            # Fixed Fractional Method (Current System)
            "quantity_fractional": quantity_fractional,
            "position_value_fractional": position_value_fractional,
            "actual_risk_fractional": actual_risk_fractional,
            "risk_percent_fractional": (actual_risk_fractional / balance) * 100,
            
            # Kelly/Fixed Risk Method (Better)
            "quantity_kelly": quantity_kelly,
            "position_value_kelly": position_value_kelly,
            "actual_risk_kelly": actual_risk_kelly,
            "risk_percent_kelly": (actual_risk_kelly / balance) * 100,
            "target_risk_amount": risk_amount,
            
            # Recommendations
            "recommended_method": "kelly" if position_value_kelly < position_value_fractional * 2 else "fractional",
            "recommended_quantity": quantity_kelly if position_value_kelly < position_value_fractional * 2 else quantity_fractional
        }
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
        daily_loss_threshold = -self.config["max_daily_loss"] / 100.0 * self.portfolio_value
        if self.daily_pnl < daily_loss_threshold:
            return False, f"daily loss limit exceeded ({self.daily_pnl:.2f})"
        
        # 4. Check maximum drawdown
        current_drawdown = (self.max_historical_value - self.portfolio_value) / self.max_historical_value
        if current_drawdown > self.config["max_drawdown"] / 100.0:
            return False, f"maximum drawdown exceeded ({current_drawdown:.3f})"
        
        # 5. Check hourly trade limit
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        hourly_trades = self.hourly_trade_count.get(current_hour, 0)
        if hourly_trades >= self.config["max_trades_per_hour"]:
            return False, f"hourly trade limit reached ({hourly_trades}/{self.config['max_trades_per_hour']})"
        
        return True, "approved"
    
    def check_market_conditions(self, symbol: str, df) -> Tuple[bool, str]:
        """
        Check if market conditions are suitable for trading with adaptive thresholds.
        Allows volatility and breakout overrides if trend indicators align.
        """
        if df is None or len(df) < 20:  # Ensure enough data for trend calculations
            return False, "insufficient market data"
    
    # Compute recent returns (volatility check)
        recent_returns = df['close'].pct_change().dropna().tail(6)
        hourly_volatility = recent_returns.std() if len(recent_returns) > 0 else 0.0

    # Calculate trend strength (simple EMA slope)
        ema_short = df['close'].ewm(span=10).mean()
        ema_long = df['close'].ewm(span=20).mean()
        trend_strength = ema_short.iloc[-1] - ema_long.iloc[-1]

        # Adaptive volatility threshold
        volatility_threshold = self.config["market_volatility_threshold"]
        if abs(trend_strength) > df['close'].iloc[-1] * 0.001:  # If trending
            volatility_threshold *= 1.5  # Loosen threshold by 50% in trending markets

        if hourly_volatility > volatility_threshold:
            return False, f"market too volatile ({hourly_volatility:.4f}), threshold: {volatility_threshold:.4f}"

        # Extreme price movement detection (breakout check)
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        emergency_stop_loss = self.config["emergency_stop_loss"] / 100.0

        # Allow breakout if trend supports it
        if abs(price_change) > emergency_stop_loss:
            if abs(trend_strength) > df['close'].iloc[-1] * 0.002:  # Strong trend override
                return True, f"breakout detected but trend strong ({price_change:.4f})"
            else:
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
            "drawdown_limit": self.config["max_drawdown"] / 100.0,
            "daily_loss_limit": self.config["max_daily_loss"] / 100.0 * self.portfolio_value,
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
        if risk_metrics["current_drawdown"] > 0.10:  # Warn at 10% drawdown
            report["warnings"].append("High drawdown detected")
        if risk_metrics["daily_pnl"] < -risk_metrics["daily_loss_limit"] * 0.8:
            report["warnings"].append("Approaching daily loss limit")
        if total_exposure > self.config["max_portfolio_exposure"] * 0.8:
            report["warnings"].append("High portfolio exposure")
        
        return report
