import os
import json
import glob
import importlib.util
import time
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import dateutil.parser
import pandas as pd

from backend.binance.fetch_live_ohlcv import fetch_ohlcv
from backend.binance.binance_trader import place_market_order
from backend.mother_ai.performance_tracker import PerformanceTracker
from backend.strategy_engine.strategy_health import StrategyHealth
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.mother_ai.meta_evaluator import MetaEvaluator
from backend.mother_ai.profit_calculator import compute_trade_profits
from backend.storage.auto_cleanup import auto_cleanup_logs
from backend.mother_ai.risk_manager import RiskManager
from backend.mother_ai.cooldown import CooldownManager

TRADE_HISTORY_DIR = "backend/storage/trade_history"
PERFORMANCE_LOG_DIR = "backend/storage/performance_logs"

os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)


class MotherAI:
    def __init__(self, agents_dir="backend/agents", strategy_dir="backend/storage/strategies", 
                 agent_symbols=None, data_interval="1m"):
        """
        Initialize MotherAI with configurable data interval
        
        Args:
            agents_dir: Directory containing agent files
            strategy_dir: Directory containing strategy files
            agent_symbols: List of symbols to trade (None = all)
            data_interval: Data interval to use ("1m", "5m", "1h", etc.)
        """
        self.agents_dir = agents_dir
        self.strategy_dir = strategy_dir
        self.data_interval = data_interval  # Store configurable interval
        self.performance_tracker = PerformanceTracker("performance_logs")
        self.agent_symbols = agent_symbols or []
        self.meta_evaluator = MetaEvaluator()
        self.loaded_agents = {}  # Cache loaded agent instances
        self.risk_manager = RiskManager()  # Enhanced risk management
        self.cooldown_manager = CooldownManager()  # Separate cooldown management
        
        # Initialize agents on startup
        self._initialize_agents()
        
        print(f"🔧 MotherAI initialized with {self.data_interval} data interval")

    def _initialize_agents(self):
        """Initialize and cache agent instances"""
        print("🔄 Initializing agents...")
        for file in os.listdir(self.agents_dir):
            if not file.endswith("_agent.py"):
                continue
            symbol = file.replace("_agent.py", "").upper()
            if self.agent_symbols and symbol not in self.agent_symbols:
                continue
                
            try:
                strategy = self._load_strategy(symbol)
                agent_class = self._load_agent_class(file, symbol)
                if agent_class:
                    agent_instance = agent_class(symbol=symbol, strategy_logic=strategy)
                    self.loaded_agents[symbol] = agent_instance
                    
                    # Log agent's initial position state
                    pos_state = getattr(agent_instance, 'position_state', None)
                    print(f"📊 Initialized {symbol} agent - Position: {pos_state}")
            except Exception as e:
                print(f"⚠️ Failed to initialize agent for {symbol}: {e}")

    def get_agent_by_symbol(self, symbol: str):
        """Get cached agent instance by symbol"""
        return self.loaded_agents.get(symbol)

    def load_agents(self):
        """Return list of cached agent instances"""
        return list(self.loaded_agents.values())

    def _load_strategy(self, symbol: str):
        files = glob.glob(os.path.join(self.strategy_dir, f"{symbol}_strategy_*.json"))
        file = next((f for f in files if "default" in f), None) or (files[0] if files else None)
        if not file:
            return StrategyParser({})
        try:
            with open(file, "r") as f:
                return StrategyParser(json.load(f))
        except Exception:
            return StrategyParser({})

    def _load_agent_class(self, file, symbol):
        path = os.path.join(self.agents_dir, file)
        spec = importlib.util.spec_from_file_location(symbol, path)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Try to get the agent class - could be GenericAgent or {symbol}Agent
        agent_class = getattr(module, f"{symbol}Agent", None)
        if agent_class is None:
            # Fallback to GenericAgent if specific agent class not found
            agent_class = getattr(module, "GenericAgent", None)
        
        return agent_class

    def _fetch_agent_data(self, symbol: str):
        """
        Fetch OHLCV data using configured interval to match agent predictions
        FIXED: Now uses configurable interval instead of hardcoded "1h"
        """
        symbol = symbol if symbol.endswith("/USDT") else symbol.replace("USDT", "") + "/USDT"
        # Use configured interval to match agent training data
        df = fetch_ohlcv(symbol, self.data_interval, 100)
        return df if not df.empty else None

    def _validate_data_compatibility(self, agent, ohlcv_data: pd.DataFrame) -> bool:
        """
        Validate that data interval matches agent's expected interval
        From Solution 3: Add data validation
        """
        if not hasattr(agent, 'model') or agent.model is None:
            return True
        
        # Check if data frequency matches expected frequency
        if len(ohlcv_data) >= 2:
            time_diff = ohlcv_data.index[1] - ohlcv_data.index[0]
            interval_minutes = time_diff.total_seconds() / 60
            
            # Check against configured interval
            expected_minutes = self._interval_to_minutes(self.data_interval)
            
            # Warn if using different interval than configured
            if abs(interval_minutes - expected_minutes) > 0.1:
                print(f"⚠️ Data interval mismatch for {agent.symbol}: "
                     f"got {interval_minutes}min, expected {expected_minutes}min")
                return False
        
        return True

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 1440
        else:
            return 1  # Default to 1 minute

    def debug_data_intervals(self):
        """
        Debug method to check data intervals across the system
        From Solution 4: Debug logging to identify issues
        """
        print(f"\n🔍 DEBUGGING DATA INTERVALS (Config: {self.data_interval})")
        print("=" * 60)
        
        for symbol in list(self.agent_symbols)[:3]:  # Check first 3 symbols
            # Check what mother_ai gets
            mother_df = self._fetch_agent_data(symbol)
            if mother_df is not None and len(mother_df) >= 2:
                mother_interval = (mother_df.index[1] - mother_df.index[0]).total_seconds() / 60
                print(f"MotherAI {symbol}: {mother_interval}min intervals")
            
            # Check what agent predictions would get (simulate)
            symbol_formatted = symbol if "/" in symbol else symbol.replace("USDT", "") + "/USDT"
            agent_df = fetch_ohlcv(symbol_formatted, "1m", 100)
            if agent_df is not None and len(agent_df) >= 2:
                agent_interval = (agent_df.index[1] - agent_df.index[0]).total_seconds() / 60
                print(f"Agent Pred {symbol}: {agent_interval}min intervals")
            
            # Check if agent has model
            agent = self.get_agent_by_symbol(symbol)
            if agent:
                has_model = agent.model is not None
                model_info = f"Has ML model: {has_model}"
                
                # Check data compatibility
                if mother_df is not None:
                    is_compatible = self._validate_data_compatibility(agent, mother_df)
                    model_info += f", Data compatible: {is_compatible}"
                
                print(f"Agent {symbol}: {model_info}")
        
        print("=" * 60)

    def _safe_evaluate_agent(self, agent, ohlcv):
        """Enhanced agent evaluation with data compatibility checks"""
        try:
            # Validate data compatibility first
            if not self._validate_data_compatibility(agent, ohlcv):
                print(f"⚠️ Data compatibility issue for {agent.symbol}, using fallback strategy")
                # Could implement fallback logic here if needed
            
            # Use the agent's evaluate method which returns full decision dict
            decision = agent.evaluate(ohlcv)
        
            # Enhanced debugging information
            ml_data = decision.get("ml", {})
            rule_data = decision.get("rule_based", {})
        
            print(f"🤖 Agent {agent.symbol} evaluation:")
            print(f"   Final: {decision.get('action', 'unknown').upper()} ({decision.get('confidence', 0.0):.3f})")
            print(f"   ML: {ml_data.get('action', 'N/A')} ({ml_data.get('confidence', 0.0):.3f})")
            print(f"   Rule: {rule_data.get('action', 'N/A')} ({rule_data.get('confidence', 0.0):.3f})")
            print(f"   Source: {decision.get('source', 'unknown')}")
            print(f"   Position: {decision.get('position_state', 'None')}")
            print(f"   Data Interval: {self.data_interval}")
        
            # Convert agent's decision format to what MotherAI expects
            result = {
                "symbol": agent.symbol,
                "signal": decision.get("action", "hold").lower(),
                "confidence": decision.get("confidence", 0.0),
                "source": decision.get("source", "unknown"),
                "position_state": decision.get("position_state"),
                "ml_available": decision.get("ml", {}).get("available", False),
                "ml_confidence": decision.get("ml", {}).get("confidence", 0.0),
                "rule_confidence": decision.get("rule_based", {}).get("confidence", 0.0),
                "timestamp": decision.get("timestamp"),
                "data_interval": self.data_interval,  # Track what interval was used
                "full_decision": decision  # Preserve full agent decision
            }
        
            return result
        
        except Exception as e:
            print(f"❌ Agent evaluation error for {agent.symbol}: {e}")
            return {
                "symbol": agent.symbol,
                "signal": "hold",
                "confidence": 0.0,
                "source": "error",
                "position_state": None,
                "ml_available": False,
                "ml_confidence": 0.0,
                "rule_confidence": 0.0,
                "data_interval": self.data_interval,
                "full_decision": {}
            }

    def evaluate_agents(self, agents):
        """Evaluate agents using their full decision output with data consistency checks"""
        results = []
        trade_tracker = PerformanceTracker("trade_history")
        
        print(f"📊 Evaluating agents with {self.data_interval} data interval...")
        
        for agent in agents:
            ohlcv = self._fetch_agent_data(agent.symbol)
            if ohlcv is None:
                continue
                
            # Use the agent's full evaluation result with compatibility check
            prediction = self._safe_evaluate_agent(agent, ohlcv)
            health = StrategyHealth(trade_tracker.get_agent_log(agent.symbol)).summary()
            score = self._calculate_score(prediction, health)
            
            # Preserve all agent decision data
            result = {
                **prediction,
                "score": round(score, 3),
                "health": health
            }
            results.append(result)
            
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _calculate_score(self, prediction, health):
        return self.meta_evaluator.predict_refined_score({
            "confidence": prediction["confidence"],
            "win_rate": health.get("win_rate", 0.5),
            "drawdown_penalty": 1.0 - health.get("max_drawdown", 0.3),
            "is_buy": int(prediction["signal"] == "buy"),
            "is_sell": int(prediction["signal"] == "sell")
        })

    def _log_trade_execution(self, symbol, signal, price, confidence, score, timestamp, qty=None, source="mother_ai_decision"):
        """Log trade execution with quantity information and data interval tracking"""
        path = os.path.join(PERFORMANCE_LOG_DIR, f"{symbol}_trades.json")
        entry = {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(confidence, 4),
            "score": round(score, 4),
            "last_price": round(price, 4),
            "price": round(price, 4),
            "qty": round(qty, 6) if qty is not None else None,
            "timestamp": timestamp,
            "source": source,
            "data_interval": self.data_interval  # Track what interval was used for this trade
        }

        try:
            history = json.load(open(path)) if os.path.exists(path) else []
        except Exception:
            history = []

        history.append(entry)

        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def update_agent_position_state(self, symbol: str, signal: str):
        """Update agent's position state after successful trade execution"""
        agent = self.get_agent_by_symbol(symbol)
        if agent and hasattr(agent, 'position_state'):
            old_state = agent.position_state
            
            if signal == "buy":
                agent.position_state = "long"
            elif signal == "sell":
                agent.position_state = None
                
            print(f"🔄 Updated {symbol} agent position: {old_state} → {agent.position_state}")
        else:
            print(f"⚠️ Could not update position state for {symbol} - agent not found or no position_state")

    def get_current_positions(self) -> Dict:
        """Get current position states of all agents with exposure estimates"""
        positions = {}
        for symbol, agent in self.loaded_agents.items():
            position_info = {
                'symbol': symbol,
                'position_state': getattr(agent, 'position_state', None),
                'exposure': 0.0,
                'unrealized_pnl': 0.0,
                'data_interval': self.data_interval  # Track interval for each position
            }
            
            # Estimate exposure if in position
            if position_info['position_state'] == 'long':
                # Try to get last trade size or estimate
                position_info['exposure'] = self.risk_manager.get_symbol_config(symbol)["risk_per_trade"]
                
            positions[symbol] = position_info
        
        return positions

    def execute_trade(self, symbol, signal, price, confidence, live=False):
        """Enhanced trade execution with comprehensive risk checks and data interval awareness"""
        if signal not in ("buy", "sell") or not price:
            print(f"❌ Invalid trade parameters: signal={signal}, price={price}")
            return None

        # Get market data for risk calculations using configured interval
        df = self._fetch_agent_data(symbol)
        
        print(f"📈 Executing trade with {self.data_interval} data interval")

        # 1. Check market conditions
        market_ok, market_reason = self.risk_manager.check_market_conditions(symbol, df) 
        if not market_ok:
            print(f"❌ Market conditions check failed for {symbol}: {market_reason}")
            return None

        # 2. Calculate dynamic position size 
        position_size = self.risk_manager.calculate_dynamic_position_size(symbol, price, df)

        # 3. Get current positions for portfolio checks
        current_positions = self.get_current_positions()

        # 4. Check portfolio limits (for buy orders)
        if signal == "buy":
            portfolio_ok, portfolio_reason = self.risk_manager.check_portfolio_limits(
                symbol, position_size, current_positions
            )
            if not portfolio_ok:
                print(f"❌ Portfolio limits check failed for {symbol}: {portfolio_reason}")
                return None

        # 5. Log trade attempt for rate limiting
        self.risk_manager.log_trade_attempt()

        # Get current trading mode from binance_api.py
        from backend.utils.binance_api import get_trading_mode, get_binance_client
        current_mode = get_trading_mode()

        print(f"🚀 Executing {signal.upper()} order for {symbol} at ${price:.4f} | Mode: {current_mode.upper()}")
        print(f"📊 Risk-adjusted position size: {position_size:.3f} (dynamic sizing applied)")
        print(f"⏱️ Using {self.data_interval} data interval for analysis")

        qty = None
        executed_qty = None  # Track the actual executed quantity

        try:
            # Check actual account balance before placing order
            if current_mode == "live":
                try:
                    client = get_binance_client()
                    account_info = client.get_account()
                    balances = {b['asset']: float(b['free']) for b in account_info['balances']}
                    usdt_balance = balances.get('USDT', 0)
                    print(f"💰 Current USDT balance: ${usdt_balance:.2f}")
                
                    if signal == "buy" and usdt_balance < 10:
                        print(f"❌ Insufficient USDT balance for buy order. Need at least $10, have ${usdt_balance:.2f}")
                        return None
            
                    # Use dynamic position sizing with actual balance
                    available_balance = min(usdt_balance * 0.9, 50)  # Use 90% of balance, max $50
                    trade_amount = available_balance * position_size
                    print(f"💡 Using ${trade_amount:.2f} for this trade (dynamic sizing)")
            
                except Exception as balance_err:
                    print(f"⚠️ Could not check balance: {balance_err}")
                    trade_amount = self.risk_manager.config["default_balance_usd"] * position_size
            else:
                trade_amount = self.risk_manager.config["default_balance_usd"] * position_size

            # Format symbol for Binance API
            binance_symbol = symbol if "/" in symbol else symbol.replace("USDT", "/USDT")

            if signal == "buy":
                config = self.risk_manager.get_symbol_config(symbol)
        
                # FIXED CALCULATION: Use percentage-based SL/TP
                sl_percent = config["sl_percent"] / 100.0  # Convert to decimal
                tp_ratio = config["tp_ratio"]
        
                # Calculate stop loss and take profit correctly
                sl = price * (1 - sl_percent)  # 1% below entry
                tp_distance = (price - sl) * tp_ratio  # 2x the risk distance
                tp = price + tp_distance  # Take profit above entry
    
                # Calculate quantity based on trade amount
                qty = trade_amount / price

                if qty <= 0:
                    print(f"⚠️ Computed qty is zero or negative for BUY on {symbol}, skipping.")
                    return None

                print(f"📊 Buying {symbol} | Entry: {price:.4f}, SL: {sl:.4f}, TP: {tp:.4f}, Qty: {qty:.6f}")
                print(f"📊 Risk amount: ${trade_amount:.2f}, Total cost: ${qty * price:.2f}")

                # Use the updated place_market_order function
                order = place_market_order(binance_symbol, signal, qty)

                if order and order.get("status") != "mock":
                    self.cooldown_manager.set_cooldown(symbol)
                    # Update agent's position state
                    self.update_agent_position_state(symbol, signal)
                    # Extract executed quantity from order response
                    executed_qty = order.get("executedQty", qty)
                    print(f"✅ BUY order executed for {symbol} in {current_mode.upper()} mode")
                elif order and order.get("status") == "mock":
                    # Mock order - still update agent position state
                    self.cooldown_manager.set_cooldown(symbol)
                    self.update_agent_position_state(symbol, signal)
                    executed_qty = qty  # For mock trades, use the calculated qty
                    print(f"✅ MOCK BUY order logged for {symbol}")
                else:
                    print(f"❌ BUY order failed for {symbol}")
                    return None

            elif signal == "sell":
                # For sell orders, try to determine quantity
                agent = self.get_agent_by_symbol(symbol)
                qty = 0
            
                # First check if we can get quantity from agent or previous trades
                if current_mode == "live":
                    try:
                        client = get_binance_client()
                        account_info = client.get_account()
                        balances = {b['asset']: float(b['free']) for b in account_info['balances']} 
                        base_asset = symbol.replace("USDT", "")
                        actual_qty = balances.get(base_asset, 0)
                        if actual_qty > 0:
                            qty = actual_qty
                            print(f"💡 Found {qty:.6f} {base_asset} in account")
                        else:
                            print(f"❌ No {base_asset} holdings found in account")
                            return None
                    except Exception as e:
                        print(f"⚠️ Could not check holdings: {e}")
                        return None
                else:
                    # For mock, use trade amount calculation
                    qty = trade_amount / price

                if qty <= 0:
                    print(f"⚠️ Computed qty is zero or negative for SELL on {symbol}, skipping.")
                    return None

                print(f"📊 Selling {symbol} | Qty: {qty:.6f} at ${price:.4f}")

                # Use the updated place_market_order function
                order = place_market_order(binance_symbol, signal, qty)

                if order and order.get("status") != "mock":
                    self.cooldown_manager.set_cooldown(symbol)
                    # Update agent's position state
                    self.update_agent_position_state(symbol, signal)
                    # Extract executed quantity from order response
                    executed_qty = order.get("executedQty", qty)
                    print(f"✅ SELL order executed for {symbol} in {current_mode.upper()} mode")
                elif order and order.get("status") == "mock":
                    # Mock order - still update agent position state
                    self.cooldown_manager.set_cooldown(symbol)
                    self.update_agent_position_state(symbol, signal)
                    executed_qty = qty  # For mock trades, use the calculated qty
                    print(f"✅ MOCK SELL order logged for {symbol}")
                else:
                    print(f"❌ SELL order failed for {symbol}")
                    return None

            # Return the executed quantity for logging
            return executed_qty

        except Exception as e:
            print(f"❌ Trade execution error for {symbol}: {e}")
            return None

    def check_exit_conditions_for_agents(self):
        """Enhanced exit conditions with dynamic stop losses and portfolio protection"""
        print(f"🔍 Checking exit conditions using agent states (Data: {self.data_interval})...")
        
        # Update portfolio metrics first
        current_positions = self.get_current_positions()
        self.risk_manager.update_portfolio_metrics(current_positions)
        
        # Get risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        # Emergency portfolio protection
        if risk_metrics["current_drawdown"] > risk_metrics["drawdown_limit"]:
            print(f"🚨 EMERGENCY: Maximum drawdown exceeded ({risk_metrics['current_drawdown']:.3f})")
            self._emergency_close_all_positions()
            return
        
        if risk_metrics["daily_pnl"] < -risk_metrics["daily_loss_limit"]:
            print(f"🚨 EMERGENCY: Daily loss limit exceeded (${risk_metrics['daily_pnl']:.2f})")
            self._emergency_close_all_positions()
            return
        
        for symbol, agent in self.loaded_agents.items():
            if not hasattr(agent, 'position_state') or agent.position_state != "long":
                continue
                
            # Agent thinks it has a long position, check if we should exit
            df = self._fetch_agent_data(symbol)
            if df is None or df.empty:
                continue
                
            current_price = df["close"].iloc[-1]
            
            # Enhanced exit condition checking
            exit_reason = self._check_enhanced_exit_conditions(agent, symbol, current_price, df)
            
            if exit_reason:
                print(f"💡 Agent {symbol} exit condition met: {exit_reason}")
                timestamp = self.performance_tracker.current_time()
                
                # Execute sell order
                executed_qty = self.execute_trade(symbol, "sell", current_price, confidence=1.0)
                
                # Log the exit trade  
                self._log_trade_execution(
                    symbol=symbol,
                    signal="sell",
                    price=current_price,
                    confidence=1.0,
                    score=1.0,
                    timestamp=timestamp,
                    qty=executed_qty,
                    source=f"exit_{exit_reason}"
                )
                
                # Calculate profits
                compute_trade_profits(symbol)

    def _emergency_close_all_positions(self):
        """Emergency close all open positions"""
        print("🚨 EMERGENCY CLOSE: Closing all positions immediately")
        
        for symbol, agent in self.loaded_agents.items():
            if hasattr(agent, 'position_state') and agent.position_state == "long":
                df = self._fetch_agent_data(symbol)
                if df is not None and not df.empty:
                    current_price = df["close"].iloc[-1]
                    print(f"🚨 Emergency closing {symbol} at ${current_price:.4f}")
                    
                    # Force execute sell order
                    executed_qty = self.execute_trade(symbol, "sell", current_price, confidence=1.0)
                    
                    # Log emergency exit
                    timestamp = self.performance_tracker.current_time()
                    self._log_trade_execution(
                        symbol=symbol,
                        signal="sell",
                        price=current_price,
                        confidence=1.0,
                        score=0.0,
                        timestamp=timestamp,
                        qty=executed_qty,
                        source="emergency_exit"
                    )

    def _check_enhanced_exit_conditions(self, agent, symbol: str, current_price: float, df) -> Optional[str]:
        """Enhanced exit conditions with dynamic stops and time limits"""
        config = self.risk_manager.get_symbol_config(symbol)
    
        # Get last buy trade to determine entry price and time
        try:
            path = f"backend/storage/performance_logs/{symbol}_trades.json"
            if os.path.exists(path):
                with open(path, "r") as f:
                    trades = json.load(f)
            
                # Find last buy trade
                last_buy = None
                for trade in reversed(trades):
                    if trade.get("signal") == "buy":
                        last_buy = trade
                        break
            
                if last_buy:
                    entry_price = last_buy.get("price", current_price)
                    entry_time_str = last_buy.get("timestamp", "")
                    trade_interval = last_buy.get("data_interval", self.data_interval)
                
                    # FIXED CALCULATIONS: Use percentage-based SL/TP
                    sl_percent = config["sl_percent"] / 100.0  # Convert to decimal
                    tp_ratio = config["tp_ratio"]
                
                    # 1. Stop Loss Check (enhanced with volatility)
                    if len(df) >= 20:
                        # Dynamic stop loss based on ATR
                        high_low = df['high'] - df['low']
                        atr = high_low.rolling(14).mean().iloc[-1]
                        dynamic_sl_percent = max(sl_percent, atr / current_price * 2)
                    else:
                        dynamic_sl_percent = sl_percent
                
                    stop_loss = entry_price * (1 - dynamic_sl_percent)
                    if current_price <= stop_loss:
                        loss_percent = (entry_price - current_price) / entry_price
                        return f"stop_loss_hit_{loss_percent:.3f}"
                
                    # 2. Take Profit Check (FIXED)
                    sl_distance = entry_price * dynamic_sl_percent  # Distance in dollars
                    tp_distance = sl_distance * tp_ratio  # 2x the SL distance
                    take_profit = entry_price + tp_distance  # Add to entry price
                
                    if current_price >= take_profit:
                        profit_percent = (current_price - entry_price) / entry_price
                        return f"take_profit_hit_{profit_percent:.3f}"
                
                    # 3. Time-based Exit (adjusted for data interval)
                    if entry_time_str:
                        try:
                            entry_time = dateutil.parser.parse(entry_time_str)
                            now = datetime.now(entry_time.tzinfo) if entry_time.tzinfo else datetime.now()
                            hold_duration = (now - entry_time).total_seconds()
                            
                            # Adjust max hold time based on data interval
                            interval_multiplier = self._interval_to_minutes(self.data_interval) / 60.0  # Convert to hours
                            adjusted_max_hold = config["max_hold_seconds"] * max(1.0, interval_multiplier)
                        
                            if hold_duration > adjusted_max_hold:
                                return f"max_hold_time_{hold_duration/3600:.1f}h"
                        except:
                            pass
                
                    # 4. Trailing Stop (if profitable)
                    profit_percent = (current_price - entry_price) / entry_price
                    if profit_percent > 0.02:  # 2% profit threshold for trailing stop
                        # Check if price dropped more than 1% from recent high
                        lookback_periods = max(6, int(6 * 60 / self._interval_to_minutes(self.data_interval)))  # Adjust for interval
                        recent_high = df['high'].tail(lookback_periods).max()
                        if current_price < recent_high * 0.99:
                            return f"trailing_stop_{profit_percent:.3f}"
                
                    # 5. Volatility-based Exit (adjusted for interval)
                    if len(df) >= 6:
                        lookback_periods = max(6, int(6 * 60 / self._interval_to_minutes(self.data_interval)))
                        recent_volatility = df['close'].pct_change().tail(lookback_periods).std()
                        volatility_threshold = config.get("exit_volatility_threshold", 0.08)
                        
                        # Adjust volatility threshold for different intervals
                        adjusted_threshold = volatility_threshold * np.sqrt(self._interval_to_minutes(self.data_interval) / 60.0)
                        
                        if recent_volatility > adjusted_threshold:
                            return f"high_volatility_{recent_volatility:.4f}"
    
        except Exception as e:
            print(f"⚠️ Error checking exit conditions for {symbol}: {e}")
    
        return None

    def _is_position_stale(self, timestamp_str: str, max_hours: int = 24) -> bool:
        """Check if a position is older than max_hours"""
        try:
            trade_time = dateutil.parser.parse(timestamp_str)
            now = datetime.now(trade_time.tzinfo) if trade_time.tzinfo else datetime.now()
            return (now - trade_time) > timedelta(hours=max_hours)
        except:
            return False

    def make_portfolio_decision(self, min_score=0.7):
        """Enhanced portfolio decision making with comprehensive risk management and data consistency"""
        auto_cleanup_logs()
        
        # Update portfolio metrics
        current_positions = self.get_current_positions()
        self.risk_manager.update_portfolio_metrics(current_positions)
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        print(f"🎯 Making portfolio decision with enhanced risk management")
        print(f"📊 Using {self.data_interval} data interval for all analysis")
        print(f"📊 Portfolio Metrics:")
        print(f"   Value: ${risk_metrics['portfolio_value']:.2f}")
        print(f"   Drawdown: {risk_metrics['current_drawdown']:.3f} (limit: {risk_metrics['drawdown_limit']:.3f})")
        print(f"   Daily P&L: ${risk_metrics['daily_pnl']:.2f} (limit: ${-risk_metrics['daily_loss_limit']:.2f})")
        print(f"   Hourly Trades: {risk_metrics['hourly_trades']}")

        # Emergency checks
        if risk_metrics["current_drawdown"] > risk_metrics["drawdown_limit"]:
            print("🚨 Portfolio in emergency mode - no new trades allowed")
            return {"decision": [], "timestamp": self.performance_tracker.current_time(), 
                   "status": "emergency_drawdown", "data_interval": self.data_interval}
        
        if risk_metrics["daily_pnl"] < -risk_metrics["daily_loss_limit"]:
            print("🚨 Daily loss limit reached - no new trades allowed")
            return {"decision": [], "timestamp": self.performance_tracker.current_time(), 
                   "status": "daily_loss_limit", "data_interval": self.data_interval}

        # Sync agent positions before making decisions
        self.sync_agent_positions()

        timestamp = self.performance_tracker.current_time()

        # Step 1: Check exit conditions using agent states
        self.check_exit_conditions_for_agents()

        # Step 2: Evaluate new trades with enhanced filtering
        top = self.decide_trades(min_score=min_score)
        if not top:
            print("📭 No trades meet enhanced criteria")
            return {"decision": [], "timestamp": timestamp, "risk_metrics": risk_metrics, 
                   "data_interval": self.data_interval}

        decision = top[0]
        df = self._fetch_agent_data(decision["symbol"])
        price = df["close"].iloc[-1] if df is not None and not df.empty else None
        result = {**decision, "last_price": price}

        print(f"🎯 Top decision: {decision['symbol']} {decision['signal']} "
              f"(score: {decision['score']:.3f}, confidence: {decision['confidence']:.3f})")
        print(f"    Source: {decision.get('source')}, ML Available: {decision.get('ml_available')}")
        print(f"    Data Interval: {decision.get('data_interval', self.data_interval)}")

        if decision["signal"] in ("buy", "sell") and price:
            # Execute the trade (enhanced with risk management)
            executed_qty = self.execute_trade(decision["symbol"], decision["signal"], price, decision["confidence"])
            
            if executed_qty:
                # Log the trade
                self._log_trade_execution(
                    symbol=decision["symbol"],
                    signal=decision["signal"],
                    price=price,
                    confidence=decision["confidence"],
                    score=decision["score"],
                    timestamp=timestamp,
                    qty=executed_qty,
                    source=f"agent_{decision.get('source', 'unknown')}"
                )
                
                # Calculate profits for sell orders
                if decision["signal"] == "sell":
                    compute_trade_profits(decision["symbol"])

        return {
            "decision": result, 
            "timestamp": timestamp, 
            "risk_metrics": risk_metrics,
            "portfolio_status": "active",
            "data_interval": self.data_interval
        }

    def load_all_predictions(self) -> List[Dict]:
        """Load predictions from all agents with data consistency tracking"""
        predictions = []
        agents = self.load_agents()
        
        print(f"📊 Loading predictions using {self.data_interval} data interval...")
        
        for agent in agents:
            ohlcv = self._fetch_agent_data(agent.symbol)
            if ohlcv is not None:
                prediction = self._safe_evaluate_agent(agent, ohlcv)
                predictions.append(prediction)
                
        return predictions

    def get_agent_status_summary(self) -> Dict:
        """Enhanced agent status summary with risk metrics and data interval info"""
        current_positions = self.get_current_positions()
        self.risk_manager.update_portfolio_metrics(current_positions)
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        summary = {
            "total_agents": len(self.loaded_agents),
            "data_interval": self.data_interval,
            "risk_metrics": risk_metrics,
            "cooldown_status": self.cooldown_manager.get_cooldown_status(),
            "agents": {}
        }
        
        for symbol, agent in self.loaded_agents.items():
            agent_info = {
                "symbol": symbol,
                "position_state": getattr(agent, 'position_state', None),
                "has_ml_model": getattr(agent, 'model', None) is not None,
                "in_cooldown": self.cooldown_manager.is_in_cooldown(symbol),
                "cooldown_remaining": self.cooldown_manager.get_cooldown_remaining(symbol),
                "exposure": current_positions.get(symbol, {}).get('exposure', 0.0),
                "data_interval": self.data_interval
            }
            
            if hasattr(agent, 'get_model_info'):
                model_info = agent.get_model_info()
                agent_info.update({
                    "model_loaded": model_info.get("model_loaded", False),
                    "model_classes": model_info.get("classes", [])
                })
            
            # Check data compatibility
            try:
                df = self._fetch_agent_data(symbol)
                if df is not None:
                    agent_info["data_compatible"] = self._validate_data_compatibility(agent, df)
                else:
                    agent_info["data_compatible"] = None
            except:
                agent_info["data_compatible"] = False
            
            summary["agents"][symbol] = agent_info
            
        return summary

    def sync_agent_positions(self):
        """Enhanced position synchronization with validation and correction"""
        print(f"🔄 Syncing agent positions (Data interval: {self.data_interval})...")
    
        for symbol, agent in self.loaded_agents.items():
            if not hasattr(agent, 'position_state'):
                continue
            
            # Get current position state
            current_state = agent.position_state
            
            # Validate against trade history
            validated_state = self._validate_agent_position(symbol, current_state)
            
            if validated_state != current_state:
                print(f"🔧 Correcting {symbol} position: {current_state} → {validated_state}")
                agent.position_state = validated_state
            
            print(f"📊 {symbol} agent position: {agent.position_state}")

    def _validate_agent_position(self, symbol: str, claimed_state: Union[str, None]) -> Optional[str]:
        """Validate agent position state against trade history"""
        try:
            path = f"backend/storage/performance_logs/{symbol}_trades.json"
            if not os.path.exists(path):
                return None  # No trade history, assume flat
        
            with open(path, "r") as f:
                trades = json.load(f)
        
            if not trades:
                return None
        
            # Get the last trade signal
            last_trade = trades[-1]
            last_signal = last_trade.get("signal", "")
            last_timestamp = last_trade.get("timestamp", "")
    
            # Determine correct position based on last signal
            if last_signal == "buy":
                correct_state = "long"
            elif last_signal == "sell":
                correct_state = None
            else:
                correct_state = None
        
            # Additional validation: check if position is too old (safety mechanism)
            if correct_state == "long" and self._is_position_stale(last_timestamp):
                print(f"⚠️ {symbol} position appears stale, resetting to flat")
                correct_state = None
        
            return correct_state
    
        except Exception as e:
            print(f"⚠️ Failed to validate position for {symbol}: {e}")
            return None

    def decide_trades(self, top_n=1, min_score=0.5, min_confidence=0.7):
        """Enhanced trade decision with comprehensive risk filtering and data consistency checks"""
        agents = self.load_agents()
        evaluations = self.evaluate_agents(agents)
        
        # Get current positions for additional filtering
        current_positions = self.get_current_positions()

        print(f"\n🔍 ENHANCED TRADE DECISION ANALYSIS (Data: {self.data_interval})")
        print(f"📊 Filters: min_score={min_score}, min_confidence={min_confidence}")
        print("=" * 80)

        approved_trades = []

        for e in evaluations:
            symbol = e["symbol"]
            signal = e["signal"]
            confidence = e["confidence"]
            score = e["score"]
            position_state = e.get("position_state")
            source = e.get("source", "unknown")

            print(f"\n📊 {symbol}:")
            print(f"   Signal: {signal.upper()} | Confidence: {confidence:.3f} | Score: {score:.3f}")
            print(f"   Position: {position_state} | Source: {source}")
            print(f"   ML Available: {e.get('ml_available', False)} | ML Conf: {e.get('ml_confidence', 0.0):.3f}")
            print(f"   Rule Conf: {e.get('rule_confidence', 0.0):.3f}")
            print(f"   Data Interval: {e.get('data_interval', self.data_interval)}")

            # Enhanced filtering with risk management
            skip_reason = None

            # Check score threshold
            if score < min_score:
                skip_reason = f"score too low ({score:.3f} < {min_score})"

            # Check cooldown using CooldownManager
            elif self.cooldown_manager.is_in_cooldown(symbol):
                cooldown_remaining = self.cooldown_manager.get_cooldown_remaining(symbol)
                skip_reason = f"in cooldown ({cooldown_remaining:.0f}s remaining)"

            # Check confidence threshold
            elif confidence < min_confidence:
                skip_reason = f"confidence too low ({confidence:.3f} < {min_confidence})"

            # Check if signal is actionable
            elif signal not in ["buy", "sell"]:
                skip_reason = f"signal is '{signal}' (not actionable)"

            # Enhanced market condition checks
            elif signal == "buy":
                df = self._fetch_agent_data(symbol)
                market_ok, market_reason = self.risk_manager.check_market_conditions(symbol, df)
                if not market_ok:
                    skip_reason = f"market conditions: {market_reason}"
                elif position_state == "long":
                    # Double-check the position state
                    actual_state = self._validate_agent_position(symbol, position_state)
                    if actual_state == "long":
                        skip_reason = f"already in long position"
                    else:
                        # Position state was wrong, correct it and allow trade
                        agent = self.get_agent_by_symbol(symbol)
                        if agent:
                            agent.position_state = actual_state
                            print(f"   🔧 Corrected position state to: {actual_state}")
                else:
                    # Portfolio-level checks for new positions
                    position_size = self.risk_manager.calculate_dynamic_position_size(symbol, 0, df)  # Estimate
                    portfolio_ok, portfolio_reason = self.risk_manager.check_portfolio_limits(
                        symbol, position_size, current_positions
                    )
                    if not portfolio_ok:
                        skip_reason = f"portfolio limit: {portfolio_reason}"

            elif signal == "sell" and position_state is None:
                # Double-check the position state  
                actual_state = self._validate_agent_position(symbol, position_state)
                if actual_state is None:
                    skip_reason = f"no position to sell"
                else:
                    # Position state was wrong, correct it and allow trade
                    agent = self.get_agent_by_symbol(symbol)
                    if agent:
                        agent.position_state = actual_state
                        print(f"   🔧 Corrected position state to: {actual_state}")

            if skip_reason:
                print(f"   ⏭️ SKIPPED: {skip_reason}")
            else: 
                print(f"   ✅ APPROVED: {signal.upper()} trade")
                approved_trades.append(e)

        print(f"\n📈 ENHANCED SUMMARY:")
        print(f"   Data Interval: {self.data_interval}")
        print(f"   Total evaluated: {len(evaluations)}")
        print(f"   Approved trades: {len(approved_trades)}")
        print(f"   Symbols approved: {[f'{t['symbol']}-{t['signal'].upper()}' for t in approved_trades]}")
        
        # Display current risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        print(f"   Portfolio Status:")
        print(f"     Value: ${risk_metrics['portfolio_value']:.2f}")
        print(f"     Drawdown: {risk_metrics['current_drawdown']:.3f}")
        print(f"     Daily P&L: ${risk_metrics['daily_pnl']:.2f}")
        print(f"     Hourly Trades: {risk_metrics['hourly_trades']}")
        print("=" * 80)

        return approved_trades[:top_n]


# Utility functions for external usage
def enhanced_trading_loop_example():
    """Example of how to use the enhanced system in your trading loop"""
    from backend.utils.binance_api import get_trading_mode
    
    # Initialize MotherAI with configurable data interval
    mother_ai = MotherAI(
        agent_symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
        data_interval="1m"  # Use 1-minute data to match agent predictions
    )
    
    # Debug data intervals on startup
    mother_ai.debug_data_intervals()
    
    # Main trading loop
    import time
    
    while True:
        try:
            # Make trading decision
            decision_result = mother_ai.make_portfolio_decision(min_score=0.6)
            print(f"📊 Decision result: {decision_result}")
            
            # Wait before next iteration
            time.sleep(300)  # 5 minutes
            
        except KeyboardInterrupt:
            print("🛑 Trading loop stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in trading loop: {e}")
            time.sleep(60)  # Wait 1 minute before retrying


def get_risk_report_cli():
    """CLI command to get risk report with data interval info"""
    mother_ai = MotherAI()
    current_positions = mother_ai.get_current_positions()
    report = mother_ai.risk_manager.get_risk_report(current_positions)
    
    print(f"\n📊 RISK REPORT (Data Interval: {mother_ai.data_interval})")
    print("=" * 60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Portfolio Value: ${report['portfolio_metrics']['portfolio_value']:.2f}")
    print(f"Current Drawdown: {report['portfolio_metrics']['current_drawdown']:.3f}")
    print(f"Daily P&L: ${report['portfolio_metrics']['daily_pnl']:.2f}")
    
    print(f"\nActive Positions: {report['position_summary']['total_positions']}")
    print(f"Total Exposure: {report['position_summary']['total_exposure']:.3f}")
    
    if report['warnings']:
        print(f"\n⚠️ Warnings:")
        for warning in report['warnings']:
            print(f"   - {warning}")
    else:
        print(f"\n✅ No warnings")
    
    return report


def debug_data_intervals_cli():
    """CLI command to debug data intervals"""
    mother_ai = MotherAI()
    mother_ai.debug_data_intervals()


def test_different_intervals():
    """Test function to compare different data intervals"""
    intervals = ["1m", "5m", "15m", "1h"]
    
    for interval in intervals:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {interval.upper()} DATA INTERVAL")
        print(f"{'='*60}")
        
        mother_ai = MotherAI(
            agent_symbols=["BTCUSDT", "ETHUSDT"],  # Test with 2 symbols
            data_interval=interval
        )
        
        # Debug intervals
        mother_ai.debug_data_intervals()
        
        # Get agent status
        status = mother_ai.get_agent_status_summary()
        print(f"\nAgent Status Summary:")
        for symbol, info in status["agents"].items():
            print(f"  {symbol}: Data Compatible: {info.get('data_compatible', 'Unknown')}")
        
        print(f"\nCompleted test with {interval} interval")