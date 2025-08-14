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
                 agent_symbols=None, data_interval="30m"):
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
        self.data_interval = data_interval
        self.performance_tracker = PerformanceTracker("performance_logs")
        self.agent_symbols = agent_symbols or []
        self.meta_evaluator = MetaEvaluator()
        self.loaded_agents = {}
        self.risk_manager = RiskManager()
        self.cooldown_manager = CooldownManager()
        
        # NEW: Track last exit check time to prevent too frequent exit checks
        self.last_exit_check = {}
        self.exit_check_interval = 300  # 5 minutes between exit checks (configurable)
        
        # NEW: Track trade entry times to enforce minimum hold periods
        self.position_entry_times = {}
        self.minimum_hold_time = 900  # 15 minutes minimum hold (configurable)

        # Initialize agents on startup
        self._initialize_agents()
        
        print(f"🔧 MotherAI initialized with {self.data_interval} data interval")
        print(f"🔧 Exit check interval: {self.exit_check_interval}s")
        print(f"🔧 Minimum hold time: {self.minimum_hold_time}s")

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
        """Fetch OHLCV data using configured interval to match agent predictions"""
        symbol = symbol if symbol.endswith("/USDT") else symbol.replace("USDT", "") + "/USDT"
        df = fetch_ohlcv(symbol, self.data_interval, 100)
        return df if not df.empty else None

    def _validate_data_compatibility(self, agent, ohlcv_data: pd.DataFrame) -> bool:
        """Validate that data interval matches agent's expected interval"""
        if not hasattr(agent, 'model') or agent.model is None:
            return True
        
        if len(ohlcv_data) >= 2:
            time_diff = ohlcv_data.index[1] - ohlcv_data.index[0]
            interval_minutes = time_diff.total_seconds() / 60
            
            expected_minutes = self._interval_to_minutes(self.data_interval)
            
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
            return 1

    def _safe_evaluate_agent(self, agent, ohlcv):
        """Enhanced agent evaluation with data compatibility checks"""
        try:
            if not self._validate_data_compatibility(agent, ohlcv):
                print(f"⚠️ Data compatibility issue for {agent.symbol}, using fallback strategy")
            
            decision = agent.evaluate(ohlcv)
        
            ml_data = decision.get("ml", {})
            rule_data = decision.get("rule_based", {})
        
            print(f"🤖 Agent {agent.symbol} evaluation:")
            print(f"   Final: {decision.get('action', 'unknown').upper()} ({decision.get('confidence', 0.0):.3f})")
            print(f"   ML: {ml_data.get('action', 'N/A')} ({ml_data.get('confidence', 0.0):.3f})")
            print(f"   Rule: {rule_data.get('action', 'N/A')} ({rule_data.get('confidence', 0.0):.3f})")
            print(f"   Source: {decision.get('source', 'unknown')}")
            print(f"   Position: {decision.get('position_state', 'None')}")
        
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
                "data_interval": self.data_interval,
                "full_decision": decision
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
                
            prediction = self._safe_evaluate_agent(agent, ohlcv)
            health = StrategyHealth(trade_tracker.get_agent_log(agent.symbol)).summary()
            score = self._calculate_score(prediction, health)
            
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
            "data_interval": self.data_interval
        }

        try:
            history = json.load(open(path)) if os.path.exists(path) else []
        except Exception:
            history = []

        history.append(entry)

        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        
        # NEW: Track entry times for minimum hold period enforcement
        if signal == "buy":
            self.position_entry_times[symbol] = datetime.now()
            print(f"📝 Recorded entry time for {symbol}: {self.position_entry_times[symbol]}")

    def update_agent_position_state(self, symbol: str, signal: str):
        """Update agent's position state after successful trade execution"""
        agent = self.get_agent_by_symbol(symbol)
        if agent and hasattr(agent, 'position_state'):
            old_state = agent.position_state
            
            if signal == "buy":
                agent.position_state = "long"
            elif signal == "sell":
                agent.position_state = None
                # NEW: Clear entry time when position is closed
                if symbol in self.position_entry_times:
                    del self.position_entry_times[symbol]
                    print(f"🗑️ Cleared entry time for {symbol}")
                
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
                'data_interval': self.data_interval,
                'entry_time': self.position_entry_times.get(symbol),  # NEW: Track entry time
                'hold_duration': None  # NEW: Track how long we've held
            }
            
            # Calculate hold duration if we have entry time
            if position_info['entry_time'] and position_info['position_state'] == 'long':
                hold_duration = (datetime.now() - position_info['entry_time']).total_seconds()
                position_info['hold_duration'] = hold_duration
            
            if position_info['position_state'] == 'long':
                position_info['exposure'] = self.risk_manager.get_symbol_config(symbol)["risk_per_trade"]
                
            positions[symbol] = position_info
        
        return positions

    def execute_trade(self, symbol, signal, price, confidence, live=False):
        """Enhanced trade execution with comprehensive risk checks"""
        if signal not in ("buy", "sell") or not price:
            print(f"❌ Invalid trade parameters: signal={signal}, price={price}")
            return None

        df = self._fetch_agent_data(symbol)
        
        print(f"📈 Executing trade with {self.data_interval} data interval")

        # Risk checks (keeping existing logic)
        market_ok, market_reason = self.risk_manager.check_market_conditions(symbol, df) 
        if not market_ok:
            print(f"❌ Market conditions check failed for {symbol}: {market_reason}")
            return None

        position_size = self.risk_manager.calculate_dynamic_position_size(symbol, price, df)
        current_positions = self.get_current_positions()

        if signal == "buy":
            portfolio_ok, portfolio_reason = self.risk_manager.check_portfolio_limits(
                symbol, position_size, current_positions
            )
            if not portfolio_ok:
                print(f"❌ Portfolio limits check failed for {symbol}: {portfolio_reason}")
                return None

        self.risk_manager.log_trade_attempt()

        # Continue with existing trade execution logic...
        from backend.utils.binance_api import get_trading_mode, get_binance_client
        current_mode = get_trading_mode()

        print(f"🚀 Executing {signal.upper()} order for {symbol} at ${price:.4f} | Mode: {current_mode.upper()}")
        print(f"📊 Risk-adjusted position size: {position_size:.3f}")

        qty = None
        executed_qty = None

        try:
            # Account balance check for live trading
            if current_mode == "live":
                try:
                    client = get_binance_client()
                    account_info = client.get_account()
                    balances = {b['asset']: float(b['free']) for b in account_info['balances']}
                    usdt_balance = balances.get('USDT', 0)
                    print(f"💰 Current USDT balance: ${usdt_balance:.2f}")
            
                    if signal == "buy" and usdt_balance < 10:
                        print(f"❌ Insufficient USDT balance for buy order")
                        return None
            
                    available_balance = min(usdt_balance * 0.9, 50)
                    trade_amount = available_balance * position_size
                    print(f"💡 Using ${trade_amount:.2f} for this trade")
                except Exception as balance_err:
                    print(f"⚠️ Could not check balance: {balance_err}")
                    trade_amount = self.risk_manager.config["default_balance_usd"] * position_size
            else:
                trade_amount = self.risk_manager.config["default_balance_usd"] * position_size

            binance_symbol = symbol if "/" in symbol else symbol.replace("USDT", "/USDT")

            if signal == "buy":
                config = self.risk_manager.get_symbol_config(symbol)
                sl_percent = config["sl_percent"] / 100.0
                tp_ratio = config["tp_ratio"]
                
                sl = price * (1 - sl_percent)
                tp_distance = (price - sl) * tp_ratio
                tp = price + tp_distance
                qty = trade_amount / price

                if qty <= 0:
                    print(f"⚠️ Computed qty is zero or negative for BUY on {symbol}")
                    return None

                print(f"📊 Buying {symbol} | Entry: {price:.4f}, SL: {sl:.4f}, TP: {tp:.4f}")
                order = place_market_order(binance_symbol, signal, qty)

                if order and order.get("status") != "mock":
                    self.cooldown_manager.set_cooldown(symbol)
                    self.update_agent_position_state(symbol, signal)
                    executed_qty = order.get("executedQty", qty)
                    print(f"✅ BUY order executed for {symbol}")
                elif order and order.get("status") == "mock":
                    self.cooldown_manager.set_cooldown(symbol)
                    self.update_agent_position_state(symbol, signal)
                    executed_qty = qty
                    print(f"✅ MOCK BUY order logged for {symbol}")
                else:
                    print(f"❌ BUY order failed for {symbol}")
                    return None

            elif signal == "sell":
                agent = self.get_agent_by_symbol(symbol)
                qty = 0
            
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
                            print(f"❌ No {base_asset} holdings found")
                            return None
                    except Exception as e:
                        print(f"⚠️ Could not check holdings: {e}")
                        return None
                else:
                    qty = trade_amount / price

                if qty <= 0:
                    print(f"⚠️ Computed qty is zero or negative for SELL on {symbol}")
                    return None

                print(f"📊 Selling {symbol} | Qty: {qty:.6f} at ${price:.4f}")
                order = place_market_order(binance_symbol, signal, qty)

                if order and order.get("status") != "mock":
                    self.cooldown_manager.set_cooldown(symbol)
                    self.update_agent_position_state(symbol, signal)
                    executed_qty = order.get("executedQty", qty)
                    print(f"✅ SELL order executed for {symbol}")
                elif order and order.get("status") == "mock":
                    self.cooldown_manager.set_cooldown(symbol)
                    self.update_agent_position_state(symbol, signal)
                    executed_qty = qty
                    print(f"✅ MOCK SELL order logged for {symbol}")
                else:
                    print(f"❌ SELL order failed for {symbol}")
                    return None

            return executed_qty

        except Exception as e:
            print(f"❌ Trade execution error for {symbol}: {e}")
            return None

    # NEW: More intelligent exit condition checking
    def should_check_exit_conditions(self, symbol: str) -> bool:
        """Determine if we should check exit conditions for a symbol"""
        current_time = datetime.now()
        
        # Check if enough time has passed since last exit check
        last_check = self.last_exit_check.get(symbol)
        if last_check and (current_time - last_check).total_seconds() < self.exit_check_interval:
            return False
        
        # Check if position meets minimum hold time
        entry_time = self.position_entry_times.get(symbol)
        if entry_time:
            hold_duration = (current_time - entry_time).total_seconds()
            if hold_duration < self.minimum_hold_time:
                print(f"⏰ {symbol} hasn't met minimum hold time ({hold_duration:.0f}s < {self.minimum_hold_time}s)")
                return False
        
        return True

    def check_exit_conditions_for_agents(self, force_check=False):
        """Enhanced exit conditions with intelligent timing and minimum hold periods"""
        if not force_check:
            print(f"🔍 Checking exit conditions (Min hold: {self.minimum_hold_time}s, Check interval: {self.exit_check_interval}s)...")
        else:
            print(f"🔍 FORCED exit condition check...")
        
        current_positions = self.get_current_positions()
        self.risk_manager.update_portfolio_metrics(current_positions)
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        # Emergency portfolio protection (always check these)
        if risk_metrics["current_drawdown"] > risk_metrics["drawdown_limit"]:
            print(f"🚨 EMERGENCY: Maximum drawdown exceeded ({risk_metrics['current_drawdown']:.3f})")
            self._emergency_close_all_positions()
            return
        
        if risk_metrics["daily_pnl"] < -risk_metrics["daily_loss_limit"]:
            print(f"🚨 EMERGENCY: Daily loss limit exceeded (${risk_metrics['daily_pnl']:.2f})")
            self._emergency_close_all_positions()
            return
        
        # Check individual positions with intelligent timing
        for symbol, agent in self.loaded_agents.items():
            if not hasattr(agent, 'position_state') or agent.position_state != "long":
                continue
            
            # NEW: Skip if not time to check yet (unless forced)
            if not force_check and not self.should_check_exit_conditions(symbol):
                continue
                
            # Update last check time
            self.last_exit_check[symbol] = datetime.now()
            
            df = self._fetch_agent_data(symbol)
            if df is None or df.empty:
                continue
                
            current_price = df["close"].iloc[-1]
            
            # Check exit conditions with enhanced logic
            exit_reason = self._check_enhanced_exit_conditions(agent, symbol, current_price, df)
            
            if exit_reason:
                print(f"💡 Agent {symbol} exit condition met: {exit_reason}")
                timestamp = self.performance_tracker.current_time()
                
                executed_qty = self.execute_trade(symbol, "sell", current_price, confidence=1.0)
                
                if executed_qty:
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
                    
                    executed_qty = self.execute_trade(symbol, "sell", current_price, confidence=1.0)
                    
                    if executed_qty:
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
        """Enhanced exit conditions with minimum hold time consideration"""
        config = self.risk_manager.get_symbol_config(symbol)
        
        # NEW: Check minimum hold time first
        entry_time = self.position_entry_times.get(symbol)
        if entry_time:
            hold_duration = (datetime.now() - entry_time).total_seconds()
            if hold_duration < self.minimum_hold_time:
                # Only allow emergency exits during minimum hold period
                profit_loss_percent = 0.0
                try:
                    path = f"backend/storage/performance_logs/{symbol}_trades.json"
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            trades = json.load(f)
                        
                        last_buy = None
                        for trade in reversed(trades):
                            if trade.get("signal") == "buy":
                                last_buy = trade
                                break
                        
                        if last_buy:
                            entry_price = last_buy.get("price", current_price)
                            profit_loss_percent = (current_price - entry_price) / entry_price
                except:
                    pass
                
                # Only emergency stop loss during minimum hold period
                if profit_loss_percent < -0.05:  # 5% emergency stop loss
                    return f"emergency_stop_loss_{profit_loss_percent:.3f}"
                
                print(f"⏰ {symbol} in minimum hold period ({hold_duration:.0f}s/{self.minimum_hold_time}s)")
                return None
        
        # Proceed with normal exit conditions after minimum hold time
        try:
            path = f"backend/storage/performance_logs/{symbol}_trades.json"
            if os.path.exists(path):
                with open(path, "r") as f:
                    trades = json.load(f)
            
                last_buy = None
                for trade in reversed(trades):
                    if trade.get("signal") == "buy":
                        last_buy = trade
                        break
            
                if last_buy:
                    entry_price = last_buy.get("price", current_price)
                    entry_time_str = last_buy.get("timestamp", "")
                    
                    sl_percent = config["sl_percent"] / 100.0
                    tp_ratio = config["tp_ratio"]
                    
                    # Dynamic stop loss with ATR
                    if len(df) >= 20:
                        high_low = df['high'] - df['low']
                        atr = high_low.rolling(14).mean().iloc[-1]
                        dynamic_sl_percent = max(sl_percent, atr / current_price * 2)
                    else:
                        dynamic_sl_percent = sl_percent
                
                    stop_loss = entry_price * (1 - dynamic_sl_percent)
                    if current_price <= stop_loss:
                        loss_percent = (entry_price - current_price) / entry_price
                        return f"stop_loss_hit_{loss_percent:.3f}"
                
                    # Take profit check
                    sl_distance = entry_price * dynamic_sl_percent
                    tp_distance = sl_distance * tp_ratio
                    take_profit = entry_price + tp_distance
                
                    if current_price >= take_profit:
                        profit_percent = (current_price - entry_price) / entry_price
                        return f"take_profit_hit_{profit_percent:.3f}"
                
                    # Time-based exit (only after minimum hold time)
                    if entry_time_str:
                        try:
                            entry_time_parsed = dateutil.parser.parse(entry_time_str)
                            now = datetime.now(entry_time_parsed.tzinfo) if entry_time_parsed.tzinfo else datetime.now()
                            hold_duration = (now - entry_time_parsed).total_seconds()
                            profit_percent = (current_price - entry_price) / entry_price
                            interval_multiplier = self._interval_to_minutes(self.data_interval) / 60.0
                            adjusted_max_hold = config["max_hold_seconds"] * max(1.0, interval_multiplier)
                        
                            if hold_duration > adjusted_max_hold:
                                return f"trailing_stop_{profit_percent:.3f}"
                        
                        except Exception as e:
                            print(f"Error: {e}")
                
                    # Volatility-based exit (adjusted for interval)
                    if len(df) >= 6:
                        lookback_periods = max(6, int(6 * 60 / self._interval_to_minutes(self.data_interval)))
                        recent_volatility = df['close'].pct_change().tail(lookback_periods).std()
                        volatility_threshold = config.get("exit_volatility_threshold", 0.08)
                        
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

    # NEW: Separate method for strategic exit checks (called less frequently)
    def check_strategic_exits(self):
        """Check for strategic exit opportunities - called separately from regular decision making"""
        print(f"🎯 Running strategic exit analysis...")
        self.check_exit_conditions_for_agents(force_check=True)

    def make_portfolio_decision(self, min_score=0.7):
        """Enhanced portfolio decision making - SEPARATED EXIT LOGIC FROM NEW TRADE LOGIC"""
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

        # Emergency portfolio checks
        if risk_metrics["current_drawdown"] > risk_metrics["drawdown_limit"]:
            print("🚨 Portfolio in emergency mode - no new trades allowed")
            return {"decision": [], "timestamp": self.performance_tracker.current_time(), 
                   "status": "emergency_drawdown", "data_interval": self.data_interval}
        
        if risk_metrics["daily_pnl"] < -risk_metrics["daily_loss_limit"]:
            print("🚨 Daily loss limit reached - no new trades allowed")
            return {"decision": [], "timestamp": self.performance_tracker.current_time(), 
                   "status": "daily_loss_limit", "data_interval": self.data_interval}

        # Sync agent positions
        self.sync_agent_positions()
        timestamp = self.performance_tracker.current_time()

        # NEW: Only check exits occasionally, not every decision cycle
        last_strategic_check = getattr(self, '_last_strategic_exit_check', None)
        strategic_check_interval = 900  # 15 minutes between strategic exit checks
        
        if (not last_strategic_check or 
            (datetime.now() - last_strategic_check).total_seconds() > strategic_check_interval):
            print(f"🎯 Time for strategic exit check (every {strategic_check_interval}s)")
            self.check_exit_conditions_for_agents(force_check=False)
            self._last_strategic_exit_check = datetime.now()
        else:
            time_since_check = (datetime.now() - last_strategic_check).total_seconds()
            print(f"⏭️ Skipping exit check (last check {time_since_check:.0f}s ago)")

        # Evaluate new trade opportunities
        top = self.decide_trades(min_score=min_score)
        if not top:
            print("📭 No new trades meet criteria")
            return {"decision": [], "timestamp": timestamp, "risk_metrics": risk_metrics, 
                   "data_interval": self.data_interval}

        decision = top[0]
        df = self._fetch_agent_data(decision["symbol"])
        price = df["close"].iloc[-1] if df is not None and not df.empty else None
        result = {**decision, "last_price": price}

        print(f"🎯 Top decision: {decision['symbol']} {decision['signal']} "
              f"(score: {decision['score']:.3f}, confidence: {decision['confidence']:.3f})")
        print(f"    Source: {decision.get('source')}, ML Available: {decision.get('ml_available')}")

        if decision["signal"] in ("buy", "sell") and price:
            executed_qty = self.execute_trade(decision["symbol"], decision["signal"], price, decision["confidence"])
            
            if executed_qty:
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
        """Enhanced agent status summary with hold times and exit check status"""
        current_positions = self.get_current_positions()
        self.risk_manager.update_portfolio_metrics(current_positions)
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        summary = {
            "total_agents": len(self.loaded_agents),
            "data_interval": self.data_interval,
            "minimum_hold_time": self.minimum_hold_time,
            "exit_check_interval": self.exit_check_interval,
            "risk_metrics": risk_metrics,
            "cooldown_status": self.cooldown_manager.get_cooldown_status(),
            "agents": {}
        }
        
        current_time = datetime.now()
        
        for symbol, agent in self.loaded_agents.items():
            position_info = current_positions.get(symbol, {})
            
            agent_info = {
                "symbol": symbol,
                "position_state": getattr(agent, 'position_state', None),
                "has_ml_model": getattr(agent, 'model', None) is not None,
                "in_cooldown": self.cooldown_manager.is_in_cooldown(symbol),
                "cooldown_remaining": self.cooldown_manager.get_cooldown_remaining(symbol),
                "exposure": position_info.get('exposure', 0.0),
                "entry_time": position_info.get('entry_time'),
                "hold_duration": position_info.get('hold_duration'),
                "meets_min_hold": False,
                "next_exit_check": None,
                "data_interval": self.data_interval
            }
            
            # Check if position meets minimum hold time
            if agent_info['entry_time'] and agent_info['position_state'] == 'long':
                hold_duration = (current_time - agent_info['entry_time']).total_seconds()
                agent_info['meets_min_hold'] = hold_duration >= self.minimum_hold_time
                
                # Calculate when next exit check is allowed
                last_check = self.last_exit_check.get(symbol)
                if last_check:
                    next_check_time = last_check + timedelta(seconds=self.exit_check_interval)
                    if next_check_time > current_time:
                        agent_info['next_exit_check'] = (next_check_time - current_time).total_seconds()
                    else:
                        agent_info['next_exit_check'] = 0
            
            # Data compatibility check
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
        """Enhanced position synchronization with entry time recovery"""
        print(f"🔄 Syncing agent positions (Data interval: {self.data_interval})...")
    
        for symbol, agent in self.loaded_agents.items():
            if not hasattr(agent, 'position_state'):
                continue
            
            current_state = agent.position_state
            validated_state = self._validate_agent_position(symbol, current_state)
            
            if validated_state != current_state:
                print(f"🔧 Correcting {symbol} position: {current_state} → {validated_state}")
                agent.position_state = validated_state
            
            # NEW: Recover entry times from trade history if missing
            if (agent.position_state == 'long' and 
                symbol not in self.position_entry_times):
                self._recover_entry_time(symbol)
            
            print(f"📊 {symbol} agent position: {agent.position_state}")

    def _recover_entry_time(self, symbol: str):
        """Recover entry time from trade history for existing positions"""
        try:
            path = f"backend/storage/performance_logs/{symbol}_trades.json"
            if os.path.exists(path):
                with open(path, "r") as f:
                    trades = json.load(f)
                
                # Find last buy trade
                for trade in reversed(trades):
                    if trade.get("signal") == "buy":
                        timestamp_str = trade.get("timestamp", "")
                        if timestamp_str:
                            entry_time = dateutil.parser.parse(timestamp_str)
                            self.position_entry_times[symbol] = entry_time
                            print(f"🔄 Recovered entry time for {symbol}: {entry_time}")
                            break
        except Exception as e:
            print(f"⚠️ Could not recover entry time for {symbol}: {e}")

    def _validate_agent_position(self, symbol: str, claimed_state: Union[str, None]) -> Optional[str]:
        """Validate agent position state against trade history"""
        try:
            path = f"backend/storage/performance_logs/{symbol}_trades.json"
            if not os.path.exists(path):
                return None
        
            with open(path, "r") as f:
                trades = json.load(f)
        
            if not trades:
                return None
        
            last_trade = trades[-1]
            last_signal = last_trade.get("signal", "")
            last_timestamp = last_trade.get("timestamp", "")
    
            if last_signal == "buy":
                correct_state = "long"
            elif last_signal == "sell":
                correct_state = None
            else:
                correct_state = None
        
            # Check if position is stale
            if correct_state == "long" and self._is_position_stale(last_timestamp):
                print(f"⚠️ {symbol} position appears stale, resetting to flat")
                correct_state = None
        
            return correct_state
    
        except Exception as e:
            print(f"⚠️ Failed to validate position for {symbol}: {e}")
            return None

    def decide_trades(self, top_n=1, min_score=0.5, min_confidence=0.7):
        """Enhanced trade decision with comprehensive filtering including hold time checks"""
        agents = self.load_agents()
        evaluations = self.evaluate_agents(agents)
        
        current_positions = self.get_current_positions()
        current_time = datetime.now()

        print(f"\n🔍 ENHANCED TRADE DECISION ANALYSIS (Data: {self.data_interval})")
        print(f"📊 Filters: min_score={min_score}, min_confidence={min_confidence}")
        print(f"⏰ Minimum hold time: {self.minimum_hold_time}s")
        print("=" * 80)

        approved_trades = []

        for e in evaluations:
            symbol = e["symbol"]
            signal = e["signal"]
            confidence = e["confidence"]
            score = e["score"]
            position_state = e.get("position_state")

            print(f"\n📊 {symbol}:")
            print(f"   Signal: {signal.upper()} | Confidence: {confidence:.3f} | Score: {score:.3f}")
            print(f"   Position: {position_state}")

            skip_reason = None

            # Existing checks
            if score < min_score:
                skip_reason = f"score too low ({score:.3f} < {min_score})"
            elif self.cooldown_manager.is_in_cooldown(symbol):
                cooldown_remaining = self.cooldown_manager.get_cooldown_remaining(symbol)
                skip_reason = f"in cooldown ({cooldown_remaining:.0f}s remaining)"
            elif confidence < min_confidence:
                skip_reason = f"confidence too low ({confidence:.3f} < {min_confidence})"
            elif signal not in ["buy", "sell"]:
                skip_reason = f"signal is '{signal}' (not actionable)"
            
            # NEW: Check minimum hold time for sell signals
            elif signal == "sell" and position_state == "long":
                entry_time = self.position_entry_times.get(symbol)
                if entry_time:
                    hold_duration = (current_time - entry_time).total_seconds()
                    if hold_duration < self.minimum_hold_time:
                        skip_reason = f"minimum hold time not met ({hold_duration:.0f}s < {self.minimum_hold_time}s)"

            elif signal == "buy":
                df = self._fetch_agent_data(symbol)
                market_ok, market_reason = self.risk_manager.check_market_conditions(symbol, df)
                if not market_ok:
                    skip_reason = f"market conditions: {market_reason}"
                elif position_state == "long":
                    actual_state = self._validate_agent_position(symbol, position_state)
                    if actual_state == "long":
                        skip_reason = f"already in long position"
                    else:
                        agent = self.get_agent_by_symbol(symbol)
                        if agent:
                            agent.position_state = actual_state
                            print(f"   🔧 Corrected position state to: {actual_state}")
                else:
                    position_size = self.risk_manager.calculate_dynamic_position_size(symbol, 0, df)
                    portfolio_ok, portfolio_reason = self.risk_manager.check_portfolio_limits(
                        symbol, position_size, current_positions
                    )
                    if not portfolio_ok:
                        skip_reason = f"portfolio limit: {portfolio_reason}"

            elif signal == "sell" and position_state is None:
                actual_state = self._validate_agent_position(symbol, position_state)
                if actual_state is None:
                    skip_reason = f"no position to sell"
                else:
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
        print(f"   Total evaluated: {len(evaluations)}")
        print(f"   Approved trades: {len(approved_trades)}")
        print(f"   Minimum hold time: {self.minimum_hold_time}s")
        
        risk_metrics = self.risk_manager.get_risk_metrics()
        print(f"   Portfolio Status:")
        print(f"     Value: ${risk_metrics['portfolio_value']:.2f}")
        print(f"     Drawdown: {risk_metrics['current_drawdown']:.3f}")
        print("=" * 80)

        return approved_trades[:top_n]

    # NEW: Configuration methods for dynamic adjustment
    def set_minimum_hold_time(self, seconds: int):
        """Dynamically adjust minimum hold time"""
        self.minimum_hold_time = seconds
        print(f"🔧 Minimum hold time set to {seconds}s ({seconds/60:.1f} minutes)")

    def set_exit_check_interval(self, seconds: int):
        """Dynamically adjust exit check interval"""
        self.exit_check_interval = seconds
        print(f"🔧 Exit check interval set to {seconds}s ({seconds/60:.1f} minutes)")

    def get_position_hold_times(self) -> Dict:
        """Get current hold times for all positions"""
        hold_times = {}
        current_time = datetime.now()
        
        for symbol, entry_time in self.position_entry_times.items():
            if entry_time:
                hold_duration = (current_time - entry_time).total_seconds()
                hold_times[symbol] = {
                    'entry_time': entry_time,
                    'hold_duration_seconds': hold_duration,
                    'hold_duration_minutes': hold_duration / 60,
                    'meets_minimum_hold': hold_duration >= self.minimum_hold_time,
                    'remaining_hold_time': max(0, self.minimum_hold_time - hold_duration)
                }
        
        return hold_times


# NEW: Enhanced utility functions
def enhanced_trading_loop_with_smart_exits():
    """Enhanced trading loop with intelligent exit timing"""
    mother_ai = MotherAI(
        agent_symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
        data_interval="1m"
    )
    
    # Configure timing
    mother_ai.set_minimum_hold_time(900)  # 15 minutes
    mother_ai.set_exit_check_interval(300)  # 5 minutes
    
    import time
    
    decision_count = 0
    
    while True:
        try:
            decision_count += 1
            print(f"\n{'='*80}")
            print(f"DECISION CYCLE #{decision_count}")
            print(f"{'='*80}")
            
            # Show current hold times
            hold_times = mother_ai.get_position_hold_times()
            if hold_times:
                print(f"📊 Current Position Hold Times:")
                for symbol, info in hold_times.items():
                    print(f"   {symbol}: {info['hold_duration_minutes']:.1f}min "
                          f"(Min hold: {'✅' if info['meets_minimum_hold'] else '❌'})")
            
            # Make decision (with smart exit logic)
            decision_result = mother_ai.make_portfolio_decision(min_score=0.6)
            
            # Every 5th cycle, run strategic exits
            if decision_count % 5 == 0:
                print(f"\n🎯 Running strategic exit analysis (cycle #{decision_count})")
                mother_ai.check_strategic_exits()
            
            print(f"📊 Decision result: {decision_result}")
            time.sleep(180)  # 3 minutes between decisions
            
        except KeyboardInterrupt:
            print("🛑 Trading loop stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in trading loop: {e}")
            time.sleep(60)


def get_position_status_report(): 
    """Get detailed position status including hold times"""
    mother_ai = MotherAI()
    
    print("\n📊 POSITION STATUS REPORT")
    print("=" * 60)
    
    status = mother_ai.get_agent_status_summary()
    hold_times = mother_ai.get_position_hold_times()
    
    print("Configuration:")
    print(f"  Minimum hold time: {status['minimum_hold_time']}s ({status['minimum_hold_time']/60:.1f}min)")
    print(f"  Exit check interval: {status['exit_check_interval']}s ({status['exit_check_interval']/60:.1f}min)")
    
    active_positions = [s for s, info in status['agents'].items() 
                        if info['position_state'] == 'long']
    
    print(f"\nActive Positions: {len(active_positions)}")
    
    for symbol in active_positions:
        info = status['agents'][symbol]
        hold_info = hold_times.get(symbol, {})
        
        print(f"\n{symbol}:")
        print(f"  Position: {info['position_state']}")
        print(f"  Hold time: {hold_info.get('hold_duration_minutes', 0):.1f} minutes")
        print(f"  Meets min hold: {'✅' if hold_info.get('meets_minimum_hold', False) else '❌'}")
        print(f"  Remaining hold: {hold_info.get('remaining_hold_time', 0):.0f}s")
        print(f"  Next exit check: {info.get('next_exit_check', 0):.0f}s")
        print(f"  In cooldown: {'Yes' if info['in_cooldown'] else 'No'}")
    
    return status, hold_times

def trailing_stop_logic(self, current_price, entry_price, df):
    """Example trailing stop calculation."""
    try:
        profit_percent = (current_price - entry_price) / entry_price
        if profit_percent > 0.02:
            lookback_periods = max(6, int(6 * 60 / self._interval_to_minutes(self.data_interval)))
            recent_high = df['high'].tail(lookback_periods).max()
            if current_price < recent_high * 0.99:
                return "Exit - trailing stop triggered"
    except Exception as e:
        print(f"Error in trailing stop logic: {e}")
    return None
