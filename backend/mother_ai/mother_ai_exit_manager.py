import os
import json
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
import dateutil.parser
from .mother_ai_trader import MotherAITrader
from backend.mother_ai.profit_calculator import compute_trade_profits


class MotherAIExitManager(MotherAITrader):
    """Exit conditions and risk management for MotherAI - handles exit logic and portfolio decisions"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
            
            # Skip if not time to check yet (unless forced)
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
        
        # Check minimum hold time first
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

    def check_strategic_exits(self):
        """Check for strategic exit opportunities - called separately from regular decision making"""
        print(f"🎯 Running strategic exit analysis...")
        self.check_exit_conditions_for_agents(force_check=True)

    def make_portfolio_decision(self, min_score=0.7):
        """Enhanced portfolio decision making - SEPARATED EXIT LOGIC FROM NEW TRADE LOGIC"""
        from backend.storage.auto_cleanup import auto_cleanup_logs
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

        # Only check exits occasionally, not every decision cycle
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

    def get_agent_status_summary(self) -> dict:
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