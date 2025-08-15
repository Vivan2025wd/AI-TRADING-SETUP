import os
import json
from dateutil import parser
from typing import Optional
from datetime import datetime, timedelta

import dateutil

from backend.binance.binance_trader import place_market_order
from .mother_ai_core import MotherAICore, PERFORMANCE_LOG_DIR

class MotherAITrader(MotherAICore):
    """Trading execution functionality for MotherAI - handles order execution and trade logging"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        
        # Track entry times for minimum hold period enforcement
        if signal == "buy":
            self.position_entry_times[symbol] = datetime.now()
            print(f"📝 Recorded entry time for {symbol}: {self.position_entry_times[symbol]}")

    def execute_trade(self, symbol, signal, price, confidence, live=False):
        """Enhanced trade execution with comprehensive risk checks"""
        if signal not in ("buy", "sell") or not price:
            print(f"❌ Invalid trade parameters: signal={signal}, price={price}")
            return None

        df = self._fetch_agent_data(symbol)
        
        print(f"📈 Executing trade with {self.data_interval} data interval")

        # Risk checks
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

        # Continue with trade execution
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
            
            # Check minimum hold time for sell signals
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
            
            # Recover entry times from trade history if missing
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

    def _validate_agent_position(self, symbol: str, claimed_state: Optional[str]) -> Optional[str]:
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

    def _is_position_stale(self, timestamp_str: str, max_hours: int = 24) -> bool:
        """Check if a position is older than max_hours"""
        try:
            trade_time = parser.parse(timestamp_str)  # using parser directly
            now = datetime.now(trade_time.tzinfo) if trade_time.tzinfo else datetime.now()
            return (now - trade_time) > timedelta(hours=max_hours)
        except Exception:
            return False