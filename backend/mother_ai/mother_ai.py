import os
import json
import glob
import importlib.util
import time
from typing import List, Dict

from backend.binance.fetch_live_ohlcv import fetch_ohlcv
from backend.binance.binance_trader import place_market_order
from backend.mother_ai.performance_tracker import PerformanceTracker
from backend.strategy_engine.strategy_health import StrategyHealth
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.mother_ai.meta_evaluator import MetaEvaluator
from backend.mother_ai.profit_calculator import compute_trade_profits
from backend.storage.auto_cleanup import auto_cleanup_logs

TRADE_HISTORY_DIR = "backend/storage/trade_history"
PERFORMANCE_LOG_DIR = "backend/storage/performance_logs"
TRADE_COOLDOWN_SECONDS = 600
MAX_HOLD_SECONDS = 21600  # 6 hours (6 * 60 * 60)
RISK_PER_TRADE = 0.5  # 1% of capital
DEFAULT_BALANCE_USD = 10  # for mock position sizing
TP_RATIO = 1.5  # Take Profit: 1.5x Risk
SL_PERCENT = 0.03  # 3% Stop Loss

os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)


class MotherAI:
    def __init__(self, agents_dir="backend/agents", strategy_dir="backend/storage/strategies", agent_symbols=None):
        self.agents_dir = agents_dir
        self.strategy_dir = strategy_dir
        self.performance_tracker = PerformanceTracker("performance_logs")
        self.agent_symbols = agent_symbols or []
        self.meta_evaluator = MetaEvaluator()
        self.cooldown_tracker = {}
        self.loaded_agents = {}  # Cache loaded agent instances
        
        # Initialize agents on startup
        self._initialize_agents()

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

    def sync_agent_positions(self):
        """Sync agent position states - called before each decision cycle"""
        print("🔄 Syncing agent positions...")
        for symbol, agent in self.loaded_agents.items():
            if hasattr(agent, 'position_state'):
                pos_state = agent.position_state
                print(f"📊 {symbol} agent position: {pos_state}")
            else:
                print(f"⚠️ {symbol} agent has no position_state attribute")

    def evaluate_agents(self, agents):
        """Evaluate agents using their full decision output"""
        results = []
        trade_tracker = PerformanceTracker("trade_history")
        
        for agent in agents:
            ohlcv = self._fetch_agent_data(agent.symbol)
            if ohlcv is None:
                continue
                
            # Use the agent's full evaluation result
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

    def _fetch_agent_data(self, symbol: str):
        symbol = symbol if symbol.endswith("/USDT") else symbol.replace("USDT", "") + "/USDT"
        df = fetch_ohlcv(symbol, "1h", 100)
        return df if not df.empty else None

    def _safe_evaluate_agent(self, agent, ohlcv):
        """Safely evaluate agent and preserve full decision context"""
        try:
            # Use the agent's evaluate method which returns full decision dict
            decision = agent.evaluate(ohlcv)
            
            # Convert agent's decision format to what MotherAI expects
            return {
                "symbol": agent.symbol,
                "signal": decision.get("action", "hold").lower(),
                "confidence": decision.get("confidence", 0.0),
                "source": decision.get("source", "unknown"),
                "position_state": decision.get("position_state"),
                "ml_available": decision.get("ml", {}).get("available", False),
                "ml_confidence": decision.get("ml", {}).get("confidence", 0.0),
                "rule_confidence": decision.get("rule_based", {}).get("confidence", 0.0),
                "timestamp": decision.get("timestamp"),
                "full_decision": decision  # Preserve full agent decision
            }
        except Exception as e:
            print(f"⚠️ Agent evaluation error for {agent.symbol}: {e}")
            return {
                "symbol": agent.symbol,
                "signal": "hold",
                "confidence": 0.0,
                "source": "error",
                "position_state": None,
                "ml_available": False,
                "ml_confidence": 0.0,
                "rule_confidence": 0.0,
                "full_decision": {}
            }

    def _calculate_score(self, prediction, health):
        return self.meta_evaluator.predict_refined_score({
            "confidence": prediction["confidence"],
            "win_rate": health.get("win_rate", 0.5),
            "drawdown_penalty": 1.0 - health.get("max_drawdown", 0.3),
            "is_buy": int(prediction["signal"] == "buy"),
            "is_sell": int(prediction["signal"] == "sell")
        })

    def _log_trade_execution(self, symbol, signal, price, confidence, score, timestamp, qty=None, source="mother_ai_decision"):
        """Log trade execution with quantity information"""
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
            "source": source
        }

        try:
            history = json.load(open(path)) if os.path.exists(path) else []
        except Exception:
            history = []

        history.append(entry)

        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def is_in_cooldown(self, symbol):
        return symbol in self.cooldown_tracker and (time.time() - self.cooldown_tracker[symbol]) < TRADE_COOLDOWN_SECONDS

    def decide_trades(self, top_n=1, min_score=0.5, min_confidence=0.7):
        """Simplified trade decision - trust agent position management"""
        agents = self.load_agents()
        evaluations = self.evaluate_agents(agents)

        filtered = []
        for e in evaluations:
            symbol = e["symbol"]
            
            # Skip if score too low or in cooldown
            if e["score"] < min_score:
                print(f"⏭️ Skipping {symbol}: score too low ({e['score']:.3f} < {min_score})")
                continue
                
            if self.is_in_cooldown(symbol):
                print(f"⏭️ Skipping {symbol}: in cooldown")
                continue

            # Skip if confidence too low
            if e["confidence"] < min_confidence:
                print(f"⏭️ Skipping {symbol}: confidence too low ({e['confidence']:.3f} < {min_confidence})")
                continue

            # Trust the agent's decision - it already handled position management
            signal = e["signal"]
            if signal in ["buy", "sell"]:
                print(f"✅ Agent decision approved: {symbol} {signal.upper()} "
                      f"(confidence: {e['confidence']:.3f}, score: {e['score']:.3f})")
                print(f"    Agent position state: {e.get('position_state')}")
                print(f"    Decision source: {e.get('source')}")
                filtered.append(e)
            else:
                print(f"⏭️ Skipping {symbol}: signal is '{signal}' (not buy/sell)")

        print(f"📊 Approved trades: {[f'{t['symbol']}-{t['signal']}' for t in filtered]}")
        return filtered[:top_n]

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

    def execute_trade(self, symbol, signal, price, confidence, live=False):
        if signal not in ("buy", "sell") or not price:
            print(f"❌ Invalid trade parameters: signal={signal}, price={price}")
            return None

        # Get current trading mode from binance_api.py
        from backend.utils.binance_api import get_trading_mode, get_binance_client
        current_mode = get_trading_mode()
    
        print(f"🚀 Executing {signal.upper()} order for {symbol} at ${price:.4f} | Mode: {current_mode.upper()}")

        qty = None

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
                    
                    # Use actual balance instead of DEFAULT_BALANCE_USD
                    available_balance = min(usdt_balance * 0.9, 20)  # Use 90% of balance, max $20
                    print(f"💡 Using ${available_balance:.2f} for this trade")
                    
                except Exception as balance_err:
                    print(f"⚠️ Could not check balance: {balance_err}")
                    available_balance = DEFAULT_BALANCE_USD
            else:
                available_balance = DEFAULT_BALANCE_USD

            # Format symbol for Binance API
            binance_symbol = symbol if "/" in symbol else symbol.replace("USDT", "/USDT")

            if signal == "buy":
                sl = price * (1 - SL_PERCENT)
                tp = price + ((price - sl) * TP_RATIO)
            
                # Calculate quantity based on available balance
                if current_mode == "live":
                    # For live trading, use smaller position size
                    risk_amount = available_balance * 0.5  # Use 50% of available
                    qty = risk_amount / price  # Simple calculation based on price
                else:
                    # Mock trading uses original calculation
                    risk_amount = DEFAULT_BALANCE_USD * RISK_PER_TRADE
                    qty = risk_amount / (price - sl)

                if qty <= 0:
                    print(f"⚠️ Computed qty is zero or negative for BUY on {symbol}, skipping.")
                    return None

                print(f"📊 Buying {symbol} | Entry: {price:.4f}, SL: {sl:.4f}, TP: {tp:.4f}, Qty: {qty:.6f}")
                print(f"📊 Risk amount: ${risk_amount:.2f}, Total cost: ${qty * price:.2f}")

                # Use the updated place_market_order function
                order = place_market_order(binance_symbol, signal, qty)

                if order and order.get("status") != "mock":
                    self.cooldown_tracker[symbol] = time.time()
                    # Update agent's position state
                    self.update_agent_position_state(symbol, signal)
                    print(f"✅ BUY order executed for {symbol} in {current_mode.upper()} mode")
                elif order and order.get("status") == "mock":
                    # Mock order - still update agent position state
                    self.cooldown_tracker[symbol] = time.time()
                    self.update_agent_position_state(symbol, signal)
                    print(f"✅ MOCK BUY order logged for {symbol}")
                else:
                    print(f"❌ BUY order failed for {symbol}")

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
                    # For mock, use default calculation
                    risk_amount = DEFAULT_BALANCE_USD * RISK_PER_TRADE
                    qty = risk_amount / price

                if qty <= 0:
                    print(f"⚠️ Computed qty is zero or negative for SELL on {symbol}, skipping.")
                    return None

                print(f"📊 Selling {symbol} | Qty: {qty:.6f} at ${price:.4f}")

                # Use the updated place_market_order function
                order = place_market_order(binance_symbol, signal, qty)

                if order and order.get("status") != "mock":
                    self.cooldown_tracker[symbol] = time.time()
                    # Update agent's position state
                    self.update_agent_position_state(symbol, signal)
                    print(f"✅ SELL order executed for {symbol} in {current_mode.upper()} mode")
                elif order and order.get("status") == "mock":
                    # Mock order - still update agent position state
                    self.cooldown_tracker[symbol] = time.time()
                    self.update_agent_position_state(symbol, signal)
                    print(f"✅ MOCK SELL order logged for {symbol}")
                else:
                    print(f"❌ SELL order failed for {symbol}")

        except Exception as e:
            print(f"❌ Trade execution error for {symbol}: {e}")

        return qty

    def check_exit_conditions_for_agents(self):
        """Check exit conditions using agent position states"""
        print("🔍 Checking exit conditions using agent states...")
        
        for symbol, agent in self.loaded_agents.items():
            if not hasattr(agent, 'position_state') or agent.position_state != "long":
                continue
                
            # Agent thinks it has a long position, check if we should exit
            df = self._fetch_agent_data(symbol)
            if df is None or df.empty:
                continue
                
            current_price = df["close"].iloc[-1]
            
            # Get position info from agent's trade history if available
            exit_reason = self._check_agent_exit_conditions(agent, current_price)
            
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

    def _check_agent_exit_conditions(self, agent, current_price):
        """Check if agent position should be closed based on SL/TP/Time"""
        # This is a simplified version - you might want to get more sophisticated
        # position data from the agent's trade history or internal state
        
        # For now, we'll use a basic time-based exit as an example
        # In a full implementation, you'd want to track entry prices and times
        
        # You could extend GenericAgent to provide position entry details
        # or read from the trade logs to determine entry price/time
        
        return None  # Placeholder - implement based on your specific needs

    def make_portfolio_decision(self, min_score=0.5):
        auto_cleanup_logs()
        print(f"🎯 Making portfolio decision with min_score={min_score}")

        # Sync agent positions before making decisions
        self.sync_agent_positions()

        timestamp = self.performance_tracker.current_time()

        # Step 1: Check exit conditions using agent states
        self.check_exit_conditions_for_agents()

        # Step 2: Evaluate new trades (agents handle their own position management)
        top = self.decide_trades(min_score=min_score)
        if not top:
            print("📭 No trades meet criteria")
            return {"decision": [], "timestamp": timestamp}

        decision = top[0]
        df = self._fetch_agent_data(decision["symbol"])
        price = df["close"].iloc[-1] if df is not None and not df.empty else None
        result = {**decision, "last_price": price}

        print(f"🎯 Top decision: {decision['symbol']} {decision['signal']} "
              f"(score: {decision['score']:.3f}, confidence: {decision['confidence']:.3f})")
        print(f"    Source: {decision.get('source')}, ML Available: {decision.get('ml_available')}")

        if decision["signal"] in ("buy", "sell") and price:
            # Execute the trade
            executed_qty = self.execute_trade(decision["symbol"], decision["signal"], price, decision["confidence"])
            
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

        return {"decision": result, "timestamp": timestamp}

    def load_all_predictions(self) -> List[Dict]:
        """Load predictions from all agents"""
        predictions = []
        agents = self.load_agents()
        
        for agent in agents:
            ohlcv = self._fetch_agent_data(agent.symbol)
            if ohlcv is not None:
                prediction = self._safe_evaluate_agent(agent, ohlcv)
                predictions.append(prediction)
                
        return predictions

    def get_agent_status_summary(self) -> Dict:
        """Get summary of all agent states"""
        summary = {
            "total_agents": len(self.loaded_agents),
            "agents": {}
        }
        
        for symbol, agent in self.loaded_agents.items():
            agent_info = {
                "symbol": symbol,
                "position_state": getattr(agent, 'position_state', None),
                "has_ml_model": getattr(agent, 'model', None) is not None,
                "in_cooldown": self.is_in_cooldown(symbol)
            }
            
            if hasattr(agent, 'get_model_info'):
                model_info = agent.get_model_info()
                agent_info.update({
                    "model_loaded": model_info.get("model_loaded", False),
                    "model_classes": model_info.get("classes", [])
                })
            
            summary["agents"][symbol] = agent_info
            
        return summary

    # Legacy method for backward compatibility
    def check_exit_conditions(self, symbol, current_price):
        """Legacy method - kept for backward compatibility"""
        return None