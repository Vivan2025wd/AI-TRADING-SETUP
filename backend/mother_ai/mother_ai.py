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

TRADE_HISTORY_DIR =  "backend/storage/trade_history"
PERFORMANCE_LOG_DIR = "backend/storage/performance_logs"
TRADE_COOLDOWN_SECONDS = 600
MAX_HOLD_SECONDS = 21600  # 6 hours (6 * 60 * 60)
RISK_PER_TRADE = 0.01  # 1% of capital
DEFAULT_BALANCE_USD = 1000  # for mock position sizing
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
        self.position_tracker = {}
        self._sync_positions_on_startup()  # ✅ Sync positions on startup

    def _sync_positions_on_startup(self):
        """Sync position tracker with agent position states"""
        print("🔄 Syncing position tracker with agent states...")
        for file in os.listdir(self.agents_dir):
            if not file.endswith("_agent.py"):
                continue
            symbol = file.replace("_agent.py", "").upper()
            
            # Try to get agent's position state by creating an instance
            try:
                strategy = self._load_strategy(symbol)
                agent_class = self._load_agent_class(file, symbol)
                if agent_class is not None:
                    # Create agent instance to access position_state
                    agent_instance = agent_class(symbol=symbol, strategy_logic=strategy)
                    if hasattr(agent_instance, 'position_state'):
                        if agent_instance.position_state == "long":
                            self.position_tracker[symbol] = {"side": "long"}
                            print(f"📊 Synced {symbol}: long position")
                        else:
                            self.position_tracker[symbol] = None
                            print(f"📊 Synced {symbol}: flat position")
                    else:
                        self.position_tracker[symbol] = None
                        print(f"📊 Synced {symbol}: no position state available")
            except Exception as e:
                print(f"⚠️ Could not sync position for {symbol}: {e}")
                self.position_tracker[symbol] = None

    def load_agents(self):
        agents = []
        for file in os.listdir(self.agents_dir):
            if not file.endswith("_agent.py"):
                continue
            symbol = file.replace("_agent.py", "").upper()
            if self.agent_symbols and symbol not in self.agent_symbols:
                continue
            strategy = self._load_strategy(symbol)
            agent_class = self._load_agent_class(file, symbol)
            if agent_class:
                agent_instance = agent_class(symbol=symbol, strategy_logic=strategy)
                agents.append(agent_instance)
                
                # ✅ Sync position state from agent instance
                if hasattr(agent_instance, 'position_state'):
                    if agent_instance.position_state == "long":
                        self.position_tracker[symbol] = {"side": "long"}
                        print(f"📊 Agent {symbol} reports: long position")
                    else:
                        self.position_tracker[symbol] = None
                        print(f"📊 Agent {symbol} reports: flat position")
                else:
                    self.position_tracker[symbol] = None
                    print(f"📊 Agent {symbol} has no position state")
                        
        return agents

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

    def evaluate_agents(self, agents):
        results = []
        trade_tracker = PerformanceTracker("trade_history")
        for agent in agents:
            ohlcv = self._fetch_agent_data(agent.symbol)
            if ohlcv is None:
                continue
            prediction = self._safe_predict(agent, ohlcv)
            health = StrategyHealth(trade_tracker.get_agent_log(agent.symbol)).summary()
            score = self._calculate_score(prediction, health)
            results.append({**prediction, "score": round(score, 3)})
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _fetch_agent_data(self, symbol: str):
        symbol = symbol if symbol.endswith("/USDT") else symbol.replace("USDT", "") + "/USDT"
        df = fetch_ohlcv(symbol, "1h", 100)
        return df if not df.empty else None

    def _safe_predict(self, agent, ohlcv):
        try:
            # Use the evaluate method from GenericAgent which returns a dict
            prediction = agent.evaluate(ohlcv)
            return {
                "symbol": agent.symbol,
                "signal": prediction.get("action", "hold").lower(),
                "confidence": prediction.get("confidence", 0.0)
            }
        except Exception as e:
            print(f"⚠️ Prediction error for {agent.symbol}: {e}")
            return {"symbol": agent.symbol, "signal": "hold", "confidence": 0.0}

    def _calculate_score(self, prediction, health):
        return self.meta_evaluator.predict_refined_score({
            "confidence": prediction["confidence"],
            "win_rate": health.get("win_rate", 0.5),
            "drawdown_penalty": 1.0 - health.get("max_drawdown", 0.3),
            "is_buy": int(prediction["signal"] == "buy"),
            "is_sell": int(prediction["signal"] == "sell")
        })

    def _log_trade_execution(self, symbol, signal, price, confidence, score, timestamp):
        path = os.path.join(PERFORMANCE_LOG_DIR, f"{symbol}_trades.json")
        entry = {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(confidence, 4),
            "score": round(score, 4),
            "last_price": round(price, 4),
            "price": round(price, 4),
            "timestamp": timestamp,
            "source": "mother_ai_decision"
        }

        try:
            history = json.load(open(path)) if os.path.exists(path) else []
        except Exception:
            history = []

        history.append(entry)

        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def _is_previous_trade_open_buy(self, symbol: str) -> bool:
        """Check if the last trade was a buy that hasn't been closed by a sell"""
        path = os.path.join(PERFORMANCE_LOG_DIR, f"{symbol}_trades.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r") as f:
                history = json.load(f)
                if not history:
                    return False
                    
                # Look through recent trades to see if last buy was closed
                for entry in reversed(history):
                    if entry["signal"] == "buy":
                        return True
                    elif entry["signal"] == "sell":
                        return False
                        
        except Exception as e:
            print(f"⚠️ Error checking previous trade for {symbol}: {e}")
            return False
        return False

    def is_in_cooldown(self, symbol):
        return symbol in self.cooldown_tracker and (time.time() - self.cooldown_tracker[symbol]) < TRADE_COOLDOWN_SECONDS

    def decide_trades(self, top_n=1, min_score=0.5, min_confidence=0.7):
        agents = self.load_agents()
        evaluations = self.evaluate_agents(agents)

        filtered = []
        for e in evaluations:
            if e["score"] < min_score or self.is_in_cooldown(e["symbol"]):
                print(f"⏭️ Skipping {e['symbol']}: score={e['score']:.3f}, in_cooldown={self.is_in_cooldown(e['symbol'])}")
                continue

            current_position = self.position_tracker.get(e["symbol"], None)
            print(f"🔍 Evaluating {e['symbol']}: signal={e['signal']}, position={current_position}, confidence={e['confidence']:.3f}")

            if e["signal"] == "buy":
                # ✅ Allow buy only if no position or position is None
                if current_position is None and e["confidence"] >= min_confidence:
                    # Check if last buy was not closed by sell
                    if not self._is_previous_trade_open_buy(e["symbol"]):
                        print(f"✅ BUY signal approved for {e['symbol']}")
                        filtered.append(e)
                    else:
                        print(f"⛔ Previous buy still open for {e['symbol']}")
                else:
                    reason = "has_position" if current_position else "low_confidence"
                    print(f"⛔ Cannot buy {e['symbol']}: {reason} (confidence={e['confidence']:.3f})")
                    
            elif e["signal"] == "sell":
                # ✅ Allow sell if we have a long position or if agent thinks it should sell
                if current_position is not None and current_position.get("side") == "long" and e["confidence"] >= min_confidence:
                    print(f"✅ SELL signal approved for {e['symbol']}")
                    filtered.append(e)
                elif current_position is None and e["confidence"] >= min_confidence:
                    # ✅ Agent might know about a position we don't track
                    print(f"⚠️ Agent wants to sell {e['symbol']} but Mother AI shows no position. Allowing sell anyway.")
                    filtered.append(e)
                else:
                    reason = "no_position" if current_position is None else "low_confidence"
                    print(f"⛔ Cannot sell {e['symbol']}: {reason} (confidence={e['confidence']:.3f})")

        print(f"📊 Filtered trades: {[f'{t["symbol"]}-{t["signal"]}' for t in filtered]}")
        return filtered[:top_n]

    def execute_trade(self, symbol, signal, price, confidence):
        if signal not in ("buy", "sell") or not price:
            print(f"❌ Invalid trade parameters: signal={signal}, price={price}")
            return
            
        print(f"🚀 Executing {signal.upper()} order for {symbol} at ${price:.4f}")
        
        try:
            if signal == "buy":
                sl = price * (1 - SL_PERCENT)
                tp = price + ((price - sl) * TP_RATIO)
                risk_amount = DEFAULT_BALANCE_USD * RISK_PER_TRADE
                qty = risk_amount / (price - sl)

                # ✅ Sanity check qty
                if qty <= 0:
                    print(f"⚠️ Computed qty is zero or negative for BUY on {symbol}, skipping.")
                    return

                print(f"📊 Buying {symbol} | Entry: {price:.4f}, SL: {sl:.4f}, TP: {tp:.4f}, Qty: {qty:.6f}")
                
                # Format symbol for Binance API
                binance_symbol = symbol.replace("USDT", "/USDT") if not "/" in symbol else symbol
                order = place_market_order(binance_symbol, signal, qty)
                
                if order:
                    self.cooldown_tracker[symbol] = time.time()
                    self.position_tracker[symbol] = {
                        "side": "long",
                        "entry_price": price,
                        "entry_time": time.time(),
                        "stop_loss": sl,
                        "take_profit": tp,
                        "qty": qty
                    }
                    print(f"✅ BUY order executed for {symbol}")
                else:
                    print(f"❌ BUY order failed for {symbol}")

            elif signal == "sell":
                pos = self.position_tracker.get(symbol, {})
                qty = pos.get("qty", 0) if isinstance(pos, dict) else 0

                # ✅ If we don't have qty tracked, use a default or get from agent
                if qty <= 0:
                    print(f"⚠️ No quantity tracked for {symbol}. Using default qty for sell.")
                    # Calculate a reasonable default quantity based on risk amount
                    risk_amount = DEFAULT_BALANCE_USD * RISK_PER_TRADE
                    qty = risk_amount / price  # Simple approximation

                print(f"📊 Selling {symbol} | Qty: {qty:.6f} at ${price:.4f}")
                
                # Format symbol for Binance API
                binance_symbol = symbol.replace("USDT", "/USDT") if not "/" in symbol else symbol
                order = place_market_order(binance_symbol, signal, qty)
                
                if order:
                    self.cooldown_tracker[symbol] = time.time()
                    self.position_tracker[symbol] = None  # Clear position
                    print(f"✅ SELL order executed for {symbol}")
                else:
                    print(f"❌ SELL order failed for {symbol}")

        except Exception as e:
            print(f"❌ Trade execution error for {symbol}: {e}")

    def make_portfolio_decision(self, min_score=0.5):
        auto_cleanup_logs()
        print(f"🎯 Making portfolio decision with min_score={min_score}")

        timestamp = self.performance_tracker.current_time()

        # Step 1: Check open positions for SL/TP/Timeout exit
        for symbol, pos in list(self.position_tracker.items()):
            if isinstance(pos, dict) and pos.get("side") == "long":
                df = self._fetch_agent_data(symbol)
                price = df["close"].iloc[-1] if df is not None and not df.empty else None
                if price:
                    # ✅ Show debug info
                    held_for = int(time.time() - pos.get("entry_time", 0))
                    print(
                        f"🔍 Checking {symbol} | Held for: {held_for}s | "
                        f"SL: {pos.get('stop_loss', 0):.4f}, TP: {pos.get('take_profit', 0):.4f}, Now: {price:.4f}"
                    )
                    
                    exit_reason = self.check_exit_conditions(symbol, price)
                    if exit_reason:
                        self._log_trade_execution(
                            symbol=symbol,
                            signal="sell",
                            price=price,
                            confidence=1.0,
                            score=1.0,
                            timestamp=timestamp
                        )
                        compute_trade_profits(symbol)

        # Step 2: Evaluate new trades
        top = self.decide_trades(min_score=min_score)
        if not top:
            print("📭 No trades meet criteria")
            return {"decision": [], "timestamp": timestamp}

        decision = top[0]
        df = self._fetch_agent_data(decision["symbol"])
        price = df["close"].iloc[-1] if df is not None and not df.empty else None
        result = {**decision, "last_price": price}

        print(f"🎯 Top decision: {decision['symbol']} {decision['signal']} (score: {decision['score']:.3f}, confidence: {decision['confidence']:.3f})")

        if decision["signal"] in ("buy", "sell") and price:
            self.execute_trade(decision["symbol"], decision["signal"], price, decision["confidence"])
            self._log_trade_execution(
                symbol=decision["symbol"],
                signal=decision["signal"],
                price=price,
                confidence=decision["confidence"],
                score=decision["score"],
                timestamp=timestamp
            )
            if decision["signal"] == "sell":
                compute_trade_profits(decision["symbol"])

        return {"decision": result, "timestamp": timestamp}

    def load_all_predictions(self) -> List[Dict]:
        return []

    def check_exit_conditions(self, symbol, current_price):
        pos = self.position_tracker.get(symbol)
        if not pos or pos.get("side") != "long":
            return None  # No open position

        # ✅ Guard for missing entry_time
        entry_time = pos.get("entry_time")
        if not entry_time:
            print(f"⚠️ Missing entry_time for {symbol}, forcing position close.")
            self.position_tracker[symbol] = None
            return "invalid_entry"

        sl_hit = current_price <= pos.get("stop_loss", 0)
        tp_hit = current_price >= pos.get("take_profit", float('inf'))
        time_expired = (time.time() - entry_time) > MAX_HOLD_SECONDS

        if sl_hit or tp_hit or time_expired:
            reason = "stop_loss" if sl_hit else "take_profit" if tp_hit else "max_hold"
            print(f"💡 Exiting {symbol} due to {reason.upper()}")
            self.execute_trade(symbol, "sell", current_price, confidence=1.0)
            return reason
        return None