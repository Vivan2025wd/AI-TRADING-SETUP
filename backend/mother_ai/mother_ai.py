import os
import json
import glob
import importlib.util
import time
from typing import List, Dict, Optional
from enum import Enum

from backend.binance.fetch_live_ohlcv import fetch_ohlcv
from backend.binance.binance_trader import place_market_order
from backend.mother_ai.performance_tracker import PerformanceTracker
from backend.strategy_engine.strategy_health import StrategyHealth
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.mother_ai.meta_evaluator import MetaEvaluator
from backend.mother_ai.profit_calculator import compute_trade_profits

TRADE_HISTORY_DIR = "backend/storage/trade_history"
PERFORMANCE_LOG_DIR = "backend/storage/performance_logs"
TRADE_COOLDOWN_SECONDS = 600

os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)


class TradeState(Enum):
    IDLE = "idle"
    SEEKING_BUY = "seeking_buy"
    HOLDING = "holding"
    SEEKING_SELL = "seeking_sell"


class TradePosition:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.state = TradeState.IDLE
        self.entry_price = None
        self.entry_time = None
        self.quantity = None
        self.confidence_at_entry = None
        self.stop_loss = None
        self.take_profit = None
        
    def enter_position(self, price: float, quantity: float, confidence: float):
        """Enter a long position"""
        self.state = TradeState.HOLDING
        self.entry_price = price
        self.entry_time = time.time()
        self.quantity = quantity
        self.confidence_at_entry = confidence
        
    def exit_position(self):
        """Exit the current position"""
        self.state = TradeState.IDLE
        self.entry_price = None
        self.entry_time = None
        self.quantity = None
        self.confidence_at_entry = None
        self.stop_loss = None
        self.take_profit = None
        
    def set_stop_loss(self, price: float):
        """Set stop loss price"""
        self.stop_loss = price
        
    def set_take_profit(self, price: float):
        """Set take profit price"""
        self.take_profit = price
        
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.entry_price and self.quantity:
            return (current_price - self.entry_price) * self.quantity
        return 0.0
        
    def get_duration_hours(self) -> float:
        """Get position duration in hours"""
        if self.entry_time:
            return (time.time() - self.entry_time) / 3600
        return 0.0


class MotherAI:
    def __init__(self, agents_dir="backend/agents", strategy_dir="backend/storage/strategies", agent_symbols=None):
        self.agents_dir = agents_dir
        self.strategy_dir = strategy_dir
        self.performance_tracker = PerformanceTracker("performance_logs")
        self.agent_symbols = agent_symbols or []
        self.meta_evaluator = MetaEvaluator()
        self.cooldown_tracker = {}
        
        # Enhanced trade cycle management
        self.positions = {}  # symbol -> TradePosition
        self.trade_cycle_config = {
            "min_buy_confidence": 0.75,
            "min_sell_confidence": 0.65,
            "max_hold_duration_hours": 24,
            "stop_loss_percentage": 0.05,  # 5% stop loss
            "take_profit_percentage": 0.10,  # 10% take profit
            "seeking_buy_threshold": 0.6,  # Lower threshold for seeking opportunities
        }

    def get_position(self, symbol: str) -> TradePosition:
        """Get or create position tracker for symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = TradePosition(symbol)
        return self.positions[symbol]

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
                agents.append(agent_class(symbol=symbol, strategy_logic=strategy))
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
        return getattr(module, f"{symbol}Agent", None)

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
            self._log_prediction(agent.symbol, prediction, health, score)
            results.append({**prediction, "score": round(score, 3)})
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _fetch_agent_data(self, symbol: str):
        symbol = symbol if symbol.endswith("/USDT") else symbol.replace("USDT", "") + "/USDT"
        df = fetch_ohlcv(symbol, "1h", 100)
        return df if not df.empty else None

    def _safe_predict(self, agent, ohlcv):
        try:
            prediction = agent.predict(ohlcv)
            return {
                "symbol": agent.symbol,
                "signal": prediction.get("action", "hold").lower(),
                "confidence": prediction.get("confidence", 0.0)
            }
        except Exception:
            return {"symbol": agent.symbol, "signal": "hold", "confidence": 0.0}

    def _calculate_score(self, prediction, health):
        return self.meta_evaluator.predict_refined_score({
            "confidence": prediction["confidence"],
            "win_rate": health.get("win_rate", 0.5),
            "drawdown_penalty": 1.0 - health.get("max_drawdown", 0.3),
            "is_buy": int(prediction["signal"] == "buy"),
            "is_sell": int(prediction["signal"] == "sell")
        })

    def _log_prediction(self, symbol, prediction, health, score, price=None):
        position = self.get_position(symbol)
        
        data = {
            **prediction,
            "win_rate": health.get("win_rate", 0.5),
            "score": round(score, 3),
            "price": price,
            "timestamp": self.performance_tracker.current_time(),
            "source": "mother_ai_decision",
            "position_state": position.state.value,
            "unrealized_pnl": position.get_unrealized_pnl(price) if price else 0.0,
            "position_duration_hours": position.get_duration_hours()
        }
        path = os.path.join(TRADE_HISTORY_DIR, f"{symbol}_predictions.json")
        try:
            existing = json.load(open(path)) if os.path.exists(path) else []
        except:
            existing = []
        existing.append(data)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

    def is_in_cooldown(self, symbol):
        return symbol in self.cooldown_tracker and (time.time() - self.cooldown_tracker[symbol]) < TRADE_COOLDOWN_SECONDS

    def check_risk_management(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if position should be closed due to risk management rules"""
        position = self.get_position(symbol)
        
        if position.state != TradeState.HOLDING:
            return None
            
        # Check stop loss
        if position.stop_loss and current_price <= position.stop_loss:
            return "stop_loss"
            
        # Check take profit
        if position.take_profit and current_price >= position.take_profit:
            return "take_profit"
            
        # Check max hold duration
        if position.get_duration_hours() > self.trade_cycle_config["max_hold_duration_hours"]:
            return "max_duration"
            
        return None

    def execute_trade_cycle(self, symbol: str, signal: str, confidence: float, current_price: float) -> Dict:
        """Execute the BUY → HOLD → SELL trade cycle"""
        position = self.get_position(symbol)
        config = self.trade_cycle_config
        
        # Check risk management first
        risk_action = self.check_risk_management(symbol, current_price)
        if risk_action:
            return self._execute_risk_management_exit(symbol, risk_action, current_price)
        
        # State machine logic
        if position.state == TradeState.IDLE:
            # Look for BUY opportunities
            if signal == "buy" and confidence >= config["min_buy_confidence"]:
                return self._execute_buy(symbol, current_price, confidence)
            elif signal == "buy" and confidence >= config["seeking_buy_threshold"]:
                position.state = TradeState.SEEKING_BUY
                return {"action": "seeking_buy", "symbol": symbol, "confidence": confidence}
            else:
                return {"action": "idle", "symbol": symbol, "confidence": confidence}
                
        elif position.state == TradeState.SEEKING_BUY:
            # Actively seeking better BUY opportunity
            if signal == "buy" and confidence >= config["min_buy_confidence"]:
                return self._execute_buy(symbol, current_price, confidence)
            elif signal in ["sell", "hold"] and confidence > 0.7:
                # Strong signal to stop seeking
                position.state = TradeState.IDLE
                return {"action": "stop_seeking", "symbol": symbol, "confidence": confidence}
            else:
                return {"action": "seeking_buy", "symbol": symbol, "confidence": confidence}
                
        elif position.state == TradeState.HOLDING:
            # In position, look for SELL opportunities
            if signal == "sell" and confidence >= config["min_sell_confidence"]:
                return self._execute_sell(symbol, current_price, confidence)
            elif signal == "sell" and confidence >= 0.5:
                position.state = TradeState.SEEKING_SELL
                return {"action": "seeking_sell", "symbol": symbol, "confidence": confidence}
            else:
                return {"action": "holding", "symbol": symbol, "confidence": confidence,
                       "unrealized_pnl": position.get_unrealized_pnl(current_price)}
                
        elif position.state == TradeState.SEEKING_SELL:
            # Actively seeking better SELL opportunity
            if signal == "sell" and confidence >= config["min_sell_confidence"]:
                return self._execute_sell(symbol, current_price, confidence)
            elif signal in ["buy", "hold"] and confidence > 0.7:
                # Strong signal to keep holding
                position.state = TradeState.HOLDING
                return {"action": "resume_holding", "symbol": symbol, "confidence": confidence}
            else:
                return {"action": "seeking_sell", "symbol": symbol, "confidence": confidence}
                
        return {"action": "unknown_state", "symbol": symbol, "confidence": confidence}

    def _execute_buy(self, symbol: str, price: float, confidence: float) -> Dict:
        """Execute BUY order and enter position"""
        position = self.get_position(symbol)
        
        try:
            qty = 20 / price  # $20 position size
            order = place_market_order(symbol.replace("USDT", "/USDT"), "buy", qty)
            
            if order:
                position.enter_position(price, qty, confidence)
                
                # Set risk management levels
                stop_loss_price = price * (1 - self.trade_cycle_config["stop_loss_percentage"])
                take_profit_price = price * (1 + self.trade_cycle_config["take_profit_percentage"])
                
                position.set_stop_loss(stop_loss_price)
                position.set_take_profit(take_profit_price)
                
                self.cooldown_tracker[symbol] = time.time()
                
                return {
                    "action": "buy_executed",
                    "symbol": symbol,
                    "price": price,
                    "quantity": qty,
                    "confidence": confidence,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price
                }
            else:
                return {"action": "buy_failed", "symbol": symbol, "confidence": confidence}
                
        except Exception as e:
            print(f"❌ Buy trade error: {e}")
            return {"action": "buy_error", "symbol": symbol, "error": str(e)}

    def _execute_sell(self, symbol: str, price: float, confidence: float) -> Dict:
        """Execute SELL order and exit position"""
        position = self.get_position(symbol)
        
        if position.quantity is None:
            return {"action": "sell_error", "symbol": symbol, "error": "No quantity to sell"}
        
        try:
            order = place_market_order(symbol.replace("USDT", "/USDT"), "sell", position.quantity)
            
            if order:
                pnl = position.get_unrealized_pnl(price)
                duration = position.get_duration_hours()
                sold_quantity = position.quantity
                
                position.exit_position()
                self.cooldown_tracker[symbol] = time.time()
                
                # Calculate profits
                compute_trade_profits(symbol)
                
                return {
                    "action": "sell_executed",
                    "symbol": symbol,
                    "price": price,
                    "quantity": sold_quantity,
                    "confidence": confidence,
                    "pnl": pnl,
                    "duration_hours": duration
                }
            else:
                return {"action": "sell_failed", "symbol": symbol, "confidence": confidence}
                
        except Exception as e:
            print(f"❌ Sell trade error: {e}")
            return {"action": "sell_error", "symbol": symbol, "error": str(e)}

    def _execute_risk_management_exit(self, symbol: str, reason: str, price: float) -> Dict:
        """Execute forced exit due to risk management"""
        position = self.get_position(symbol)
        
        if position.quantity is None:
            return {"action": "risk_exit_error", "symbol": symbol, "error": "No quantity to sell"}
        
        try:
            order = place_market_order(symbol.replace("USDT", "/USDT"), "sell", position.quantity)
            
            if order:
                pnl = position.get_unrealized_pnl(price)
                duration = position.get_duration_hours()
                sold_quantity = position.quantity
                
                position.exit_position()
                self.cooldown_tracker[symbol] = time.time()
                
                return {
                    "action": "risk_exit",
                    "symbol": symbol,
                    "price": price,
                    "quantity": sold_quantity,
                    "reason": reason,
                    "pnl": pnl,
                    "duration_hours": duration
                }
            else:
                return {"action": "risk_exit_failed", "symbol": symbol, "reason": reason}
                
        except Exception as e:
            print(f"❌ Risk management exit error: {e}")
            return {"action": "risk_exit_error", "symbol": symbol, "error": str(e)}

    def decide_trades(self, top_n=1, min_score=0.5):
        """Enhanced trade decision making with cycle management"""
        agents = self.load_agents()
        evaluations = self.evaluate_agents(agents)

        decisions = []
        for evaluation in evaluations[:top_n]:
            if evaluation["score"] < min_score or self.is_in_cooldown(evaluation["symbol"]):
                continue
                
            # Get current price
            df = fetch_ohlcv(evaluation["symbol"], "1h", 1)
            current_price = df["close"].iloc[-1] if not df.empty else None
            
            if current_price:
                # Execute trade cycle logic
                cycle_result = self.execute_trade_cycle(
                    evaluation["symbol"],
                    evaluation["signal"],
                    evaluation["confidence"],
                    current_price
                )
                
                decisions.append({
                    **evaluation,
                    "current_price": current_price,
                    "cycle_result": cycle_result
                })

        return decisions

    def make_portfolio_decision(self, min_score=0.5):
        """Make portfolio decision with enhanced trade cycle"""
        decisions = self.decide_trades(min_score=min_score)
        timestamp = self.performance_tracker.current_time()
        
        if not decisions:
            return {"decisions": [], "timestamp": timestamp}

        # Process all decisions
        results = []
        for decision in decisions:
            self._log_prediction(
                decision["symbol"],
                decision,
                {"win_rate": decision.get("win_rate", 0.0)},
                decision["score"],
                decision["current_price"]
            )
            results.append(decision)

        return {"decisions": results, "timestamp": timestamp}

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        status = {
            "total_positions": 0,
            "positions_by_state": {},
            "unrealized_pnl": 0.0,
            "positions": []
        }
        
        for symbol, position in self.positions.items():
            if position.state != TradeState.IDLE:
                status["total_positions"] += 1
                
                state_name = position.state.value
                if state_name not in status["positions_by_state"]:
                    status["positions_by_state"][state_name] = 0
                status["positions_by_state"][state_name] += 1
                
                # Get current price for unrealized P&L
                df = fetch_ohlcv(symbol, "1h", 1)
                current_price = df["close"].iloc[-1] if not df.empty else position.entry_price
                
                if current_price:
                    unrealized_pnl = position.get_unrealized_pnl(current_price)
                    status["unrealized_pnl"] += unrealized_pnl
                    
                    status["positions"].append({
                        "symbol": symbol,
                        "state": state_name,
                        "entry_price": position.entry_price,
                        "current_price": current_price,
                        "quantity": position.quantity,
                        "unrealized_pnl": unrealized_pnl,
                        "duration_hours": position.get_duration_hours(),
                        "stop_loss": position.stop_loss,
                        "take_profit": position.take_profit
                    })
        
        return status

    def load_all_predictions(self) -> List[Dict]:
        all_preds = []
        for file in glob.glob(os.path.join(TRADE_HISTORY_DIR, "*_predictions.json")):
            try:
                with open(file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_preds.extend(data)
            except:
                continue
        return all_preds