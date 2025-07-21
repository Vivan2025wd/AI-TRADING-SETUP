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
        path = os.path.join(PERFORMANCE_LOG_DIR, f"{symbol}_trades.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r") as f:
                history = json.load(f)
                for entry in reversed(history):
                    if entry["signal"] == "buy":
                        return True
                    if entry["signal"] == "sell":
                        return False
        except Exception:
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
                continue

            current_position = self.position_tracker.get(e["symbol"], None)

            if e["signal"] == "buy":
                if current_position is None and e["confidence"] >= min_confidence:
                    # Check if last buy was not closed by sell
                    if not self._is_previous_trade_open_buy(e["symbol"]):
                        filtered.append(e)
            elif e["signal"] == "sell":
                if current_position == "long" and e["confidence"] >= min_confidence:
                    filtered.append(e)

        return filtered[:top_n]

    def execute_trade(self, symbol, signal, price, confidence):
        if signal not in ("buy", "sell") or not price:
            return
        try:
            qty = 20 / price
            order = place_market_order(symbol.replace("USDT", "/USDT"), signal, qty)
            if order:
                self.cooldown_tracker[symbol] = time.time()
                if signal == "buy":
                    self.position_tracker[symbol] = "long"
                elif signal == "sell":
                    self.position_tracker[symbol] = None
        except Exception as e:
            print(f"❌ Trade error: {e}")

    def make_portfolio_decision(self, min_score=0.5):
        auto_cleanup_logs()

        top = self.decide_trades(min_score=min_score)
        timestamp = self.performance_tracker.current_time()
        if not top:
            return {"decision": [], "timestamp": timestamp}

        decision = top[0]
        df = fetch_ohlcv(decision["symbol"], "1h", 1)
        price = df["close"].iloc[-1] if not df.empty else None

        result = {**decision, "last_price": price}

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
