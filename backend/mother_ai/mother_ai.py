import os
import json
import glob
import random
import importlib.util
from fastapi import APIRouter, HTTPException, Query

from backend.binance.fetch_live_ohlcv import fetch_ohlcv
from backend.mother_ai.performance_tracker import PerformanceTracker
from backend.strategy_engine.strategy_health import StrategyHealth
from backend.strategy_engine.strategy_parser import StrategyParser

router = APIRouter()

class MotherAI:
    def __init__(self, agents_dir="backend/agents", strategy_dir="backend/storage/strategies", agent_symbols=None):
        self.agents_dir = agents_dir
        self.strategy_dir = strategy_dir
        # Log to performance_logs folder
        self.performance_tracker = PerformanceTracker(log_dir="backend/storage/performance_logs")
        self.agent_symbols = agent_symbols or []

    def load_agents(self):
        agents = []
        agent_files = [
            f for f in os.listdir(self.agents_dir)
            if f.endswith(".py") and f not in ("__init__.py", "generic_agent.py")
        ]

        for agent_file in agent_files:
            base_name = agent_file.replace(".py", "")
            if not base_name.endswith("_agent"):
                continue

            symbol_prefix = base_name[:-6]
            symbol = symbol_prefix.upper()

            if self.agent_symbols and symbol not in self.agent_symbols:
                continue

            strategy_pattern = os.path.join(self.strategy_dir, f"{symbol}_strategy_*.json")
            matched_files = glob.glob(strategy_pattern)

            strategy_file = None
            for f in matched_files:
                if "default" in os.path.basename(f):
                    strategy_file = f
                    break
            if not strategy_file and matched_files:
                strategy_file = matched_files[0]

            if strategy_file:
                try:
                    with open(strategy_file, "r") as f:
                        strategy_data = json.load(f)
                except Exception as e:
                    print(f"⚠️ Failed to load strategy for {symbol}: {e}")
                    strategy_data = {}
            else:
                strategy_data = {}

            strategy_logic = StrategyParser(strategy_data)
            agent_module_path = os.path.join(self.agents_dir, agent_file)

            spec = importlib.util.spec_from_file_location(base_name, agent_module_path)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            agent_class_name = symbol + "Agent"
            agent_class = getattr(module, agent_class_name, None)
            if not agent_class:
                continue

            agent_instance = agent_class(symbol=symbol, strategy_logic=strategy_logic)
            agents.append(agent_instance)

        return agents

    def evaluate_agents(self, agents):
        results = []
        for agent in agents:
            try:
                symbol_ccxt = agent.symbol[:-4] + "/USDT"
                ohlcv_data = fetch_ohlcv(symbol_ccxt, interval="1h", limit=100)
                if ohlcv_data.empty:
                    continue

                prediction = agent.predict(ohlcv_data)
                signal = prediction.get("action", "hold").lower()
                confidence = prediction.get("confidence", 0.0)

                # Add jitter ±5% to confidence
                confidence += random.uniform(-0.05, 0.05)
                confidence = max(0.0, min(confidence, 1.0))

            except Exception as e:
                print(f"⚠️ Error in predict for {agent.symbol}: {e}")
                signal, confidence = "hold", 0.0

            history = self.performance_tracker.get_agent_log(agent.symbol)

            if not history:
                win_rate = 0.5
                health_penalty = 1.0
            else:
                health = StrategyHealth(history).summary()
                win_rate = health.get("win_rate", 0.5)
                drawdown = health.get("max_drawdown", 0.3)
                health_penalty = 1.0 - drawdown  # penalize agents with high drawdowns

            score = self.calculate_confidence_score(confidence, win_rate, health_penalty)

            print(f"✅ Agent {agent.symbol}: confidence={confidence:.2f}, win_rate={win_rate:.2f}, drawdown_penalty={health_penalty:.2f}, score={score:.3f}")

            results.append({
                "symbol": agent.symbol,
                "signal": signal,
                "confidence": confidence,
                "win_rate": win_rate,
                "score": round(score, 3)
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def calculate_confidence_score(self, confidence, win_rate, health_penalty, alpha=0.5, beta=0.3, gamma=0.2):
        return (alpha * confidence) + (beta * win_rate) + (gamma * health_penalty)

    def decide_trades(self, top_n=1, min_score=0.5):  # lowered from 0.7 to 0.5
        agents = self.load_agents()
        evaluations = self.evaluate_agents(agents)
        top_trades = [e for e in evaluations if e["score"] >= min_score][:top_n]
        return top_trades

    def make_portfolio_decision(self, min_score=0.5):  # lowered from 0.7 to 0.5
        top_trades = self.decide_trades(top_n=1, min_score=min_score)
        timestamp = self.performance_tracker.current_time()

        if not top_trades:
            return {"decision": [], "timestamp": timestamp}

        trade = top_trades[0]
        symbol_ccxt = trade["symbol"][:-4] + "/USDT"
        df = fetch_ohlcv(symbol_ccxt, interval="1h", limit=1)
        last_price = df["close"].iloc[-1] if not df.empty else None

        decision_obj = {
            "symbol": trade["symbol"],
            "signal": trade["signal"],
            "confidence": trade["confidence"],
            "win_rate": trade["win_rate"],
            "score": trade["score"],
            "last_price": last_price,
        }

        trade_log_entry = {
            "timestamp": timestamp,
            "symbol": trade["symbol"],
            "signal": trade["signal"],
            "price": last_price,
            "confidence": trade["confidence"],
            "win_rate": trade["win_rate"],
            "score": trade["score"]
        }
        # Log trade to performance_logs/{symbol}_trades.json
        self.performance_tracker.log_trade(trade["symbol"], trade_log_entry)

        return {"decision": decision_obj, "timestamp": timestamp}


# --- FastAPI Routes ---
@router.get("/trades")
async def get_trade_logs(limit: int = Query(100, ge=1)):
    log_dir = "backend/storage/performance_logs"
    tracker = PerformanceTracker(log_dir=log_dir)
    trades = []

    for filename in os.listdir(log_dir):
        if filename.endswith("_trades.json"):
            symbol = filename.replace("_trades.json", "")
            trades.extend(tracker.get_agent_log(symbol, limit=limit))

    trades = sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)

    return {
        "data": trades[:limit],
        "total": len(trades)
    }


@router.get("/decision")
async def get_decision(min_score: float = Query(0.5, ge=0.0, le=1.0)):  # lowered from 0.7
    mother_ai = MotherAI()
    decision = mother_ai.make_portfolio_decision(min_score=min_score)
    if not decision or not decision.get("decision"):
        raise HTTPException(status_code=404, detail="No decision found")
    return decision
