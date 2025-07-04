import os
import json
import importlib.util
from fastapi import APIRouter, HTTPException, Query
from backend.binance.fetch_live_ohlcv import fetch_ohlcv

from backend.mother_ai.performance_tracker import PerformanceTracker
from backend.strategy_engine.strategy_health import StrategyHealth
from backend.strategy_engine.strategy_parser import StrategyParser

router = APIRouter()

class MotherAI:
    def __init__(self, agents_dir="backend/agents", strategy_dir="backend/strategy_engine/strategies", agent_symbols=None):
        self.agents_dir = agents_dir
        self.strategy_dir = strategy_dir
        self.performance_tracker = PerformanceTracker()
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
                print(f"Agent filename '{agent_file}' does not end with '_agent', skipping.")
                continue
            symbol_prefix = base_name[:-6]  # remove "_agent"
            symbol = symbol_prefix.upper() + "USDT"

            if self.agent_symbols and symbol not in self.agent_symbols:
                continue

            strategy_path = os.path.join(self.strategy_dir, f"{symbol}.json")
            try:
                with open(strategy_path, "r") as f:
                    strategy_data = json.load(f)
            except FileNotFoundError:
                strategy_data = {}

            strategy_logic = StrategyParser(strategy_data)

            agent_module_path = os.path.join(self.agents_dir, agent_file)

            # FIXED: Remove "agents." prefix, use only base_name as module name
            spec = importlib.util.spec_from_file_location(base_name, agent_module_path)
            if spec is None or spec.loader is None:
                print(f"Could not load spec or loader for module '{base_name}', skipping.")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            agent_class_name = symbol + "Agent"
            agent_class = getattr(module, agent_class_name, None)
            if not agent_class:
                print(f"Agent class '{agent_class_name}' not found in {agent_module_path}, skipping.")
                continue

            agent_instance = agent_class(symbol=symbol, strategy_logic=strategy_logic)
            agents.append(agent_instance)

        return agents

    def evaluate_agents(self, agents):
        results = []
        for agent in agents:
            try:
                signal, confidence = agent.predict()
            except Exception as e:
                signal, confidence = "HOLD", 0.0
                print(f"Error in predict for {agent.symbol}: {e}")

            history = self.performance_tracker.get_agent_log(agent.symbol)
            health = StrategyHealth(history).summary()

            score = self.calculate_confidence_score(confidence, health.get("win_rate", 0))

            results.append({
                "symbol": agent.symbol,
                "signal": signal,
                "confidence": confidence,
                "win_rate": health.get("win_rate", 0),
                "score": round(score, 3)
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def calculate_confidence_score(self, confidence, win_rate, alpha=0.6, beta=0.4):
        return (alpha * confidence) + (beta * win_rate)

    def decide_trades(self, top_n=1, min_score=0.7):
        agents = self.load_agents()
        evaluations = self.evaluate_agents(agents)
        top_trades = [e for e in evaluations if e["score"] >= min_score][:top_n]
        return top_trades

    def make_portfolio_decision(self, min_score=0.7):
        top_trades = self.decide_trades(top_n=1, min_score=min_score)
        if not top_trades:
            return {}
        # Return only the single highest scoring trade
        return {
            "decision": top_trades[0],
            "timestamp": self.performance_tracker.current_time()
        }


@router.get("/trades")
async def get_trade_logs(limit: int = Query(100, ge=1)):
    log_dir = "backend/storage/trade_history"
    tracker = PerformanceTracker(log_dir=log_dir)

    trades = []

    for filename in os.listdir(log_dir):
        if filename.endswith("_log.json"):
            symbol = filename.replace("_log.json", "")
            symbol_trades = tracker.get_agent_log(symbol, limit=limit)
            trades.extend(symbol_trades)

    trades = sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)

    return {
        "data": trades[:limit],
        "total": len(trades)
    }


@router.get("/decision")
async def get_decision():
    mother_ai = MotherAI()
    decision = mother_ai.make_portfolio_decision()
    if not decision or not decision.get("decision"):
        raise HTTPException(status_code=404, detail="No decision found")
    return decision

def make_portfolio_decision(self, min_score=0.7):
    top_trades = self.decide_trades(top_n=1, min_score=min_score)
    timestamp = self.performance_tracker.current_time()

    if not top_trades:
        # Return empty decision but with timestamp
        return {"decision": [], "timestamp": timestamp}

    trade = top_trades[0]

    # Fetch latest close price from Binance
    symbol_ccxt = trade["symbol"][:-4] + "/USDT"  # e.g. "BTCUSDT" -> "BTC/USDT"
    df = fetch_ohlcv(symbol_ccxt, interval="1h", limit=1)
    last_price = None
    if not df.empty:
        last_price = df["close"].iloc[-1]

    # Compose decision object with live price included
    decision_obj = {
        "symbol": trade["symbol"],
        "signal": trade["signal"],
        "confidence": trade["confidence"],
        "win_rate": trade["win_rate"],
        "score": trade["score"],
        "last_price": last_price,
    }

    # Log this trade decision (simulate execution)
    trade_log_entry = {
        "timestamp": timestamp,
        "symbol": trade["symbol"],
        "signal": trade["signal"],
        "price": last_price,
        "confidence": trade["confidence"],
        "win_rate": trade["win_rate"],
        "score": trade["score"]
    }
    self.performance_tracker.log_trade(trade_log_entry)

    return {"decision": decision_obj, "timestamp": timestamp}
