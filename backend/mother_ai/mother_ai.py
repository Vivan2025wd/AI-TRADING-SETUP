import os
import json
import glob
import random
import importlib.util
from fastapi import APIRouter, HTTPException, Query, FastAPI
from apscheduler.schedulers.background import BackgroundScheduler

from backend.binance.fetch_live_ohlcv import fetch_ohlcv
from backend.mother_ai.performance_tracker import PerformanceTracker
from backend.strategy_engine.strategy_health import StrategyHealth
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.mother_ai.profit_calculator import compute_trade_profits

router = APIRouter()

TRADE_HISTORY_DIR = "backend/storage/trade_history"
PERFORMANCE_LOG_DIR = "backend/storage/performance_logs"
os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)


class MotherAI:
    def __init__(self, agents_dir="backend/agents", strategy_dir="backend/storage/strategies", agent_symbols=None):
        self.agents_dir = agents_dir
        self.strategy_dir = strategy_dir
        self.performance_tracker = PerformanceTracker(log_dir_type="performance_logs")
        self.agent_symbols = agent_symbols or []

    def load_agents(self):
        agents = []
        agent_files = [f for f in os.listdir(self.agents_dir) if f.endswith(".py") and f not in ("__init__.py", "generic_agent.py")]

        for agent_file in agent_files:
            base_name = agent_file.replace(".py", "")
            if not base_name.endswith("_agent"):
                continue

            symbol = base_name[:-6].upper()

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
                symbol_ccxt = agent.symbol if agent.symbol.endswith("USDT") else agent.symbol + "USDT"
                symbol_ccxt = symbol_ccxt[:-4] + "/USDT"

                ohlcv_data = fetch_ohlcv(symbol_ccxt, interval="1h", limit=100)
                if ohlcv_data.empty:
                    continue

                prediction = agent.predict(ohlcv_data)
                signal = prediction.get("action", "hold").lower()
                confidence = prediction.get("confidence", 0.0)
                confidence += random.uniform(-0.05, 0.05)
                confidence = max(0.0, min(confidence, 1.0))

            except Exception as e:
                print(f"⚠️ Error in predict for {agent.symbol}: {e}")
                signal, confidence = "hold", 0.0

            # Load past predictions to compute win_rate
            tracker = PerformanceTracker(log_dir_type="trade_history")
            history = tracker.get_agent_log(agent.symbol)

            if not history:
                win_rate = 0.5
                health_penalty = 1.0
            else:
                health = StrategyHealth(history).summary()
                win_rate = health.get("win_rate", 0.5)
                drawdown = health.get("max_drawdown", 0.3)
                health_penalty = 1.0 - drawdown

            score = self.calculate_confidence_score(confidence, win_rate, health_penalty)

            self.log_agent_prediction(
                agent.symbol,
                {
                    "timestamp": self.performance_tracker.current_time(),
                    "symbol": agent.symbol,
                    "signal": signal,
                    "confidence": confidence,
                    "win_rate": win_rate,
                    "score": round(score, 3),
                    "source": "agent_prediction",
                },
            )

            print(f"✅ Agent {agent.symbol}: confidence={confidence:.2f}, win_rate={win_rate:.2f}, drawdown_penalty={health_penalty:.2f}, score={score:.3f}")

            results.append({
                "symbol": agent.symbol,
                "signal": signal,
                "confidence": confidence,
                "win_rate": win_rate,
                "score": round(score, 3),
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def calculate_confidence_score(self, confidence, win_rate, health_penalty, alpha=0.5, beta=0.3, gamma=0.2):
        return (alpha * confidence) + (beta * win_rate) + (gamma * health_penalty)

    def decide_trades(self, top_n=1, min_score=0.5):
        agents = self.load_agents()
        evaluations = self.evaluate_agents(agents)
        top_trades = [e for e in evaluations if e["score"] >= min_score][:top_n]
        return top_trades

    def make_portfolio_decision(self, min_score=0.5):
        top_trades = self.decide_trades(top_n=1, min_score=min_score)
        timestamp = self.performance_tracker.current_time()

        if not top_trades:
            return {"decision": [], "timestamp": timestamp}

        trade = top_trades[0]
        symbol = trade["symbol"]
        symbol_ccxt = symbol if symbol.endswith("USDT") else symbol + "USDT"
        df = fetch_ohlcv(symbol_ccxt, interval="1h", limit=1)
        last_price = df["close"].iloc[-1] if not df.empty else None

        decision_obj = {
            "symbol": symbol,
            "signal": trade["signal"],
            "confidence": trade["confidence"],
            "win_rate": trade["win_rate"],
            "score": trade["score"],
            "last_price": last_price,
        }

        trade_log_entry = {
            "timestamp": timestamp,
            "symbol": symbol,
            "signal": trade["signal"],
            "price": last_price,
            "confidence": trade["confidence"],
            "win_rate": trade["win_rate"],
            "score": trade["score"],
            "source": "mother_ai_decision",
        }

        self.performance_tracker.log_trade(symbol, trade_log_entry)

        if trade["signal"].lower() == "sell":
            print(f"📊 SELL decision made for {symbol}, updating profit summary...")
            compute_trade_profits(symbol)

        return {"decision": decision_obj, "timestamp": timestamp}

    def log_agent_prediction(self, symbol, data):
        path = os.path.join(TRADE_HISTORY_DIR, f"{symbol}_predictions.json")
        existing = []
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        existing.append(data)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)


# --- Scheduler and FastAPI app setup ---

mother_ai_instance = MotherAI()
latest_decision = None

def scheduled_trade_decision():
    global latest_decision
    print("⏰ Running scheduled MotherAI decision...")
    decision = mother_ai_instance.make_portfolio_decision(min_score=0.5)
    latest_decision = decision
    print(f"Decision made at {decision.get('timestamp')} for symbol: {decision['decision'].get('symbol') if decision.get('decision') else 'None'}")

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_trade_decision, "interval", hours=2)
scheduler.start()

app = FastAPI(title="Mother AI Trading API")

@app.on_event("startup")
async def startup_event():
    print("Starting scheduler for MotherAI trading decisions...")

@app.get("/api/mother-ai/latest-decision")
async def get_latest_decision():
    if latest_decision is None:
        return {"message": "No decision made yet. Waiting for scheduled run."}
    return latest_decision

@app.post("/api/mother-ai/trigger-decision")
async def trigger_decision():
    global latest_decision
    latest_decision = mother_ai_instance.make_portfolio_decision(min_score=0.5)
    return latest_decision

# Existing routes exposed under the router
@router.get("/trades")
async def get_trade_logs(limit: int = Query(100, ge=1)):
    tracker = PerformanceTracker(log_dir_type="performance_logs")
    trades = []

    for filename in os.listdir(PERFORMANCE_LOG_DIR):
        if filename.endswith("_trades.json"):
            symbol = filename.replace("_trades.json", "")
            trades.extend(tracker.get_agent_log(symbol, limit=limit))

    trades = sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)

    return {
        "data": trades[:limit],
        "total": len(trades),
    }

@router.get("/decision")
async def get_decision(min_score: float = Query(0.5, ge=0.0, le=1.0)):
    mother_ai = MotherAI()
    decision = mother_ai.make_portfolio_decision(min_score=min_score)
    if not decision or not decision.get("decision"):
        raise HTTPException(status_code=404, detail="No decision found")
    return decision

# Include the router for all Mother AI endpoints except /latest-decision and /trigger-decision
app.include_router(router, prefix="/api/mother-ai", tags=["Mother AI"])
