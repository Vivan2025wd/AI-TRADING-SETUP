import os
import json
import glob
import importlib.util
import asyncio
import time
from typing import List, Dict
from fastapi import APIRouter, HTTPException, Query, FastAPI

from backend.binance.fetch_live_ohlcv import fetch_ohlcv
from backend.binance.binance_trader import place_market_order
from backend.mother_ai.performance_tracker import PerformanceTracker
from backend.strategy_engine.strategy_health import StrategyHealth
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.mother_ai.profit_calculator import compute_trade_profits
from backend.mother_ai.meta_evaluator import MetaEvaluator

router = APIRouter()

TRADE_HISTORY_DIR = "backend/storage/trade_history"
PERFORMANCE_LOG_DIR = "backend/storage/performance_logs"
os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)

TRADE_COOLDOWN_SECONDS = 600
AUTO_DECISION_INTERVAL = 120


class MotherAI:
    def __init__(self, agents_dir="backend/agents", strategy_dir="backend/storage/strategies", agent_symbols=None):
        self.agents_dir = agents_dir
        self.strategy_dir = strategy_dir
        self.performance_tracker = PerformanceTracker(log_dir_type="performance_logs")
        self.agent_symbols = agent_symbols or []
        self.meta_evaluator = MetaEvaluator()
        self.cooldown_tracker = {}

    def load_agents(self):
        print("📥 Loading agents from directory...")
        agents = []
        agent_files = [
            f for f in os.listdir(self.agents_dir)
            if f.endswith(".py") and f not in ("__init__.py", "generic_agent.py")
        ]

        for agent_file in agent_files:
            base_name = agent_file.replace(".py", "")
            if not base_name.endswith("_agent"):
                continue

            symbol = base_name[:-6].upper()
            if self.agent_symbols and symbol not in self.agent_symbols:
                continue

            print(f"🔍 Loading strategy for agent: {symbol}")
            strategy_pattern = os.path.join(self.strategy_dir, f"{symbol}_strategy_*.json")
            matched_files = glob.glob(strategy_pattern)
            strategy_file = next((f for f in matched_files if "default" in os.path.basename(f)), None)
            if not strategy_file and matched_files:
                strategy_file = matched_files[0]

            strategy_data = {}
            if strategy_file:
                try:
                    with open(strategy_file, "r") as f:
                        strategy_data = json.load(f)
                    print(f"✅ Loaded strategy file: {strategy_file}")
                except Exception as e:
                    print(f"⚠️ Failed to load strategy for {symbol}: {e}")

            strategy_logic = StrategyParser(strategy_data)
            agent_module_path = os.path.join(self.agents_dir, agent_file)
            spec = importlib.util.spec_from_file_location(base_name, agent_module_path)
            if not spec or not spec.loader:
                print(f"⚠️ Failed to load module for {symbol}")
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            agent_class_name = symbol + "Agent"
            agent_class = getattr(module, agent_class_name, None)
            if not agent_class:
                print(f"⚠️ No class found for agent: {agent_class_name}")
                continue

            agent_instance = agent_class(symbol=symbol, strategy_logic=strategy_logic)
            agents.append(agent_instance)
            print(f"✅ Loaded agent class: {agent_class_name}")

        print(f"📦 Total agents loaded: {len(agents)}")
        return agents

    def evaluate_agents(self, agents):
        print("🔬 Starting agent evaluation...")
        results = []
        tracker = PerformanceTracker(log_dir_type="trade_history")

        for agent in agents:
            print(f"🧠 Evaluating agent: {agent.symbol}")
            try:
                symbol_ccxt = agent.symbol
                if not symbol_ccxt.endswith("/USDT"):
                    if symbol_ccxt.endswith("USDT"):
                        symbol_ccxt = symbol_ccxt[:-4] + "/USDT"
                    else:
                        symbol_ccxt += "/USDT"

                ohlcv_data = fetch_ohlcv(symbol_ccxt, interval="1h", limit=100)
                if ohlcv_data.empty:
                    print(f"⚠️ No OHLCV data for {agent.symbol}, skipping")
                    continue

                prediction = agent.predict(ohlcv_data)
                signal = prediction.get("action", "hold").lower()
                confidence = prediction.get("confidence", 0.0)

            except Exception as e:
                print(f"⚠️ Error during prediction for {agent.symbol}: {e}")
                signal, confidence = "hold", 0.0

            history = tracker.get_agent_log(agent.symbol)
            if not history:
                win_rate = 0.5
                health_penalty = 1.0
            else:
                health = StrategyHealth(history).summary()
                win_rate = health.get("win_rate", 0.5)
                drawdown = health.get("max_drawdown", 0.3)
                health_penalty = 1.0 - drawdown

            meta_features = {
                "confidence": confidence,
                "win_rate": win_rate,
                "drawdown_penalty": health_penalty,
                "is_buy": 1 if signal == "buy" else 0,
                "is_sell": 1 if signal == "sell" else 0,
            }
            score = self.meta_evaluator.predict_refined_score(meta_features)

            print(f"📝 Logging prediction for {agent.symbol}: signal={signal}, confidence={confidence:.2f}, score={score:.3f}")
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

            results.append({
                "symbol": agent.symbol,
                "signal": signal,
                "confidence": confidence,
                "win_rate": win_rate,
                "score": round(score, 3),
            })

        print("✅ Agent evaluation complete.\n")
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def log_agent_prediction(self, symbol, data):
        path = os.path.join(TRADE_HISTORY_DIR, f"{symbol}_predictions.json")
        print(f"📁 Logging prediction to {path}...")
        existing = []
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
                print(f"📂 Loaded {len(existing)} previous predictions")
            except json.JSONDecodeError:
                print(f"⚠️ Corrupted JSON in {path}, resetting file")
                existing = []
        existing.append(data)
        try:
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"✅ Prediction saved for {symbol}")
        except Exception as e:
            print(f"❌ Failed to write prediction for {symbol}: {e}")

    def is_in_cooldown(self, symbol):
        last_time = self.cooldown_tracker.get(symbol)
        return last_time and (time.time() - last_time) < TRADE_COOLDOWN_SECONDS

    def decide_trades(self, top_n=1, min_score=0.5):
        agents = self.load_agents()
        evaluations = self.evaluate_agents(agents)
        return [
            e for e in evaluations
            if e["score"] >= min_score and not self.is_in_cooldown(e["symbol"])
        ][:top_n]

    def execute_trade(self, symbol: str, signal: str, price: float, confidence: float):
        if signal.lower() not in ("buy", "sell") or not price:
            print(f"⚠️ Skipping invalid trade: {symbol} {signal} at {price}")
            return
        print(f"🚀 Executing {signal.upper()} trade for {symbol} at ${price:.2f} (confidence={confidence:.2f})")

        try:
            binance_symbol = symbol.replace("USDT", "/USDT")
            usdt_amount = 20
            quantity = usdt_amount / price

            order = place_market_order(binance_symbol, signal, quantity)
            if order:
                print(f"✅ Trade successful: {order['id']}")
                self.cooldown_tracker[symbol] = time.time()
            else:
                print("❌ Trade execution failed")
        except Exception as e:
            print(f"❌ Exception during trade: {e}")

    def make_portfolio_decision(self, min_score=0.5):
        print("📊 Making portfolio decision...")
        top_trades = self.decide_trades(top_n=1, min_score=min_score)
        timestamp = self.performance_tracker.current_time()
        if not top_trades:
            print("⚠️ No trades met the minimum score threshold.")
            return {"decision": [], "timestamp": timestamp}

        trade = top_trades[0]
        symbol = trade["symbol"]
        df = fetch_ohlcv(symbol if symbol.endswith("USDT") else symbol + "USDT", interval="1h", limit=1)
        last_price = df["close"].iloc[-1] if not df.empty else None

        decision_obj = {
            "symbol": symbol,
            "signal": trade["signal"],
            "confidence": trade["confidence"],
            "win_rate": trade["win_rate"],
            "score": trade["score"],
            "last_price": last_price,
        }

        log_entry = {**decision_obj, "timestamp": timestamp, "price": last_price, "source": "mother_ai_decision"}
        self.performance_tracker.log_trade(symbol, log_entry)

        if trade["signal"].lower() in ("buy", "sell") and last_price is not None:
            self.execute_trade(symbol, trade["signal"], last_price, trade["confidence"])
            if trade["signal"].lower() == "sell":
                print(f"📈 Sell executed for {symbol}. Computing profit...")
                compute_trade_profits(symbol)

        return {"decision": decision_obj, "timestamp": timestamp}

    # --- New methods to read all prediction files and decide based on confidence ---

    def load_all_predictions(self) -> List[Dict]:
        """
        Load all predictions from all *_predictions.json files in trade_history directory.
        Returns a list of prediction dicts.
        """
        predictions = []
        files = [f for f in os.listdir(TRADE_HISTORY_DIR) if f.endswith("_predictions.json")]
        for filename in files:
            path = os.path.join(TRADE_HISTORY_DIR, filename)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        predictions.extend(data)
                    else:
                        print(f"⚠️ Data in {filename} is not a list, skipping")
            except Exception as e:
                print(f"⚠️ Failed to load predictions from {filename}: {e}")
        print(f"✅ Loaded {len(predictions)} total predictions from {len(files)} files")
        return predictions

    def make_decision_from_predictions(self, min_confidence=0.3, top_n=1):
        """
        Make trading decision based on aggregated predictions with confidence threshold.
        Returns a list of top N decisions sorted by confidence descending.
        """
        predictions = self.load_all_predictions()
        # Filter by confidence threshold
        filtered = [p for p in predictions if p.get("confidence", 0) >= min_confidence]
        if not filtered:
            print("⚠️ No predictions meet the confidence threshold")
            return []

        # Sort by confidence descending
        filtered.sort(key=lambda x: x["confidence"], reverse=True)

        # Pick top N predictions
        top_predictions = filtered[:top_n]

        decisions = []
        for pred in top_predictions:
            symbol = pred.get("symbol")
            signal = pred.get("signal")
            confidence = pred.get("confidence")
            timestamp = pred.get("timestamp")

            # Additional logic like fetching price can be added here

            decisions.append({
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "timestamp": timestamp,
            })

        return decisions


# FastAPI + Auto Loop

mother_ai_instance = MotherAI()
latest_decision = None

app = FastAPI(title="Mother AI Trading API")


@app.on_event("startup")
async def startup_event():
    print("✅ MotherAI backend started with auto-trade loop")

    async def auto_trade_loop():
        global latest_decision
        while True:
            print("🔁 Auto-evaluating trading decisions...")
            try:
                latest_decision = mother_ai_instance.make_portfolio_decision(min_score=0.5)
                if latest_decision.get("decision"):
                    print(f"🤖 Auto decision made: {latest_decision['decision']['symbol']}")
                else:
                    print("⚠️ No trade executed this cycle")
            except Exception as e:
                print(f"❌ Auto loop exception: {e}")
            await asyncio.sleep(AUTO_DECISION_INTERVAL)

    asyncio.create_task(auto_trade_loop())


@app.get("/api/mother-ai/latest-decision")
async def get_latest_decision():
    if latest_decision is None:
        return {"status": "waiting", "message": "Evaluating...", "decision": None, "timestamp": None}
    if not latest_decision.get("decision"):
        return {"status": "no_signal", "message": "No trades qualified", "decision": None, "timestamp": latest_decision.get("timestamp")}
    return {"status": "success", "message": "Decision made", **latest_decision}


@app.post("/api/mother-ai/trigger-decision")
async def trigger_decision():
    global latest_decision
    latest_decision = mother_ai_instance.make_portfolio_decision(min_score=0.5)
    return latest_decision


@router.get("/trades")
async def get_trade_logs(limit: int = Query(100, ge=1)):
    tracker = PerformanceTracker(log_dir_type="performance_logs")
    trades = []
    for filename in os.listdir(PERFORMANCE_LOG_DIR):
        if filename.endswith("_trades.json"):
            symbol = filename.replace("_trades.json", "")
            trades.extend(tracker.get_agent_log(symbol, limit=limit))
    trades = sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"data": trades[:limit], "total": len(trades)}


@router.get("/decision")
async def get_decision(min_score: float = Query(0.5, ge=0.0, le=1.0)):
    decision = mother_ai_instance.make_portfolio_decision(min_score=min_score)
    if not decision or not decision.get("decision"):
        raise HTTPException(status_code=404, detail="No decision found")
    return decision


app.include_router(router, prefix="/api/mother-ai", tags=["Mother AI"])
