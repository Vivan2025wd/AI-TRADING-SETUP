from backend.mother_ai.performance_tracker import PerformanceTracker
from backend.agents.generic_agent import GenericAgent
from backend.strategy_engine.strategy_health import StrategyHealth

import os
import json
import glob

class MotherAI:
    def __init__(self, agent_symbols, strategy_dir="backend/strategy_engine/strategies"):
        self.agent_symbols = agent_symbols
        self.strategy_dir = strategy_dir
        self.performance_tracker = PerformanceTracker()

    def load_agents(self):
        agents = []
        for symbol in self.agent_symbols:
            strategy_path = os.path.join(self.strategy_dir, f"{symbol}.json")
            with open(strategy_path, "r") as f:
                strategy_logic = json.load(f)
            agent = GenericAgent(symbol=symbol, strategy_logic=strategy_logic)
            agents.append(agent)
        return agents

    def evaluate_agents(self, agents):
        results = []
        for agent in agents:
            signal, confidence = agent.predict_live()
            history = self.performance_tracker.get_agent_log(agent.symbol)
            health = StrategyHealth(history).summary()
            
            score = self.calculate_confidence_score(confidence, health["win_rate"])
            
            results.append({
                "symbol": agent.symbol,
                "signal": signal,
                "confidence": confidence,
                "win_rate": health["win_rate"],
                "score": round(score, 3)
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def calculate_confidence_score(self, confidence, win_rate, alpha=0.6, beta=0.4):
        """
        Weighted score based on current prediction confidence and historical win rate
        """
        return (alpha * confidence) + (beta * win_rate)

    def decide_trades(self, top_n=1, min_score=0.7):
        agents = self.load_agents()
        evaluations = self.evaluate_agents(agents)
        top_trades = [e for e in evaluations if e["score"] >= min_score][:top_n]

        return {
            "decision": top_trades,
            "timestamp": self.performance_tracker.current_time()
        }

    def make_portfolio_decision(self, top_n=3, min_score=0.75):
        """
        Run the full decision pipeline and return top N trade decisions above min_score.
        """
        trades = self.decide_trades(top_n=top_n, min_score=min_score)
        return trades
