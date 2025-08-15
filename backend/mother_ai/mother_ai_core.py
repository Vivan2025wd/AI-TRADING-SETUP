import os
import json
import glob
import importlib.util
import time
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import dateutil.parser
import pandas as pd

from backend.binance.fetch_live_ohlcv import fetch_ohlcv
from backend.binance.binance_trader import place_market_order
from backend.mother_ai.performance_tracker import PerformanceTracker
from backend.strategy_engine.strategy_health import StrategyHealth
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.mother_ai.meta_evaluator import MetaEvaluator
from backend.mother_ai.profit_calculator import compute_trade_profits
from backend.storage.auto_cleanup import auto_cleanup_logs
from backend.mother_ai.risk_manager import RiskManager
from backend.mother_ai.cooldown import CooldownManager

TRADE_HISTORY_DIR = "backend/storage/trade_history"
PERFORMANCE_LOG_DIR = "backend/storage/performance_logs"

os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)


class MotherAICore:
    """Core functionality for MotherAI - handles initialization and basic operations"""
    
    def __init__(self, agents_dir="backend/agents", strategy_dir="backend/storage/strategies", 
                 agent_symbols=None, data_interval="30m"):
        """
        Initialize MotherAI Core with configurable data interval
        
        Args:
            agents_dir: Directory containing agent files
            strategy_dir: Directory containing strategy files
            agent_symbols: List of symbols to trade (None = all)
            data_interval: Data interval to use ("1m", "5m", "1h", etc.)
        """
        self.agents_dir = agents_dir
        self.strategy_dir = strategy_dir
        self.data_interval = data_interval
        self.performance_tracker = PerformanceTracker("performance_logs")
        self.agent_symbols = agent_symbols or []
        self.meta_evaluator = MetaEvaluator()
        self.loaded_agents = {}
        self.risk_manager = RiskManager()
        self.cooldown_manager = CooldownManager()
        
        # Track last exit check time to prevent too frequent exit checks
        self.last_exit_check = {}
        self.exit_check_interval = 300  # 5 minutes between exit checks (configurable)
        
        # Track trade entry times to enforce minimum hold periods
        self.position_entry_times = {}
        self.minimum_hold_time = 900  # 15 minutes minimum hold (configurable)

        # Initialize agents on startup
        self._initialize_agents()
        
        print(f"🔧 MotherAI Core initialized with {self.data_interval} data interval")
        print(f"🔧 Exit check interval: {self.exit_check_interval}s")
        print(f"🔧 Minimum hold time: {self.minimum_hold_time}s")

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
        """Load strategy configuration for a symbol"""
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
        """Load agent class from file"""
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

    def _fetch_agent_data(self, symbol: str):
        """Fetch OHLCV data using configured interval to match agent predictions"""
        symbol = symbol if symbol.endswith("/USDT") else symbol.replace("USDT", "") + "/USDT"
        df = fetch_ohlcv(symbol, self.data_interval, 100)
        return df if not df.empty else None

    def _validate_data_compatibility(self, agent, ohlcv_data: pd.DataFrame) -> bool:
        """Validate that data interval matches agent's expected interval"""
        if not hasattr(agent, 'model') or agent.model is None:
            return True
        
        if len(ohlcv_data) >= 2:
            time_diff = ohlcv_data.index[1] - ohlcv_data.index[0]
            interval_minutes = time_diff.total_seconds() / 60
            
            expected_minutes = self._interval_to_minutes(self.data_interval)
            
            if abs(interval_minutes - expected_minutes) > 0.1:
                print(f"⚠️ Data interval mismatch for {agent.symbol}: "
                     f"got {interval_minutes}min, expected {expected_minutes}min")
                return False
        
        return True

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 1440
        else:
            return 1

    def _safe_evaluate_agent(self, agent, ohlcv):
        """Enhanced agent evaluation with data compatibility checks"""
        try:
            if not self._validate_data_compatibility(agent, ohlcv):
                print(f"⚠️ Data compatibility issue for {agent.symbol}, using fallback strategy")
            
            decision = agent.evaluate(ohlcv)
        
            ml_data = decision.get("ml", {})
            rule_data = decision.get("rule_based", {})
        
            print(f"🤖 Agent {agent.symbol} evaluation:")
            print(f"   Final: {decision.get('action', 'unknown').upper()} ({decision.get('confidence', 0.0):.3f})")
            print(f"   ML: {ml_data.get('action', 'N/A')} ({ml_data.get('confidence', 0.0):.3f})")
            print(f"   Rule: {rule_data.get('action', 'N/A')} ({rule_data.get('confidence', 0.0):.3f})")
            print(f"   Source: {decision.get('source', 'unknown')}")
            print(f"   Position: {decision.get('position_state', 'None')}")
        
            result = {
                "symbol": agent.symbol,
                "signal": decision.get("action", "hold").lower(),
                "confidence": decision.get("confidence", 0.0),
                "source": decision.get("source", "unknown"),
                "position_state": decision.get("position_state"),
                "ml_available": decision.get("ml", {}).get("available", False),
                "ml_confidence": decision.get("ml", {}).get("confidence", 0.0),
                "rule_confidence": decision.get("rule_based", {}).get("confidence", 0.0),
                "timestamp": decision.get("timestamp"),
                "data_interval": self.data_interval,
                "full_decision": decision
            }
        
            return result
        
        except Exception as e:
            print(f"❌ Agent evaluation error for {agent.symbol}: {e}")
            return {
                "symbol": agent.symbol,
                "signal": "hold",
                "confidence": 0.0,
                "source": "error",
                "position_state": None,
                "ml_available": False,
                "ml_confidence": 0.0,
                "rule_confidence": 0.0,
                "data_interval": self.data_interval,
                "full_decision": {}
            }

    def evaluate_agents(self, agents):
        """Evaluate agents using their full decision output with data consistency checks"""
        results = []
        trade_tracker = PerformanceTracker("trade_history")
        
        print(f"📊 Evaluating agents with {self.data_interval} data interval...")
        
        for agent in agents:
            ohlcv = self._fetch_agent_data(agent.symbol)
            if ohlcv is None:
                continue
                
            prediction = self._safe_evaluate_agent(agent, ohlcv)
            health = StrategyHealth(trade_tracker.get_agent_log(agent.symbol)).summary()
            score = self._calculate_score(prediction, health)
            
            result = {
                **prediction,
                "score": round(score, 3),
                "health": health
            }
            results.append(result)
            
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _calculate_score(self, prediction, health):
        """Calculate meta score for prediction"""
        return self.meta_evaluator.predict_refined_score({
            "confidence": prediction["confidence"],
            "win_rate": health.get("win_rate", 0.5),
            "drawdown_penalty": 1.0 - health.get("max_drawdown", 0.3),
            "is_buy": int(prediction["signal"] == "buy"),
            "is_sell": int(prediction["signal"] == "sell")
        })

    def get_current_positions(self) -> Dict:
        """Get current position states of all agents with exposure estimates"""
        positions = {}
        for symbol, agent in self.loaded_agents.items():
            position_info = {
                'symbol': symbol,
                'position_state': getattr(agent, 'position_state', None),
                'exposure': 0.0,
                'unrealized_pnl': 0.0,
                'data_interval': self.data_interval,
                'entry_time': self.position_entry_times.get(symbol),
                'hold_duration': None
            }
            
            # Calculate hold duration if we have entry time
            if position_info['entry_time'] and position_info['position_state'] == 'long':
                hold_duration = (datetime.now() - position_info['entry_time']).total_seconds()
                position_info['hold_duration'] = hold_duration
            
            if position_info['position_state'] == 'long':
                position_info['exposure'] = self.risk_manager.get_symbol_config(symbol)["risk_per_trade"]
                
            positions[symbol] = position_info
        
        return positions

    def update_agent_position_state(self, symbol: str, signal: str):
        """Update agent's position state after successful trade execution"""
        agent = self.get_agent_by_symbol(symbol)
        if agent and hasattr(agent, 'position_state'):
            old_state = agent.position_state
            
            if signal == "buy":
                agent.position_state = "long"
            elif signal == "sell":
                agent.position_state = None
                # Clear entry time when position is closed
                if symbol in self.position_entry_times:
                    del self.position_entry_times[symbol]
                    print(f"🗑️ Cleared entry time for {symbol}")
                
            print(f"🔄 Updated {symbol} agent position: {old_state} → {agent.position_state}")
        else:
            print(f"⚠️ Could not update position state for {symbol} - agent not found or no position_state")

    def load_all_predictions(self) -> List[Dict]:
        """Load predictions from all agents with data consistency tracking"""
        predictions = []
        agents = self.load_agents()
        
        print(f"📊 Loading predictions using {self.data_interval} data interval...")
        
        for agent in agents:
            ohlcv = self._fetch_agent_data(agent.symbol)
            if ohlcv is not None:
                prediction = self._safe_evaluate_agent(agent, ohlcv)
                predictions.append(prediction)
                
        return predictions

    # Configuration methods for dynamic adjustment
    def set_minimum_hold_time(self, seconds: int):
        """Dynamically adjust minimum hold time"""
        self.minimum_hold_time = seconds
        print(f"🔧 Minimum hold time set to {seconds}s ({seconds/60:.1f} minutes)")

    def set_exit_check_interval(self, seconds: int):
        """Dynamically adjust exit check interval"""
        self.exit_check_interval = seconds
        print(f"🔧 Exit check interval set to {seconds}s ({seconds/60:.1f} minutes)")

    def get_position_hold_times(self) -> Dict:
        """Get current hold times for all positions"""
        hold_times = {}
        current_time = datetime.now()
        
        for symbol, entry_time in self.position_entry_times.items():
            if entry_time:
                hold_duration = (current_time - entry_time).total_seconds()
                hold_times[symbol] = {
                    'entry_time': entry_time,
                    'hold_duration_seconds': hold_duration,
                    'hold_duration_minutes': hold_duration / 60,
                    'meets_minimum_hold': hold_duration >= self.minimum_hold_time,
                    'remaining_hold_time': max(0, self.minimum_hold_time - hold_duration)
                }
        
        return hold_times