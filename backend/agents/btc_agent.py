# backend/agents/btcusdt_agent.py
from .generic_agent import GenericAgent
from backend.strategy_engine.json_strategy_parser import load_strategy_for_symbol

class BTCUSDTAgent(GenericAgent):
    def __init__(self):
        strategy = load_strategy_for_symbol("BTCUSDT")
        super().__init__(symbol="BTCUSDT", strategy_logic=strategy)