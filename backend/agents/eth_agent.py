# backend/agents/ethusdt_agent.py
from .generic_agent import GenericAgent
from backend.strategy_engine.json_strategy_parser import load_strategy_for_symbol

class ETHUSDTAgent(GenericAgent):
    def __init__(self):
        strategy = load_strategy_for_symbol("ETHUSDT")
        super().__init__(symbol="ETHUSDT", strategy_logic=strategy)