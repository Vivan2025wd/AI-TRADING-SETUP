from backend.agents.generic_agent import GenericAgent
from backend.strategy_engine.json_strategy_parser import load_strategy_for_symbol
from backend.strategy_engine.strategy_parser import StrategyParser

class BTCUSDTAgent(GenericAgent):
    def __init__(self, symbol=None, strategy_logic=None):
        if symbol is None:
            symbol = "BTCUSDT"
        if strategy_logic is None:
            strategy_dict = load_strategy_for_symbol(symbol)
            strategy_logic = StrategyParser(strategy_dict)
        super().__init__(symbol=symbol, strategy_logic=strategy_logic)