from backend.agents.generic_agent import GenericAgent
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.strategy_engine.json_strategy_parser import load_strategy_for_symbol
from typing import Optional

class BTCUSDTAgent(GenericAgent):
    symbol: str = "BTCUSDT"

    def __init__(self, symbol: Optional[str] = None, strategy_logic: Optional[StrategyParser] = None):
        resolved_symbol = symbol or self.symbol
        if strategy_logic is None:
            strategy_dict = load_strategy_for_symbol(resolved_symbol)
            strategy_logic = StrategyParser(strategy_dict)

        super().__init__(symbol=resolved_symbol, strategy_logic=strategy_logic)
