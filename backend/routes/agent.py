from fastapi import APIRouter, HTTPException
from backend.agents.generic_agent import GenericAgent
from backend.strategy_registry import get_strategy_by_symbol
from backend.backtester.runner import BacktestRunner
from typing import List

router = APIRouter(prefix="/agent", tags=["Agent"])

@router.get("/{symbol}/predict")
def get_agent_prediction(symbol: str):
    """
    Return a prediction (BUY/SELL/HOLD + confidence) from an agent based on the latest OHLCV data and strategy.
    """
    try:
        # 🔁 Load parsed strategy JSON for the given symbol
        strategy_json = get_strategy_by_symbol(symbol)
        if not strategy_json:
            raise HTTPException(status_code=404, detail=f"No strategy found for symbol: {symbol}")

        # 📊 Load OHLCV data for symbol
        backtest_runner = BacktestRunner()
        ohlcv_data = backtest_runner.load_ohlcv(symbol)
        if ohlcv_data is None or len(ohlcv_data) == 0:
            raise HTTPException(status_code=400, detail=f"No OHLCV data found for symbol: {symbol}")

        # 🧠 Create agent instance with symbol and strategy logic
        agent = GenericAgent(symbol=symbol, strategy_logic=strategy_json)

        # 🧾 Get prediction (returns tuple: (action, confidence))
        signal, confidence = agent.predict(ohlcv_data)

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence
        }

    except HTTPException:
        raise  # Re-raise known HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

AVAILABLE_AGENTS = ["BTC", "ETH", "SOL", "AAPL", "TSLA", "GOOG"]

@router.get("", response_model=List[str])
def list_available_agents():
    return AVAILABLE_AGENTS