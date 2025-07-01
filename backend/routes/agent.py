from fastapi import APIRouter, HTTPException
from backend.agents.generic_agent import GenericAgent
from backend.strategy_registry import get_strategy_by_symbol
from backend.backtester.runner import BacktestRunner  # To load OHLCV data

router = APIRouter(prefix="/ai/agent", tags=["Agent"])

@router.get("/{symbol}/predict")
def get_agent_prediction(symbol: str):
    try:
        # Load user's strategy JSON (parsed)
        strategy_json = get_strategy_by_symbol(symbol)
        
        # Load OHLCV data for symbol (needed by agent)
        backtest_runner = BacktestRunner()
        ohlcv_data = backtest_runner.load_ohlcv(symbol)
        
        # Create agent with parsed strategy object (should be parsed accordingly)
        # Assuming strategy_json is already parsed or you parse it before passing
        agent = GenericAgent(symbol, strategy_json)
        
        # Call predict with OHLCV data
        prediction = agent.predict(ohlcv_data)
        
        return {"symbol": symbol, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
