from fastapi import APIRouter, HTTPException
from backend.agents.generic_agent import GenericAgent
from backend.strategy_registry import get_strategy_by_symbol
from backend.backtester.runner import BacktestRunner
from backend.strategy_engine.strategy_parser import StrategyParser # Added
from typing import List

router = APIRouter(tags=["Agent"])

@router.get("/{symbol}/predict")
def get_agent_prediction(symbol: str):
    """
    Return a prediction (BUY/SELL/HOLD + confidence) from an agent based on the latest OHLCV data and strategy.
    """
    try:
        # 🔁 Load strategy configurations for the given symbol
        strategies_for_symbol = get_strategy_by_symbol(symbol)
        if not strategies_for_symbol:
            raise HTTPException(status_code=404, detail=f"No strategies found for symbol: {symbol}")

        # For now, use the first strategy found for the symbol
        # TODO: Add logic to select a specific or default strategy if multiple exist
        strategy_dict = strategies_for_symbol[0]

        strategy_runner = StrategyParser(strategy_dict)

        # 📊 Load OHLCV data for symbol
        backtest_runner = BacktestRunner() # Consider making this a dependency or singleton
        ohlcv_data = backtest_runner.load_ohlcv(symbol)
        if ohlcv_data is None or ohlcv_data.empty:
            raise HTTPException(status_code=400, detail=f"No OHLCV data found for symbol: {symbol}")

        # 🧠 Create agent instance with symbol and strategy logic
        agent = GenericAgent(symbol=symbol, strategy_logic=strategy_runner)

        # 🧾 Get prediction (agent.predict() returns a dict)
        prediction_result = agent.predict(ohlcv_data)
        action = prediction_result["action"]
        confidence_value = prediction_result["confidence"] # This is 0.0 - 1.0

        return {
            "symbol": symbol,
            "signal": action, # 'buy', 'sell', 'hold'
            "confidence": confidence_value * 100 # Return as percentage 0-100
        }

    except HTTPException:
        raise  # Re-raise known HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

AVAILABLE_AGENTS = ["BTC", "ETH", "SOL", "AAPL", "TSLA", "GOOG"]

@router.get("", response_model=List[str])
def list_available_agents():
    return AVAILABLE_AGENTS

@router.get("/predictions")
def get_all_agent_predictions():
    """
    Return predictions for all available agents.
    Matches the structure expected by AgentPredictionCard.jsx.
    """
    predictions = []
    for symbol in AVAILABLE_AGENTS:
        try:
            strategies_for_symbol = get_strategy_by_symbol(symbol)
            if not strategies_for_symbol:
                print(f"No strategies found for symbol: {symbol}, skipping for aggregated predictions.")
                # Optionally add a specific "no strategy" entry to predictions list
                predictions.append({
                    "agentName": f"{symbol} Agent",
                    "prediction": "No Strategy",
                    "confidence": 0,
                    "tradeDetails": {"symbol": symbol, "entryPrice": 0, "targetPrice": 0, "stopLoss": 0},
                })
                continue

            # Use the first strategy found
            strategy_dict = strategies_for_symbol[0]
            strategy_runner = StrategyParser(strategy_dict)

            backtest_runner = BacktestRunner()
            ohlcv_data = backtest_runner.load_ohlcv(symbol)
            if ohlcv_data is None or ohlcv_data.empty:
                print(f"No OHLCV data found for symbol: {symbol}, skipping for aggregated predictions.")
                predictions.append({
                    "agentName": f"{symbol} Agent",
                    "prediction": "No Data",
                    "confidence": 0,
                    "tradeDetails": {"symbol": symbol, "entryPrice": 0, "targetPrice": 0, "stopLoss": 0},
                })
                continue

            agent = GenericAgent(symbol=symbol, strategy_logic=strategy_runner)
            prediction_result = agent.predict(ohlcv_data)
            action = prediction_result["action"] # 'buy', 'sell', 'hold'
            confidence_value = prediction_result["confidence"] # 0.0 - 1.0

            # Mimic the structure expected by AgentPredictionCard.jsx
            # The 'tradeDetails' part is not fully available from agent.predict directly
            # We'll have to make some assumptions or simplify it for now.
            # The frontend expects: agentName, prediction, confidence, tradeDetails: {symbol, entryPrice, targetPrice, stopLoss}
            # For now, we'll use placeholder for tradeDetails as it's not directly part of the agent's prediction output.
            # A more robust solution would involve fetching current price for entry, and having strategy define targets/stops.

            # Simplification: Use latest close price as a proxy for entryPrice
            entry_price = ohlcv_data['close'].iloc[-1] if not ohlcv_data.empty else 0 # Assuming 'close' column

            predictions.append({
                "agentName": f"{symbol} Agent",
                "prediction": action, # 'buy', 'sell', or 'hold' string
                "confidence": confidence_value * 100, # Frontend expects 0-100
                "tradeDetails": {
                    "symbol": symbol,
                    "entryPrice": entry_price,
                    "targetPrice": entry_price * 1.05 if action == "buy" else entry_price * 0.95, # Placeholder
                    "stopLoss": entry_price * 0.98 if action == "buy" else entry_price * 1.02, # Placeholder
                }
            })
        except Exception as e:
            print(f"Error getting prediction for {symbol}: {str(e)}")
            # Optionally, add an error state for this agent in the response
            predictions.append({
                "agentName": f"{symbol} Agent",
                "prediction": "Error",
                "confidence": 0,
                "tradeDetails": {"symbol": symbol, "entryPrice": 0, "targetPrice": 0, "stopLoss": 0},
                "error": str(e)
            })

    return predictions