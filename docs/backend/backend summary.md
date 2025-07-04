Core Backend Functions & Classes
Trading Agents (BTCUSDTAgent, ETHUSDTAgent, GenericAgent)

Specialized agents for specific symbols.

Load symbol-specific strategies.

Evaluate OHLCV data with strategy logic.

Return trading actions (buy, sell, hold) with confidence and timestamps.

BacktestRunner

Load historical OHLCV data for a symbol.

Run backtests on strategies.

Simulate trades and track balance and performance.

Return summary statistics and trades.

StrategyParser

Initialize with strategy JSON (symbol + indicator configs).

Apply technical indicators (RSI, EMA) to OHLCV data.

Evaluate buy/sell/hold signals row-by-row.

Provide latest trading decision based on data.

ML Engine

Feature Extractor: Compute RSI, EMAs, MACD indicators on OHLCV.

Indicators: Standalone functions to calculate RSI and EMA.

Trainer: Train a Random Forest model on OHLCV data to predict price movement.

MotherAI

Manage multiple trading agents and aggregate their predictions.

Load strategies and evaluate live signals.

Calculate weighted confidence combining prediction and historical win rate.

Decide which trades to execute based on scores.

Provide portfolio-level decision making.

PerformanceTracker

Log and retrieve trade performance.

Calculate win rate, loss rate, and total trades.

Clear or summarize logs.

MockTradeExecutor

Simulate trades with a mock portfolio balance.

Buy/sell logic with USD balance and holdings.

Save and load mock portfolio state.

Storage Utilities

Save/load/delete trade logs and strategies as JSON files.

List available strategies for symbols.

Strategy Registry

Register/unregister strategies.

Maintain an index of available strategies.

Load/save/delete strategy files.

Retrieve strategies by symbol.

Binance Connector

Connect user API keys to Binance.

Fetch OHLCV candlestick data using connected client.

Manage singleton Binance client instance.

Logger Utility

Set up centralized logging.

Log to console and file with timestamps and levels.

Strategy Engine JSON Parser

Load and validate strategy JSON files or strings.

Ensure correct keys and data types.

Parse raw JSON into usable strategy dicts.

Strategy Health

Compute health metrics for strategies from trade logs.

Calculate win rate, average profit, drawdown.

Summarize strategy performance over recent trades.

FastAPI Routes & Endpoints
Agent Routes

GET /agents: List available agents (e.g., BTC, ETH, SOL).

Agent Prediction Routes

GET /{symbol}/predict: Get prediction (buy/sell/hold + confidence) for a symbol.

GET /predictions: Aggregate predictions for all supported symbols.

Backtest Routes

POST /backtest/: Run backtest with given symbol, strategy JSON, and date range.

GET /backtest/results: Retrieve recent backtest trade logs.

Binance Routes

POST /connect: Connect user Binance API keys and validate connection.

Mother AI Routes

GET /decision: Get top portfolio trade decisions from MotherAI.

GET /trades: Retrieve aggregated trade logs for all MotherAI agents.

Strategy Routes

GET /list: List all available strategy JSON files with metadata.

Strategy Management Routes

POST /save: Save a new or updated strategy (validates before saving).

GET /list: List all registered strategies.

GET /{symbol}: Get all strategies for a specific symbol.

DELETE /strategies/{symbol}/{strategy_id}: Delete a strategy by ID.

GET /{symbol}/{strategy_id}/performance: Get performance summary of a strategy.