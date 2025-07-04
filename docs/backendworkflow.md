Here’s a complete backend workflow breakdown for your AI Trading System — modular, scalable, and tailored for your self-hosted SaaS architecture with local storage:


---

🔧 Backend Workflow Overview

💡 Core Goal:

Automate AI-based crypto trading across multiple symbols, allowing user-defined strategy injection, machine learning model predictions, and a Mother AI to control execution based on confidence scoring.


---

1. 🌐 FastAPI App Entry – main.py

Starts the backend server.

Includes all API routers (agent.py, mother_ai.py, strategy.py, backtest.py).

Loads .env settings for configurations.

Sets up CORS to allow frontend access.



---

2. 🧠 Agent Layer – agents/

Each file is an independent AI agent managing a symbol.

generic_agent.py

Loads models or strategies.

Reads latest OHLCV data (via binance_api.py).

Extracts features (via feature_extractor.py).

Runs logic:

ML prediction (model.predict())

OR rule-based strategy parsed from JSON.


Outputs decision: "buy" | "sell" | "hold" with a confidence score.


btc_agent.py, eth_agent.py

Symbol-specific agents using custom logic or models.


Output:

→ Trade decision + metadata → returned to API or passed to Mother AI


---

3. 📈 Strategy Engine – strategy_engine/

json_strategy_parser.py

Parses frontend user strategy JSON (e.g., RSI > 70 → sell).

Translates into Python logic.


strategy_parser.py

Executes parsed strategy rules on price data.


strategy_health.py

Evaluates performance of strategies (win/loss rate, PnL, etc).



---

4. 🧬 ML Engine – ml_engine/

feature_extractor.py

Adds technical indicators (RSI, EMA, MACD) to price data.


trainer.py

Trains ML models on labeled OHLCV datasets.

Exports .pkl model files for agents to load.



---

5. 🧩 Mother AI – mother_ai/

mother_ai.py

Aggregates decisions from all agents.

Evaluates recent performance via performance_tracker.py.

Selects high-confidence trades only for execution.

Implements cooldowns, volatility filters, or trade limits.


performance_tracker.py

Tracks each agent’s win rate, ROI, and execution history.

Feeds this data into the Mother AI's decision-making.



---

6. 🧪 Backtesting – backtester/

runner.py

Runs user strategies over historical data (CSV from /data/).

Collects and returns trade logs + metrics (PnL, Sharpe ratio, etc).



---

7. 💾 Local Storage Interface – db/

storage.py

Stores:

Trade logs (storage/trade_logs/{symbol}_trades.json)

Strategy JSONs (storage/strategies/{symbol}/{strategy_name}.json)


Ensures offline-first, self-contained SaaS behavior.

Simple file-based, user-accessible format.


models.py

Contains Pydantic models and schemas for consistent data exchange.



---

8. 🧭 API Routes – routes/

agent.py:

/ai/agent/{symbol} – run predictions for a symbol

/ai/agent/list – list available agents


mother_ai.py:

/mother-ai/decision – get top trades from agents

/mother-ai/status – current execution/filtering status


strategy.py:

/strategies/{symbol}/{strategy} – save/load/list strategies

Handles custom logic uploads from frontend


backtest.py:

/backtest/run – run historical backtest

/backtest/report – serve previous results




---

9. 🔗 Utils – utils/

binance_api.py

Pulls live OHLCV data from Binance (1m, 5m, etc).

Can be swapped with websocket for real-time updates.


logger.py

Central logging utility.



---

10. 📦 Modular Strategy Registry – strategy_registry.py

Loads all available strategy modules dynamically.

Keeps registry of valid strategy functions.

Can hot-swap logic without server restart.



---

11. 🗂️ Persistent Folder Structure

storage/
├── trade_logs/
│   ├── btcusdt_trades.json
│   └── ethusdt_trades.json
└── strategies/
    ├── btcusdt/
    │   └── rsi_macd_combo.json
    └── ethusdt/
        └── ema_cross_strategy.json


---

🔒 Security Notes (for Self-Hosted SaaS)

No DB server needed – stores user data locally in files

Can be zipped and moved with the app

API keys handled client-side (or stored encrypted locally)



---

📤 Workflow Summary (Runtime)

Frontend (Strategy Builder) 
   ↓
POST JSON strategy → Backend /strategies/
   ↓
Agent reads strategy → Applies on live OHLCV
   ↓
Returns prediction → Mother AI evaluates
   ↓
Only high-confidence trades executed/logged
   ↓
Logs saved locally → visible in dashboard




Backend Workflow Overview
1. User or System Input
User interacts via API routes (FastAPI endpoints).

Inputs include symbol names, strategy JSONs, dates, Binance API keys, or requests for predictions/trades/backtests.

2. Strategy Management & Loading
Strategy Management Routes handle saving, listing, deleting, and retrieving strategies.

Uses Strategy Registry to:

Register/unregister strategies.

Load strategies by symbol or ID.

Manage strategy files on disk.

Strategies are parsed and validated using Strategy Engine JSON Parser.

3. Data Acquisition & Preparation
Binance Connector:

Connects to Binance with user keys.

Fetches OHLCV candlestick data for requested symbols.

Storage Utilities:

Save/load trade logs and strategies as JSON.

Support data persistence for backtests and performance tracking.

4. Strategy Evaluation & Signal Generation
Trading Agents (BTCUSDTAgent, ETHUSDTAgent, GenericAgent):

Load symbol-specific strategy via registry or loader.

Use StrategyParser to apply indicators (RSI, EMA, etc.) on OHLCV data.

Evaluate buy/sell/hold signals for the current market data.

Return trading action with confidence and timestamp.

ML Engine (optional/advanced):

Extracts features (RSI, EMA, MACD).

Can train models to enhance prediction accuracy.

5. Aggregation and Decision Making
MotherAI:

Loads multiple trading agents.

Aggregates their predictions and confidence scores.

Combines live prediction confidence with historical win rates (via PerformanceTracker).

Computes weighted confidence scores.

Makes portfolio-level trade decisions, selects best trades to execute.

6. Execution and Simulation
MockTradeExecutor:

Simulates trade execution using mock portfolio balances.

Executes buy/sell logic based on agent signals.

Updates mock portfolio state.

Logs trade outcomes.

PerformanceTracker:

Logs and retrieves trade performance.

Summarizes win/loss rates and trade history.

7. Backtesting
BacktestRunner:

Loads historical OHLCV data for a symbol.

Runs strategy on historical data.

Simulates trades and computes balance/performance metrics.

Returns detailed backtest results.

Backtest routes allow users to start backtests and retrieve results.

8. Logging & Monitoring
Logger Utility:

Centralized logging for debugging and info.

Logs saved to console and file with timestamps.

Monitoring of system activity and errors.

Summary of API Routes Workflow
Route Group	Key Operations	Connects to
Agent Routes	List available agents	Agent registry
Agent Prediction Routes	Get predictions per symbol or aggregate	Trading Agents, StrategyParser, OHLCV data
Backtest Routes	Run backtests, get backtest logs	BacktestRunner, Storage Utilities
Binance Routes	Connect to Binance API, fetch OHLCV	Binance Connector
Mother AI Routes	Get portfolio trade decisions and aggregated trade logs	MotherAI, PerformanceTracker
Strategy Routes	List available strategies	Strategy Registry, JSON Parser
Strategy Management	Save, delete, list strategies; get performance	Strategy Registry, Strategy Health

This workflow ensures a modular, clear flow:

Input → Strategy Load → Data Fetch → Signal Generation → Aggregation → Execution/Simulation → Logging/Backtesting