# backend/agents/btcusdt_agent.py

## Logic
- Defines a specialized trading agent `BTCUSDTAgent` for the BTCUSDT symbol.
- Inherits from `GenericAgent`.
- On initialization, loads the trading strategy specifically for "BTCUSDT" using `load_strategy_for_symbol`.
- Passes the symbol and loaded strategy logic to the parent `GenericAgent` constructor.

## Routes
- None (this file only defines an agent class, no API routes).

# backend/agents/ethusdt_agent.py

## Logic
- Defines a specialized trading agent `ETHUSDTAgent` for the ETHUSDT symbol.
- Inherits from `GenericAgent`.
- On initialization, loads the trading strategy for "ETHUSDT" using `load_strategy_for_symbol`.
- Passes the symbol and strategy logic to the parent `GenericAgent` constructor.

## Routes
- None (agent class only, no API routes).

# backend/agents/generic_agent.py

## Logic
- Defines the `GenericAgent` class, a base trading agent.
- Initialized with:
  - `symbol`: trading pair symbol (e.g., BTCUSDT).
  - `strategy_logic`: an instance of `StrategyParser` representing the trading strategy.
- `evaluate(ohlcv_data: pd.DataFrame)` method:
  - Uses the `strategy_logic` to evaluate OHLCV market data.
  - Returns an action (`buy`, `sell`, or `hold`), a default confidence score (0.75), and a timestamp of the last data point.
- `predict(ohlcv_data: pd.DataFrame)` method:
  - Alias for `evaluate`, returns the same output.

## Routes
- None (class definition only, no API routes).

# backend/backtester/backtest_runner.py

## Logic
- Defines `BacktestRunner` class for running strategy backtests on historical OHLCV data.
- Loads OHLCV data for a given symbol from CSV files stored in a `data` directory.
- Supports filtering data by optional `start_date` and `end_date`.
- Runs backtest by:
  - Parsing a JSON strategy via `StrategyParser`.
  - Iterating through the data (skipping first 50 rows for indicator warmup).
  - Evaluating strategy signals (`buy`, `sell`, or none).
  - Simulating trades with a simple position model (only one long position at a time).
  - Calculating balance changes based on trade profits.
- Returns a summary including final balance, trade list, total trades, symbol, and date range.

- Helper function `run_backtest()` instantiates and runs `BacktestRunner` for FastAPI integration.

## Routes
- None defined here, but `run_backtest()` is likely used as backend logic called by API routes.

# backend/models/trading_models.py

## Logic
- Defines Pydantic data models (schemas) used throughout the backend for validation and data exchange:

### TradeLog
- Represents a single trade record.
- Includes symbol, entry/exit times and prices, position type (long/short), result (win/loss/breakeven), strategy name, indicators used, confidence score, ROI, drawdown, duration, and optional notes.

### IndicatorCondition
- Defines a single indicator condition in a strategy.
- Includes indicator type (`rsi`, `macd`, `ema`), condition expression as string, and timeframe (default "1m").

### StrategyConfig
- Defines a full trading strategy configuration.
- Includes strategy name, symbol, entry and exit conditions (lists of `IndicatorCondition`), creation time, and optional notes.

### StrategyPerformance
- Holds performance summary of a strategy.
- Includes strategy name, symbol, total trades, win rate, average ROI, max drawdown, optional Sharpe ratio, and last tested date.

### FeatureVector (Optional for ML)
- Schema for machine learning features.
- Includes timestamp, symbol, indicator values (rsi, macd, ema), and optional label (`buy`, `sell`, `hold`).

## Routes
- None (data models only).

# backend/storage/storage_utils.py

## Logic
- Manages local JSON file storage for trade logs and strategies under `/storage` folder.
- Trade Logs:
  - `save_trade_log(symbol, trade_data)`: Appends trade data to `{symbol}_trades.json`.
  - `load_trade_logs(symbol)`: Loads all trade logs for a symbol or returns empty list if none.
- Strategies:
  - `save_strategy(symbol, strategy_name, strategy_data)`: Saves strategy JSON under `/strategies/{symbol}/{strategy_name}.json`.
  - `load_strategy(symbol, strategy_name)`: Loads strategy JSON; raises error if not found.
  - `list_strategies(symbol)`: Lists all strategy names (without `.json`) for a symbol.

## Routes
- None (utility functions only).

# backend/ml_engine/feature_extractor.py

## Logic
- Adds key technical indicators to OHLCV DataFrame for ML and strategy use:
  - Converts 'close' prices to numeric.
  - Calculates RSI (14-period) using average gains and losses.
  - Calculates two EMAs (12 and 26 periods).
  - Computes MACD as difference of EMAs and its signal line (9-period EMA).
- Fills any NaN values with 0 before returning the enriched DataFrame.

## Routes
- None (utility function only).

# backend/ml_engine/indicators.py

## Logic
- Provides standalone functions to calculate technical indicators:

### `calculate_rsi(series, period=14)`
- Calculates the Relative Strength Index (RSI) over the given period on a Pandas Series.

### `calculate_ema(series, period=20)`
- Calculates the Exponential Moving Average (EMA) over the given period on a Pandas Series.

## Routes
- None (utility indicator functions only).


# backend/ml_engine/trainer.py

## Logic
- Trains a Random Forest classifier on OHLCV CSV data to predict next candle price movement.
- Feature engineering:
  - Adds `price_change` as percent change in close price.
  - Labels data with binary target: 1 if next candle price increases, else 0.
- Splits data into train/test sets (80/20).
- Trains Random Forest with 100 trees.
- Evaluates accuracy on test set and prints it.
- Saves trained model to specified path using `joblib`.

## Routes
- None (training utility only).


# backend/mother_ai/mother_ai.py

## Logic
- Defines `MotherAI` class, orchestrating multiple trading agents and strategy evaluation.
- Initialized with a list of agent symbols and strategy directory.
- Methods:
  - `load_agents()`: Loads strategy JSON for each symbol, creates `GenericAgent` instances.
  - `evaluate_agents(agents)`: For each agent, predicts live signal and confidence, fetches historical performance, calculates combined confidence score weighted by current confidence and historical win rate.
  - `calculate_confidence_score(confidence, win_rate, alpha=0.6, beta=0.4)`: Computes weighted score from confidence and win rate.
  - `decide_trades(top_n=1, min_score=0.7)`: Selects top N agents with score above threshold as trade decisions.
  - `make_portfolio_decision(top_n=3, min_score=0.75)`: Runs full pipeline to return top N trade decisions.

## Routes
- None (core AI logic module).


# backend/mother_ai/performance_tracker.py

## Logic
- Manages trade performance logs for each trading symbol.
- Stores logs as JSON files under `backend/mother_ai/logs`.
- Methods:
  - `log_trade(symbol, trade_data)`: Appends a trade record to symbol’s log file.
  - `get_agent_log(symbol, limit=100)`: Retrieves the latest `limit` trade logs.
  - `_load_logs(path)`: Helper to read logs from file or return empty list.
  - `current_time()`: Returns current UTC ISO timestamp.
  - `clear_log(symbol)`: Deletes the log file for the symbol.
  - `log_summary(symbol)`: Calculates and returns win rate, loss rate, and total trades from logs.

## Routes
- None (internal performance tracking utility).


# backend/mock_trading/mock_trade_executor.py

## Logic
- Simulates trades and manages a mock portfolio balance stored in JSON.
- Uses a local file `mock_balance.json` for USD balance and holdings.
- Functions:
  - `load_mock_balance()`: Loads current mock portfolio, initializes if missing.
  - `save_mock_balance(data)`: Saves portfolio state to file.
  - `execute_mock_trade(symbol, action, price, confidence)`: Simulates a buy or sell trade:
    - Buy: Uses 10% of USD balance to buy amount of symbol.
    - Sell: Sells all holdings of symbol if present.
    - Updates balances accordingly and logs results.
    - Returns trade execution status and details.
  - `get_mock_portfolio()`: Returns current mock portfolio state.

## Routes
- None (mock trading utility).

# backend/routes/agent_routes.py

## Logic
- Defines a FastAPI router with a simple endpoint.

## Routes
- `GET /agents`: Returns a static list of agent names: `["BTC", "ETH", "SOL"]`.


# backend/routes/agent_prediction_routes.py

## Logic
- Provides FastAPI routes for trading agent predictions based on strategies and OHLCV data.

## Routes
- `GET /{symbol}/predict`
  - Returns prediction (`buy`, `sell`, or `hold`) with confidence % for a given symbol.
  - Loads the first strategy for the symbol, loads OHLCV data, creates a `GenericAgent`, and predicts action.

- `GET /` (root)
  - Returns a static list of available agent symbols: `["BTC", "ETH", "SOL", "AAPL", "TSLA", "GOOG"]`.

- `GET /predictions`
  - Returns aggregated predictions for all available agents.
  - For each symbol:
    - Loads strategy and OHLCV data.
    - Creates agent and predicts action/confidence.
    - Returns data structured for frontend consumption including `agentName`, `prediction`, `confidence` (0-100), and placeholder trade details (`entryPrice`, `targetPrice`, `stopLoss`).
  - Handles missing data or errors gracefully by adding appropriate placeholders or error info.

## Notes
- Uses `StrategyParser`, `BacktestRunner`, and `GenericAgent` internally.
- Placeholder trade details can be enhanced with real price targets/stop-loss from strategy.

# backend/routes/backtest_routes.py

## Logic
- Provides FastAPI routes for running backtests and retrieving recent backtest trade logs.

## Routes
- `POST /backtest/`
  - Accepts a JSON payload with:
    - `symbol` (str)
    - `strategy_json` (dict)
    - `start_date` (str)
    - `end_date` (str)
  - Runs backtest using `run_backtest()` helper.
  - Returns backtest results including final balance, trades, etc.
  - Handles exceptions with HTTP 500 errors.

- `GET /backtest/results`
  - Reads recent trade logs from `storage/Performance_logs` directory.
  - Aggregates and tags trades by symbol.
  - Returns the 10 most recent trades sorted by timestamp.
  - Returns 404 if logs directory not found.

## Notes
- Depends on local filesystem storage for trade logs.


# backend/routes/binance.py

## Logic
- Provides FastAPI route to connect user to Binance API using their API keys.
- Uses `connect_user_api` utility for the actual connection and validation.

## Routes
- `POST /connect`
  - Accepts `apiKey` and `secretKey` in request body.
  - Attempts to connect to Binance with provided keys.
  - Returns success message if connected.
  - Raises HTTP 400 if keys are invalid.
  - Raises HTTP 500 for other errors.

## Notes
- No route prefix; full path depends on main app inclusion.


# backend/routes/mother_ai_routes.py

## Logic
- Provides API endpoints to interact with the `MotherAI` decision engine and trade logs.

## Routes
- `GET /decision`
  - Calls `MotherAI.make_portfolio_decision()` to get top trade decisions for configured agents (`AAPL`, `TSLA`, `GOOG`).
  - Returns decision data or HTTP 500 on error.

- `GET /trades`
  - Aggregates trade logs for all agents managed by `MotherAI`.
  - Returns all trades sorted by timestamp descending.
  - Handles errors with HTTP 500.

## Notes
- `MotherAI` instance is created with predefined agent symbols.


# backend/routes/strategy_routes.py

## Logic
- Provides an endpoint to list all available strategy JSON files in `backend/backtester/strategies`.
- Reads each `.json` file, extracts the symbol and full strategy JSON.

## Routes
- `GET /list`
  - Returns a list of strategies with:
    - `strategy_id` (filename without extension)
    - `symbol` (from strategy JSON, or "UNKNOWN" if missing)
    - `strategy_json` (full JSON content)
  - Handles file read errors gracefully by logging and skipping.

## Notes
- Uses local filesystem access to load strategies.


# backend/routes/strategy_management_routes.py

## Logic
- Manages saving, listing, deleting, and performance retrieval of user trading strategies.
- Validates strategies using `parse_strategy_json` before saving.
- Registers strategies to internal registry for lookup.
- Loads strategy performance stats via `StrategyHealth`.

## Routes
- `POST /save`
  - Accepts `strategy_id`, `symbol`, and `strategy_json`.
  - Validates and saves the strategy file.
  - Registers the strategy.
  - Returns success or HTTP errors.

- `GET /list`
  - Returns a list of all registered strategies.

- `GET /{symbol}`
  - Returns all strategies registered for the specified symbol (case-insensitive).
  - 404 if none found.

- `DELETE /strategies/{symbol}/{strategy_id}`
  - Deletes the specified strategy file.
  - Accepts full or partial strategy ID; normalizes accordingly.
  - Returns success or 404 if not found.

- `GET /{symbol}/{strategy_id}/performance`
  - Returns performance summary stats for the given strategy based on stored logs.
  - 404 if no performance data found.

## Notes
- Uses local filesystem paths for storage.
- Symbol inputs are normalized to uppercase.


# backend/strategy_engine/json_strategy_parser.py

## Logic
- Provides functions to load, parse, and validate trading strategy JSON files.
- Supports loading default strategy, user strategy files, or raw JSON strings.
- Validates presence and type of required keys (`symbol`, `indicators`).

## Functions
- `load_strategy_for_symbol(symbol: str)`: loads default strategy JSON file for a symbol.
- `load_strategy_from_file(symbol: str, strategy_id: str)`: loads a user strategy JSON file by ID.
- `load_strategy_from_json_string(json_str: str)`: loads and validates strategy from raw JSON string.
- `validate_strategy(strategy: Dict)`: internal validation of strategy keys and types.
- `parse_strategy_json(json_str: str)`: frontend-facing parser calling validation.

## Notes
- Strategy files expected in `backend/backtester/strategies` folder.
- File naming convention: `{symbol}_strategy_{id}.json`.
- Raises exceptions for missing files or invalid format.


# backend/strategy_engine/strategy_health.py

## Logic
- Analyzes a strategy's performance log (list of trade dicts).
- Calculates key health metrics over a recent subset (`lookback`) of trades.

## Methods
- `win_rate(lookback)`: ratio of winning trades to total trades in lookback.
- `avg_profit(lookback)`: average ROI (%) of trades in lookback.
- `recent_drawdown(lookback)`: max absolute loss ROI in lookback.
- `summary(lookback)`: returns a dict of all above metrics rounded to 3 decimals and count of trades analyzed.

## Notes
- Defaults to last 20 trades.
- Useful for evaluating strategy effectiveness and risk.


# StrategyParser Overview

## Purpose
The `StrategyParser` class processes a trading strategy JSON configuration and applies its logic to OHLCV price data to generate trading signals (`buy`, `sell`, or `hold`).

---

## Core Logic

- **Initialization**:  
  Loads strategy JSON containing `symbol` and `indicators` (e.g., RSI, EMA) configurations.

- **apply_indicators(df)**:  
  Adds technical indicator columns (like RSI, EMA) to the input OHLCV DataFrame based on the strategy parameters.

- **evaluate_conditions(df)**:  
  Iterates over the DataFrame rows to evaluate buy/sell/hold signals per row according to indicator thresholds:
  - RSI triggers buy if below `buy_below` threshold (default 30), sell if above `sell_above` (default 70).
  - EMA triggers buy if price crosses above EMA, sell if price crosses below EMA.

- **evaluate(df)**:  
  Returns the signal for the most recent data point (`buy`, `sell`, or `hold`).

---

## Usage in API Routes

- **Agent Prediction Route** (`GET /{symbol}/predict`):  
  Loads strategy JSON → Creates `StrategyParser` → Loads OHLCV data → Applies `evaluate()` to get current trading signal and confidence.

- **Aggregated Predictions Route** (`GET /predictions`):  
  Iterates over available symbols → Runs `StrategyParser.evaluate()` for each → Returns signals with confidence and placeholder trade details.

---

## Example Strategy JSON Structure

```json
{
  "symbol": "BTCUSDT",
  "indicators": {
    "rsi": {
      "period": 14,
      "buy_below": 30,
      "sell_above": 70
    },
    "ema": {
      "period": 20,
      "buy_crosses_above": true,
      "sell_crosses_below": true
    }
  }
}


binance_connector.py
Purpose
Handles connection to the Binance API, manages a single-user client instance, and provides utility functions to fetch OHLCV candlestick data.

🔧 Key Functions
connect_user_api(api_key, secret_key)

Connects to Binance using given or default .env credentials.

Returns a success/failure message with logs.

Validates credentials by calling get_account().

get_binance_client()

Returns the initialized Binance client.

Raises an error if not connected.

fetch_ohlcv(symbol, interval, limit)

Fetches OHLCV candlestick data (timestamp, open, high, low, close, volume).

Defaults to 1-minute interval and 100 candles.

Uses the connected client.

🔗 Dependencies
.env file for default API keys

binance.client.Client – Binance SDK

backend.utils.logger – Custom logging utility

📌 Notes
Maintains a global client instance user_binance_client for reusability.

Uses structured error logging for debugging.

Adds the project root to sys.path for clean imports.


/logger.py
Purpose
Sets up a centralized logging system for the trading backend, logging messages to both the console and a file.

🛠 Key Features
Creates storage/logs/ directory if it doesn't exist.

Configures a logger named "trading_system" with two handlers:

Console handler (INFO level)

File handler (DEBUG level, logs to app.log)

Applies a standard formatter:
timestamp - level - message

Exposes a simple log shortcut for global use.

🔗 Used In
Other modules can import the logger via:

python
Copy code
from backend.utils.logger import log
📁 Log Output
File: storage/logs/app.log

Includes debug, info, warning, error, and critical logs depending on context.

strategy_registry.py
Purpose
Manages saving, loading, deleting, and registering trading strategies stored as JSON files, organized by symbol. Acts as a local strategy database for the AI trading system.

📁 Directory Structure
Strategies stored in: backend/backtester/strategies/

Registry file: registry.json
Tracks available strategy IDs per symbol.

🧩 Key Functions
Function	Description
list_strategies()	Lists all strategy files grouped by symbol.
load_strategy(symbol, strategy_id)	Loads a specific strategy JSON file.
save_strategy(symbol, strategy_id, data)	Saves a strategy to disk and registers it.
delete_strategy(symbol, strategy_id)	Deletes a strategy file and unregisters it.
get_strategy_by_symbol(symbol)	Loads all strategies for a given symbol.
register_strategy(symbol, strategy_id)	Adds a strategy ID to registry.json.
_unregister_strategy(symbol, strategy_id)	Removes a strategy ID from the registry.
get_registered_strategies(symbol=None)	Returns full or filtered registry content.

🧠 Behavior Summary
Automatically creates strategies/ folder if missing.

Each strategy is stored as <symbol>_strategy_<id>.json.

Registry keeps a clean index of available strategies.

Provides resilience via error handling on file I/O.

🔗 Integration
Used by:

Frontend: Sends strategy JSON to backend via API.

Backtester/Executor: Loads strategies to simulate or trade.

main.py
Purpose
Entry point for the AI Trading Dashboard Backend, initializing the FastAPI app, enabling CORS, and registering all feature-specific API routers.

🚀 App Initialization
python
Copy code
app = FastAPI(
    title="AI Trading Dashboard API",
    description="Backend API for AI-driven trading system with multi-agent architecture.",
    version="1.0.0"
)
🔗 Enabled CORS
Allows frontend requests from Vite on http://localhost:5173

Accepts all headers, methods, and credentials.

🩺 Health Check
python
Copy code
@app.get("/")
async def read_root():
    return {"message": "Backend is running!"}
🧩 Included Routers
Prefix	Tag	Module
/api/agent	Agent	routes/agent.py
/api/strategy	Strategy	routes/strategy.py
/api/mother-ai	Mother AI	routes/mother_ai.py
/api/backtest	Backtesting	routes/backtest.py
/api/binance	Binance	routes/binance.py
/api/agents	Agent Registry	routes/agent_registry.py
/api/strategies	Strategy Files	routes/strategy_file_loader.py

✅ All routes are cleanly namespaced with clear tagging.

🌐 Example API Structure
bash
Copy code
GET    /api/agent/all
POST   /api/strategy/save
GET    /api/mother-ai/predict
POST   /api/backtest/run
GET    /api/binance/ohlcv
