# generic_agent.py 
__init__ / Initialize agent with symbol and strategy parser / N/A

evaluate / Evaluate OHLCV data to generate buy/sell/hold signal / Input: OHLCV DataFrame, Output: dict (signal, timestamp)

predict / Alias for evaluate / Same as evaluate

Data comes from OHLCV DataFrame input, no internal storage.

# backtester/runner.py
__init__ / Set data directory / N/A

load_ohlcv / Load OHLCV CSV data for symbol / From CSV file (data/{symbol}.csv)

run / Run backtest with strategy JSON, simulate trades, return results / Uses loaded OHLCV data and strategy JSON input

run_backtest / FastAPI helper to run backtest / Calls BacktestRunner.run

# binance/fetch_live_ohlvc.py
fetch_ohlcv / Fetch OHLCV data from Binance API via ccxt / Data from Binance public API, returns DataFrame

# db/models.py 
TradeLog / Define trade record schema / Stored as trade log entries

IndicatorCondition / Define single indicator condition for strategies / Used in strategy config

StrategyConfig / Define strategy rules and metadata / Stored as strategy configuration

StrategyPerformance / Summarize strategy backtest performance / Stored as performance reports

FeatureVector / Define ML input features schema / Used for ML model inputs

# db/storage.py
save_trade_log / Append trade data to symbol’s trade log JSON / Stored in storage/trade_logs/{symbol}_trades.json

load_trade_logs / Load trade logs for symbol / From storage/trade_logs/{symbol}_trades.json

save_strategy / Save strategy JSON file / Stored in storage/strategies/{symbol}_strategy_{id}.json

load_strategy / Load specific strategy JSON / From storage/strategies/{symbol}_strategy_{id}.json

list_strategies / List strategy IDs for symbol / Reads filenames from storage/strategies/

# ml_engine/feature_extractor.py
add_technical_indicators / Add RSI, EMA, MACD indicators to OHLC DataFrame / Input/output: DataFrame in-memory

# ml_engine/indicators.py 
calculate_rsi / Compute RSI from price series / Uses in-memory series

calculate_ema / Compute EMA from price series / Uses in-memory series

# ml_engine/trainer.py 
train_model / Train RandomForest model on OHLCV CSV and save it / Reads from CSV, saves model to file via joblib

# mother_ai/mother_ai.py 
load_agents / Dynamically load agent classes and strategies / From backend/agents/, strategy_engine/strategies/

evaluate_agents / Evaluate all agents and score their performance / Uses agent prediction + PerformanceTracker

decide_trades / Select top trade decisions based on score threshold / From evaluated agents

make_portfolio_decision / Return best decision with price, log it / Uses Binance via fetch_ohlcv, stores in tracker

get_trade_logs / API route to get recent trades / From storage/trade_history/

get_decision / API route to get latest decision from MotherAI / Returns top-scoring signal decision

# mother_ai/oerformance_tracker.py
log_trade / Save trade entry to symbol log / Stored in storage/trade_history/{symbol}_log.json

get_agent_log / Load recent trades for symbol / From storage/trade_history/

clear_log / Delete trade log file / From storage/trade_history/

log_summary / Compute win/loss rate summary / From trade logs

current_time / Get current UTC timestamp / In-memory only

# mother_ai/trade_executer.py
load_mock_balance / Load mock portfolio balance / From storage/mock_balance.json

save_mock_balance / Save mock balance data / To storage/mock_balance.json

execute_mock_trade / Simulate buy/sell trade and update balance / Reads & writes mock_balance.json

get_mock_portfolio / Return current mock balance / From mock_balance.json

# routes/agent_registry.py 
list_agents / List all agent class names dynamically / From backend/agents/*.py (excluding generic/init)

# routes/agent.py
get_real_agents / List custom agent classes from agents/ / From backend/agents/*.py

list_available_agents / API to return available custom agent symbols / In-memory

get_agent_prediction / API to get prediction from single generic agent / Strategy from storage/strategies/, OHLCV from Binance

get_all_agent_predictions / API to get paginated predictions from all agents / Strategies from file, OHLCV from Binance

# routes/backtest.py
execute_backtest / Run a strategy backtest on given symbol and date range / Uses run_backtest, input via request body

get_recent_backtest_results / Get paginated backtest trade logs / From storage/performance_logs/*_trades.json

# routes/binance.py
connect_to_binance / Connect user to Binance API with provided keys / Uses connect_user_api

get_latest_price / Get latest price for a trading symbol / From Binance API via user_binance_client

# routes/mother_ai.py
get_mother_ai_decision / Returns highest-confidence trade decision from Mother AI / Uses all loaded agent predictions

get_trades / Returns paginated trade history / From backend/storage/trade_history via PerformanceTracker

# routes/strategy_file_loader.py
load_all_strategies / Load all strategy JSONs from storage / From backend/storage/strategies folder

list_strategies / Paginated API to list strategies / Calls load_all_strategies

get_strategy_performance / Get win rate and total trades for a strategy / Reads JSON from backend/storage/performance_logs

delete_strategy / Delete strategy and its performance files / Deletes from backend/storage/strategies and performance_logs

# routes/strategy.py
save_user_strategy / Validate and save strategy JSON / Stores in backend/storage/strategies/{symbol}_strategy_{id}.json

list_registered_strategies / Paginated list of all saved strategies / Reads from backend/storage/strategies

list_strategies_by_symbol / List strategies for a symbol / Reads from backend/storage/strategies

delete_strategy_api / Delete a strategy file by key / Deletes from backend/storage/strategies

get_strategy_performance / Get performance summary for strategy / Reads from backend/storage/performance_logs/{symbol}_strategy_{id}.json

# strategy_engine/json_strategy_parser.py
save_strategy_to_file / Save strategy dict as JSON / Stores in backend/storage/strategies/{symbol}_strategy_{id}.json

load_strategy_for_symbol / Load default strategy JSON for symbol / Reads backend/storage/strategies/{symbol}_strategy_default.json

load_strategy_from_file / Load user strategy JSON by symbol & id / Reads backend/storage/strategies/{symbol}_strategy_{id}.json

load_strategy_from_json_string / Parse and validate strategy from JSON string input / From frontend or API

validate_strategy / Validate strategy dict structure / Ensures keys symbol and indicators exist and valid

parse_strategy_json / Wrapper to parse and validate JSON string / Calls load_strategy_from_json_string

# strategy_engine/strategy_health.py
__init__ / Initialize with trade logs list / Input: list of trades with 'result' & 'roi'

win_rate / Calculate win rate over last N trades / Data from performance_log

avg_profit / Average ROI over last N trades / Data from performance_log

recent_drawdown / Max loss magnitude in last N trades / Data from performance_log

summary / Return dict summary of win rate, avg profit, drawdown, count / Uses last N trades from performance_log

# strategy_engine/strategy_parser.py 
__init__ / Load strategy JSON, extract symbol & indicators / Input: strategy dict

apply_indicators / Add indicators (RSI, EMA) to OHLCV DataFrame / Input: DataFrame with 'close'

evaluate_conditions / Compute signals per row ("buy"/"sell"/"hold") based on indicators / Uses DataFrame with indicators

parse / Static method to create instance from dict / Input: strategy dict

evaluate / Return last signal for latest data row / Uses apply_indicators + evaluate_conditions

# utils/binance_api.py 
connect_user_api(api_key, secret_key) / Init Binance client with keys, test connection / Uses .env fallback keys

get_binance_client() / Return connected Binance client or error if none / Uses global user_binance_client

fetch_ohlcv(symbol, interval, limit) / Fetch OHLCV candle data from Binance / Uses connected client, returns list of dicts

get_symbol_price(symbol) / Get current ticker price from Binance / Uses connected client, returns float price

# utils/logger.py
logger / Configured logging for console (INFO) & file (DEBUG) output / Logs saved in storage/logs/app.log

log / Shortcut to logger instance for app-wide logging

# main.py
app / FastAPI instance for AI Trading Dashboard backend / Hosts all API routes and middleware

CORSMiddleware / Enables frontend access from http://localhost:5173 / Controls cross-origin requests

read_root() / Health check endpoint returning status message / No data storage, simple GET "/"

include_router() calls / Register feature-specific route modules under /api/* prefixes / Routes handle agents, strategies, mother AI, backtesting, Binance API, agent registry, strategy files

# strategy_registry.py 
load_strategy(symbol, strategy_id) / Load specific strategy JSON from backend/storage/strategies/{symbol}_strategy_{strategy_id}.json / returns dict or None on error

save_strategy(symbol, strategy_id, data) / Save strategy JSON file and register ID in registry / files stored in backend/storage/strategies/

delete_strategy(symbol, strategy_id) / Delete strategy file and unregister from registry

list_strategies() / List all strategies grouped by symbol from files in backend/storage/strategies/

get_strategy_by_symbol(symbol) / Load all strategy JSONs for a symbol

register_strategy(symbol, strategy_id) / Add strategy ID to registry.json in strategies folder

_unregister_strategy(symbol, strategy_id) / Remove strategy ID from registry.json

get_registered_strategies(symbol=None) / Read registry JSON fully or filtered by symbol