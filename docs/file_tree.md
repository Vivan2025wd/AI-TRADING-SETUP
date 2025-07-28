.:
backend
data
docs
eslint.config.js
index.html
manage.py
package-lock.json
package.json
postcss.config.js
requirements.txt
run_server.py
scripts
src
tailwind.config.js
vite.config.js

./backend:
__init__.py
__pycache__
agents
backtester
binance
config
db
fetch_and_save_ohlcv.py
main.py
ml_engine
mother_ai
position_sync
requirements.txt
routes
startup_trainer.py
storage
strategy_engine
strategy_registry.py
utils

./backend/__pycache__:
__init__.cpython-313.pyc
fetch_and_save_ohlcv.cpython-313.pyc
main.cpython-313.pyc
startup_trainer.cpython-313.pyc
strategy_registry.cpython-313.pyc

./backend/agents:
__init__.py
__pycache__
adausdt_agent.py
agent_training.py
avaxusdt_agent.py
bchusdt_agent.py
btcusdt_agent.py
dogeusdt_agent.py
dotusdt_agent.py
ethusdt_agent.py
generic_agent.py
ltcusdt_agent.py
models
solusdt_agent.py
xrpusdt_agent.py

./backend/agents/__pycache__:
__init__.cpython-313.pyc
ada_agent.cpython-313.pyc
adausdt_agent.cpython-313.pyc
agent_training.cpython-313.pyc
avax_agent.cpython-313.pyc
avaxusdt_agent.cpython-313.pyc
batch_train_agents.cpython-313.pyc
bch_agent.cpython-313.pyc
bchusdt_agent.cpython-313.pyc
btc_agent.cpython-313.pyc
btcusdt_agent.cpython-313.pyc
dodge_agent.cpython-313.pyc
dodgeusdt_agent.cpython-313.pyc
dogeusdt_agent.cpython-313.pyc
dot_agent.cpython-313.pyc
dotusdt_agent.cpython-313.pyc
eth_agent.cpython-313.pyc
ethusdt_agent.cpython-313.pyc
generate_labels.cpython-313.pyc
generic_agent.cpython-313.pyc
ltc_agent.cpython-313.pyc
ltcusdt_agent.cpython-313.pyc
sol_agent.cpython-313.pyc
solusdt_agent.cpython-313.pyc
xrp_agent.cpython-313.pyc
xrpusdt_agent.cpython-313.pyc

./backend/agents/models:
__pycache__
adausdt_model.pkl
avaxusdt_model.pkl
bchusdt_model.pkl
btcusdt_model.pkl
dogeusdt_model.pkl
dotusdt_model.pkl
ethusdt_model.pkl
ltcusdt_model.pkl
model_utils.py
solusdt_model.pkl
xrpusdt_model.pkl

./backend/agents/models/__pycache__:
model_utils.cpython-313.pyc

./backend/backtester:
__init__.py
__pycache__
runner.py

./backend/backtester/__pycache__:
__init__.cpython-313.pyc
runner.cpython-313.pyc

./backend/binance:
__init__.py
__pycache__
binance_trader.py
fetch_live_ohlcv.py

./backend/binance/__pycache__:
__init__.cpython-313.pyc
binance_trader.cpython-313.pyc
fetch_live_ohlcv.cpython-313.pyc

./backend/config:
training_config.py

./backend/db:
__init__.py
models.py
storage.py

./backend/ml_engine:
__init__.py
__pycache__
batch_train_agents.py
feature_extractor.py
generate_labels.py
indicators.py

./backend/ml_engine/__pycache__:
__init__.cpython-313.pyc
feature_extractor.cpython-313.pyc
generate_labels.cpython-313.pyc
indicators.cpython-313.pyc

./backend/mother_ai:
__init__.py
__pycache__
meta_evaluator.py
mother_ai.py
performance_tracker.py
profit_calculator.py
trade_executer.py

./backend/mother_ai/__pycache__:
__init__.cpython-313.pyc
meta_evaluator.cpython-313.pyc
mother_ai.cpython-313.pyc
performance_tracker.cpython-313.pyc
profit_calculator.cpython-313.pyc
trade_executer.cpython-313.pyc

./backend/position_sync:
position_synchronizer.py

./backend/routes:
__init__.py
__pycache__
agent.py
agent_registry.py
backtest.py
binance.py
mother_ai.py
strategy.py
strategy_file_loader.py

./backend/routes/__pycache__:
__init__.cpython-313.pyc
agent.cpython-313.pyc
agent_registry.cpython-313.pyc
backtest.cpython-313.pyc
binance.cpython-313.pyc
mother_ai.cpython-313.pyc
strategy.cpython-313.pyc
strategy_file_loader.cpython-313.pyc

./backend/storage:
__pycache__
auto_cleanup.py
performance_logs
risk_config.json
strategies
trade_history
trade_profits
training_summary.json

./backend/storage/__pycache__:
auto_cleanup.cpython-313.pyc

./backend/storage/performance_logs:
ADAUSDT_trades.json
AVAXUSDT_trades.json
BCHUSDT_trades.json
BTCUSDT_trades.json
DOGEUSDT_trades.json
DOTUSDT_trades.json
LTCUSDT_trades.json
SOLUSDT_trades.json
XRPUSDT_trades.json

./backend/storage/strategies:
ADAUSDT_strategy_default.json
AVAXUSDT_strategy_default.json
BCHUSDT_strategy_default.json
BTCUSDT_strategy_test btc.json
DOGEUSDT_strategy_default.json
DOTUSDT_strategy_default.json
ETHUSDT_strategy_default.json
LTCUSDT_strategy_default.json
SOLUSDT_strategy_default.json
XRPUSDT_strategy_default.json

./backend/storage/trade_history:
ADAUSDT_predictions.json
AVAXUSDT_predictions.json
BCHUSDT_predictions.json
BTCUSDT_predictions.json
DOGEUSDT_predictions.json
DOTUSDT_predictions.json
ETHUSDT_predictions.json
LTCUSDT_predictions.json
SOLUSDT_predictions.json
XRPUSDT_predictions.json

./backend/storage/trade_profits:
ADAUSDT_summary.json
AVAXUSDT_summary.json
BCHUSDT_summary.json
BTCUSDT_summary.json
DOGEUSDT_summary.json
DOTUSDT_summary.json
LTCUSDT_summary.json
SOLUSDT_summary.json
XRPUSDT_summary.json

./backend/strategy_engine:
__init__.py
__pycache__
json_strategy_parser.py
strategy_health.py
strategy_parser.py

./backend/strategy_engine/__pycache__:
__init__.cpython-313.pyc
json_strategy_parser.cpython-313.pyc
strategy_health.cpython-313.pyc
strategy_parser.cpython-313.pyc

./backend/utils:
__init__.py
__pycache__
binance_api.py
logger.py

./backend/utils/__pycache__:
__init__.cpython-313.pyc
binance_api.cpython-313.pyc
logger.cpython-313.pyc

./data:
labels
ohlcv

./data/labels:
ADAUSDT_labels.csv
ADAUSDT_outcome_features.csv
ADAUSDT_outcome_labels.csv
AVAXUSDT_labels.csv
AVAXUSDT_outcome_features.csv
AVAXUSDT_outcome_labels.csv
BCHUSDT_labels.csv
BCHUSDT_outcome_features.csv
BCHUSDT_outcome_labels.csv
BTCUSDT_outcome_features.csv
BTCUSDT_outcome_labels.csv
DOGEUSDT_labels.csv
DOGEUSDT_outcome_features.csv
DOGEUSDT_outcome_labels.csv
DOTUSDT_labels.csv
DOTUSDT_outcome_features.csv
DOTUSDT_outcome_labels.csv
ETHUSDT_labels.csv
ETHUSDT_outcome_features.csv
ETHUSDT_outcome_labels.csv
LTCUSDT_labels.csv
LTCUSDT_outcome_features.csv
LTCUSDT_outcome_labels.csv
SOLUSDT_labels.csv
SOLUSDT_outcome_features.csv
SOLUSDT_outcome_labels.csv
XRPUSDT_labels.csv
XRPUSDT_outcome_features.csv
XRPUSDT_outcome_labels.csv
labeling_summary.csv
training_labels_summary.csv

./data/ohlcv:
ADAUSDT_1h.csv
AVAXUSDT_1h.csv
BCHUSDT_1h.csv
BTCUSDT_1h.csv
DOGEUSDT_1h.csv
DOTUSDT_1h.csv
ETHUSDT_1h.csv
LTCUSDT_1h.csv
SOLUSDT_1h.csv
XRPUSDT_1h.csv

./docs:
Folder Setup.md
backend
backend_tree.md
file_tree.md
readme.md

./docs/backend:
backend summary.md
backend.md
backendworkflow.md
data flow.md
diagram.md
progress sedtup.md
txt backend miles.md

./scripts:
build_all.sh
init_db.sh
start_dev.sh

./src:
App.jsx
components
main.jsx
styles

./src/components:
AgentPredictionCard.jsx
BacktestResults.jsx
BinanceAPISetup.jsx
ChartDisplay.jsx
DashboardPanel.jsx
ErrorBoundary.jsx
MotherAIDecisionCard.jsx
StrategyBuilder.jsx
StrategyPerformance.jsx

./src/styles:
index.css
