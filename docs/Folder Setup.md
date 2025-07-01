# Project Root
.
├── README.md
├── .env                   # Global environment variables (ignored in .gitignore)
├── docker-compose.yml     # To run backend, frontend, and DB in one command
├── requirements.txt       # Python dependencies
├── package.json           # Frontend dependencies
├── .gitignore
├── scripts/                # Dev & deployment utilities
│   ├── init_db.sh          # MongoDB init or setup
│   ├── build_all.sh        # Build frontend & backend
│   └── start_dev.sh        # One-click dev startup


# frontend
frontend/
├── public/
│   └── index.html
├── src/
│   ├── App.jsx             # Main app with routing
│   ├── main.jsx            # React DOM render
│   ├── components/         # All reusable components here
│   ├── pages/              # Route-level page components
│   ├── services/           # API logic
│   └── styles/
│       └── index.css
├── package.json
└── vite.config.js


src/
├── App.jsx(done)
├── main.jsx(done)
├── components/
│   ├── DashboardPanel.jsx(done)
│   ├── StrategyBuilder.jsx (done)        # Converts logic into JSON
│   ├── ChartDisplay.jsx(done)            # ChartJS or TradingView widget
│   ├── BacktestResults.jsx(done)         # Displays historical trades
│   ├── AgentPredictionCard.jsx(done)    # Displays agent prediction
│   └── MotherAIDecisionCard.jsx (done)   # Displays Mother AI trade pick
├── pages/
│   └── Dashboard.jsx(done)               # Composes the entire view
├── styles/
│   └── index.css




# Backend (FastAPI + Agents + ML)
backend/
├── main.py     (done)             # FastAPI app
├── .env                     # Backend env vars (Mongo URI, API keys)
├── Dockerfile
├── agents/
│   ├── btc_agent.py(done)
│   ├── eth_agent.py(done)
│   ├── generic_agent.py(done)
│   └── models/              # .pkl ML models
├── strategy_engine/
│   ├── strategy_parser.py (done)
│   ├── json_strategy_parser.py(done)
│   └── strategy_health.py(done)
├── mother_ai/
│   ├── mother_ai.py(done)
│   └── performance_tracker.py(done)
├── backtester/
│   ├── strategies/
│   ├── runner.py(done)
│   ├── data/
│   └── utils/
├── routes/
│   ├── agent.py(done)
│   ├── strategy.py(done)
│   ├── mother_ai.py(done)
│   └── backtest.py(done)
├── db/
│   ├── storage.py(done)
│   └── models.py(done)
├── ml_engine/
│   ├── indicators.py(done)
│   ├── trainer.py(done)
│   └── feature_extractor.py(done)
└── utils/
|    ├── binance_api.py
|    └── logger.py
|-strategy_registry.py(done)


# storage
storage/
├── trade_logs/
│   ├── BTCUSDT_trades.json
│   └── ETHUSDT_trades.json
├── strategies/
│   ├── BTCUSDT/
│   │   └── rsi_macd_combo.json
│   └── ETHUSDT/
│       └── ema_cross.json


# Tests
__tests__/
├── test_agents.py
├── test_backtest.py
├── test_strategy_parser.py
└── test_api.py