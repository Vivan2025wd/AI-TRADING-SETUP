# Project Structure Tree

```
crypto-trading-platform/
├── 📁 backend/                          # Python backend services
│   ├── 🤖 agents/                       # ML trading agents for each cryptocurrency
│   │   ├── models/                      # Trained ML models (.pkl files)
│   │   │   ├── adausdt_model.pkl
│   │   │   ├── btcusdt_model.pkl
│   │   │   ├── ethusdt_model.pkl
│   │   │   └── ... (other crypto models)
│   │   ├── adausdt_agent.py
│   │   ├── btcusdt_agent.py
│   │   ├── generic_agent.py             # Base agent class
│   │   └── agent_training.py            # Training orchestration
│   │
│   ├── 🧠 mother_ai/                    # Meta-AI system for strategy coordination
│   │   ├── mother_ai.py                 # Main orchestration logic
│   │   ├── meta_evaluator.py            # Strategy evaluation
│   │   ├── performance_tracker.py       # Performance monitoring
│   │   ├── profit_calculator.py         # P&L calculations
│   │   └── trade_executer.py            # Trade execution logic
│   │
│   ├── 📊 ml_engine/                    # Machine learning infrastructure
│   │   ├── feature_extractor.py         # Technical indicator extraction
│   │   ├── generate_labels.py           # Training label generation
│   │   ├── indicators.py                # Technical indicators library
│   │   └── batch_train_agents.py        # Batch training system
│   │
│   ├── 📈 strategy_engine/              # Strategy parsing and management
│   │   ├── json_strategy_parser.py      # JSON strategy configuration
│   │   ├── strategy_parser.py           # Strategy logic parser
│   │   └── strategy_health.py           # Strategy health monitoring
│   │
│   ├── 🔄 backtester/                   # Historical strategy testing
│   │   └── runner.py                    # Backtest execution engine
│   │
│   ├── 💱 binance/                      # Exchange integration
│   │   ├── binance_trader.py            # Trading interface
│   │   └── fetch_live_ohlcv.py          # Live data fetching
│   │
│   ├── 🌐 routes/                       # API endpoints
│   │   ├── agent.py                     # Agent management endpoints
│   │   ├── backtest.py                  # Backtesting endpoints
│   │   ├── binance.py                   # Exchange endpoints
│   │   ├── mother_ai.py                 # Mother AI endpoints
│   │   └── strategy.py                  # Strategy management endpoints
│   │
│   ├── 💾 storage/                      # Data persistence layer
│   │   ├── performance_logs/            # Trading performance logs
│   │   ├── strategies/                  # Strategy configurations
│   │   ├── trade_history/               # Historical predictions
│   │   └── trade_profits/               # Profit summaries
│   │
│   ├── 🔧 utils/                        # Utility functions
│   │   ├── binance_api.py               # Binance API wrapper
│   │   └── logger.py                    # Logging utilities
│   │
│   └── 📝 config/                       # Configuration files
│       └── training_config.py           # ML training parameters
│
├── 📊 data/                             # Training and market data
│   ├── ohlcv/                          # OHLCV market data (CSV files)
│   │   ├── BTCUSDT_1h.csv
│   │   ├── ETHUSDT_1h.csv
│   │   └── ... (other crypto pairs)
│   │
│   └── labels/                         # ML training labels
│       ├── outcome_features.csv         # Feature datasets
│       ├── outcome_labels.csv           # Training labels
│       └── labeling_summary.csv         # Label generation summary
│
├── 🎨 src/                              # React frontend
│   ├── components/                      # React components
│   │   ├── AgentPredictionCard.jsx      # Agent prediction display
│   │   ├── BacktestResults.jsx          # Backtesting results
│   │   ├── BinanceAPISetup.jsx          # API configuration
│   │   ├── ChartDisplay.jsx             # Price chart visualization
│   │   ├── DashboardPanel.jsx           # Main dashboard
│   │   ├── MotherAIDecisionCard.jsx     # Mother AI insights
│   │   ├── StrategyBuilder.jsx          # Strategy creation interface
│   │   └── StrategyPerformance.jsx      # Performance analytics
│   │
│   ├── styles/                         # CSS styling
│   │   └── index.css
│   │
│   ├── App.jsx                         # Main React application
│   └── main.jsx                        # React entry point
│
├── 📖 docs/                            # Documentation
│   ├── backend/                        # Backend documentation
│   │   ├── backend summary.md
│   │   ├── backendworkflow.md
│   │   └── data flow.md
│   │
│   └── readme.md                       # Project documentation
│
├── 🔧 scripts/                         # Build and deployment scripts
│   ├── build_all.sh                   # Full build script
│   ├── init_db.sh                     # Database initialization
│   └── start_dev.sh                   # Development server startup
│
├── ⚙️ Configuration Files
│   ├── package.json                    # Node.js dependencies
│   ├── requirements.txt                # Python dependencies
│   ├── vite.config.js                 # Vite build configuration
│   ├── tailwind.config.js             # Tailwind CSS configuration
│   ├── eslint.config.js               # ESLint configuration
│   └── postcss.config.js              # PostCSS configuration
│
└── 🚀 Entry Points
    ├── index.html                      # Frontend entry point
    ├── main.py                         # Backend main server
    ├── manage.py                       # Django-style management
    └── run_server.py                   # Server startup script
```

## Key Components Overview

### 🤖 **Agents System**
- Individual ML agents for each cryptocurrency pair (BTC, ETH, ADA, etc.)
- Trained models stored as pickle files
- Generic agent base class for consistency

### 🧠 **Mother AI**
- Meta-level decision making system
- Coordinates multiple agents
- Manages risk and portfolio allocation

### 📊 **ML Engine**
- Feature extraction from market data
- Technical indicator calculations
- Automated label generation for training

### 💱 **Exchange Integration**
- Binance API integration
- Live data fetching
- Trade execution capabilities

### 🎨 **Frontend Dashboard**
- React-based user interface
- Real-time charts and analytics
- Strategy builder and backtesting tools