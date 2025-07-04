# 📁 `backend/` — AI Trading System Backend

This backend powers your **AI-driven trading dashboard**. It's built with **FastAPI**, has a **multi-agent architecture**, and includes **strategy parsing**, **backtesting**, **machine learning**, and **trade execution**.

---

## 🧠 Root Files

| File                   | Purpose                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `main.py`              | FastAPI entry point. Loads all routes: agent, strategy, mother AI, backtest, and Binance. |
| `.env`                 | Holds environment variables like API keys, DB config, etc.                                |
| `requirements.txt`     | Lists Python dependencies.                                                                |
| `strategy_registry.py` | Central strategy loader/registry used across modules.                                     |
| `__init__.py`          | Marks this folder as a Python package.                                                    |

---

## 🔌 `routes/` – API Endpoints

| File           | Purpose                                               |
| -------------- | ----------------------------------------------------- |
| `agent.py`     | Endpoints to run or fetch AI agents (e.g., BTC, ETH). |
| `backtest.py`  | API to run backtests with selected strategies.        |
| `strategy.py`  | Save, load, update strategies from frontend.          |
| `mother_ai.py` | Trigger Mother AI decisions or evaluations.           |
| `binance.py`   | Fetch market data from Binance (candles, prices).     |

---

## 🤖 `agents/` – Symbol-Specific AI Modules

| File                           | Purpose                                        |
| ------------------------------ | ---------------------------------------------- |
| `btc_agent.py`, `eth_agent.py` | Symbol-specific agents for prediction.         |
| `generic_agent.py`             | Base logic shared across all agents.           |
| `models/`                      | ML model files (likely saved `.pkl` or `.pt`). |

---

## 📊 `backtester/` – Strategy Simulation

| File          | Purpose                                               |
| ------------- | ----------------------------------------------------- |
| `runner.py`   | Runs a strategy over OHLCV data and simulates trades. |
| `data/`       | CSV or OHLCV data for different trading pairs.        |
| `strategies/` | Built-in strategies (e.g., SMA crossover).            |
| `utils/`      | Helper functions (e.g., indicators, math ops).        |

---

## 🧠 `ml_engine/` – Machine Learning Core

| File                   | Purpose                                         |
| ---------------------- | ----------------------------------------------- |
| `trainer.py`           | Trains AI models on past market data.           |
| `indicators.py`        | Compute RSI, MACD, SMA, etc. for use in models. |
| `feature_extractor.py` | Converts raw data into ML-ready features.       |

---

## 🧬 `mother_ai/` – Central Decision Brain

| File                     | Purpose                                             |
| ------------------------ | --------------------------------------------------- |
| `mother_ai.py`           | Combines predictions from agents, selects the best. |
| `performance_tracker.py` | Tracks agent or system performance over time.       |
| `trade_executer.py`      | Simulates or places trades (mock or real).          |
| `logs/`                  | Stores decision logs or trade history.              |

---

## 🗃️ `db/` – Local Database Layer

| File         | Purpose                                        |
| ------------ | ---------------------------------------------- |
| `models.py`  | SQLAlchemy models for trades, strategies, etc. |
| `storage.py` | Save/load data from local storage or DB.       |

---

## 🧹 `strategy_engine/` – Custom Strategy System

| File                      | Purpose                                                    |
| ------------------------- | ---------------------------------------------------------- |
| `strategy_parser.py`      | Parses and executes user-defined JSON strategies.          |
| `json_strategy_parser.py` | Converts UI strategy blocks into backend-executable logic. |
| `strategy_health.py`      | Validates and scores strategies before use.                |

---

## 💾 `storage/` – Local Strategy & Trade Data

| Folder                | Purpose                                     |
| --------------------- | ------------------------------------------- |
| `strategies/BTCUSDT/` | Saved strategy files for BTC.               |
| `trade_logs/`         | Simulated or real trade logs for analytics. |

---

## 🛠️ `utils/` – Shared Utilities

| File             | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| `binance_api.py` | Fetch OHLCV, symbols, and prices from Binance. |
| `logger.py`      | Logging utilities for backend monitoring.      |

---

## ✅ Integration Ready

* All APIs are registered in `main.py` with clean `/api/*` routes.
* Modular and scalable for new agents, models, or exchanges.
* Fully supports self-hosted, local-first SaaS design.
