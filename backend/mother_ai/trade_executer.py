import os
import json
import glob
import importlib.util
import asyncio
import time
from fastapi import APIRouter, HTTPException, Query, FastAPI

from backend.binance.fetch_live_ohlcv import fetch_ohlcv
from backend.binance.binance_trader import place_market_order
from backend.mother_ai.performance_tracker import PerformanceTracker
from backend.strategy_engine.strategy_health import StrategyHealth
from backend.strategy_engine.strategy_parser import StrategyParser
from backend.mother_ai.profit_calculator import compute_trade_profits
from backend.mother_ai.meta_evaluator import MetaEvaluator

# <-- ADD THIS IMPORT
from backend.mother_ai.mother_ai import MotherAI

TRADE_HISTORY_DIR = "backend/storage/trade_history"
PERFORMANCE_LOG_DIR = "backend/storage/performance_logs"
os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)

TRADE_COOLDOWN_SECONDS = 600
AUTO_DECISION_INTERVAL = 120

def execute_mother_ai_decision(decision_result):
    """
    Executes trades based on a decision result dict from MotherAI.
    Returns a list of executed trade summaries.
    """
    mother_ai = MotherAI()  # Create a new instance or consider passing existing one for reuse
    decision = decision_result.get("decision", {})
    executed_trades = []

    if not decision:
        print("⚠️ No decision to execute.")
        return executed_trades

    symbol = decision.get("symbol")
    signal = decision.get("signal")
    confidence = decision.get("confidence", 0.0)
    price = decision.get("last_price")

    if not (symbol and signal and price):
        print("⚠️ Incomplete decision data, cannot execute trade.")
        return executed_trades

    print(f"🚀 Executing trade from execute_mother_ai_decision: {signal} {symbol} @ {price}")
    mother_ai.execute_trade(symbol, signal, price, confidence)

    executed_trades.append({
        "symbol": symbol,
        "signal": signal,
        "price": price,
        "confidence": confidence
    })

    # If selling, recompute profits
    if signal.lower() == "sell":
        compute_trade_profits(symbol)
        print(f"📈 Profits recomputed after sell for {symbol}")

    return executed_trades
