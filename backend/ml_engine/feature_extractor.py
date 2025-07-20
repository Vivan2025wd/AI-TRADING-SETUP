import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Base directory where strategy JSON files are stored
STRATEGY_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / "strategies"


def load_strategy(symbol: str, strategy_id: str) -> Dict[str, Any]:
    """
    Loads strategy JSON file for given symbol and strategy_id.
    """
    filename = f"{symbol.upper()}_strategy_{strategy_id}.json"
    file_path = STRATEGY_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        strategy = json.load(f)
    return strategy


def add_technical_indicators(df: pd.DataFrame, strategy: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Add technical indicators with optional parameters from strategy JSON.
    """
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # Default parameters or from strategy
    rsi_period = 14
    if strategy and "indicators" in strategy and "rsi" in strategy["indicators"]:
        rsi_period = strategy["indicators"]["rsi"].get("period", 14)

    delta = pd.to_numeric(df['close'].diff(), errors='coerce')
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # EMA periods
    ema_12_span = 12
    ema_26_span = 26
    if strategy and "indicators" in strategy and "ema" in strategy["indicators"]:
        ema_12_span = strategy["indicators"]["ema"].get("ema_12_span", 12)
        ema_26_span = strategy["indicators"]["ema"].get("ema_26_span", 26)

    df['ema_12'] = df['close'].ewm(span=ema_12_span, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=ema_26_span, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    df.fillna(0, inplace=True)
    return df


def extract_features(df: pd.DataFrame, strategy: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Extract features from OHLCV data using strategy params if provided.
    """
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['return'] = df['close'].pct_change()

    # Simple moving average window from strategy or default 10
    sma_window = 10
    if strategy and "indicators" in strategy and "sma" in strategy["indicators"]:
        sma_window = strategy["indicators"]["sma"].get("period", 10)

    df['sma_10'] = df['close'].rolling(sma_window).mean()

    # EMA window from strategy or default 20
    ema_span = 20
    if strategy and "indicators" in strategy and "ema" in strategy["indicators"]:
        ema_span = strategy["indicators"]["ema"].get("period", 20)

    df['ema_20'] = df['close'].ewm(span=ema_span, adjust=False).mean()

    # RSI calculation with period from strategy or default 14
    rsi_period = 14
    if strategy and "indicators" in strategy and "rsi" in strategy["indicators"]:
        rsi_period = strategy["indicators"]["rsi"].get("period", 14)

    delta = pd.to_numeric(df['close'].diff(), errors='coerce')
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    df = df.dropna()

    # Select only relevant columns
    return df[['return', 'sma_10', 'ema_20', 'rsi']]
