# backend/ml_engine/feature_extractor.py

import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure 'close' is numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    # RSI
    delta = pd.to_numeric(df['close'].diff(), errors='coerce')
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # EMA
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    df.fillna(0, inplace=True)
    return df