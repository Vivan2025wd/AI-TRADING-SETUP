import pandas as pd

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series, period=20):
    return series.rolling(window=period).mean()

def calculate_ema_crossover(short_series, long_series):
    """
    Returns two boolean Series:
    - crossover: short EMA crosses above long EMA
    - crossunder: short EMA crosses below long EMA
    """
    crossover = (short_series > long_series) & (short_series.shift(1) <= long_series.shift(1))
    crossunder = (short_series < long_series) & (short_series.shift(1) >= long_series.shift(1))
    return crossover, crossunder

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = calculate_ema(series, period=fast_period)
    ema_slow = calculate_ema(series, period=slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
