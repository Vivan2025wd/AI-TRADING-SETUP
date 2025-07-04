import json
import pandas as pd
import os
from backend.strategy_engine.strategy_parser import StrategyParser
from datetime import datetime
from typing import Optional

class BacktestRunner:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        """
        Loads OHLCV data for the given symbol from a CSV file.
        Assumes files are stored as 'data/{symbol}.csv'.
        """
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"OHLCV data file not found: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df

    def run(
        self,
        symbol: str,
        strategy_json: str,
        initial_balance: float = 1000.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        df = self.load_ohlcv(symbol)

        # Filter by date range if provided
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        if df.empty:
            raise ValueError("No data available in the specified date range.")

        strategy_dict = json.loads(strategy_json)
        parser = StrategyParser(strategy_dict)

        balance = initial_balance
        trades = []
        position = None
        entry_price = 0.0

        for i in range(50, len(df)):  # skip first 50 rows for indicators
            row = df.iloc[i]
            signals = parser.evaluate_conditions(df.iloc[:i+1])
            decision = signals[-1] if signals and len(signals) > 0 else None  # last signal for current window

            timestamp = row.name
            try:
                timestamp_dt = pd.to_datetime(str(timestamp))
                timestamp_str = timestamp_dt.isoformat() if timestamp_dt is not None else ""
            except Exception:
                timestamp_str = ""

            if decision == "buy" and position is None:
                entry_price = row["close"]
                position = "long"
                trades.append({
                    "type": "BUY",
                    "timestamp": timestamp_str,
                    "price": entry_price
                })

            elif decision == "sell" and position == "long":
                exit_price = row["close"]
                profit = (exit_price - entry_price) / entry_price * 100
                balance += balance * (profit / 100)
                trades.append({
                    "type": "SELL",
                    "timestamp": timestamp_str,
                    "price": exit_price,
                    "profit_percent": round(profit, 2),
                    "balance": round(balance, 2)
                })
                position = None

        return {
            "final_balance": round(balance, 2),
            "trades": trades,
            "total_trades": len(trades),
            "symbol": symbol,
            "start_date": df.index[0].isoformat() if not df.empty else "",
            "end_date": df.index[-1].isoformat() if not df.empty else ""
        }

# Helper function for FastAPI
def run_backtest(
    symbol: str,
    strategy_json: str,
    initial_balance: float = 1000.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    runner = BacktestRunner()
    return runner.run(symbol, strategy_json, initial_balance, start_date, end_date)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        