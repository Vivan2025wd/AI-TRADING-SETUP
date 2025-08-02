import json
import pandas as pd
import os
from backend.strategy_engine.strategy_parser import StrategyParser
from datetime import datetime
from typing import Optional

class BacktestRunner:
    def __init__(self, data_dir="data/ohlcv"):
        self.data_dir = data_dir

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, f"{symbol}_1h.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"OHLCV data file not found: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df

    def run(
        self,
        symbol: str,
        strategy_json: str,
        initial_balance: float = 100.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        risk_per_trade: float = 0.10,     # Allocate 10% of balance per trade
        leverage: float = 1.0             # Leverage multiplier (1.0 = no leverage)
    ):
        df = self.load_ohlcv(symbol)

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
        allocated_capital = 0.0
        trade_size = 0.0
        capital_over_time = []

        for i in range(50, len(df)):  # skip first 50 rows for indicators
            row = df.iloc[i]
            signals = parser.evaluate_conditions_with_confidence(df.iloc[:i+1])
            decision = signals[-1] if signals and len(signals) > 0 else None  # last signal for current window

            timestamp = row.name
            timestamp_str = pd.to_datetime(str(timestamp)).isoformat()

            if decision == "buy" and position is None:
                entry_price = row["close"]
                position = "long"
                allocated_capital = balance * risk_per_trade  # Risked capital per trade
                trade_size = allocated_capital / entry_price  # Units of asset bought
                trades.append({
                    "type": "BUY",
                    "timestamp": timestamp_str,
                    "price": entry_price,
                    "allocated_capital": round(allocated_capital, 2),
                    "trade_size": round(trade_size, 6)
                })

            elif decision == "sell" and position == "long":
                exit_price = row["close"]
                raw_profit_pct = (exit_price - entry_price) / entry_price * 100
                leveraged_profit_pct = raw_profit_pct * leverage  # Apply leverage if any
                pnl_usd = allocated_capital * (leveraged_profit_pct / 100)

                balance += pnl_usd  # Only add profit/loss, not entire balance

                trades.append({
                    "type": "SELL",
                    "timestamp": timestamp_str,
                    "price": exit_price,
                    "raw_profit_percent": round(raw_profit_pct, 2),
                    "leveraged_profit_percent": round(leveraged_profit_pct, 2),
                    "profit_usd": round(pnl_usd, 2),
                    "balance": round(balance, 2)
                })
                position = None
                allocated_capital = 0.0
                trade_size = 0.0

            capital_over_time.append({
                "timestamp": timestamp_str,
                "capital": round(balance, 2)
            })

        return {
            "final_balance": round(balance, 2),
            "trades": trades,
            "total_trades": len(trades),
            "symbol": symbol,
            "start_date": df.index[0].isoformat() if not df.empty else "",
            "end_date": df.index[-1].isoformat() if not df.empty else "",
            "capital_over_time": capital_over_time
        }

# Helper function for FastAPI
def run_backtest(
    symbol: str,
    strategy_json: str,
    initial_balance: float = 100.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    risk_per_trade: float = 0.10,
    leverage: float = 1.0
):
    runner = BacktestRunner()
    return runner.run(symbol, strategy_json, initial_balance, start_date, end_date, risk_per_trade, leverage)
