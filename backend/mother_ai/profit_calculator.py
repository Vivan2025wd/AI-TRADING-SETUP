import os
import json
from datetime import datetime

TRADE_LOG_DIR = "backend/storage/performance_logs"
PROFIT_LOG_DIR = "backend/storage/trade_profits"
os.makedirs(PROFIT_LOG_DIR, exist_ok=True)

def compute_trade_profits(symbol: str):
    path = os.path.join(TRADE_LOG_DIR, f"{symbol}_trades.json")
    if not os.path.exists(path):
        print(f"⚠️ No trade log found for {symbol}")
        return None

    with open(path, "r") as f:
        trades = json.load(f)

    # Sort trades by timestamp
    trades = sorted(trades, key=lambda x: x.get("timestamp", ""))

    position = None
    trade_log = []
    total_profit = 0.0
    total_trades = 0
    wins = 0
    losses = 0

    for trade in trades:
        signal = trade.get("signal", "").lower()
        price = trade.get("price")
        timestamp = trade.get("timestamp")

        # Skip invalid entries
        if price is None or signal not in {"buy", "sell"}:
            continue

        # Buy: open position if none is open
        if signal == "buy" and position is None:
            position = {
                "entry_price": price,
                "entry_time": timestamp
            }

        # Sell: close position if one is open
        elif signal == "sell" and position is not None:
            pnl = price - position["entry_price"]
            total_profit += pnl
            total_trades += 1

            if pnl >= 0:
                wins += 1
            else:
                losses += 1

            trade_log.append({
                "entry_time": position["entry_time"],
                "entry_price": round(position["entry_price"], 4),
                "exit_time": timestamp,
                "exit_price": round(price, 4),
                "pnl": round(pnl, 4)
            })

            position = None  # Reset after closing trade

    win_rate = round((wins / total_trades) * 100, 2) if total_trades > 0 else 0.0

    summary = {
        "symbol": symbol,
        "total_profit": round(total_profit, 4),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "trades": trade_log
    }

    # Save output
    output_path = os.path.join(PROFIT_LOG_DIR, f"{symbol}_summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Profit summary saved: {output_path}")
    return summary
