import os
import json

TRADE_LOG_DIR = "backend/storage/performance_logs"
PROFIT_LOG_DIR = "backend/storage/trade_profits"
os.makedirs(PROFIT_LOG_DIR, exist_ok=True)

def compute_trade_profits(symbol: str):
    path = os.path.join(TRADE_LOG_DIR, f"{symbol}_trades.json")
    if not os.path.exists(path):
        print(f"⚠️ No trade log for {symbol}")
        return None

    with open(path, "r") as f:
        trades = json.load(f)

    trades = sorted(trades, key=lambda x: x["timestamp"])
    total_profit = 0.0
    total_trades = 0
    wins = 0
    losses = 0
    position = None

    trade_log = []

    for trade in trades:
        if trade["signal"] == "buy":
            if position is None:
                position = {
                    "price": trade["price"],
                    "timestamp": trade["timestamp"]
                }
        elif trade["signal"] == "sell" and position:
            entry_price = position["price"]
            exit_price = trade["price"]
            pnl = exit_price - entry_price
            total_profit += pnl
            total_trades += 1
            if pnl > 0:
                wins += 1
            else:
                losses += 1

            trade_log.append({
                "entry_time": position["timestamp"],
                "entry_price": entry_price,
                "exit_time": trade["timestamp"],
                "exit_price": exit_price,
                "pnl": round(pnl, 4)
            })
            position = None

    summary = {
        "symbol": symbol,
        "total_profit": round(total_profit, 4),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / total_trades) * 100, 2) if total_trades > 0 else 0.0,
        "trades": trade_log
    }

    # Save to file
    output_path = os.path.join(PROFIT_LOG_DIR, f"{symbol}_summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"💰 Profit summary for {symbol} saved.")
    return summary
