import os
import json
from datetime import datetime

TRADE_LOG_DIR = "backend/storage/performance_logs"
PROFIT_LOG_DIR = "backend/storage/trade_profits"
os.makedirs(PROFIT_LOG_DIR, exist_ok=True)

def compute_trade_profits(symbol: str, trading_fee_percent: float = 0.1):
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
        
        # NEW: Get quantity from trade data if available
        qty = trade.get("qty", None)

        # Skip invalid entries
        if price is None or signal not in {"buy", "sell"}:
            continue

        # Buy: open position if none is open
        if signal == "buy" and position is None:
            # Calculate quantity if not provided
            if qty is None:
                # Fallback calculation using the same logic as MotherAI
                RISK_PER_TRADE = 1.0
                DEFAULT_BALANCE_USD = 10000
                SL_PERCENT = 0.5

                risk_amount = DEFAULT_BALANCE_USD * RISK_PER_TRADE / 100  # $100
                sl_price = price * (1 - SL_PERCENT / 100)
                qty = risk_amount / (price - sl_price)
                
                print(f"📊 Calculated qty for {symbol}: {qty:.6f} (risk: ${risk_amount})")

            # Calculate buy fee
            entry_value = price * qty
            buy_fee = entry_value * (trading_fee_percent / 100)
            
            position = {
                "entry_price": price,
                "entry_time": timestamp,
                "qty": qty,
                "capital_invested": entry_value  # Track actual capital invested
            }

        # Sell: close position if one is open
        elif signal == "sell" and position is not None:
            # Use the same quantity as the buy order
            sell_qty = qty if qty is not None else position.get("qty", 0)
            
            # Calculate ACTUAL profit in dollars
            entry_value = position["entry_price"] * sell_qty
            exit_value = price * sell_qty
            
            # Calculate fees and deduct from profit
            buy_fee = entry_value * (trading_fee_percent / 100)
            sell_fee = exit_value * (trading_fee_percent / 100)
            total_fees = buy_fee + sell_fee
            
            # Net profit after fees
            pnl_dollars = (exit_value - entry_value) - total_fees
            
            # Calculate percentage return on invested capital
            pnl_percentage = (pnl_dollars / entry_value) * 100 if entry_value > 0 else 0
            
            total_profit += pnl_dollars
            total_trades += 1

            if pnl_dollars >= 0:
                wins += 1
            else:
                losses += 1

            trade_log.append({
                "entry_time": position["entry_time"],
                "entry_price": round(position["entry_price"], 6),
                "exit_time": timestamp,
                "exit_price": round(price, 6),
                "qty": round(sell_qty, 6),
                "entry_value": round(entry_value, 4),
                "exit_value": round(exit_value, 4),
                "pnl_dollars": round(pnl_dollars, 4),
                "pnl_percentage": round(pnl_percentage, 2)
            })

            print(f"💰 {symbol} Trade: Entry=${position['entry_price']:.6f}, Exit=${price:.6f}, "
                  f"Qty={sell_qty:.6f}, P&L=${pnl_dollars:.4f} ({pnl_percentage:.2f}%)")

            position = None  # Reset after closing trade

    win_rate = round((wins / total_trades) * 100, 2) if total_trades > 0 else 0.0
    avg_profit_per_trade = round(total_profit / total_trades, 4) if total_trades > 0 else 0.0

    summary = {
        "symbol": symbol,
        "total_profit_dollars": round(total_profit, 4),
        "avg_profit_per_trade": avg_profit_per_trade,
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


# Alternative function with different fee structures
def compute_trade_profits_advanced_fees(symbol: str, maker_fee: float = 0.1, taker_fee: float = 0.1, 
                                       bnb_discount: bool = False):
    """
    Advanced version with separate maker/taker fees and BNB discount option.
    
    Args:
        symbol: Trading pair symbol
        maker_fee: Maker fee percentage (default 0.1%)
        taker_fee: Taker fee percentage (default 0.1%)
        bnb_discount: Whether to apply 25% BNB discount (default False)
    """
    # Apply BNB discount if enabled
    if bnb_discount:
        maker_fee *= 0.75  # 25% discount
        taker_fee *= 0.75
    
    # For simplicity, assume all trades are taker orders (market orders)
    # You could enhance this by adding order type detection
    return compute_trade_profits(symbol, trading_fee_percent=taker_fee)