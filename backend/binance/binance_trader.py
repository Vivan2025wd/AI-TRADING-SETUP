import ccxt
import os

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

USE_TESTNET = True

if BINANCE_API_KEY and BINANCE_API_SECRET:
    if USE_TESTNET:
        exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        exchange.set_sandbox_mode(True)
    else:
        exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
else:
    exchange = None
    print("⚠️ Warning: Missing BINANCE_API_KEY or BINANCE_API_SECRET, using mock orders.")

def place_market_order(symbol: str, side: str, amount: float):
    if exchange is None:
        # Mock behavior
        print(f"ℹ️ Mock order: {side.upper()} {amount} {symbol} (No real trade executed)")
        return {
            "status": "mock",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "message": "No real trade executed due to missing API keys."
        }
    else:
        # Real behavior
        try:
            side_clean = side.lower()
            if side_clean not in ("buy", "sell"):
                raise ValueError("side must be either 'buy' or 'sell'")
            order = exchange.create_market_order(symbol=symbol, side=side_clean, amount=amount)
            print(f"✅ Binance {side_clean.upper()} order executed: {order}")
            return order
        except Exception as e:
            print(f"❌ Binance order failed: {e}")
            return None
