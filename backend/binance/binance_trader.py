import ccxt
import os
from dotenv import load_dotenv

load_dotenv()  # Load API keys from .env

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in the environment variables.")

# ✅ Use testnet for paper trading (change to 'binance' for real trading)
USE_TESTNET = True

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


def place_market_order(symbol: str, side: str, amount: float):
    """
    Places a market buy/sell order on Binance.
    symbol: "BTC/USDT"
    side: "buy" or "sell"
    amount: float (amount of base currency, e.g., BTC)
    """
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
