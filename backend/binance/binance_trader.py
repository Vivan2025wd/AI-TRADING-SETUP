import ccxt
import os
from backend.utils.binance_api import get_trading_mode, get_binance_client

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

def get_ccxt_exchange():
    """Get CCXT exchange instance based on current trading mode and API connection"""
    try:
        # Try to use the connected Binance client from binance_api.py
        client = get_binance_client()
        trading_mode = get_trading_mode()
        
        # Extract API keys from the connected client (if available)
        api_key = getattr(client, 'API_KEY', BINANCE_API_KEY)
        api_secret = getattr(client, 'API_SECRET', BINANCE_API_SECRET)
        
        if not api_key or not api_secret:
            print("⚠️ Warning: Missing API keys, using mock orders.")
            return None
            
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            # Add more verbose error reporting
            'verbose': False,  # Set to True for more detailed logs
        })
        
        # Set sandbox mode based on trading mode from binance_api.py
        if trading_mode == "mock":
            exchange.set_sandbox_mode(True)
            print("🧪 Using Binance Testnet (Mock Mode)")
        else:
            exchange.set_sandbox_mode(False)
            print("⚠️ Using Binance LIVE Trading")
            
        # Test the connection with a simple API call
        try:
            account_info = exchange.fetch_balance()
            print(f"✅ API Connection verified. Account type: {account_info.get('info', {}).get('accountType', 'Unknown')}")
        except Exception as test_error:
            print(f"⚠️ API Connection test failed: {test_error}")
            # Don't return None here - let the actual order attempt show the real error
            
        return exchange
        
    except Exception as e:
        print(f"⚠️ Could not connect to Binance API: {e}")
        return None

def place_market_order(symbol: str, side: str, amount: float):
    """Place market order using current trading mode settings"""
    exchange = get_ccxt_exchange()
    
    if exchange is None:
        # Mock behavior
        print(f"ℹ️ Mock order: {side.upper()} {amount} {symbol} (No real trade executed)")
        return {
            "status": "mock",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "message": "No real trade executed due to missing API connection."
        }
    else:
        # Real behavior (testnet or live based on trading mode)
        try:
            side_clean = side.lower()
            if side_clean not in ("buy", "sell"):
                raise ValueError("side must be either 'buy' or 'sell'")
                
            # Format symbol for CCXT (remove / if present, then add back)
            ccxt_symbol = symbol.replace("/", "") if "/" in symbol else symbol
            if not ccxt_symbol.endswith("USDT"):
                ccxt_symbol += "USDT"
            ccxt_symbol = ccxt_symbol[:len(ccxt_symbol)-4] + "/" + ccxt_symbol[-4:]  # BTC/USDT format
            
            print(f"🔍 Attempting to place order: {side_clean.upper()} {amount} {ccxt_symbol}")
            
            # Check account balance before placing order
            try:
                balance = exchange.fetch_balance()
                if side_clean == "buy":
                    usdt_balance = float(balance.get('USDT', {}).get('free', 0) or 0)
                    ticker_price = float(exchange.fetch_ticker(ccxt_symbol)['last'] or 0)
                    required_amount = float(amount) * ticker_price
                    print(f"💰 USDT Balance: {usdt_balance}, Required: {required_amount}")
                    if usdt_balance < required_amount:
                        raise Exception(f"Insufficient USDT balance. Have: {usdt_balance}, Need: {required_amount}")
                else:  # sell
                    base_asset = ccxt_symbol.split('/')[0]
                    asset_balance = float(balance.get(base_asset, {}).get('free', 0) or 0)
                    print(f"💰 {base_asset} Balance: {asset_balance}, Required: {amount}")
                    if asset_balance < float(amount):
                        raise Exception(f"Insufficient {base_asset} balance. Have: {asset_balance}, Need: {amount}")
            except Exception as balance_error:
                print(f"⚠️ Balance check failed: {balance_error}")
                # Continue with order attempt
            
            order = exchange.create_market_order(symbol=ccxt_symbol, side=side_clean, amount=amount)
            
            trading_mode = get_trading_mode()
            mode_text = "TESTNET" if trading_mode == "mock" else "LIVE"
            print(f"✅ Binance {mode_text} {side_clean.upper()} order executed: {order}")
            
            return order
            
        except ccxt.BaseError as ccxt_error:
            error_message = str(ccxt_error)
            print(f"❌ CCXT Error: {error_message}")
            
            # Provide specific guidance based on error type
            if "Invalid API-key" in error_message:
                print("🔧 SOLUTION: Check your API key permissions:")
                print("   1. Go to Binance → API Management")
                print("   2. Ensure 'Enable Spot & Margin Trading' is checked")
                print("   3. Check IP restrictions (disable or whitelist your IP)")
                print("   4. Verify you're using mainnet keys for live trading")
            elif "Insufficient" in error_message:
                print("🔧 SOLUTION: Add funds to your Binance account")
            elif "MIN_NOTIONAL" in error_message:
                print("🔧 SOLUTION: Increase order size (minimum $10-20 typically)")
            
            return None
            
        except Exception as e:
            print(f"❌ Binance order failed: {e}")
            return None

def test_api_permissions():
    """Test API permissions and provide detailed diagnostics"""
    exchange = get_ccxt_exchange()
    if not exchange:
        return {"status": "error", "message": "No exchange connection"}
    
    try:
        # Test 1: Account info
        account = exchange.fetch_balance()
        print(f"✅ Account access: OK")
        
        # Test 2: Market data
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"✅ Market data access: OK (BTC price: ${ticker['last']})")
        
        # Test 3: Trading permissions (dry run)
        trading_mode = get_trading_mode()
        print(f"✅ Trading mode: {trading_mode}")
        
        return {
            "status": "success",
            "account_type": account.get('info', {}).get('accountType'),
            "trading_mode": trading_mode,
            "balances": {k: float(v.get('free', 0) or 0) for k, v in account.items() 
                        if isinstance(v, dict) and float(v.get('free', 0) or 0) > 0}
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}