import ccxt
import os
import time
import logging
from typing import Dict, Any, Optional, List
from backend.utils.binance_api import get_trading_mode, get_binance_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Fallback configuration
FALLBACK_CONFIG = {
    'max_retries': 3,
    'retry_delay': 2,  # seconds
    'connection_timeout': 30,
    'request_timeout': 15,
    'rate_limit_backoff': 5,  # seconds to wait on rate limit
    'alternative_exchanges': ['kucoin', 'okx'],  # Alternative exchanges if Binance fails
}

class ExchangeConnectionError(Exception):
    """Custom exception for exchange connection issues"""
    pass

def get_ccxt_exchange(retry_count: int = 0) -> Optional[ccxt.Exchange]:
    """Get CCXT exchange instance with comprehensive fallback mechanisms"""
    max_retries = FALLBACK_CONFIG['max_retries']
    
    if retry_count >= max_retries:
        logger.error(f"❌ Max retries ({max_retries}) exceeded for Binance connection")
        return None
    
    try:
        # Fallback 1: Try to use the connected Binance client from binance_api.py
        try:
            client = get_binance_client()
            trading_mode = get_trading_mode()
        except Exception as e:
            logger.warning(f"⚠️ Could not get binance client, using fallback config: {e}")
            client = None
            trading_mode = "mock"  # Default fallback
        
        # Fallback 2: Extract API keys with multiple sources
        api_key = None
        api_secret = None
        
        if client:
            api_key = getattr(client, 'API_KEY', None)
            api_secret = getattr(client, 'API_SECRET', None)
        
        # Fallback to environment variables
        if not api_key or not api_secret:
            api_key = BINANCE_API_KEY
            api_secret = BINANCE_API_SECRET
        
        # Fallback 3: Alternative API key environment variable names
        if not api_key or not api_secret:
            api_key = api_key or os.getenv("BINANCE_KEY") or os.getenv("BN_API_KEY")
            api_secret = api_secret or os.getenv("BINANCE_SECRET") or os.getenv("BN_API_SECRET")
        
        if not api_key or not api_secret:
            logger.warning("⚠️ No API keys found, will use mock mode")
            return None
            
        # Fallback 4: Create exchange with robust configuration
        exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'timeout': FALLBACK_CONFIG['connection_timeout'] * 1000,  # ccxt uses milliseconds
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,  # Handle server time differences
            },
            'verbose': False,
        }
        
        # Fallback 5: Add proxy support if configured
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        if proxy:
            exchange_config['proxies'] = {'http': proxy, 'https': proxy}
            logger.info(f"🌐 Using proxy: {proxy}")
        
        exchange = ccxt.binance(exchange_config)  # type: ignore
        
        # Fallback 6: Set sandbox mode with error handling
        try:
            if trading_mode == "mock":
                exchange.set_sandbox_mode(True)
                logger.info("🧪 Using Binance Testnet (Mock Mode)")
            else:
                exchange.set_sandbox_mode(False)
                logger.info("⚠️ Using Binance LIVE Trading")
        except Exception as sandbox_error:
            logger.warning(f"⚠️ Could not set sandbox mode: {sandbox_error}")
            # Continue anyway - some configurations don't support sandbox mode
            
        # Fallback 7: Test connection with multiple validation levels
        if not _test_exchange_connection(exchange, retry_count):
            if retry_count < max_retries - 1:
                logger.info(f"🔄 Retrying connection... ({retry_count + 1}/{max_retries})")
                time.sleep(FALLBACK_CONFIG['retry_delay'] * (retry_count + 1))  # Exponential backoff
                return get_ccxt_exchange(retry_count + 1)
            else:
                logger.error("❌ All connection tests failed")
                return None
            
        return exchange
        
    except ccxt.NetworkError as e:
        logger.warning(f"🌐 Network error (attempt {retry_count + 1}): {e}")
        if retry_count < max_retries - 1:
            time.sleep(FALLBACK_CONFIG['retry_delay'])
            return get_ccxt_exchange(retry_count + 1)
        return None
        
    except ccxt.ExchangeError as e:
        logger.error(f"🏦 Exchange error: {e}")
        return None
        
    except Exception as e:
        logger.error(f"⚠️ Unexpected error connecting to Binance API: {e}")
        if retry_count < max_retries - 1:
            time.sleep(FALLBACK_CONFIG['retry_delay'])
            return get_ccxt_exchange(retry_count + 1)
        return None

def _test_exchange_connection(exchange: ccxt.Exchange, retry_count: int) -> bool:
    """Test exchange connection with progressive validation levels"""
    tests = [
        ("Basic connectivity", lambda: exchange.load_markets()),
        ("Account access", lambda: exchange.fetch_balance()),
        ("Market data", lambda: exchange.fetch_ticker('BTC/USDT')),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            logger.info(f"✅ {test_name}: OK")
            if test_name == "Account access" and result:
                account_type = result.get('info', {}).get('accountType', 'Unknown')
                logger.info(f"📊 Account type: {account_type}")
        except ccxt.RateLimitExceeded:
            logger.warning(f"⏳ Rate limit hit during {test_name}, waiting...")
            time.sleep(FALLBACK_CONFIG['rate_limit_backoff'])
            return False
        except ccxt.AuthenticationError as e:
            logger.error(f"🔐 Authentication failed during {test_name}: {e}")
            return False
        except ccxt.NetworkError as e:
            logger.warning(f"🌐 Network error during {test_name}: {e}")
            return False
        except Exception as e:
            logger.warning(f"⚠️ {test_name} failed: {e}")
            if retry_count == 0:  # Only require all tests to pass on first attempt
                return False
    
    return True

def place_market_order(symbol: str, side: str, amount: float, retry_count: int = 0) -> Optional[Dict[str, Any]]:
    """Place market order with comprehensive fallback mechanisms"""
    max_retries = FALLBACK_CONFIG['max_retries']
    
    if retry_count >= max_retries:
        logger.error(f"❌ Max retries ({max_retries}) exceeded for order placement")
        return _create_mock_order_response(symbol, side, amount, "Max retries exceeded")
    
    # Fallback 1: Get exchange with fallback
    exchange = get_ccxt_exchange()
    
    if exchange is None:
        return _create_mock_order_response(symbol, side, amount, "No API connection available")
    
    try:
        # Fallback 2: Input validation and normalization
        side_clean = side.lower().strip()
        if side_clean not in ("buy", "sell"):
            raise ValueError("side must be either 'buy' or 'sell'")
        
        amount = float(amount)
        if amount <= 0:
            raise ValueError("amount must be positive")
        
        # Fallback 3: Symbol normalization with multiple formats
        ccxt_symbol = _normalize_symbol(symbol)
        
        logger.info(f"🔍 Attempting to place order: {side_clean.upper()} {amount} {ccxt_symbol}")
        
        # Fallback 4: Pre-order validation
        if not _validate_order_requirements(exchange, ccxt_symbol, side_clean, amount):
            if retry_count < max_retries - 1:
                logger.info(f"🔄 Retrying order after validation failure... ({retry_count + 1}/{max_retries})")
                time.sleep(FALLBACK_CONFIG['retry_delay'])
                return place_market_order(symbol, side, amount, retry_count + 1)
            return _create_mock_order_response(symbol, side, amount, "Order validation failed")
        
        # Fallback 5: Place order with error handling
        order = exchange.create_market_order(symbol=ccxt_symbol, side=side_clean, amount=amount)
        
        trading_mode = get_trading_mode() if get_trading_mode else "unknown"
        mode_text = "TESTNET" if trading_mode == "mock" else "LIVE"
        logger.info(f"✅ Binance {mode_text} {side_clean.upper()} order executed")
        
        return order
        
    except ccxt.InsufficientFunds as e:
        logger.error(f"💰 Insufficient funds: {e}")
        return _handle_insufficient_funds_error(exchange, symbol, side, amount)
        
    except ccxt.RateLimitExceeded as e:
        logger.warning(f"⏳ Rate limit exceeded: {e}")
        if retry_count < max_retries - 1:
            wait_time = FALLBACK_CONFIG['rate_limit_backoff'] * (retry_count + 1)
            logger.info(f"⏳ Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            return place_market_order(symbol, side, amount, retry_count + 1)
        return _create_mock_order_response(symbol, side, amount, "Rate limit exceeded")
        
    except ccxt.NetworkError as e:
        logger.warning(f"🌐 Network error: {e}")
        if retry_count < max_retries - 1:
            time.sleep(FALLBACK_CONFIG['retry_delay'])
            return place_market_order(symbol, side, amount, retry_count + 1)
        return _create_mock_order_response(symbol, side, amount, "Network error")
        
    except ccxt.AuthenticationError as e:
        logger.error(f"🔐 Authentication error: {e}")
        _provide_auth_error_guidance()
        return _create_mock_order_response(symbol, side, amount, "Authentication failed")
        
    except ccxt.BadRequest as e:
        logger.error(f"📝 Bad request: {e}")
        _provide_bad_request_guidance(str(e))
        return _create_mock_order_response(symbol, side, amount, "Invalid request")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if retry_count < max_retries - 1:
            time.sleep(FALLBACK_CONFIG['retry_delay'])
            return place_market_order(symbol, side, amount, retry_count + 1)
        return _create_mock_order_response(symbol, side, amount, f"Unexpected error: {str(e)}")

def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol format with fallback options"""
    symbol = symbol.upper().strip()
    
    # Remove existing separators
    symbol = symbol.replace("/", "").replace("-", "").replace("_", "")
    
    # Add USDT if not present
    if not any(symbol.endswith(quote) for quote in ["USDT", "BUSD", "BTC", "ETH", "BNB"]):
        symbol += "USDT"
    
    # Convert to CCXT format (BTC/USDT)
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT"
    elif symbol.endswith("BUSD"):
        base = symbol[:-4]
        return f"{base}/BUSD"
    elif symbol.endswith("BTC"):
        base = symbol[:-3]
        return f"{base}/BTC"
    elif symbol.endswith("ETH"):
        base = symbol[:-3]
        return f"{base}/ETH"
    else:
        return symbol

def _validate_order_requirements(exchange: ccxt.Exchange, symbol: str, side: str, amount: float) -> bool:
    """Validate order requirements with fallback checks"""
    try:
        # Check 1: Market exists
        markets = exchange.load_markets()
        if symbol not in markets:
            logger.error(f"❌ Symbol {symbol} not found in markets")
            return False
        
        market = markets[symbol]
        
        # Check 2: Market is active
        if not market.get('active', True):
            logger.error(f"❌ Market {symbol} is not active")
            return False
        
        # Check 3: Minimum order size
        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
        if min_amount and amount < min_amount:
            logger.error(f"❌ Order amount {amount} below minimum {min_amount}")
            return False
        
        # Check 4: Balance validation with fallback
        try:
            balance = exchange.fetch_balance()
            
            if side == "buy":
                quote_asset = symbol.split('/')[1]  # USDT in BTC/USDT
                available = float(balance.get(quote_asset, {}).get('free', 0) or 0)
                
                # Estimate cost with fallback price calculation
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    price = float(ticker.get('last') or ticker.get('close') or 0)
                except:
                    # Fallback: use order book
                    try:
                        orderbook = exchange.fetch_order_book(symbol, 5)
                        price = float(orderbook['asks'][0][0]) if orderbook['asks'] else 0
                    except:
                        logger.warning("⚠️ Could not fetch price, skipping balance check")
                        return True
                
                required = amount * price * 1.01  # Add 1% buffer for fees
                if available < required:
                    logger.error(f"💰 Insufficient {quote_asset}: {available} < {required}")
                    return False
                    
            else:  # sell
                base_asset = symbol.split('/')[0]  # BTC in BTC/USDT
                available = float(balance.get(base_asset, {}).get('free', 0) or 0)
                if available < amount:
                    logger.error(f"💰 Insufficient {base_asset}: {available} < {amount}")
                    return False
                    
        except Exception as balance_error:
            logger.warning(f"⚠️ Balance check failed, continuing anyway: {balance_error}")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Order validation failed: {e}")
        return False

def _create_mock_order_response(symbol: str, side: str, amount: float, reason: str) -> Dict[str, Any]:
    """Create a standardized mock order response"""
    logger.info(f"ℹ️ Mock order: {side.upper()} {amount} {symbol} - {reason}")
    return {
        "status": "mock",
        "symbol": symbol,
        "side": side,
        "amount": amount,
        "reason": reason,
        "message": f"No real trade executed: {reason}",
        "timestamp": int(time.time() * 1000),
        "id": f"mock_{int(time.time())}"
    }

def _handle_insufficient_funds_error(exchange: ccxt.Exchange, symbol: str, side: str, amount: float) -> Dict[str, Any]:
    """Handle insufficient funds with detailed information"""
    try:
        balance = exchange.fetch_balance()
        if side.lower() == "buy":
            quote_asset = symbol.split('/')[1]
            available = balance.get(quote_asset, {}).get('free', 0)
            logger.info(f"💰 Available {quote_asset}: {available}")
        else:
            base_asset = symbol.split('/')[0]
            available = balance.get(base_asset, {}).get('free', 0)
            logger.info(f"💰 Available {base_asset}: {available}")
    except:
        logger.warning("Could not fetch balance details")
    
    logger.info("🔧 SOLUTION: Add funds to your Binance account or reduce order size")
    return _create_mock_order_response(symbol, side, amount, "Insufficient funds")

def _provide_auth_error_guidance():
    """Provide detailed authentication error guidance"""
    logger.info("🔧 AUTHENTICATION ERROR SOLUTIONS:")
    logger.info("   1. Go to Binance → API Management")
    logger.info("   2. Ensure 'Enable Spot & Margin Trading' is checked")
    logger.info("   3. Check IP restrictions (disable or whitelist your IP)")
    logger.info("   4. Verify you're using correct API keys")
    logger.info("   5. Check if API keys are for testnet vs mainnet")

def _provide_bad_request_guidance(error_msg: str):
    """Provide guidance based on specific bad request errors"""
    if "MIN_NOTIONAL" in error_msg:
        logger.info("🔧 SOLUTION: Increase order size (minimum $10-20 typically)")
    elif "LOT_SIZE" in error_msg:
        logger.info("🔧 SOLUTION: Adjust order quantity to match lot size requirements")
    elif "PRICE_FILTER" in error_msg:
        logger.info("🔧 SOLUTION: Check price precision and tick size requirements")
    else:
        logger.info("🔧 SOLUTION: Check order parameters and market requirements")

def test_api_permissions() -> Dict[str, Any]:
    """Test API permissions with comprehensive fallback diagnostics"""
    logger.info("🔍 Starting comprehensive API diagnostics...")
    
    # Test 1: Exchange connection
    exchange = get_ccxt_exchange()
    if not exchange:
        return {
            "status": "error", 
            "message": "No exchange connection available",
            "fallbacks_used": ["mock_mode"]
        }
    
    results = {
        "status": "success",
        "tests": {},
        "trading_mode": "unknown",
        "fallbacks_used": []
    }
    
    # Test 2: Account access with fallback
    try:
        account = exchange.fetch_balance()
        results["tests"]["account_access"] = "✅ OK"
        results["account_type"] = account.get('info', {}).get('accountType', 'Unknown')
        
        # Extract meaningful balance information
        balances = {}
        for asset, balance_info in account.items():
            if isinstance(balance_info, dict):
                free = float(balance_info.get('free', 0) or 0)
                if free > 0:
                    balances[asset] = free
        results["balances"] = balances
        
    except Exception as e:
        results["tests"]["account_access"] = f"❌ Failed: {e}"
        results["fallbacks_used"].append("account_access_failed")
    
    # Test 3: Market data access
    try:
        ticker = exchange.fetch_ticker('BTC/USDT')
        price = ticker.get('last', 'Unknown')
        results["tests"]["market_data"] = f"✅ OK (BTC: ${price})"
    except Exception as e:
        results["tests"]["market_data"] = f"❌ Failed: {e}"
        results["fallbacks_used"].append("market_data_failed")
    
    # Test 4: Trading mode detection
    try:
        trading_mode = get_trading_mode()
        if trading_mode is not None:
            results["trading_mode"] = trading_mode
            results["tests"]["trading_mode"] = f"✅ {trading_mode.upper()}"
        else:
            results["tests"]["trading_mode"] = "⚠️ Trading mode is None"
            results["fallbacks_used"].append("trading_mode_none")
    except Exception as e:
        results["tests"]["trading_mode"] = f"⚠️ Could not determine: {e}"
        results["fallbacks_used"].append("trading_mode_unknown")
    
    # Test 5: Order validation (dry run)
    try:
        markets = exchange.load_markets()
        if 'BTC/USDT' in markets:
            market_info = markets['BTC/USDT']
            min_amount = market_info.get('limits', {}).get('amount', {}).get('min', 'Unknown')
            results["tests"]["order_validation"] = f"✅ OK (Min order: {min_amount} BTC)"
        else:
            results["tests"]["order_validation"] = "⚠️ BTC/USDT market not available"
    except Exception as e:
        results["tests"]["order_validation"] = f"❌ Failed: {e}"
        results["fallbacks_used"].append("order_validation_failed")
    
    logger.info("🏁 API diagnostics completed")
    return results

# Additional utility functions for fallback scenarios

def get_alternative_exchange(exchange_id: Optional[str] = None) -> Optional[ccxt.Exchange]:
    """Get alternative exchange as fallback (placeholder for future implementation)"""
    # This could be implemented to use KuCoin, OKX, or other exchanges as fallbacks
    logger.info(f"🔄 Alternative exchange fallback not implemented yet")
    return None

def save_failed_order_for_retry(order_data: Dict[str, Any]):
    """Save failed order data for manual retry later (placeholder)"""
    # This could save to a file or database for later processing
    logger.info(f"💾 Order retry mechanism not implemented yet")
    pass