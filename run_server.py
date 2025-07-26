#!/usr/bin/env python3
"""
AI Trading Dashboard - Complete Server Startup Script
"""
import os
import sys
import logging
import asyncio
from pathlib import Path
import uvicorn
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
log_format = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.basicConfig(
    level=getattr(logging, log_level),
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend/storage/logs/server.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

def validate_environment():
    """Validate required environment and directories"""
    logger.info("🔍 Validating environment...")
    
    # Required directories
    required_dirs = [
        'data/ohlcv',
        'data/labels', 
        'backend/agents/models',
        'backend/storage/logs',
        'backend/storage/training_logs',
        'backend/storage/performance_logs'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directory ready: {dir_path}")
    
    # Check for critical configuration
    critical_settings = {
        'API_HOST': os.getenv('API_HOST', '0.0.0.0'),
        'API_PORT': int(os.getenv('API_PORT', '8000')),
        'SKIP_STARTUP_TRAINING': os.getenv('SKIP_STARTUP_TRAINING', 'false'),
        'TRAINING_MAX_WORKERS': int(os.getenv('TRAINING_MAX_WORKERS', '3'))
    }
    
    logger.info("📋 Configuration:")
    for key, value in critical_settings.items():
        logger.info(f"  {key}: {value}")
    
    return critical_settings

def check_data_availability():
    """Check available training data"""
    logger.info("📊 Checking data availability...")
    
    ohlcv_dir = project_root / 'data' / 'ohlcv'
    available_symbols = []
    
    if ohlcv_dir.exists():
        for file_path in ohlcv_dir.glob("*_1h.csv"):
            symbol = file_path.stem.replace('_1h', '')
            try:
                # Quick validation
                import pandas as pd
                df = pd.read_csv(file_path, nrows=5)
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    available_symbols.append(symbol)
                    logger.info(f"✅ Data available: {symbol}")
            except Exception as e:
                logger.warning(f"⚠️ Invalid data file: {file_path} - {e}")
    
    if not available_symbols:
        logger.warning("⚠️ No valid OHLCV data found!")
        logger.info("💡 To get started, place OHLCV CSV files in data/ohlcv/")
        logger.info("   Expected format: SYMBOL_1h.csv (e.g., BTCUSDT_1h.csv)")
    
    return available_symbols

def print_startup_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🤖 AI Trading Dashboard - Server Starting 🚀          ║
║                                                              ║
║  🧠 Multi-Agent Trading System                               ║
║  📊 Automated ML Model Training                              ║
║  ⚡ Real-time Predictions                                    ║
║  📈 Performance Tracking                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def run_startup_checks():
    """Run all pre-startup checks"""
    logger.info("🔧 Running startup checks...")
    
    try:
        # Validate environment
        config = validate_environment()
        
        # Check data availability
        available_symbols = check_data_availability()
        
        # Import and validate core modules
        logger.info("📦 Validating core modules...")
        
        try:
            from backend.main import app
            logger.info("✅ FastAPI app imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import FastAPI app: {e}")
            raise
        
        try:
            from backend.startup_trainer import get_training_status
            status = get_training_status()
            logger.info(f"✅ Training system ready - {len(status)} symbols configured")
        except ImportError as e:
            logger.error(f"❌ Failed to import training system: {e}")
            raise
        
        # Pre-flight summary
        logger.info("=" * 60)
        logger.info("🎯 PRE-FLIGHT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"🏠 Project Root: {project_root}")
        logger.info(f"📊 Available Data: {len(available_symbols)} symbols")
        logger.info(f"🔧 Skip Training: {config['SKIP_STARTUP_TRAINING']}")
        logger.info(f"⚡ Server: {config['API_HOST']}:{config['API_PORT']}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Startup checks failed: {e}")
        return False

def main():
    """Main entry point"""
    print_startup_banner()
    
    try:
        # Run startup checks
        checks_passed = asyncio.run(run_startup_checks())
        
        if not checks_passed:
            logger.error("❌ Startup checks failed - exiting")
            sys.exit(1)
        
        # Import the FastAPI app
        from backend.main import app
        
        # Get server configuration
        host = os.getenv('API_HOST', '0.0.0.0')
        port = int(os.getenv('API_PORT', '8000'))
        debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Additional uvicorn settings
        reload = debug  # Enable auto-reload in debug mode
        workers = 1 if debug else int(os.getenv('WORKERS', '1'))
        
        logger.info(f"🚀 Starting server on http://{host}:{port}")
        logger.info(f"🔧 Debug mode: {debug}")
        logger.info(f"🔧 Auto-reload: {reload}")
        
        # Start the server
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level.lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested by user")
    except Exception as e:
        logger.error(f"💥 Server startup failed: {e}")
        sys.exit(1)
    finally:
        logger.info("👋 Server shutdown complete")

if __name__ == "__main__":
    main()