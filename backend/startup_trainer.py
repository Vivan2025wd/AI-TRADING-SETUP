import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoTrainingSystem:
    """Automated system for fetching live data, generating labels and training models on startup"""
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.ohlcv_dir = self.base_dir / "backend" / "data" / "ohlcv"
        self.labels_dir = self.base_dir / "backend" / "data" / "labels"
        self.models_dir = self.base_dir / "backend" / "agents" / "models"
        self.logs_dir = self.base_dir / "backend" / "storage" / "training_logs"
        
        # Create directories
        for dir_path in [self.ohlcv_dir, self.labels_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.symbols = [
            "DOGEUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT",
            "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"
        ]
        
        self.training_status = {symbol: "pending" for symbol in self.symbols}
        self._training_lock = threading.Lock()

    def fetch_live_ohlcv_for_symbol(self, symbol: str, force_refresh: bool = False) -> bool:
        """Fetch live OHLCV data for a specific symbol"""
        try:
            ohlcv_path = self.ohlcv_dir / f"{symbol}_1h.csv"
            
            # Check if we need to fetch new data
            if not force_refresh and ohlcv_path.exists():
                # Check if data is recent (less than 1 hour old)
                data_age_hours = (datetime.now().timestamp() - ohlcv_path.stat().st_mtime) / 3600
                if data_age_hours < 1:
                    logger.info(f"📋 {symbol}: Using cached data (age: {data_age_hours:.1f}h)")
                    return True
            
            logger.info(f"📡 Fetching live OHLCV data for {symbol}...")
            
            # Import the live OHLCV fetcher
            from backend.binance.fetch_live_ohlcv import fetch_ohlcv
            
            # Fetch data with sufficient history for feature extraction (1000 candles = ~41 days)
            df = fetch_ohlcv(symbol, interval="1h", limit=1000)
            
            if df.empty:
                logger.error(f"❌ {symbol}: No OHLCV data received")
                return False
            
            # Validate data quality
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"❌ {symbol}: Invalid OHLCV format - missing columns")
                return False
            
            # Check for sufficient data
            if len(df) < 100:
                logger.error(f"❌ {symbol}: Insufficient data - only {len(df)} candles")
                return False
            
            # Save to CSV
            df.to_csv(ohlcv_path)
            logger.info(f"✅ {symbol}: Saved {len(df)} candles to {ohlcv_path}")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ {symbol}: Could not import live OHLCV fetcher - {e}")
            logger.info("💡 Make sure backend.binance.fetch_live_ohlcv module exists")
            return False
        except Exception as e:
            logger.error(f"❌ {symbol}: Failed to fetch live OHLCV - {e}")
            return False

    def check_data_availability(self, force_refresh: bool = False) -> Dict[str, bool]:
        """Check which symbols have OHLCV data available, fetch if needed"""
        available_data = {}
        
        logger.info("🔍 Checking data availability and fetching live OHLCV...")
        
        for symbol in self.symbols:
            # Try to fetch/update live data first
            fetch_success = self.fetch_live_ohlcv_for_symbol(symbol, force_refresh)
            
            if fetch_success:
                # Verify the saved data
                ohlcv_path = self.ohlcv_dir / f"{symbol}_1h.csv"
                try:
                    df = pd.read_csv(ohlcv_path, nrows=5)
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    available_data[symbol] = all(col in df.columns for col in required_cols)
                    
                    if available_data[symbol]:
                        logger.info(f"✅ {symbol}: Data ready for training")
                    else:
                        logger.warning(f"❌ {symbol}: Invalid data format after fetch")
                        
                except Exception as e:
                    available_data[symbol] = False
                    logger.error(f"❌ {symbol}: Failed to validate fetched data - {e}")
            else:
                available_data[symbol] = False
        
        return available_data

    def check_existing_models(self) -> Dict[str, bool]:
        """Check which symbols already have trained models"""
        existing_models = {}
        
        for symbol in self.symbols:
            model_path = self.models_dir / f"{symbol.lower()}_model.pkl"
            existing_models[symbol] = model_path.exists()
        
        return existing_models

    def should_retrain_model(self, symbol: str) -> bool:
        """Determine if a model should be retrained based on age and performance"""
        model_path = self.models_dir / f"{symbol.lower()}_model.pkl"
        metadata_path = self.models_dir / "training_summary.json"
        
        if not model_path.exists():
            return True
        
        try:
            # Check model age
            model_age_days = (datetime.now().timestamp() - model_path.stat().st_mtime) / 86400
            
            if model_age_days > 7:  # Retrain if older than 7 days
                logger.info(f"🔄 {symbol}: Model is {model_age_days:.1f} days old, retraining")
                return True
            
            # Check model performance from metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                if symbol in metadata:
                    model_info = metadata[symbol]
                    accuracy = model_info.get('validation_accuracy', 0)
                    
                    if accuracy < 0.6:  # Retrain if accuracy is below 60%
                        logger.info(f"🔄 {symbol}: Model accuracy {accuracy:.3f} is low, retraining")
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ {symbol}: Could not check model status - {e}")
            return True

    def generate_labels_for_symbol(self, symbol: str) -> bool:
        """Generate training labels for a specific symbol"""
        try:
            logger.info(f"🏷️ Generating labels for {symbol}...")
        
            # Import the correct label generator class
            from backend.ml_engine.generate_labels import TradingLabelGenerator
        
            generator = TradingLabelGenerator(
                forward_window=24,
                min_return_threshold=0.025,
                stop_loss_threshold=-0.04,
                hold_threshold=0.008
            )
        
            # Generate labels using the outcome method
            labels_df, stats = generator.generate_labels(symbol, method='outcome')
        
            logger.info(f"✅ {symbol}: Generated {stats['total_samples']} labels, "
                        f"balance ratio: {stats['balance_ratio']:.3f}")
        
            with self._training_lock:
                self.training_status[symbol] = "labels_ready"
        
            return True
        
        except Exception as e:
            logger.error(f"❌ {symbol}: Label generation failed - {e}")
            with self._training_lock:
                self.training_status[symbol] = "label_failed"
            return False

    def train_model_for_symbol(self, symbol: str) -> bool:
        """Train ML model for a specific symbol"""
        try:
            logger.info(f"🧠 Training model for {symbol}...")
            
            # Import necessary components
            from backend.ml_engine.feature_extractor import extract_features
            from backend.agents.generic_agent import GenericAgent
            from backend.strategy_engine.strategy_parser import StrategyParser
            from backend.strategy_engine.json_strategy_parser import load_strategy_for_symbol
            
            # Load labels
            labels_path = self.labels_dir / f"{symbol}_labels.csv"
            if not labels_path.exists():
                raise FileNotFoundError(f"Labels not found for {symbol}")
            
            labels_df = pd.read_csv(labels_path, index_col=0, parse_dates=True)
            
            # Load OHLCV data
            ohlcv_path = self.ohlcv_dir / f"{symbol}_1h.csv"
            ohlcv_df = pd.read_csv(ohlcv_path, index_col=0, parse_dates=True)
            
            # Extract features for the labeled timestamps
            features_df = extract_features(ohlcv_df)
            
            # Align features with labels
            common_index = features_df.index.intersection(labels_df.index)
            if len(common_index) == 0:
                raise ValueError(f"No common timestamps between features and labels for {symbol}")
            
            # Create training dataset
            training_features = features_df.loc[common_index]
            training_labels = labels_df.loc[common_index, 'action']
            
            # Combine into training dataset
            training_data = training_features.copy()
            training_data['action'] = training_labels
            
            # Create and train agent
            try:
                strategy_dict = load_strategy_for_symbol(symbol + "USDT")
            except FileNotFoundError:
                try:
                    strategy_dict = load_strategy_for_symbol(symbol)
                except FileNotFoundError:
                    # Create a default strategy if none exists
                    strategy_dict = self._create_default_strategy()
            
            strategy_parser = StrategyParser(strategy_dict)
            agent = GenericAgent(symbol=symbol, strategy_logic=strategy_parser)
            
            # Train the model
            agent.train_model(training_data)
            
            # Log training completion
            self._log_training_completion(symbol, len(training_data))
            
            logger.info(f"✅ {symbol}: Model training completed successfully")
            
            with self._training_lock:
                self.training_status[symbol] = "trained"
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {symbol}: Model training failed - {e}")
            with self._training_lock:
                self.training_status[symbol] = "training_failed"
            return False

    def _create_default_strategy(self) -> Dict:
        """Create a default strategy for symbols without existing strategies"""
        return {
            "name": "Default Technical Strategy",
            "description": "Basic technical analysis strategy",
            "rules": [
                {
                    "type": "technical",
                    "indicator": "sma",
                    "period": 20,
                    "condition": "price_above",
                    "weight": 0.3
                },
                {
                    "type": "technical", 
                    "indicator": "rsi",
                    "period": 14,
                    "condition": "oversold",
                    "threshold": 30,
                    "weight": 0.4
                },
                {
                    "type": "volume",
                    "condition": "above_average",
                    "period": 10,
                    "weight": 0.3
                }
            ]
        }

    def _log_training_completion(self, symbol: str, sample_count: int):
        """Log training completion to file"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "sample_count": sample_count,
            "status": "completed"
        }
        
        log_path = self.logs_dir / f"{symbol}_training.json"
        try:
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to log training completion for {symbol}: {e}")

    async def run_parallel_training(self, symbols_to_train: List[str], max_workers: int = 3):
        """Run training for multiple symbols in parallel"""
        logger.info(f"🚀 Starting parallel training for {len(symbols_to_train)} symbols...")
        
        def train_symbol_pipeline(symbol: str) -> bool:
            """Complete training pipeline for one symbol"""
            try:
                # Step 1: Generate labels
                if not self.generate_labels_for_symbol(symbol):
                    return False
                
                # Step 2: Train model
                if not self.train_model_for_symbol(symbol):
                    return False
                
                return True
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Training pipeline failed - {e}")
                return False

        # Run training in parallel
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, train_symbol_pipeline, symbol)
                for symbol in symbols_to_train
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Summarize results
        successful = sum(1 for result in results if result is True)
        failed = len(symbols_to_train) - successful
        
        logger.info(f"📊 Training completed: {successful} successful, {failed} failed")
        
        return successful, failed

    def get_training_status(self) -> Dict[str, str]:
        """Get current training status for all symbols"""
        with self._training_lock:
            return self.training_status.copy()

    async def initialize_system(self, force_retrain: bool = False, force_refresh_data: bool = False):
        """Main initialization method"""
        logger.info("🎯 Initializing automated training system with live OHLCV fetching...")
        
        # Check data availability and fetch live data
        available_data = self.check_data_availability(force_refresh=force_refresh_data)
        available_symbols = [symbol for symbol, available in available_data.items() if available]
        
        logger.info(f"📊 Successfully fetched data for {len(available_symbols)}/{len(self.symbols)} symbols")
        
        if not available_symbols:
            logger.error("❌ No OHLCV data available for training")
            logger.info("💡 Check your internet connection and Binance API access")
            return
        
        # Determine which symbols need training
        if force_retrain:
            symbols_to_train = available_symbols
            logger.info("🔄 Force retraining enabled - will train all symbols")
        else:
            existing_models = self.check_existing_models()
            symbols_to_train = []
            
            for symbol in available_symbols:
                if not existing_models.get(symbol, False) or self.should_retrain_model(symbol):
                    symbols_to_train.append(symbol)
                else:
                    logger.info(f"✅ {symbol}: Model up to date, skipping")
                    with self._training_lock:
                        self.training_status[symbol] = "up_to_date"
        
        if not symbols_to_train:
            logger.info("✅ All models are up to date")
            return
        
        logger.info(f"🚀 Will train models for: {', '.join(symbols_to_train)}")
        
        # Run parallel training
        successful, failed = await self.run_parallel_training(symbols_to_train)
        
        # Final status report
        logger.info("=" * 60)
        logger.info("🎯 TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successfully trained: {successful} models")
        logger.info(f"❌ Failed to train: {failed} models")
        
        final_status = self.get_training_status()
        for symbol, status in final_status.items():
            status_emoji = {
                "trained": "✅",
                "up_to_date": "📋", 
                "training_failed": "❌",
                "label_failed": "⚠️",
                "pending": "⏳"
            }.get(status, "❓")
            logger.info(f"{status_emoji} {symbol}: {status}")
        
        logger.info("=" * 60)

    async def refresh_data_only(self):
        """Refresh OHLCV data for all symbols without training"""
        logger.info("🔄 Refreshing live OHLCV data for all symbols...")
        
        available_data = self.check_data_availability(force_refresh=True)
        successful = sum(1 for available in available_data.values() if available)
        
        logger.info(f"📊 Data refresh completed: {successful}/{len(self.symbols)} symbols updated")
        return available_data


# Global instance
auto_trainer = AutoTrainingSystem()


async def run_startup_training(force_retrain: bool = False, force_refresh_data: bool = False):
    """Main function to run on startup"""
    try:
        await auto_trainer.initialize_system(
            force_retrain=force_retrain, 
            force_refresh_data=force_refresh_data
        )
        logger.info("🎉 Startup training completed successfully")
    except Exception as e:
        logger.error(f"💥 Startup training failed: {e}")
        raise


async def refresh_ohlcv_data():
    """Standalone function to refresh OHLCV data only"""
    try:
        return await auto_trainer.refresh_data_only()
    except Exception as e:
        logger.error(f"💥 Data refresh failed: {e}")
        raise


def get_training_status():
    """Get current training status (for API endpoints)"""
    return auto_trainer.get_training_status()


# Additional utility functions for manual control
async def train_specific_symbols(symbols: List[str]):
    """Train models for specific symbols only"""
    try:
        logger.info(f"🎯 Training specific symbols: {', '.join(symbols)}")
        
        # Filter to valid symbols
        valid_symbols = [s for s in symbols if s in auto_trainer.symbols]
        if not valid_symbols:
            logger.error("❌ No valid symbols provided")
            return
        
        # Fetch data for these symbols
        available_data = {}
        for symbol in valid_symbols:
            available_data[symbol] = auto_trainer.fetch_live_ohlcv_for_symbol(symbol, force_refresh=True)
        
        available_symbols = [s for s, available in available_data.items() if available]
        
        if not available_symbols:
            logger.error("❌ No data available for requested symbols")
            return
        
        # Run training
        successful, failed = await auto_trainer.run_parallel_training(available_symbols)
        logger.info(f"🎉 Specific training completed: {successful} successful, {failed} failed")
        
    except Exception as e:
        logger.error(f"💥 Specific training failed: {e}")
        raise