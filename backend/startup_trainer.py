import os
import time
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
    """Automated system for fetching live 1h data, generating labels and training models on startup"""
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.ohlcv_dir = self.base_dir / "data" / "ohlcv"
        self.labels_dir = self.base_dir / "data" / "labels"
        self.models_dir = self.base_dir / "agents" / "models"
        self.logs_dir = self.base_dir / "storage" / "training_logs"

        # Create directories
        for dir_path in [self.ohlcv_dir, self.labels_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.symbols = [
            "ATOMUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT",
            "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"
        ]
        
        self.training_status = {symbol: "pending" for symbol in self.symbols}
        self._training_lock = threading.Lock()

    def fetch_live_ohlcv_for_symbol(self, symbol: str, force_refresh: bool = False) -> bool:
        """Fetch live 1h OHLCV data for a specific symbol"""
        try:
            # Use 1h filename
            ohlcv_path = self.ohlcv_dir / f"{symbol}_1h.csv"
            
            # Check if we need to fetch new data
            if not force_refresh and ohlcv_path.exists():
                # Check if data is recent (less than 60 minutes old for 1h data)
                data_age_minutes = (datetime.now().timestamp() - ohlcv_path.stat().st_mtime) / 60
                if data_age_minutes < 60:
                    logger.info(f"📋 {symbol}: Using cached 1h data (age: {data_age_minutes:.1f}m)")
                    return True
            
            logger.info(f"📡 Fetching live 1h OHLCV data for {symbol}...")
            
            # Import the live OHLCV fetcher
            from backend.binance.fetch_live_ohlcv import fetch_ohlcv
            
            # Fetch 1h data with sufficient history (2000 candles = ~83 days)
            df = fetch_ohlcv(symbol, interval="1h", limit=2000)
            
            if df.empty:
                logger.error(f"❌ {symbol}: No 1h OHLCV data received")
                return False
            
            # Validate data quality
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"❌ {symbol}: Invalid OHLCV format - missing columns")
                return False
            
            # Check for sufficient data
            if len(df) < 100:  # Reduced from 168 to 100 hours (4+ days of data)
                logger.error(f"❌ {symbol}: Insufficient 1h data - only {len(df)} candles")
                return False
            
            # Save to CSV with 1h filename
            df.to_csv(ohlcv_path)
            logger.info(f"✅ {symbol}: Saved {len(df)} 1h candles to {ohlcv_path}")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ {symbol}: Could not import live OHLCV fetcher - {e}")
            logger.info("💡 Make sure backend.binance.fetch_live_ohlcv module exists")
            return False
        except Exception as e:
            logger.error(f"❌ {symbol}: Failed to fetch live 1h OHLCV - {e}")
            return False

    def check_data_availability(self, force_refresh: bool = False) -> Dict[str, bool]:
        """Check which symbols have 1h OHLCV data available, fetch if needed"""
        available_data = {}
        
        logger.info("🔍 Checking 1h data availability and fetching live OHLCV...")
        
        for symbol in self.symbols:
            # Try to fetch/update live 1h data first
            fetch_success = self.fetch_live_ohlcv_for_symbol(symbol, force_refresh)
            
            if fetch_success:
                # Verify the saved 1h data
                ohlcv_path = self.ohlcv_dir / f"{symbol}_1h.csv"
                try:
                    df = pd.read_csv(ohlcv_path, nrows=5)
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    available_data[symbol] = all(col in df.columns for col in required_cols)
                    
                    if available_data[symbol]:
                        logger.info(f"✅ {symbol}: 1h data ready for training")
                    else:
                        logger.warning(f"❌ {symbol}: Invalid 1h data format after fetch")
                        
                except Exception as e:
                    available_data[symbol] = False
                    logger.error(f"❌ {symbol}: Failed to validate fetched 1h data - {e}")
            else:
                available_data[symbol] = False
        
        return available_data

    def check_existing_models(self) -> Dict[str, bool]:
        """Check which symbols already have trained 1h models"""
        existing_models = {}
        
        for symbol in self.symbols:
            # Check for 1h models (standard model filename)
            model_path_1h = self.models_dir / f"{symbol.lower()}_model.pkl"
            
            if model_path_1h.exists():
                existing_models[symbol] = True
                logger.info(f"📋 {symbol}: Found existing 1h model")
            else:
                existing_models[symbol] = False
                logger.info(f"📋 {symbol}: No 1h model found")
        
        return existing_models

    def should_retrain_model(self, symbol: str) -> bool:
        """Determine if a 1h model should be retrained based on age and performance"""
        model_path_1h = self.models_dir / f"{symbol.lower()}_model.pkl"
        metadata_path = self.models_dir / "training_summary.json"
        
        if not model_path_1h.exists():
            return True
        
        try:
            # Check model age (retrain weekly for 1h models)
            model_age_days = (datetime.now().timestamp() - model_path_1h.stat().st_mtime) / 86400
            
            if model_age_days > 7:  # Retrain if older than 7 days
                logger.info(f"🔄 {symbol}: 1h model is {model_age_days:.1f} days old, retraining")
                return True
            
            # Check model performance from metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                if symbol in metadata.get('results', {}):
                    model_info = metadata['results'][symbol]
                    test_metrics = model_info.get('test_metrics', {})
                    accuracy = test_metrics.get('accuracy', 0)
                    
                    if accuracy < 0.60:  # Higher threshold for 1h models
                        logger.info(f"🔄 {symbol}: 1h model accuracy {accuracy:.3f} is low, retraining")
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ {symbol}: Could not check 1h model status - {e}")
            return True

    def generate_labels_for_symbol(self, symbol: str) -> bool:
        """Generate 1h training labels for a specific symbol"""
        try:
            logger.info(f"🏷️ Generating 1h labels for {symbol}...")
        
            # Import the correct label generator class
            from backend.ml_engine.generate_labels import TradingLabelGenerator
        
            # Use 1h-optimized parameters (more lenient for better sample count)
            generator = TradingLabelGenerator(
                forward_window=24,              # 24 hours for 1h data
                min_return_threshold=0.015,     # 1.5% minimum return (more lenient)
                stop_loss_threshold=-0.03,      # -3% stop loss (more lenient)
                hold_threshold=0.008,           # 0.8% for hold signals (more lenient)
                timeframe="1h"                  # Specify 1h timeframe
            )
        
            # Generate labels using the outcome method
            labels_df, stats = generator.generate_labels(symbol, method='outcome')
        
            logger.info(f"✅ {symbol}: Generated {stats['total_samples']} 1h labels, "
                        f"balance ratio: {stats['balance_ratio']:.3f}, "
                        f"timeframe: {stats.get('timeframe', '1h')}")
        
            with self._training_lock:
                self.training_status[symbol] = "labels_ready"
        
            return True
        
        except Exception as e:
            logger.error(f"❌ {symbol}: 1h label generation failed - {e}")
            with self._training_lock:
                self.training_status[symbol] = "label_failed"
            return False

    def train_model_for_symbol(self, symbol: str) -> bool:
        """Train 1h ML model for a specific symbol"""
        try:
            logger.info(f"🧠 Training 1h model for {symbol}...")
            
            # Import necessary components
            from backend.ml_engine.feature_extractor import extract_features
            from backend.agents.generic_agent import GenericAgent
            from backend.strategy_engine.strategy_parser import StrategyParser
            from backend.strategy_engine.json_strategy_parser import load_strategy_for_symbol
            
            # Look for 1h label files
            labels_path = None
            possible_paths = [
                self.labels_dir / f"{symbol}_1h_outcome_labels.csv",
                self.labels_dir / f"{symbol}_outcome_labels.csv",
                self.labels_dir / f"{symbol}_1h_labels.csv",
                self.labels_dir / f"{symbol}_labels.csv",
                self.labels_dir / f"{symbol}_1h_outcome_features.csv",
                self.labels_dir / f"{symbol}_hybrid_labels.csv",
                self.labels_dir / f"{symbol}_trade_labels.csv"
            ]
            
            for path in possible_paths:
                if path.exists():
                    labels_path = path
                    break
            
            if labels_path is None:
                raise FileNotFoundError(f"No 1h label files found for {symbol}. Checked: {[str(p) for p in possible_paths]}")
            
            logger.info(f"📋 Loading 1h labels from: {labels_path}")
            labels_df = pd.read_csv(labels_path, index_col=0, parse_dates=True)
            
            # Check if the labels_df has the correct column name
            if 'label' in labels_df.columns and 'action' not in labels_df.columns:
                labels_df['action'] = labels_df['label']
                logger.info(f"📝 Renamed 'label' column to 'action' for compatibility")
            elif 'action' not in labels_df.columns:
                raise ValueError(f"Labels file must contain either 'action' or 'label' column. Found columns: {list(labels_df.columns)}")
            
            # Load 1h OHLCV data
            ohlcv_path = self.ohlcv_dir / f"{symbol}_1h.csv"
            if not ohlcv_path.exists():
                raise FileNotFoundError(f"1h OHLCV data not found: {ohlcv_path}")
                
            ohlcv_df = pd.read_csv(ohlcv_path, index_col=0, parse_dates=True)
            
            # Extract features for the labeled timestamps
            logger.info(f"🔧 Extracting 1h features...")
            features_df = extract_features(ohlcv_df)
            
            # Align features with labels
            common_index = features_df.index.intersection(labels_df.index)
            if len(common_index) == 0:
                raise ValueError(f"No common timestamps between features and labels for {symbol}")
            
            logger.info(f"📊 Found {len(common_index)} common timestamps between 1h features and labels")
            
            # Create training dataset
            training_features = features_df.loc[common_index]
            training_labels = labels_df.loc[common_index, 'action']
            
            # Check for sufficient data (reduced threshold for 1h data)
            if len(training_features) < 100:  # Reduced from 200 to 100 for 1h data
                raise ValueError(f"Insufficient 1h training data: {len(training_features)} samples < 100 minimum")
            
            # Combine into training dataset
            training_data = training_features.copy()
            training_data['action'] = training_labels
            
            logger.info(f"📈 1h training dataset: {len(training_data)} samples, {len(training_features.columns)} features")
            logger.info(f"📊 Label distribution: {training_labels.value_counts().to_dict()}")
            
            # Create and train agent
            try:
                strategy_dict = load_strategy_for_symbol(symbol + "USDT")
            except FileNotFoundError:
                try:
                    strategy_dict = load_strategy_for_symbol(symbol)
                except FileNotFoundError:
                    # Create a default strategy if none exists
                    logger.info(f"🔧 Creating default strategy for {symbol}")
                    strategy_dict = self._create_default_strategy()
            
            strategy_parser = StrategyParser(strategy_dict)
            agent = GenericAgent(symbol=symbol, strategy_logic=strategy_parser)
            
            # Train the model
            logger.info(f"🎯 Starting 1h model training...")
            agent.train_model(training_data)
            
            # Save model with standard filename (no timeframe suffix for 1h)
            model_filename = f"{symbol.lower()}_model.pkl"
            
            # Log training completion
            self._log_training_completion(symbol, len(training_data), "1h")
            
            logger.info(f"✅ {symbol}: 1h model training completed successfully")
            
            with self._training_lock:
                self.training_status[symbol] = "trained"
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {symbol}: 1h model training failed - {e}")
            logger.error(f"🔍 Debug info:")
            logger.error(f"   Labels dir: {self.labels_dir}")
            logger.error(f"   Expected files: {[str(self.labels_dir / f'{symbol}_1h_outcome_labels.csv'), str(self.labels_dir / f'{symbol}_outcome_labels.csv')]}")
            logger.error(f"   Files in labels dir: {list(self.labels_dir.glob('*.csv')) if self.labels_dir.exists() else 'Directory does not exist'}")
            
            with self._training_lock:
                self.training_status[symbol] = "training_failed"
            return False

    def _create_default_strategy(self) -> Dict:
        """Create a default strategy for symbols without existing strategies (1h optimized)"""
        return {
            "name": "Default 1h Technical Strategy",
            "description": "Basic technical analysis strategy optimized for 1h timeframe",
            "rules": [
                {
                    "type": "technical",
                    "indicator": "sma",
                    "period": 20,  # 20 hours SMA
                    "condition": "price_above",
                    "weight": 0.3
                },
                {
                    "type": "technical", 
                    "indicator": "rsi",
                    "period": 14,  # 14 hours RSI
                    "condition": "oversold",
                    "threshold": 30,
                    "weight": 0.4
                },
                {
                    "type": "volume",
                    "condition": "above_average",
                    "period": 10,  # 10 hours volume average
                    "weight": 0.3
                }
            ]
        }

    def _log_training_completion(self, symbol: str, sample_count: int, timeframe: str = "1h"):
        """Log 1h training completion to file"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "sample_count": sample_count,
            "status": "completed"
        }
        
        log_path = self.logs_dir / f"{symbol}_1h_training.json"
        try:
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to log 1h training completion for {symbol}: {e}")

    async def run_parallel_training(self, symbols_to_train: List[str], max_workers: int = 3):
        """Run 1h training for multiple symbols in parallel"""
        logger.info(f"🚀 Starting parallel 1h training for {len(symbols_to_train)} symbols...")
        
        def train_symbol_pipeline(symbol: str) -> bool:
            """Complete 1h training pipeline for one symbol"""
            try:
                # Step 1: Generate 1h labels
                if not self.generate_labels_for_symbol(symbol):
                    return False
                
                # Step 2: Train 1h model
                if not self.train_model_for_symbol(symbol):
                    return False
                
                return True
                
            except Exception as e:
                logger.error(f"❌ {symbol}: 1h training pipeline failed - {e}")
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
        
        logger.info(f"📊 1h training completed: {successful} successful, {failed} failed")
        
        return successful, failed

    def get_training_status(self) -> Dict[str, str]:
        """Get current 1h training status for all symbols"""
        with self._training_lock:
            return self.training_status.copy()

    async def initialize_system(self, force_retrain: bool = False, force_refresh_data: bool = False):
        """Main initialization method for 1h training system"""
        logger.info("🎯 Initializing automated 1h training system with live OHLCV fetching...")
        
        # Check 1h data availability and fetch live data
        available_data = self.check_data_availability(force_refresh=force_refresh_data)
        available_symbols = [symbol for symbol, available in available_data.items() if available]
        
        logger.info(f"📊 Successfully fetched 1h data for {len(available_symbols)}/{len(self.symbols)} symbols")
        
        if not available_symbols:
            logger.error("❌ No 1h OHLCV data available for training")
            logger.info("💡 Check your internet connection and Binance API access")
            return
        
        # Determine which symbols need 1h training
        if force_retrain:
            symbols_to_train = available_symbols
            logger.info("🔄 Force retraining enabled - will train all 1h symbols")
        else:
            existing_models = self.check_existing_models()
            symbols_to_train = []
            
            for symbol in available_symbols:
                if not existing_models.get(symbol, False) or self.should_retrain_model(symbol):
                    symbols_to_train.append(symbol)
                else:
                    logger.info(f"✅ {symbol}: 1h model up to date, skipping")
                    with self._training_lock:
                        self.training_status[symbol] = "up_to_date"
        
        if not symbols_to_train:
            logger.info("✅ All 1h models are up to date")
            return
        
        logger.info(f"🚀 Will train 1h models for: {', '.join(symbols_to_train)}")
        
        # Run parallel 1h training
        successful, failed = await self.run_parallel_training(symbols_to_train)
        
        # Final status report
        logger.info("=" * 60)
        logger.info("🎯 1H TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successfully trained: {successful} 1h models")
        logger.info(f"❌ Failed to train: {failed} 1h models")
        
        final_status = self.get_training_status()
        for symbol, status in final_status.items():
            status_emoji = {
                "trained": "✅",
                "up_to_date": "📋", 
                "training_failed": "❌",
                "label_failed": "⚠️",
                "pending": "⏳"
            }.get(status, "❓")
            logger.info(f"{status_emoji} {symbol}: {status} (1h)")
        
        logger.info("=" * 60)

    async def refresh_data_only(self):
        """Refresh 1h OHLCV data for all symbols without training"""
        logger.info("🔄 Refreshing live 1h OHLCV data for all symbols...")
        
        available_data = self.check_data_availability(force_refresh=True)
        successful = sum(1 for available in available_data.values() if available)
        
        logger.info(f"📊 1h data refresh completed: {successful}/{len(self.symbols)} symbols updated")
        return available_data


# Global instance
auto_trainer = AutoTrainingSystem()


async def run_startup_training(force_retrain: bool = False, force_refresh_data: bool = False):
    """Main function to run 1h startup training"""
    try:
        await auto_trainer.initialize_system(
            force_retrain=force_retrain, 
            force_refresh_data=force_refresh_data
        )
        logger.info("🎉 1h startup training completed successfully")
    except Exception as e:
        logger.error(f"💥 1h startup training failed: {e}")
        raise


async def refresh_ohlcv_data():
    """Standalone function to refresh 1h OHLCV data only"""
    try:
        return await auto_trainer.refresh_data_only()
    except Exception as e:
        logger.error(f"💥 1h data refresh failed: {e}")
        raise


def get_training_status():
    """Get current 1h training status (for API endpoints)"""
    return auto_trainer.get_training_status()


# Additional utility functions for manual control
async def train_specific_symbols(symbols: List[str]):
    """Train 1h models for specific symbols only"""
    try:
        logger.info(f"🎯 Training 1h models for specific symbols: {', '.join(symbols)}")
        
        # Filter to valid symbols
        valid_symbols = [s for s in symbols if s in auto_trainer.symbols]
        if not valid_symbols:
            logger.error("❌ No valid symbols provided")
            return
        
        # Fetch 1h data for these symbols
        available_data = {}
        for symbol in valid_symbols:
            available_data[symbol] = auto_trainer.fetch_live_ohlcv_for_symbol(symbol, force_refresh=True)
        
        available_symbols = [s for s, available in available_data.items() if available]
        
        if not available_symbols:
            logger.error("❌ No 1h data available for requested symbols")
            return
        
        # Run 1h training
        successful, failed = await auto_trainer.run_parallel_training(available_symbols)
        logger.info(f"🎉 Specific 1h training completed: {successful} successful, {failed} failed")
        
    except Exception as e:
        logger.error(f"💥 Specific 1h training failed: {e}")