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
    """Automated system for fetching live 30m data, generating labels and training models on startup"""
    
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
            "DOGEUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT",
            "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"
        ]
        
        self.training_status = {symbol: "pending" for symbol in self.symbols}
        self._training_lock = threading.Lock()

    def fetch_live_ohlcv_for_symbol(self, symbol: str, force_refresh: bool = False) -> bool:
        """Fetch live 30m OHLCV data for a specific symbol"""
        try:
            # Changed to 30m filename
            ohlcv_path = self.ohlcv_dir / f"{symbol}_30m.csv"
            
            # Check if we need to fetch new data
            if not force_refresh and ohlcv_path.exists():
                # Check if data is recent (less than 30 minutes old for 30m data)
                data_age_minutes = (datetime.now().timestamp() - ohlcv_path.stat().st_mtime) / 60
                if data_age_minutes < 30:
                    logger.info(f"📋 {symbol}: Using cached 30m data (age: {data_age_minutes:.1f}m)")
                    return True
            
            logger.info(f"📡 Fetching live 30m OHLCV data for {symbol}...")
            
            # Import the live OHLCV fetcher
            from backend.binance.fetch_live_ohlcv import fetch_ohlcv
            
            # Fetch 30m data with more candles for sufficient history (2000 candles = ~41 days)
            df = fetch_ohlcv(symbol, interval="30m", limit=2000)
            
            if df.empty:
                logger.error(f"❌ {symbol}: No 30m OHLCV data received")
                return False
            
            # Validate data quality
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"❌ {symbol}: Invalid OHLCV format - missing columns")
                return False
            
            # Check for sufficient data (need more samples for 30m)
            if len(df) < 200:
                logger.error(f"❌ {symbol}: Insufficient 30m data - only {len(df)} candles")
                return False
            
            # Save to CSV with 30m filename
            df.to_csv(ohlcv_path)
            logger.info(f"✅ {symbol}: Saved {len(df)} 30m candles to {ohlcv_path}")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ {symbol}: Could not import live OHLCV fetcher - {e}")
            logger.info("💡 Make sure backend.binance.fetch_live_ohlcv module exists")
            return False
        except Exception as e:
            logger.error(f"❌ {symbol}: Failed to fetch live 30m OHLCV - {e}")
            return False

    def check_data_availability(self, force_refresh: bool = False) -> Dict[str, bool]:
        """Check which symbols have 30m OHLCV data available, fetch if needed"""
        available_data = {}
        
        logger.info("🔍 Checking 30m data availability and fetching live OHLCV...")
        
        for symbol in self.symbols:
            # Try to fetch/update live 30m data first
            fetch_success = self.fetch_live_ohlcv_for_symbol(symbol, force_refresh)
            
            if fetch_success:
                # Verify the saved 30m data
                ohlcv_path = self.ohlcv_dir / f"{symbol}_30m.csv"
                try:
                    df = pd.read_csv(ohlcv_path, nrows=5)
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    available_data[symbol] = all(col in df.columns for col in required_cols)
                    
                    if available_data[symbol]:
                        logger.info(f"✅ {symbol}: 30m data ready for training")
                    else:
                        logger.warning(f"❌ {symbol}: Invalid 30m data format after fetch")
                        
                except Exception as e:
                    available_data[symbol] = False
                    logger.error(f"❌ {symbol}: Failed to validate fetched 30m data - {e}")
            else:
                available_data[symbol] = False
        
        return available_data

    def check_existing_models(self) -> Dict[str, bool]:
        """Check which symbols already have trained 30m models"""
        existing_models = {}
        
        for symbol in self.symbols:
            # Check for 30m models first, then fallback to 1h models
            model_path_30m = self.models_dir / f"{symbol.lower()}_30m_model.pkl"
            model_path_1h = self.models_dir / f"{symbol.lower()}_model.pkl"
            
            if model_path_30m.exists():
                existing_models[symbol] = True
                logger.info(f"📋 {symbol}: Found existing 30m model")
            elif model_path_1h.exists():
                existing_models[symbol] = False  # Has 1h model but need 30m
                logger.info(f"📋 {symbol}: Has 1h model, need 30m model")
            else:
                existing_models[symbol] = False
                logger.info(f"📋 {symbol}: No model found")
        
        return existing_models

    def should_retrain_model(self, symbol: str) -> bool:
        """Determine if a 30m model should be retrained based on age and performance"""
        model_path_30m = self.models_dir / f"{symbol.lower()}_30m_model.pkl"
        metadata_path = self.models_dir / "30m_training_summary.json"
        
        if not model_path_30m.exists():
            return True
        
        try:
            # Check model age (retrain more frequently for 30m models)
            model_age_days = (datetime.now().timestamp() - model_path_30m.stat().st_mtime) / 86400
            
            if model_age_days > 5:  # Retrain if older than 5 days (more frequent for 30m)
                logger.info(f"🔄 {symbol}: 30m model is {model_age_days:.1f} days old, retraining")
                return True
            
            # Check model performance from metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                if symbol in metadata:
                    model_info = metadata[symbol]
                    accuracy = model_info.get('validation_accuracy', 0)
                    
                    if accuracy < 0.55:  # Slightly lower threshold for 30m models
                        logger.info(f"🔄 {symbol}: 30m model accuracy {accuracy:.3f} is low, retraining")
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ {symbol}: Could not check 30m model status - {e}")
            return True

    def generate_labels_for_symbol(self, symbol: str) -> bool:
        """Generate 30m training labels for a specific symbol"""
        try:
            logger.info(f"🏷️ Generating 30m labels for {symbol}...")
        
            # Import the correct label generator class
            from backend.ml_engine.generate_labels import TradingLabelGenerator
        
            # Use 30m-optimized parameters
            generator = TradingLabelGenerator(
                forward_window=48,            # 24 hours for 30m data
                min_return_threshold=0.015,   # 1.5% minimum return (adjusted for 30m)
                stop_loss_threshold=-0.035,   # -3.5% stop loss
                hold_threshold=0.006,         # 0.6% for hold signals
                timeframe="30m"               # Specify 30m timeframe
            )
        
            # Generate labels using the outcome method (works best for 30m)
            labels_df, stats = generator.generate_labels(symbol, method='outcome')
        
            logger.info(f"✅ {symbol}: Generated {stats['total_samples']} 30m labels, "
                        f"balance ratio: {stats['balance_ratio']:.3f}, "
                        f"timeframe: {stats.get('timeframe', '30m')}")
        
            with self._training_lock:
                self.training_status[symbol] = "labels_ready"
        
            return True
        
        except Exception as e:
            logger.error(f"❌ {symbol}: 30m label generation failed - {e}")
            with self._training_lock:
                self.training_status[symbol] = "label_failed"
            return False

    def train_model_for_symbol(self, symbol: str) -> bool:
        """Train 30m ML model for a specific symbol"""
        try:
            logger.info(f"🧠 Training 30m model for {symbol}...")
            
            # Import necessary components
            from backend.ml_engine.feature_extractor import extract_features
            from backend.agents.generic_agent import GenericAgent
            from backend.strategy_engine.strategy_parser import StrategyParser
            from backend.strategy_engine.json_strategy_parser import load_strategy_for_symbol
            
            # Look for 30m label files first
            labels_path = None
            possible_paths = [
                self.labels_dir / f"{symbol}_30m_outcome_labels.csv",
                self.labels_dir / f"{symbol}_outcome_labels.csv",
                self.labels_dir / f"{symbol}_30m_labels.csv",
                self.labels_dir / f"{symbol}_labels.csv",
                self.labels_dir / f"{symbol}_30m_outcome_features.csv",
                self.labels_dir / f"{symbol}_hybrid_labels.csv",
                self.labels_dir / f"{symbol}_trade_labels.csv"
            ]
            
            for path in possible_paths:
                if path.exists():
                    labels_path = path
                    break
            
            if labels_path is None:
                raise FileNotFoundError(f"No 30m label files found for {symbol}. Checked: {[str(p) for p in possible_paths]}")
            
            logger.info(f"📋 Loading 30m labels from: {labels_path}")
            labels_df = pd.read_csv(labels_path, index_col=0, parse_dates=True)
            
            # Check if the labels_df has the correct column name
            if 'label' in labels_df.columns and 'action' not in labels_df.columns:
                labels_df['action'] = labels_df['label']
                logger.info(f"📝 Renamed 'label' column to 'action' for compatibility")
            elif 'action' not in labels_df.columns:
                raise ValueError(f"Labels file must contain either 'action' or 'label' column. Found columns: {list(labels_df.columns)}")
            
            # Load 30m OHLCV data
            ohlcv_path = self.ohlcv_dir / f"{symbol}_30m.csv"
            if not ohlcv_path.exists():
                raise FileNotFoundError(f"30m OHLCV data not found: {ohlcv_path}")
                
            ohlcv_df = pd.read_csv(ohlcv_path, index_col=0, parse_dates=True)
            
            # Extract features for the labeled timestamps
            logger.info(f"🔧 Extracting 30m features...")
            features_df = extract_features(ohlcv_df)
            
            # Align features with labels
            common_index = features_df.index.intersection(labels_df.index)
            if len(common_index) == 0:
                raise ValueError(f"No common timestamps between features and labels for {symbol}")
            
            logger.info(f"📊 Found {len(common_index)} common timestamps between 30m features and labels")
            
            # Create training dataset
            training_features = features_df.loc[common_index]
            training_labels = labels_df.loc[common_index, 'action']
            
            # Check for sufficient data (lower threshold for 30m since we have more data points)
            if len(training_features) < 100:
                raise ValueError(f"Insufficient 30m training data: {len(training_features)} samples < 100 minimum")
            
            # Combine into training dataset
            training_data = training_features.copy()
            training_data['action'] = training_labels
            
            logger.info(f"📈 30m training dataset: {len(training_data)} samples, {len(training_features.columns)} features")
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
            logger.info(f"🎯 Starting 30m model training...")
            agent.train_model(training_data)
            
            # Save model with 30m indicator
            model_filename = f"{symbol.lower()}_30m_model.pkl"
            
            # Log training completion
            self._log_training_completion(symbol, len(training_data), "30m")
            
            logger.info(f"✅ {symbol}: 30m model training completed successfully")
            
            with self._training_lock:
                self.training_status[symbol] = "trained"
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {symbol}: 30m model training failed - {e}")
            logger.error(f"🔍 Debug info:")
            logger.error(f"   Labels dir: {self.labels_dir}")
            logger.error(f"   Expected files: {[str(self.labels_dir / f'{symbol}_30m_outcome_labels.csv'), str(self.labels_dir / f'{symbol}_outcome_labels.csv')]}")
            logger.error(f"   Files in labels dir: {list(self.labels_dir.glob('*.csv')) if self.labels_dir.exists() else 'Directory does not exist'}")
            
            with self._training_lock:
                self.training_status[symbol] = "training_failed"
            return False

    def _create_default_strategy(self) -> Dict:
        """Create a default strategy for symbols without existing strategies (30m optimized)"""
        return {
            "name": "Default 30m Technical Strategy",
            "description": "Basic technical analysis strategy optimized for 30m timeframe",
            "rules": [
                {
                    "type": "technical",
                    "indicator": "sma",
                    "period": 40,  # Adjusted for 30m (20 hours)
                    "condition": "price_above",
                    "weight": 0.3
                },
                {
                    "type": "technical", 
                    "indicator": "rsi",
                    "period": 28,  # Adjusted for 30m (14 hours)
                    "condition": "oversold",
                    "threshold": 30,
                    "weight": 0.4
                },
                {
                    "type": "volume",
                    "condition": "above_average",
                    "period": 20,  # Adjusted for 30m (10 hours)
                    "weight": 0.3
                }
            ]
        }

    def _log_training_completion(self, symbol: str, sample_count: int, timeframe: str = "30m"):
        """Log 30m training completion to file"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "sample_count": sample_count,
            "status": "completed"
        }
        
        log_path = self.logs_dir / f"{symbol}_30m_training.json"
        try:
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to log 30m training completion for {symbol}: {e}")

    async def run_parallel_training(self, symbols_to_train: List[str], max_workers: int = 3):
        """Run 30m training for multiple symbols in parallel"""
        logger.info(f"🚀 Starting parallel 30m training for {len(symbols_to_train)} symbols...")
        
        def train_symbol_pipeline(symbol: str) -> bool:
            """Complete 30m training pipeline for one symbol"""
            try:
                # Step 1: Generate 30m labels
                if not self.generate_labels_for_symbol(symbol):
                    return False
                
                # Step 2: Train 30m model
                if not self.train_model_for_symbol(symbol):
                    return False
                
                return True
                
            except Exception as e:
                logger.error(f"❌ {symbol}: 30m training pipeline failed - {e}")
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
        
        logger.info(f"📊 30m training completed: {successful} successful, {failed} failed")
        
        return successful, failed

    def get_training_status(self) -> Dict[str, str]:
        """Get current 30m training status for all symbols"""
        with self._training_lock:
            return self.training_status.copy()

    async def initialize_system(self, force_retrain: bool = False, force_refresh_data: bool = False):
        """Main initialization method for 30m training system"""
        logger.info("🎯 Initializing automated 30m training system with live OHLCV fetching...")
        
        # Check 30m data availability and fetch live data
        available_data = self.check_data_availability(force_refresh=force_refresh_data)
        available_symbols = [symbol for symbol, available in available_data.items() if available]
        
        logger.info(f"📊 Successfully fetched 30m data for {len(available_symbols)}/{len(self.symbols)} symbols")
        
        if not available_symbols:
            logger.error("❌ No 30m OHLCV data available for training")
            logger.info("💡 Check your internet connection and Binance API access")
            return
        
        # Determine which symbols need 30m training
        if force_retrain:
            symbols_to_train = available_symbols
            logger.info("🔄 Force retraining enabled - will train all 30m symbols")
        else:
            existing_models = self.check_existing_models()
            symbols_to_train = []
            
            for symbol in available_symbols:
                if not existing_models.get(symbol, False) or self.should_retrain_model(symbol):
                    symbols_to_train.append(symbol)
                else:
                    logger.info(f"✅ {symbol}: 30m model up to date, skipping")
                    with self._training_lock:
                        self.training_status[symbol] = "up_to_date"
        
        if not symbols_to_train:
            logger.info("✅ All 30m models are up to date")
            return
        
        logger.info(f"🚀 Will train 30m models for: {', '.join(symbols_to_train)}")
        
        # Run parallel 30m training
        successful, failed = await self.run_parallel_training(symbols_to_train)
        
        # Final status report
        logger.info("=" * 60)
        logger.info("🎯 30M TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successfully trained: {successful} 30m models")
        logger.info(f"❌ Failed to train: {failed} 30m models")
        
        final_status = self.get_training_status()
        for symbol, status in final_status.items():
            status_emoji = {
                "trained": "✅",
                "up_to_date": "📋", 
                "training_failed": "❌",
                "label_failed": "⚠️",
                "pending": "⏳"
            }.get(status, "❓")
            logger.info(f"{status_emoji} {symbol}: {status} (30m)")
        
        logger.info("=" * 60)

    async def refresh_data_only(self):
        """Refresh 30m OHLCV data for all symbols without training"""
        logger.info("🔄 Refreshing live 30m OHLCV data for all symbols...")
        
        available_data = self.check_data_availability(force_refresh=True)
        successful = sum(1 for available in available_data.values() if available)
        
        logger.info(f"📊 30m data refresh completed: {successful}/{len(self.symbols)} symbols updated")
        return available_data


# Global instance
auto_trainer = AutoTrainingSystem()


async def run_startup_training(force_retrain: bool = False, force_refresh_data: bool = False):
    """Main function to run 30m startup training"""
    try:
        await auto_trainer.initialize_system(
            force_retrain=force_retrain, 
            force_refresh_data=force_refresh_data
        )
        logger.info("🎉 30m startup training completed successfully")
    except Exception as e:
        logger.error(f"💥 30m startup training failed: {e}")
        raise


async def refresh_ohlcv_data():
    """Standalone function to refresh 30m OHLCV data only"""
    try:
        return await auto_trainer.refresh_data_only()
    except Exception as e:
        logger.error(f"💥 30m data refresh failed: {e}")
        raise


def get_training_status():
    """Get current 30m training status (for API endpoints)"""
    return auto_trainer.get_training_status()


# Additional utility functions for manual control
async def train_specific_symbols(symbols: List[str]):
    """Train 30m models for specific symbols only"""
    try:
        logger.info(f"🎯 Training 30m models for specific symbols: {', '.join(symbols)}")
        
        # Filter to valid symbols
        valid_symbols = [s for s in symbols if s in auto_trainer.symbols]
        if not valid_symbols:
            logger.error("❌ No valid symbols provided")
            return
        
        # Fetch 30m data for these symbols
        available_data = {}
        for symbol in valid_symbols:
            available_data[symbol] = auto_trainer.fetch_live_ohlcv_for_symbol(symbol, force_refresh=True)
        
        available_symbols = [s for s, available in available_data.items() if available]
        
        if not available_symbols:
            logger.error("❌ No 30m data available for requested symbols")
            return
        
        # Run 30m training
        successful, failed = await auto_trainer.run_parallel_training(available_symbols)
        logger.info(f"🎉 Specific 30m training completed: {successful} successful, {failed} failed")
        
    except Exception as e:
        logger.error(f"💥 Specific 30m training failed: {e}")