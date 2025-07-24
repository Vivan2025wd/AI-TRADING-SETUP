import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OHLCV_DIR = "data/ohlcv"
TRADES_DIR = "backend/storage/trade_history"
LABELS_DIR = "data/labels"

os.makedirs(LABELS_DIR, exist_ok=True)

class OptimizedLabelGenerator:
    """Optimized label generator specifically for training compatibility"""
    
    def __init__(self, 
                 forward_window: int = 24,
                 min_return_threshold: float = 0.025,
                 stop_loss_threshold: float = -0.04,
                 min_samples_per_class: int = 100):
        self.forward_window = forward_window
        self.min_return_threshold = min_return_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.min_samples_per_class = min_samples_per_class

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Load and validate OHLCV data"""
        path = os.path.join(OHLCV_DIR, f"{symbol}_1h.csv")
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.sort_index()
            
            # Basic validation
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            # Remove any invalid data
            df = df.dropna()
            df = df[df['volume'] > 0]  # Remove zero volume periods
            
            logger.info(f"Loaded {len(df)} OHLCV records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load OHLCV for {symbol}: {e}")
            raise

    def calculate_volatility_adjusted_thresholds(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Calculate dynamic thresholds based on market volatility"""
        df = ohlcv.copy()
        
        # Calculate rolling volatility (24-hour window)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(24).std()
        
        # Calculate ATR (Average True Range) for additional context
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(14).mean() / df['close']
        
        # Dynamic thresholds
        df['buy_threshold'] = np.maximum(
            self.min_return_threshold,
            df['volatility'] * 2.5  # 2.5x volatility for buy signals
        )
        
        df['sell_threshold'] = np.minimum(
            self.stop_loss_threshold,
            -df['volatility'] * 3.0  # 3x volatility for sell signals
        )
        
        return df

    def generate_outcome_labels(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Generate labels based on future price movements"""
        df = self.calculate_volatility_adjusted_thresholds(ohlcv)
        labels = pd.Series(index=df.index, dtype=object)
        
        for i in range(len(df) - self.forward_window):
            current_price = df['close'].iloc[i]
            
            # Look forward for price action
            future_prices = df['close'].iloc[i+1:i+self.forward_window+1]
            
            if len(future_prices) == 0:
                continue
            
            # Calculate forward returns
            max_return = (future_prices.max() / current_price) - 1
            min_return = (future_prices.min() / current_price) - 1
            final_return = (future_prices.iloc[-1] / current_price) - 1
            
            # Get dynamic thresholds
            buy_threshold = df['buy_threshold'].iloc[i]
            sell_threshold = df['sell_threshold'].iloc[i]
            
            # Ensure thresholds are valid
            if pd.isna(buy_threshold) or pd.isna(sell_threshold):
                buy_threshold = self.min_return_threshold
                sell_threshold = self.stop_loss_threshold
            
            # Label logic with risk management
            if max_return > buy_threshold and min_return > sell_threshold:
                # Strong upward movement without significant drawdown
                labels.iloc[i] = 'buy'
            elif min_return < sell_threshold:
                # Significant downward movement
                labels.iloc[i] = 'sell'
            # We skip 'hold' labels to focus on actionable buy/sell signals
        
        return labels.dropna()

    def balance_labels(self, labels: pd.Series) -> pd.Series:
        """Balance the dataset while maintaining sufficient samples"""
        value_counts = labels.value_counts()
        logger.info(f"Original distribution: {value_counts.to_dict()}")
        
        # Ensure we have minimum samples for each class
        valid_classes = []
        for class_name, count in value_counts.items():
            if count >= self.min_samples_per_class:
                valid_classes.append(class_name)
            else:
                logger.warning(f"Class '{class_name}' has only {count} samples, minimum required: {self.min_samples_per_class}")
        
        if len(valid_classes) < 2:
            raise ValueError("Not enough samples for balanced training")
        
        # Filter to valid classes only
        filtered_labels = labels[labels.isin(valid_classes)]
        
        # Balance to the smallest class
        min_count = filtered_labels.value_counts().min()
        
        balanced_indices = []
        np.random.seed(42)  # For reproducibility
        
        for class_name in valid_classes:
            class_indices = filtered_labels[filtered_labels == class_name].index
            if len(class_indices) > min_count:
                sampled_indices = np.random.choice(class_indices, min_count, replace=False)
                balanced_indices.extend(sampled_indices)
            else:
                balanced_indices.extend(class_indices)
        
        balanced_labels = filtered_labels[balanced_indices].sort_index()
        logger.info(f"Balanced distribution: {balanced_labels.value_counts().to_dict()}")
        
        return balanced_labels

    def generate_training_labels(self, symbol: str) -> Tuple[pd.Series, Dict]:
        """Generate labels optimized for training"""
        logger.info(f"Generating training labels for {symbol}...")
        
        try:
            # Load data
            ohlcv = self.load_ohlcv(symbol)
            
            # Generate outcome-based labels
            raw_labels = self.generate_outcome_labels(ohlcv)
            
            if len(raw_labels) == 0:
                raise ValueError(f"No labels generated for {symbol}")
            
            # Balance the dataset
            balanced_labels = self.balance_labels(raw_labels)
            
            # Create statistics
            stats = {
                'symbol': symbol,
                'total_samples': len(balanced_labels),
                'date_range_start': str(balanced_labels.index.min()),
                'date_range_end': str(balanced_labels.index.max()),
                'distribution': balanced_labels.value_counts().to_dict(),
                'balance_ratio': balanced_labels.value_counts().min() / balanced_labels.value_counts().max()
            }
            
            # Save in format expected by batch_train.py
            self.save_training_labels(balanced_labels, symbol)
            
            return balanced_labels, stats
            
        except Exception as e:
            logger.error(f"Failed to generate training labels for {symbol}: {e}")
            raise

    def save_training_labels(self, labels: pd.Series, symbol: str):
        """Save labels in the format expected by batch_train.py"""
        # Create a DataFrame with 'action' column (expected by batch_train.py)
        df = pd.DataFrame({'action': labels})
        
        # Save with the expected filename format
        output_path = os.path.join(LABELS_DIR, f"{symbol}_labels.csv")
        df.to_csv(output_path)
        
        logger.info(f"Saved {len(df)} training labels to {output_path}")
        
        # Also save the outcome-based version for reference
        outcome_path = os.path.join(LABELS_DIR, f"{symbol}_outcome_labels.csv")
        df.to_csv(outcome_path)

def main():
    """Generate training labels for all symbols"""
    symbols = [
        "DOGEUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT",
        "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"
    ]
    
    generator = OptimizedLabelGenerator(
        forward_window=24,
        min_return_threshold=0.025,  # 2.5%
        stop_loss_threshold=-0.04,   # -4%
        min_samples_per_class=100    # Minimum samples per class
    )
    
    all_stats = []
    successful_symbols = []
    
    for symbol in symbols:
        try:
            labels, stats = generator.generate_training_labels(symbol)
            all_stats.append(stats)
            successful_symbols.append(symbol)
            
            logger.info(f"✅ {symbol}: {stats['total_samples']} samples, "
                       f"balance ratio: {stats['balance_ratio']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {symbol}: {e}")
    
    # Save summary
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_path = os.path.join(LABELS_DIR, "training_labels_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"📊 Generated labels for {len(successful_symbols)}/{len(symbols)} symbols")
        logger.info(f"📈 Total samples: {summary_df['total_samples'].sum()}")
        logger.info(f"📊 Average balance ratio: {summary_df['balance_ratio'].mean():.3f}")
        logger.info(f"💾 Summary saved to {summary_path}")
        
        # Print ready-to-train symbols
        print(f"\n🚀 Ready for training: {', '.join(successful_symbols)}")
    else:
        logger.error("❌ No labels generated for any symbol")

if __name__ == "__main__":
    main()

