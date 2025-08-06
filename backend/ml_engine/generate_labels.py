import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OHLCV_DIR = "backend/data/ohlcv"
TRADES_DIR = "backend/storage/trade_history"
LABELS_DIR = "backend/data/labels"

os.makedirs(LABELS_DIR, exist_ok=True)

class TradingLabelGenerator:
    def __init__(self, 
                 forward_window: int = 24,     # Look 24 periods ahead (24h for 1h data)
                 backward_window: int = 0,     # No backward labeling to prevent leakage
                 min_return_threshold: float = 0.015,  # 1.5% minimum return (more lenient for 1h)
                 stop_loss_threshold: float = -0.03,   # -3% stop loss (more lenient)
                 hold_threshold: float = 0.008,        # 0.8% for hold signals (more lenient)
                 timeframe: str = "1h"):               # Default to 1h
        self.forward_window = forward_window
        self.backward_window = backward_window
        self.min_return_threshold = min_return_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.hold_threshold = hold_threshold
        self.timeframe = timeframe

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Load OHLCV data with support for 1h timeframe (prioritize 1h over 30m)"""
        # Try 1h first, then fallback to 30m
        paths_to_try = [
            os.path.join(OHLCV_DIR, f"{symbol}_1h.csv"),
            os.path.join(OHLCV_DIR, f"{symbol}_30m.csv")
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, index_col=0, parse_dates=True)
                    df = df.sort_index()  # Ensure chronological order
                    
                    # Determine actual timeframe from filename
                    if "1h" in path:
                        self.actual_timeframe = "1h"
                        # Use 24 periods for 1h data (24 periods = 24 hours)
                        self.forward_window = 24
                    else:
                        self.actual_timeframe = "30m"
                        # Adjust forward window for 30m data (48 periods = 24 hours)
                        self.forward_window = 48
                    
                    logger.info(f"Loaded {self.actual_timeframe} data for {symbol} ({len(df)} candles)")
                    
                    # Add technical indicators for better labeling
                    df = self._add_technical_indicators(df)
                    return df
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    continue
        
        # If no data found
        raise FileNotFoundError(f"No OHLCV data found for {symbol}")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to help with labeling context (adjusted for timeframe)"""
        # Adjust periods based on timeframe
        if hasattr(self, 'actual_timeframe') and self.actual_timeframe == "1h":
            # For 1h data, use standard periods
            sma_short = 20   # 20 hours
            sma_long = 50    # 50 hours
            volatility_window = 24  # 24 hours
            momentum_period = 6     # 6 hours
        else:
            # For 30m data, use longer periods to get similar time coverage
            sma_short = 40   # 20 hours
            sma_long = 100   # 50 hours
            volatility_window = 48  # 24 hours
            momentum_period = 12    # 6 hours
        
        # Simple moving averages
        df['sma_short'] = df['close'].rolling(sma_short).mean()
        df['sma_long'] = df['close'].rolling(sma_long).mean()
        
        # Volatility (used for dynamic thresholds)
        df['volatility'] = df['close'].pct_change().rolling(volatility_window).std()
        
        # Price momentum
        df['momentum'] = df['close'].pct_change(periods=momentum_period)
        
        return df

    def load_trades(self, symbol: str) -> pd.DataFrame:
        """Load trades with validation"""
        path = os.path.join(TRADES_DIR, f"{symbol}_predictions.json")
        if not os.path.exists(path):
            logger.warning(f"Trades file not found for {symbol}")
            return pd.DataFrame(columns=["timestamp", "signal", "price", "confidence"])

        try:
            trades = pd.read_json(path)
            trades["timestamp"] = pd.to_datetime(trades["timestamp"])
            trades = trades.sort_values("timestamp")
            
            # Validate trade data
            trades = trades[trades["signal"].isin(["buy", "sell", "hold"])]
            return trades
        except Exception as e:
            logger.error(f"Failed to load trades for {symbol}: {e}")
            return pd.DataFrame(columns=["timestamp", "signal", "price", "confidence"])

    def calculate_future_returns(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Calculate forward-looking returns for outcome-based labeling (adjusted for timeframe)"""
        returns_df = ohlcv[['close']].copy()
        
        # Adjust horizons based on timeframe
        if hasattr(self, 'actual_timeframe') and self.actual_timeframe == "1h":
            # For 1h data: 6, 12, 24, 48 periods = 6h, 12h, 24h, 48h
            horizons = [(6, '6h'), (12, '12h'), (24, '24h'), (48, '48h')]
        else:
            # For 30m data: 12, 24, 48, 96 periods = 6h, 12h, 24h, 48h
            horizons = [(12, '6h'), (24, '12h'), (48, '24h'), (96, '48h')]
        
        # Calculate returns at multiple horizons
        for horizon_periods, horizon_name in horizons:
            returns_df[f'return_{horizon_name}'] = (
                ohlcv['close'].shift(-horizon_periods) / ohlcv['close'] - 1
            )
            
            # Calculate max adverse excursion (worst drawdown during period)
            returns_df[f'mae_{horizon_name}'] = (
                ohlcv['low'].rolling(window=horizon_periods).min().shift(-horizon_periods) / ohlcv['close'] - 1
            )
            
            # Calculate maximum favorable excursion (best gain during period)
            returns_df[f'mfe_{horizon_name}'] = (
                ohlcv['high'].rolling(window=horizon_periods).max().shift(-horizon_periods) / ohlcv['close'] - 1
            )
        
        return returns_df

    def generate_outcome_based_labels(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Generate labels based on future price movements (adjusted for 1h timeframe)"""
        returns_df = self.calculate_future_returns(ohlcv)
        labels = pd.Series('hold', index=ohlcv.index)
        
        # Use 24h return as primary signal (main forward window)
        return_col = 'return_24h'
        mae_col = 'mae_24h'
        
        for i in range(len(ohlcv)):
            if i >= len(ohlcv) - self.forward_window:
                labels.iloc[i] = None  # Can't predict at end of data
                continue
                
            # Use volatility-adjusted thresholds
            vol = ohlcv['volatility'].iloc[i] if not pd.isna(ohlcv['volatility'].iloc[i]) else 0.02
            
            # Dynamic thresholds based on volatility and timeframe
            if hasattr(self, 'actual_timeframe') and self.actual_timeframe == "1h":
                # More lenient thresholds for 1h data to get more samples
                buy_threshold = max(self.min_return_threshold * 0.8, vol * 1.5)  # More lenient
                sell_threshold = min(self.stop_loss_threshold * 0.8, -vol * 2)   # More lenient
                hold_threshold = min(self.hold_threshold * 1.2, vol * 1.5)       # More lenient
            else:
                # Standard thresholds for 30m data
                buy_threshold = max(self.min_return_threshold, vol * 1.5)
                sell_threshold = min(self.stop_loss_threshold, -vol * 2.5)
                hold_threshold = min(self.hold_threshold, vol * 0.8)
            
            # Check forward return
            future_return = returns_df[return_col].iloc[i]
            max_adverse = returns_df[mae_col].iloc[i]
            
            if pd.isna(future_return):
                labels.iloc[i] = None
                continue
            
            # Label logic with risk management
            if future_return > buy_threshold and max_adverse > sell_threshold:
                labels.iloc[i] = 'buy'
            elif future_return < sell_threshold:
                labels.iloc[i] = 'sell'
            elif abs(future_return) < hold_threshold:
                labels.iloc[i] = 'hold'
            else:
                # Default to hold for ambiguous cases
                labels.iloc[i] = 'hold'
                
        return labels

    def validate_trade_outcomes(self, ohlcv: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        """Validate historical trades and filter out poor performers (adjusted for timeframe)"""
        validated_trades = []
        
        # Adjust validation window based on timeframe
        validation_periods = self.forward_window  # Use the same forward window
        
        for _, trade in trades.iterrows():
            if trade['signal'] not in ['buy', 'sell']:
                continue
                
            trade_time = trade['timestamp']
            
            # Find closest OHLCV timestamp
            try:
                closest_idx = ohlcv.index.get_indexer([trade_time], method='nearest')[0]
                entry_price = ohlcv['close'].iloc[closest_idx]
                
                # Calculate actual outcome over validation period
                if closest_idx < len(ohlcv) - validation_periods:
                    exit_price = ohlcv['close'].iloc[closest_idx + validation_periods]
                    actual_return = (exit_price / entry_price - 1)
                    
                    # Adjust validation criteria for timeframe
                    min_profitable_return = 0.01 if hasattr(self, 'actual_timeframe') and self.actual_timeframe == "1h" else 0.008
                    
                    # Validate trade quality
                    if trade['signal'] == 'buy' and actual_return > min_profitable_return:  # Profitable buy
                        validated_trades.append({
                            **trade.to_dict(),
                            'actual_return': actual_return,
                            'validated': True
                        })
                    elif trade['signal'] == 'sell' and actual_return < -min_profitable_return:  # Good sell signal
                        validated_trades.append({
                            **trade.to_dict(),
                            'actual_return': actual_return,
                            'validated': True
                        })
                    # Skip trades that didn't work out
                        
            except Exception as e:
                logger.warning(f"Could not validate trade at {trade_time}: {e}")
                continue
                
        return pd.DataFrame(validated_trades)

    def balance_dataset(self, labels: pd.Series) -> pd.Series:
        """Balance the dataset to prevent class imbalance"""
        # Count each class
        value_counts = labels.value_counts()
        
        if len(value_counts) == 0:
            logger.warning("No labels found to balance")
            return labels
            
        min_count = value_counts.min()
        
        logger.info(f"Original distribution: {value_counts.to_dict()}")
        
        # For 1h data, be more conservative with balancing to preserve samples
        if hasattr(self, 'actual_timeframe') and self.actual_timeframe == "1h":
            # Use 80% of minimum count to get more balanced but still sufficient data
            target_count = max(min_count, int(min(value_counts) * 0.8))
        else:
            # Use higher minimum for 30m data
            target_count = max(min_count, int(min(value_counts) * 1.2))
        
        # Don't balance if we have very few samples - just use all available
        if min_count < 50:
            logger.info(f"Very few samples ({min_count}), skipping balancing to preserve data")
            return labels
        
        # Sample each class to match the target count
        balanced_indices = []
        for class_label in value_counts.index:
            class_indices = labels[labels == class_label].index
            sample_count = int(min(len(class_indices), target_count))
            
            if len(class_indices) > sample_count:
                # Randomly sample
                np.random.seed(42)  # For reproducibility
                sampled = np.random.choice(class_indices, sample_count, replace=False)
                balanced_indices.extend(sampled)
            else:
                balanced_indices.extend(class_indices)
        
        balanced_labels = labels[balanced_indices].sort_index()
        logger.info(f"Balanced distribution: {balanced_labels.value_counts().to_dict()}")
        
        return balanced_labels

    def generate_labels(self, symbol: str, method: str = 'hybrid') -> Tuple[pd.DataFrame, Dict]:
        """
        Generate labels using different methods:
        - 'outcome': Pure outcome-based labeling (recommended)
        - 'trade': Historical trade-based labeling  
        - 'hybrid': Combination of both
        """
        logger.info(f"Generating {method} labels for {symbol} ({self.timeframe} timeframe)...")

        try:
            ohlcv = self.load_ohlcv(symbol)
            trades = self.load_trades(symbol)
            
            if method == 'outcome':
                labels = self.generate_outcome_based_labels(ohlcv)
                
            elif method == 'trade':
                # Validate trades first
                validated_trades = self.validate_trade_outcomes(ohlcv, trades)
                labels = self._create_trade_based_labels(ohlcv, validated_trades)
                
            elif method == 'hybrid':
                # Combine both approaches
                outcome_labels = self.generate_outcome_based_labels(ohlcv)
                validated_trades = self.validate_trade_outcomes(ohlcv, trades)
                trade_labels = self._create_trade_based_labels(ohlcv, validated_trades)
                
                # Prefer trade labels where available, outcome labels elsewhere
                labels = outcome_labels.copy()
                labels.update(trade_labels.dropna())
                
            else:
                raise ValueError(f"Unknown method: {method}")

            # Remove None labels and balance
            labels = labels.dropna()
            if len(labels) > 0:
                labels = self.balance_dataset(labels)
            
            # Create features dataframe aligned with labels
            features = ohlcv.loc[labels.index].copy()
            
            # Add label column
            result_df = features.copy()
            result_df['label'] = labels
            
            # Calculate statistics
            stats = self._calculate_label_stats(result_df, symbol)
            
            # Save results
            self._save_results(result_df, symbol, method)
            
            return result_df, stats
            
        except Exception as e:
            logger.error(f"Failed to generate labels for {symbol}: {e}")
            raise

    def _create_trade_based_labels(self, ohlcv: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
        """Create labels based on validated historical trades"""
        labels = pd.Series(dtype=object, index=ohlcv.index)
        
        for _, trade in trades.iterrows():
            if not trade.get('validated', False):
                continue
                
            trade_time = trade['timestamp']
            signal = trade['signal'].lower()
            
            # Find closest index
            try:
                idx = ohlcv.index.get_indexer([trade_time], method='nearest')[0]
                
                # Only label backward to prevent leakage
                start_idx = max(0, idx - self.backward_window)
                end_idx = idx + 1
                
                labels.iloc[start_idx:end_idx] = signal
                
            except Exception as e:
                logger.warning(f"Could not process trade at {trade_time}: {e}")
                continue
        
        return labels

    def _calculate_label_stats(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate comprehensive statistics about the labels"""
        if 'label' not in df.columns or len(df) == 0:
            return {}
            
        distribution = df['label'].value_counts(normalize=True).round(4) * 100
        
        timeframe_info = getattr(self, 'actual_timeframe', self.timeframe)
        
        stats = {
            'symbol': symbol,
            'timeframe': timeframe_info,
            'total_samples': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'distribution': distribution.to_dict(),
            'balance_ratio': distribution.min() / distribution.max() if len(distribution) > 1 else 1.0
        }
        
        # Add return statistics by label
        return_cols = [col for col in df.columns if col.startswith('return_')]
        if return_cols:
            primary_return_col = 'return_24h' if 'return_24h' in df.columns else return_cols[0]
            for label in df['label'].unique():
                if pd.notna(label):
                    subset = df[df['label'] == label][primary_return_col].dropna()
                    if len(subset) > 0:
                        stats[f'{label}_avg_return'] = subset.mean()
                        stats[f'{label}_return_std'] = subset.std()
        
        return stats

    def _save_results(self, df: pd.DataFrame, symbol: str, method: str):
        """Save labeled data and metadata with timeframe indicator"""
        timeframe_info = getattr(self, 'actual_timeframe', self.timeframe)
        
        if 'label' in df.columns:
            df_to_save = df.copy()
            df_to_save['action'] = df_to_save['label']  # Add action column for training compatibility
    
        # Save main dataset with timeframe in filename
        output_path = os.path.join(LABELS_DIR, f"{symbol}_{timeframe_info}_{method}_labels.csv") 
        df_to_save.to_csv(output_path)
        logger.info(f"Saved {len(df_to_save)} labeled samples to {output_path}")
    
        # Save feature-only version (without price data that could cause leakage)
        feature_cols = [col for col in df_to_save.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        if feature_cols:
            features_path = os.path.join(LABELS_DIR, f"{symbol}_{timeframe_info}_{method}_features.csv")
            df_to_save[feature_cols].to_csv(features_path)
            logger.info(f"Saved features to {features_path}")
    
        # Save a training-ready dataset with just features and action
        training_cols = [col for col in df_to_save.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'label']]
        if 'action' in df_to_save.columns and len(training_cols) > 1:
            training_path = os.path.join(LABELS_DIR, f"{symbol}_{timeframe_info}_{method}_training.csv")
            df_to_save[training_cols].to_csv(training_path)
            logger.info(f"Saved training dataset to {training_path}")
            
        # BACKWARD COMPATIBILITY: Also save with original naming for existing code
        if timeframe_info == "1h":
            legacy_path = os.path.join(LABELS_DIR, f"{symbol}_{method}_labels.csv")
            df_to_save.to_csv(legacy_path)
            logger.info(f"Saved legacy-compatible labels to {legacy_path}")

def main():
    """Generate labels for all trading pairs with 1h timeframe preference"""
    symbols = [
        "DOGEUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT",
        "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"
    ]
    
    # Create generator with 1h-optimized parameters (more lenient for better sample count)
    generator = TradingLabelGenerator(
        forward_window=24,           # 24 hours for 1h data
        min_return_threshold=0.015,  # 1.5% minimum return (more lenient)
        stop_loss_threshold=-0.03,   # -3% stop loss (more lenient)
        hold_threshold=0.008,        # 0.8% for hold signals (more lenient)
        timeframe="1h"
    )
    
    all_stats = []
    
    for symbol in symbols:
        try:
            # Try outcome-based method first (recommended)
            df, stats = generator.generate_labels(symbol, method='outcome')
            all_stats.append(stats)
            
            logger.info(f"✅ {symbol} ({stats.get('timeframe', '1h')}): {stats['total_samples']} samples, balance ratio: {stats['balance_ratio']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {symbol}: {e}")
    
    # Save summary statistics
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_path = os.path.join(LABELS_DIR, "1h_labeling_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"📊 Saved 1h summary statistics to {summary_path}")
        
        # Print overall statistics
        total_samples = summary_df['total_samples'].sum()
        avg_balance = summary_df['balance_ratio'].mean()
        timeframes_used = summary_df['timeframe'].value_counts()
        
        logger.info(f"📈 Total samples across all symbols: {total_samples}")
        logger.info(f"📊 Average balance ratio: {avg_balance:.3f}")
        logger.info(f"📋 Timeframes used: {timeframes_used.to_dict()}")

if __name__ == "__main__":
    main()