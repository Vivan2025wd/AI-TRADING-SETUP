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

class TradingLabelGenerator:
    def __init__(self, 
                 forward_window: int = 24,  # Look 24h ahead for outcomes
                 backward_window: int = 0,   # No backward labeling to prevent leakage
                 min_return_threshold: float = 0.02,  # 2% minimum return
                 stop_loss_threshold: float = -0.05,  # -5% stop loss
                 hold_threshold: float = 0.005):      # 0.5% for hold signals
        self.forward_window = forward_window
        self.backward_window = backward_window
        self.min_return_threshold = min_return_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.hold_threshold = hold_threshold

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Load OHLCV data with better error handling"""
        path = os.path.join(OHLCV_DIR, f"{symbol}_1h.csv")
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.sort_index()  # Ensure chronological order
            
            # Add technical indicators for better labeling
            df = self._add_technical_indicators(df)
            return df
        except Exception as e:
            logger.error(f"Failed to load OHLCV for {symbol}: {e}")
            raise

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to help with labeling context"""
        # Simple moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Volatility (used for dynamic thresholds)
        df['volatility'] = df['close'].pct_change().rolling(24).std()
        
        # Price momentum
        df['momentum'] = df['close'].pct_change(periods=6)
        
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
        """Calculate forward-looking returns for outcome-based labeling"""
        returns_df = ohlcv[['close']].copy()
        
        # Calculate returns at multiple horizons
        for horizon in [6, 12, 24, 48]:  # 6h, 12h, 24h, 48h
            returns_df[f'return_{horizon}h'] = (
                ohlcv['close'].shift(-horizon) / ohlcv['close'] - 1
            )
            
            # Calculate max adverse excursion (worst drawdown during period)
            returns_df[f'mae_{horizon}h'] = (
                ohlcv['low'].rolling(window=horizon).min().shift(-horizon) / ohlcv['close'] - 1
            )
            
            # Calculate maximum favorable excursion (best gain during period)
            returns_df[f'mfe_{horizon}h'] = (
                ohlcv['high'].rolling(window=horizon).max().shift(-horizon) / ohlcv['close'] - 1
            )
        
        return returns_df

    def generate_outcome_based_labels(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Generate labels based on future price movements (more robust than trade timing)"""
        returns_df = self.calculate_future_returns(ohlcv)
        labels = pd.Series('hold', index=ohlcv.index)
        
        for i in range(len(ohlcv)):
            if i >= len(ohlcv) - self.forward_window:
                labels.iloc[i] = None  # Can't predict at end of data
                continue
                
            # Use volatility-adjusted thresholds
            vol = ohlcv['volatility'].iloc[i] if not pd.isna(ohlcv['volatility'].iloc[i]) else 0.02
            
            # Dynamic thresholds based on volatility
            buy_threshold = max(self.min_return_threshold, vol * 2)
            sell_threshold = min(self.stop_loss_threshold, -vol * 3)
            hold_threshold = min(self.hold_threshold, vol)
            
            # Check 24h forward return
            future_return = returns_df[f'return_24h'].iloc[i]
            max_adverse = returns_df[f'mae_24h'].iloc[i]
            
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
        """Validate historical trades and filter out poor performers"""
        validated_trades = []
        
        for _, trade in trades.iterrows():
            if trade['signal'] not in ['buy', 'sell']:
                continue
                
            trade_time = trade['timestamp']
            
            # Find closest OHLCV timestamp
            try:
                closest_idx = ohlcv.index.get_indexer([trade_time], method='nearest')[0]
                entry_price = ohlcv['close'].iloc[closest_idx]
                
                # Calculate actual outcome over next 24 hours
                if closest_idx < len(ohlcv) - 24:
                    exit_price = ohlcv['close'].iloc[closest_idx + 24]
                    actual_return = (exit_price / entry_price - 1)
                    
                    # Validate trade quality
                    if trade['signal'] == 'buy' and actual_return > 0.01:  # Profitable buy
                        validated_trades.append({
                            **trade.to_dict(),
                            'actual_return': actual_return,
                            'validated': True
                        })
                    elif trade['signal'] == 'sell' and actual_return < -0.01:  # Good sell signal
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
        min_count = value_counts.min()
        
        logger.info(f"Original distribution: {value_counts.to_dict()}")
        
        # Sample each class to match the minority class
        balanced_indices = []
        for class_label in value_counts.index:
            class_indices = labels[labels == class_label].index
            if len(class_indices) > min_count:
                # Randomly sample
                np.random.seed(42)  # For reproducibility
                sampled = np.random.choice(class_indices, min_count, replace=False)
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
        logger.info(f"Generating {method} labels for {symbol}...")

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
        
        stats = {
            'symbol': symbol,
            'total_samples': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'distribution': distribution.to_dict(),
            'balance_ratio': distribution.min() / distribution.max() if len(distribution) > 1 else 1.0
        }
        
        # Add return statistics by label
        if 'return_24h' in df.columns:
            for label in df['label'].unique():
                if pd.notna(label):
                    subset = df[df['label'] == label]['return_24h'].dropna()
                    if len(subset) > 0:
                        stats[f'{label}_avg_return'] = subset.mean()
                        stats[f'{label}_return_std'] = subset.std()
        
        return stats

    def _save_results(self, df: pd.DataFrame, symbol: str, method: str):
        """Save labeled data and metadata"""
        # Save main dataset
        output_path = os.path.join(LABELS_DIR, f"{symbol}_{method}_labels.csv")
        df.to_csv(output_path)
        logger.info(f"Saved {len(df)} labeled samples to {output_path}")
        
        # Save feature-only version (without price data that could cause leakage)
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        if feature_cols:
            features_path = os.path.join(LABELS_DIR, f"{symbol}_{method}_features.csv")
            df[feature_cols].to_csv(features_path)
            logger.info(f"Saved features to {features_path}")

def main():
    """Generate labels for all trading pairs"""
    symbols = [
        "DOGEUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT",
        "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"
    ]
    
    generator = TradingLabelGenerator(
        forward_window=24,
        min_return_threshold=0.025,  # 2.5% minimum return
        stop_loss_threshold=-0.04,   # -4% stop loss
        hold_threshold=0.008         # 0.8% for hold signals
    )
    
    all_stats = []
    
    for symbol in symbols:
        try:
            # Try outcome-based method first (recommended)
            df, stats = generator.generate_labels(symbol, method='outcome')
            all_stats.append(stats)
            
            logger.info(f"✅ {symbol}: {stats['total_samples']} samples, balance ratio: {stats['balance_ratio']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {symbol}: {e}")
    
    # Save summary statistics
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_path = os.path.join(LABELS_DIR, "labeling_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"📊 Saved summary statistics to {summary_path}")
        
        # Print overall statistics
        total_samples = summary_df['total_samples'].sum()
        avg_balance = summary_df['balance_ratio'].mean()
        logger.info(f"📈 Total samples across all symbols: {total_samples}")
        logger.info(f"📊 Average balance ratio: {avg_balance:.3f}")

if __name__ == "__main__":
    main()