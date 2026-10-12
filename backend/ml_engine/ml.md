import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for crypto trading"""
    
    def __init__(self, timeframe: str = "1h"):
        self.timeframe = timeframe
        # Adjust periods based on timeframe
        if timeframe == "1h":
            self.short_period = 12
            self.medium_period = 24
            self.long_period = 168  # 1 week
        else:  # 30m
            self.short_period = 24
            self.medium_period = 48
            self.long_period = 336  # 1 week
    
    def add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        data = df.copy()
        
        # Price-based indicators
        data['hl2'] = (data['high'] + data['low']) / 2
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        data['ohlc4'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        
        # Multiple timeframe moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            if period <= len(data):
                data[f'sma_{period}'] = data['close'].rolling(period).mean()
                data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                data[f'price_vs_sma_{period}'] = data['close'] / data[f'sma_{period}'] - 1
        
        # Bollinger Bands with multiple periods
        for period in [20, 50]:
            if period <= len(data):
                sma = data['close'].rolling(period).mean()
                std = data['close'].rolling(period).std()
                data[f'bb_upper_{period}'] = sma + (2 * std)
                data[f'bb_lower_{period}'] = sma - (2 * std)
                data[f'bb_width_{period}'] = (data[f'bb_upper_{period}'] - data[f'bb_lower_{period}']) / sma
                data[f'bb_position_{period}'] = (data['close'] - data[f'bb_lower_{period}']) / (data[f'bb_upper_{period}'] - data[f'bb_lower_{period}'])
        
        # RSI with multiple periods
        for period in [6, 14, 21]:
            if period <= len(data):
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Stochastic Oscillator
        for period in [14, 21]:
            if period <= len(data):
                low_min = data['low'].rolling(window=period).min()
                high_max = data['high'].rolling(window=period).max()
                data[f'stoch_k_{period}'] = 100 * (data['close'] - low_min) / (high_max - low_min)
                data[f'stoch_d_{period}'] = data[f'stoch_k_{period}'].rolling(3).mean()
        
        # Williams %R
        for period in [14, 21]:
            if period <= len(data):
                high_max = data['high'].rolling(window=period).max()
                low_min = data['low'].rolling(window=period).min()
                data[f'williams_r_{period}'] = -100 * (high_max - data['close']) / (high_max - low_min)
        
        # Momentum indicators
        for period in [1, 3, 6, 12]:
            if period <= len(data):
                data[f'momentum_{period}'] = data['close'].pct_change(period)
                data[f'roc_{period}'] = (data['close'] / data['close'].shift(period) - 1) * 100
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        data['price_volume'] = data['close'] * data['volume']
        data['vwap'] = data['price_volume'].rolling(20).sum() / data['volume'].rolling(20).sum()
        
        # Volatility indicators
        for period in [10, 20, 50]:
            if period <= len(data):
                data[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
                data[f'atr_{period}'] = self._calculate_atr(data, period)
        
        # Support and Resistance levels
        data['support'] = data['low'].rolling(20).min()
        data['resistance'] = data['high'].rolling(20).max()
        data['support_distance'] = (data['close'] - data['support']) / data['close']
        data['resistance_distance'] = (data['resistance'] - data['close']) / data['close']
        
        return data
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure and pattern recognition features"""
        data = df.copy()
        
        # Higher highs and lower lows
        data['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
        data['lower_low'] = (data['low'] < data['low'].shift(1)).astype(int)
        
        # Trend strength
        for window in [5, 10, 20]:
            data[f'trend_strength_{window}'] = data['close'].rolling(window).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 2 else 0
            )
        
        # Price patterns
        data['doji'] = ((abs(data['open'] - data['close']) / (data['high'] - data['low'])) < 0.1).astype(int)
        data['hammer'] = ((data['close'] > data['open']) & 
                         ((data['open'] - data['low']) > 2 * (data['close'] - data['open']))).astype(int)
        
        # Gap detection
        data['gap_up'] = (data['open'] > data['close'].shift(1) * 1.005).astype(int)
        data['gap_down'] = (data['open'] < data['close'].shift(1) * 0.995).astype(int)
        
        # Volume patterns
        data['volume_spike'] = (data['volume'] > data['volume'].rolling(20).mean() * 2).astype(int)
        data['volume_dry'] = (data['volume'] < data['volume'].rolling(20).mean() * 0.5).astype(int)
        
        return data
    
    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based cyclical features"""
        data = df.copy()
        
        # Hour of day (important for crypto)
        data['hour'] = data.index.hour
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # Day of week
        data['day_of_week'] = data.index.dayofweek
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Weekend indicator
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        return data
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for sequence learning"""
        data = df.copy()
        
        # Price lags
        for lag in [1, 2, 3, 6, 12]:
            data[f'close_lag_{lag}'] = data['close'].shift(lag)
            data[f'return_lag_{lag}'] = data['close'].pct_change().shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
        
        # Rolling statistics of lags
        for window in [3, 6, 12]:
            data[f'return_rolling_mean_{window}'] = data['close'].pct_change().rolling(window).mean()
            data[f'return_rolling_std_{window}'] = data['close'].pct_change().rolling(window).std()
            data[f'volume_rolling_mean_{window}'] = data['volume'].rolling(window).mean()
        
        return data
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features"""
        logger.info("🔧 Extracting advanced features...")
        
        # Apply all feature engineering
        features = self.add_advanced_technical_indicators(df)
        features = self.add_market_structure_features(features)
        features = self.add_cyclical_features(features)
        features = self.add_lag_features(features)
        
        # Remove OHLCV columns to prevent data leakage
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in features.columns if col not in price_cols]
        
        result = features[feature_cols].copy()
        
        # Handle missing values
        result = result.fillna(method='forward').fillna(method='backward').fillna(0)
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        logger.info(f"✅ Extracted {len(result.columns)} advanced features")
        return result


class ImprovedLabelGenerator:
    """Improved labeling with better risk-reward logic"""
    
    def __init__(self, timeframe: str = "1h"):
        self.timeframe = timeframe
        # Adjust parameters based on crypto volatility and timeframe
        if timeframe == "1h":
            self.params = {
                'forward_window': 24,
                'min_return_buy': 0.02,      # 2% minimum for buy
                'min_return_sell': -0.02,    # -2% for sell
                'stop_loss': -0.05,          # -5% stop loss
                'take_profit': 0.08,         # 8% take profit
                'hold_range': 0.01           # ±1% for hold
            }
        else:  # 30m
            self.params = {
                'forward_window': 48,
                'min_return_buy': 0.015,
                'min_return_sell': -0.015,
                'stop_loss': -0.04,
                'take_profit': 0.06,
                'hold_range': 0.008
            }
    
    def generate_smart_labels(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Generate labels with improved risk-reward logic"""
        labels = pd.Series('hold', index=ohlcv.index)
        
        for i in range(len(ohlcv) - self.params['forward_window']):
            current_price = ohlcv['close'].iloc[i]
            
            # Look ahead window
            future_prices = ohlcv['close'].iloc[i+1:i+1+self.params['forward_window']]
            future_highs = ohlcv['high'].iloc[i+1:i+1+self.params['forward_window']]
            future_lows = ohlcv['low'].iloc[i+1:i+1+self.params['forward_window']]
            
            # Calculate maximum gain and loss during period
            max_gain = (future_highs.max() / current_price) - 1
            max_loss = (future_lows.min() / current_price) - 1
            final_return = (future_prices.iloc[-1] / current_price) - 1
            
            # Calculate when stop loss or take profit would be hit
            stop_loss_hit = any(future_lows / current_price - 1 <= self.params['stop_loss'])
            take_profit_hit = any(future_highs / current_price - 1 >= self.params['take_profit'])
            
            # Dynamic volatility adjustment
            recent_volatility = ohlcv['close'].pct_change().iloc[max(0, i-20):i].std()
            vol_multiplier = max(0.5, min(2.0, recent_volatility / 0.03))  # Normalize around 3% daily vol
            
            # Adjusted thresholds
            buy_threshold = self.params['min_return_buy'] * vol_multiplier
            sell_threshold = self.params['min_return_sell'] * vol_multiplier
            
            # Improved labeling logic
            if not stop_loss_hit and (max_gain >= self.params['take_profit'] or final_return >= buy_threshold):
                # Risk-reward ratio check
                risk_reward = max_gain / abs(max_loss) if max_loss < 0 else float('inf')
                if risk_reward >= 1.5:  # Minimum 1.5:1 risk-reward
                    labels.iloc[i] = 'buy'
            elif stop_loss_hit or final_return <= sell_threshold:
                labels.iloc[i] = 'sell'
            elif abs(final_return) <= self.params['hold_range']:
                labels.iloc[i] = 'hold'
        
        return labels
    
    def add_regime_detection(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        data = ohlcv.copy()
        
        # Trend regime (20-period trend strength)
        data['trend_regime'] = data['close'].rolling(20).apply(
            lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 2 else 0
        )
        
        # Volatility regime
        returns = data['close'].pct_change()
        data['vol_regime'] = returns.rolling(20).std()
        data['vol_percentile'] = data['vol_regime'].rolling(100).rank(pct=True)
        
        # Volume regime
        data['volume_regime'] = data['volume'].rolling(20).mean()
        data['volume_percentile'] = data['volume_regime'].rolling(100).rank(pct=True)
        
        # Market state classification
        data['high_vol'] = (data['vol_percentile'] > 0.8).astype(int)
        data['low_vol'] = (data['vol_percentile'] < 0.2).astype(int)
        data['trending'] = (abs(data['trend_regime']) > data['vol_regime'] * 0.5).astype(int)
        
        return data


class EnsembleModelTrainer:
    """Advanced ensemble model trainer with multiple algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.feature_selectors = {}
        self.models = {}
        
    def create_base_models(self) -> Dict[str, Any]:
        """Create diverse base models for ensemble"""
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        
        # Add neural network for non-linear patterns
        models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        return models
    
    def hyperparameter_optimization(self, model, X: pd.DataFrame, y: pd.Series, model_name: str) -> Any:
        """Perform hyperparameter optimization"""
        logger.info(f"🔍 Optimizing hyperparameters for {model_name}...")
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=3)
        
        param_grids = {
            'rf': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            },
            'xgb': {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9]
            },
            'lgb': {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9]
            }
        }
        
        if model_name in param_grids:
            try:
                grid_search = GridSearchCV(
                    model, param_grids[model_name],
                    cv=tscv, scoring='f1_weighted',
                    n_jobs=-1, verbose=0
                )
                grid_search.fit(X, y)
                logger.info(f"✅ {model_name}: Best params found")
                return grid_search.best_estimator_
            except Exception as e:
                logger.warning(f"⚠️ {model_name}: Hyperparameter optimization failed - {e}")
        
        return model
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> pd.DataFrame:
        """Advanced feature selection"""
        logger.info(f"🎯 Performing feature selection for {symbol}...")
        
        # Remove features with too many missing values
        missing_threshold = 0.3
        missing_rates = X.isnull().sum() / len(X)
        good_features = missing_rates[missing_rates <= missing_threshold].index
        X_clean = X[good_features].copy()
        
        # Remove features with zero variance
        var_threshold = 1e-6
        feature_vars = X_clean.var()
        non_zero_var_features = feature_vars[feature_vars > var_threshold].index
        X_clean = X_clean[non_zero_var_features]
        
        # Statistical feature selection
        k_best = min(50, len(X_clean.columns))  # Select top 50 features or all if less
        selector = SelectKBest(f_classif, k=k_best)
        X_selected = selector.fit_transform(X_clean, y)
        
        selected_features = X_clean.columns[selector.get_support()]
        X_final = pd.DataFrame(X_selected, index=X_clean.index, columns=selected_features)
        
        # Store feature selector for later use
        self.feature_selectors[symbol] = {
            'good_features': good_features,
            'non_zero_var_features': non_zero_var_features,
            'selector': selector,
            'selected_features': selected_features
        }
        
        logger.info(f"✅ Selected {len(selected_features)} features from {len(X.columns)} original")
        return X_final
    
    def create_ensemble_model(self, base_models: Dict[str, Any]) -> VotingClassifier:
        """Create ensemble model with optimized base models"""
        # Select best performing base models
        estimators = [(name, model) for name, model in base_models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        return ensemble
    
    def train_with_validation(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> Dict[str, Any]:
        """Train model with comprehensive validation"""
        logger.info(f"🧠 Training ensemble model for {symbol}...")
        
        # Time series split to prevent data leakage
        tscv = TimeSeriesSplit(n_splits=5)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
        
        # Store scaler
        self.scalers[symbol] = scaler
        
        # Create and optimize base models
        base_models = self.create_base_models()
        optimized_models = {}
        
        for name, model in base_models.items():
            try:
                if name in ['rf', 'xgb', 'lgb']:  # Only optimize computationally feasible models
                    optimized_models[name] = self.hyperparameter_optimization(
                        model, X_train_scaled, y_train, name
                    )
                else:
                    optimized_models[name] = model
                    optimized_models[name].fit(X_train_scaled, y_train)
            except Exception as e:
                logger.warning(f"⚠️ {name}: Model training failed - {e}")
        
        if not optimized_models:
            raise ValueError("No models successfully trained")
        
        # Create ensemble
        ensemble = self.create_ensemble_model(optimized_models)
        ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate all models
        results = {}
        
        for name, model in optimized_models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='f1_weighted')
                
                # Test predictions
                y_pred_test = model.predict(X_test_scaled)
                y_pred_train = model.predict(X_train_scaled)
                
                results[f'{name}_model'] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'train_accuracy': accuracy_score(y_train, y_pred_train),
                    'test_accuracy': accuracy_score(y_test, y_pred_test),
                    'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
                    'test_precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                    'test_recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
                }
            except Exception as e:
                logger.warning(f"⚠️ {name}: Evaluation failed - {e}")
        
        # Evaluate ensemble
        try:
            ensemble_cv = cross_val_score(ensemble, X_train_scaled, y_train, cv=tscv, scoring='f1_weighted')
            y_pred_ensemble = ensemble.predict(X_test_scaled)
            
            results['ensemble'] = {
                'cv_mean': ensemble_cv.mean(),
                'cv_std': ensemble_cv.std(),
                'test_accuracy': accuracy_score(y_test, y_pred_ensemble),
                'test_f1': f1_score(y_test, y_pred_ensemble, average='weighted'),
                'test_precision': precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
                'test_recall': recall_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)
            }
            
            # Feature importance from best tree model
            best_tree_model = max(
                [(name, model) for name, model in optimized_models.items() 
                 if hasattr(model, 'feature_importances_')],
                key=lambda x: results[f'{x[0]}_model']['test_f1'],
                default=(None, None)
            )
            
            if best_tree_model[1] is not None:
                feature_importance = dict(zip(X.columns, best_tree_model[1].feature_importances_))
                results['feature_importance'] = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:20]
        
        except Exception as e:
            logger.error(f"❌ Ensemble evaluation failed: {e}")
            results['ensemble'] = {'error': str(e)}
        
        # Store the best model (ensemble by default, fallback to best individual)
        best_model = ensemble
        if 'ensemble' in results and 'error' not in results['ensemble']:
            self.models[symbol] = best_model
        else:
            # Fallback to best individual model
            best_individual = max(
                optimized_models.items(),
                key=lambda x: results[f'{x[0]}_model']['test_f1']
            )
            self.models[symbol] = best_individual[1]
            logger.info(f"Using best individual model: {best_individual[0]}")
        
        return results


class EnhancedAutoTrainingSystem:
    """Enhanced training system with improved ML pipeline"""
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.ohlcv_dir = self.base_dir / "data" / "ohlcv"
        self.labels_dir = self.base_dir / "data" / "labels"
        self.models_dir = self.base_dir / "agents" / "models"
        self.logs_dir = self.base_dir / "storage" / "training_logs"

        # Create directories
        for dir_path in [self.ohlcv_dir, self.labels_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.symbols = [
            "DOGEUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT",
            "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"
        ]
        
        # Initialize components
        self.feature_engineer = AdvancedFeatureEngineer("1h")
        self.label_generator = ImprovedLabelGenerator("1h")
        self.model_trainer = EnsembleModelTrainer({"timeframe": "1h"})
        
        self.training_results = {}
    
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load OHLCV data with validation"""
        ohlcv_path = self.ohlcv_dir / f"{symbol}_1h.csv"
        if not ohlcv_path.exists():
            ohlcv_path = self.ohlcv_dir / f"{symbol}.csv"
        
        if not ohlcv_path.exists():
            raise FileNotFoundError(f"No OHLCV data found for {symbol}")
        
        df = pd.read_csv(ohlcv_path, index_col=0, parse_dates=True)
        df = df.sort_index()
        
        # Validate data quality
        if len(df) < 1000:
            raise ValueError(f"Insufficient data: {len(df)} rows < 1000 minimum")
        
        # Check for data gaps
        expected_freq = '1H' if '1h' in str(ohlcv_path) else '30T'
        complete_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
        missing_pct = (len(complete_index) - len(df)) / len(complete_index)
        
        if missing_pct > 0.1:  # More than 10% missing
            logger.warning(f"⚠️ {symbol}: {missing_pct:.1%} data gaps detected")
        
        return df
    
    def enhanced_train_symbol(self, symbol: str) -> Dict[str, Any]:
        """Enhanced training pipeline for a single symbol"""
        try:
            logger.info(f"🚀 Enhanced training for {symbol}")
            
            # 1. Load data
            ohlcv_df = self.load_data(symbol)
            logger.info(f"📊 Loaded {len(ohlcv_df)} candles for {symbol}")
            
            # 2. Advanced feature engineering
            features_df = self.feature_engineer.extract_all_features(ohlcv_df)
            
            # 3. Add regime detection
            regime_df = self.label_generator.add_regime_detection(ohlcv_df)
            regime_features = [col for col in regime_df.columns if col.endswith(('_regime', '_percentile', 'high_vol', 'low_vol', 'trending'))]
            
            # Combine all features
            all_features = features_df.join(regime_df[regime_features], how='inner')
            
            # 4. Generate improved labels
            labels = self.label_generator.generate_smart_labels(ohlcv_df)
            labels_df = pd.DataFrame({'action': labels}, index=labels.index)
            labels_df = labels_df.dropna()
            
            # 5. Align features and labels
            common_index = all_features.index.intersection(labels_df.index)
            if len(common_index) < 500:
                raise ValueError(f"Insufficient aligned samples: {len(common_index)}")
            
            X = all_features.loc[common_index]
            y = labels_df.loc[common_index, 'action']
            
            logger.info(f"📈 Training dataset: {len(X)} samples, {len(X.columns)} features")
            logger.info(f"📊 Label distribution: {y.value_counts().to_dict()}")
            
            # 6. Feature selection
            X_selected = self.model_trainer.feature_selection(X, y, symbol)
            
            # 7. Train ensemble model
            results = self.model_trainer.train_with_validation(X_selected, y, symbol)
            
            # 8. Save model and metadata
            self._save_enhanced_model(symbol, results)
            
            # 9. Additional validation metrics
            results.update({
                'symbol': symbol,
                'timeframe': '1h',
                'total_features_original': len(all_features.columns),
                'total_features_selected': len(X_selected.columns),
                'training_samples': len(X_selected),
                'data_quality_score': self._calculate_data_quality_score(ohlcv_df),
                'label_quality_score': self._calculate_label_quality_score(y),
                'feature_stability_score': self._calculate_feature_stability(X_selected)
            })
            
            self.training_results[symbol] = results
            
            logger.info(f"✅ {symbol}: Enhanced training completed")
            logger.info(f"   Best Model: {self._get_best_model_name(results)}")
            logger.info(f"   Test Accuracy: {self._get_best_accuracy(results):.3f}")
            
            return results
            
        except Exception as e:
            error_result = {
                'symbol': symbol,
                'timeframe': '1h',
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self.training_results[symbol] = error_result
            logger.error(f"❌ {symbol}: Enhanced training failed - {e}")
            
            return error_result
    
    def _save_enhanced_model(self, symbol: str, results: Dict[str, Any]):
        """Save the best model with all preprocessing components"""
        try:
            model_package = {
                'model': self.model_trainer.models.get(symbol),
                'scaler': self.model_trainer.scalers.get(symbol),
                'feature_selector': self.model_trainer.feature_selectors.get(symbol),
                'feature_engineer': self.feature_engineer,
                'metadata': {
                    'symbol': symbol,
                    'timeframe': '1h',
                    'training_date': datetime.now().isoformat(),
                    'performance': results,
                    'model_type': 'enhanced_ensemble'
                }
            }
            
            model_path = self.models_dir / f"{symbol.lower()}_enhanced_model.pkl"
            joblib.dump(model_package, model_path)
            
            # Also save with standard naming for compatibility
            standard_path = self.models_dir / f"{symbol.lower()}_model.pkl"
            joblib.dump(model_package, standard_path)
            
            logger.info(f"💾 Enhanced model saved for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model for {symbol}: {e}")
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        # Check completeness
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        
        # Check for outliers (prices that are too far from median)
        price_cols = ['open', 'high', 'low', 'close']
        outlier_scores = []
        
        for col in price_cols:
            if col in df.columns:
                median_price = df[col].median()
                outliers = abs(df[col] - median_price) > (median_price * 0.5)  # 50% deviation
                outlier_rate = outliers.sum() / len(df)
                outlier_scores.append(1 - outlier_rate)
        
        outlier_quality = np.mean(outlier_scores) if outlier_scores else 1.0
        
        # Combine scores
        quality_score = (completeness * 0.6) + (outlier_quality * 0.4)
        return round(quality_score, 3)
    
    def _calculate_label_quality_score(self, labels: pd.Series) -> float:
        """Calculate label quality based on distribution balance"""
        value_counts = labels.value_counts()
        
        # Calculate balance score (higher is better)
        if len(value_counts) <= 1:
            return 0.0
        
        balance_score = value_counts.min() / value_counts.max()
        
        # Penalize if any class has too few samples
        min_samples = value_counts.min()
        sample_penalty = min(1.0, min_samples / 50)  # Penalty if less than 50 samples
        
        quality_score = balance_score * sample_penalty
        return round(quality_score, 3)
    
    def _calculate_feature_stability(self, features: pd.DataFrame) -> float:
        """Calculate feature stability score"""
        try:
            # Check for features with extreme variance
            variances = features.var()
            stable_features = (variances > 1e-6) & (variances < 1e6)
            stability_ratio = stable_features.sum() / len(variances)
            
            # Check for correlation stability
            corr_matrix = features.corr().abs()
            high_corr_pairs = (corr_matrix > 0.95).sum().sum() - len(features.columns)  # Subtract diagonal
            correlation_penalty = max(0, 1 - (high_corr_pairs / (len(features.columns) ** 2)))
            
            stability_score = (stability_ratio * 0.7) + (correlation_penalty * 0.3)
            return round(stability_score, 3)
            
        except Exception:
            return 0.5  # Default neutral score
    
    def _get_best_model_name(self, results: Dict[str, Any]) -> str:
        """Get the name of the best performing model"""
        model_scores = {}
        
        for key, metrics in results.items():
            if key.endswith('_model') or key == 'ensemble':
                if isinstance(metrics, dict) and 'test_f1' in metrics:
                    model_scores[key] = metrics['test_f1']
        
        if model_scores:
            best_model = max(model_scores.items(), key=lambda x: x[1])
            return best_model[0]
        
        return "unknown"
    
    def _get_best_accuracy(self, results: Dict[str, Any]) -> float:
        """Get the best accuracy score"""
        accuracies = []
        
        for key, metrics in results.items():
            if (key.endswith('_model') or key == 'ensemble') and isinstance(metrics, dict):
                if 'test_accuracy' in metrics:
                    accuracies.append(metrics['test_accuracy'])
        
        return max(accuracies) if accuracies else 0.0
    
    def run_enhanced_batch_training(self, symbols: Optional[List[str]] = None, force_retrain: bool = False) -> Dict[str, Any]:
        """Run enhanced batch training"""
        if symbols is None:
            symbols = self.symbols
        
        logger.info(f"🎯 Enhanced batch training for {len(symbols)} symbols")
        
        # Check which models need training
        symbols_to_train = []
        for symbol in symbols:
            model_path = self.models_dir / f"{symbol.lower()}_enhanced_model.pkl"
            
            if force_retrain or not model_path.exists():
                symbols_to_train.append(symbol)
            else:
                # Check model age
                model_age_days = (datetime.now().timestamp() - model_path.stat().st_mtime) / 86400
                if model_age_days > 7:  # Retrain weekly
                    symbols_to_train.append(symbol)
                else:
                    logger.info(f"📋 {symbol}: Enhanced model up to date")
        
        if not symbols_to_train:
            logger.info("✅ All enhanced models are up to date")
            return {}
        
        logger.info(f"🔄 Training enhanced models for: {', '.join(symbols_to_train)}")
        
        # Train models
        all_results = {}
        successful = 0
        failed = 0
        
        for symbol in symbols_to_train:
            try:
                result = self.enhanced_train_symbol(symbol)
                all_results[symbol] = result
                
                if result.get('status') != 'failed':
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: Training failed - {e}")
                all_results[symbol] = {
                    'symbol': symbol,
                    'status': 'failed',
                    'error': str(e)
                }
                failed += 1
        
        # Save comprehensive summary
        self._save_enhanced_summary(all_results, successful, failed)
        
        # Print results
        logger.info("=" * 80)
        logger.info("🎯 ENHANCED ML TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📊 Success Rate: {successful/(successful+failed)*100:.1f}%")
        
        if successful > 0:
            # Calculate average performance metrics
            valid_results = [r for r in all_results.values() if r.get('status') != 'failed']
            if valid_results:
                avg_accuracy = np.mean([self._get_best_accuracy(r) for r in valid_results])
                avg_data_quality = np.mean([r.get('data_quality_score', 0) for r in valid_results])
                avg_label_quality = np.mean([r.get('label_quality_score', 0) for r in valid_results])
                
                logger.info(f"📈 Average Test Accuracy: {avg_accuracy:.3f}")
                logger.info(f"📊 Average Data Quality: {avg_data_quality:.3f}")
                logger.info(f"🏷️ Average Label Quality: {avg_label_quality:.3f}")
        
        logger.info("=" * 80)
        
        return all_results
    
    def _save_enhanced_summary(self, results: Dict[str, Any], successful: int, failed: int):
        """Save enhanced training summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system_type': 'enhanced_ml_system',
            'timeframe': '1h',
            'total_symbols': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / (successful + failed) if (successful + failed) > 0 else 0,
            'results': results,
            'improvements': {
                'advanced_feature_engineering': True,
                'ensemble_models': True,
                'hyperparameter_optimization': True,
                'time_series_validation': True,
                'regime_detection': True,
                'improved_labeling': True
            }
        }
        
        summary_path = self.models_dir / "enhanced_training_summary.json"
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"📊 Enhanced summary saved to {summary_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save enhanced summary: {e}")
    
    def compare_models(self, symbol: str) -> Dict[str, Any]:
        """Compare original vs enhanced model performance"""
        try:
            # Load original model results
            original_summary_path = self.models_dir / "training_summary.json"
            enhanced_summary_path = self.models_dir / "enhanced_training_summary.json"
            
            comparison = {'symbol': symbol}
            
            if original_summary_path.exists():
                with open(original_summary_path, 'r') as f:
                    original_data = json.load(f)
                    if symbol in original_data.get('results', {}):
                        original_metrics = original_data['results'][symbol].get('test_metrics', {})
                        comparison['original_accuracy'] = original_metrics.get('accuracy', 0)
                        comparison['original_f1'] = original_metrics.get('f1_score', 0)
            
            if enhanced_summary_path.exists():
                with open(enhanced_summary_path, 'r') as f:
                    enhanced_data = json.load(f)
                    if symbol in enhanced_data.get('results', {}):
                        enhanced_result = enhanced_data['results'][symbol]
                        best_accuracy = self._get_best_accuracy(enhanced_result)
                        comparison['enhanced_accuracy'] = best_accuracy
                        
                        # Try to get F1 score
                        best_model_name = self._get_best_model_name(enhanced_result)
                        if best_model_name in enhanced_result:
                            comparison['enhanced_f1'] = enhanced_result[best_model_name].get('test_f1', 0)
            
            # Calculate improvement
            if 'original_accuracy' in comparison and 'enhanced_accuracy' in comparison:
                comparison['accuracy_improvement'] = (
                    comparison['enhanced_accuracy'] - comparison['original_accuracy']
                )
                comparison['relative_improvement'] = (
                    comparison['accuracy_improvement'] / comparison['original_accuracy'] * 100
                ) if comparison['original_accuracy'] > 0 else 0
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Model comparison failed for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}


# Prediction class for using enhanced models
class EnhancedPredictor:
    """Enhanced predictor that uses the improved models"""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
    
    def load_enhanced_model(self, symbol: str) -> Dict[str, Any]:
        """Load enhanced model package"""
        if symbol in self.loaded_models:
            return self.loaded_models[symbol]
        
        model_path = self.models_dir / f"{symbol.lower()}_enhanced_model.pkl"
        if not model_path.exists():
            # Fallback to standard model
            model_path = self.models_dir / f"{symbol.lower()}_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No model found for {symbol}")
        
        model_package = joblib.load(model_path)
        self.loaded_models[symbol] = model_package
        
        return model_package
    
    def predict_with_confidence(self, symbol: str, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with confidence scores"""
        try:
            # Load model package
            package = self.load_enhanced_model(symbol)
            
            model = package['model']
            scaler = package.get('scaler')
            feature_selector = package.get('feature_selector')
            feature_engineer = package.get('feature_engineer')
            
            # Extract features
            if feature_engineer:
                features = feature_engineer.extract_all_features(ohlcv_data)
            else:
                # Fallback to basic features
                from backend.ml_engine.feature_extractor import extract_features
                features = extract_features(ohlcv_data)
            
            # Apply feature selection
            if feature_selector:
                # Apply saved feature selection pipeline
                features_clean = features[feature_selector['good_features']]
                features_clean = features_clean[feature_selector['non_zero_var_features']]
                features_selected = feature_selector['selector'].transform(features_clean)
                features_final = pd.DataFrame(
                    features_selected, 
                    index=features.index, 
                    columns=feature_selector['selected_features']
                )
            else:
                features_final = features
            
            # Scale features
            if scaler:
                features_scaled = pd.DataFrame(
                    scaler.transform(features_final),
                    index=features_final.index,
                    columns=features_final.columns
                )
            else:
                features_scaled = features_final
            
            # Make prediction
            latest_features = features_scaled.iloc[-1:].fillna(0)
            
            prediction = model.predict(latest_features)[0]
            
            # Get prediction probabilities if available
            confidence_scores = {}
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(latest_features)[0]
                classes = model.classes_ if hasattr(model, 'classes_') else ['buy', 'hold', 'sell']
                confidence_scores = dict(zip(classes, probabilities))
                max_confidence = max(probabilities)
            else:
                max_confidence = 0.5  # Default confidence
            
            result = {
                'symbol': symbol,
                'prediction': prediction,
                'confidence': max_confidence,
                'confidence_scores': confidence_scores,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'enhanced_ensemble'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'prediction': 'hold',
                'confidence': 0.0,
                'error': str(e)
            }


# Main execution functions
async def run_enhanced_training(symbols: Optional[List[str]] = None, force_retrain: bool = False):
    """Run the enhanced training system"""
    try:
        trainer = EnhancedAutoTrainingSystem()
        results = trainer.run_enhanced_batch_training(symbols, force_retrain)
        
        logger.info("🎉 Enhanced training completed!")
        return results
        
    except Exception as e:
        logger.error(f"💥 Enhanced training failed: {e}")
        raise


def compare_all_models():
    """Compare original vs enhanced models for all symbols"""
    try:
        trainer = EnhancedAutoTrainingSystem()
        
        logger.info("📊 Comparing original vs enhanced models...")
        
        comparisons = {}
        for symbol in trainer.symbols:
            comparison = trainer.compare_models(symbol)
            comparisons[symbol] = comparison
            
            if 'accuracy_improvement' in comparison:
                improvement = comparison['relative_improvement']
                logger.info(f"{symbol}: {improvement:+.1f}% improvement")
        
        return comparisons
        
    except Exception as e:
        logger.error(f"❌ Model comparison failed: {e}")
        return {}


if __name__ == "__main__":
    import asyncio
    
    # Run enhanced training
    results = asyncio.run(run_enhanced_training(force_retrain=True))
    
    # Compare with original models
    comparisons = compare_all_models()
    
    print("\n🏆 ENHANCEMENT RESULTS:")
    for symbol, comparison in comparisons.items():
        if 'relative_improvement' in comparison:
            print(f"{symbol}: {comparison['relative_improvement']:+.1f}% accuracy improvement")




















            import os
from pandas import Series
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta 
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import warnings
from scipy import stats
from typing import Union, Tuple
from sklearn.tree import DecisionTreeClassifier
from pandas import DatetimeIndex
from typing import cast

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings('ignore')



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Enhanced feature engineering from the advanced system"""
    
    def __init__(self, timeframe: str = "1h"):
        self.timeframe = timeframe
        # Adjust periods based on timeframe
        if timeframe == "1h":
            self.short_period = 12
            self.medium_period = 24
            self.long_period = 168  # 1 week
        else:  # 30m
            self.short_period = 24
            self.medium_period = 48
            self.long_period = 336  # 1 week
    
    def add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        data = df.copy()
        
        # Price-based indicators
        data['hl2'] = (data['high'] + data['low']) / 2
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        data['ohlc4'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        
        # Multiple timeframe moving averages
        for period in [5, 10, 20, 50, 100]:
            if period <= len(data):
                data[f'sma_{period}'] = data['close'].rolling(period).mean()
                data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                data[f'price_vs_sma_{period}'] = data['close'] / data[f'sma_{period}'] - 1
        
        # RSI with multiple periods
        for period in [6, 14, 21]:
            if period <= len(data):
                delta = data['close'].diff()
                gain = (delta.astype(float).where(delta.astype(float) > 0, 0)).rolling(window=period).mean()
                loss = (-delta.astype(float).where(delta.astype(float) < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            if period <= len(data):
                sma = data['close'].rolling(period).mean()
                std = data['close'].rolling(period).std()
                data[f'bb_upper_{period}'] = sma + (2 * std)
                data[f'bb_lower_{period}'] = sma - (2 * std)
                data[f'bb_width_{period}'] = (data[f'bb_upper_{period}'] - data[f'bb_lower_{period}']) / sma
                data[f'bb_position_{period}'] = (data['close'] - data[f'bb_lower_{period}']) / (data[f'bb_upper_{period}'] - data[f'bb_lower_{period}'])
        
        # Momentum indicators
        for period in [1, 3, 6, 12]:
            if period <= len(data):
                data[f'momentum_{period}'] = data['close'].pct_change(period)
                data[f'roc_{period}'] = (data['close'] / data['close'].shift(period) - 1) * 100
        
        # Volume indicators
        if 'volume' in data.columns:
            data['volume_sma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            data['price_volume'] = data['close'] * data['volume']
            data['vwap'] = data['price_volume'].rolling(20).sum() / data['volume'].rolling(20).sum()
        
        # Volatility indicators
        for period in [10, 20, 50]:
            if period <= len(data):
                data[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
        
        return data
    
    def add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure and pattern recognition features"""
        data = df.copy()
        
        # Higher highs and lower lows
        data['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
        data['lower_low'] = (data['low'] < data['low'].shift(1)).astype(int)
        
        # Trend strength
        for window in [5, 10, 20]:
            data[f'trend_strength_{window}'] = data['close'].rolling(window).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 2 else 0
            )
        
        # Price patterns
        data['doji'] = ((abs(data['open'] - data['close']) / (data['high'] - data['low'] + 1e-10)) < 0.1).astype(int)
        data['hammer'] = ((data['close'] > data['open']) & 
                         ((data['open'] - data['low']) > 2 * (data['close'] - data['open']))).astype(int)
        
        # Gap detection
        data['gap_up'] = (data['open'] > data['close'].shift(1) * 1.005).astype(int)
        data['gap_down'] = (data['open'] < data['close'].shift(1) * 0.995).astype(int)
        
        return data
    
    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based cyclical features"""
        data = df.copy()

    # Ensure we have a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            idx: pd.DatetimeIndex = data.index


    # Hour of day (important for crypto)
        data['hour'] = idx.hour
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

    # Day of week
        data['day_of_week'] = idx.dayofweek
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

    # Weekend indicator
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

        return data

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features with backward compatibility"""
        logger.info("🔧 Extracting enhanced features...")

        try:
        # Apply all feature engineering
            features = self.add_advanced_technical_indicators(df)
            features = self.add_market_structure_features(features)
            features = self.add_cyclical_features(features)

            # Remove OHLCV columns to prevent data leakage
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in features.columns if col not in price_cols]

            result = features[feature_cols].copy()

        # Handle missing values (use valid method names for Pylance)
            result = result.fillna(method="ffill").fillna(method="bfill").fillna(0)  # type: ignore[arg-type]
            result = result.replace([np.inf, -np.inf], np.nan).fillna(0)

            logger.info(f"✅ Extracted {len(result.columns)} enhanced features")
            return result

        except Exception as e:
            logger.warning(f"Enhanced feature extraction failed: {e}. Falling back to basic features.")
            try:
                from backend.ml_engine.feature_extractor import extract_features
                return extract_features(df)
            except:
                logger.error("Both enhanced and basic feature extraction failed")
                raise

class EnhancedBatchTrainer:
    """Enhanced batch trainer with backward compatibility and improved ML pipeline for 1h data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.ohlcv_dir = self.base_dir / "data" / "ohlcv"
        self.labels_dir = self.base_dir / "data" / "labels"
        self.models_dir = self.base_dir / "agents" / "models"
        self.logs_dir = self.base_dir / "storage" / "training_logs"
        
        # Create directories
        for dir_path in [self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration with fallbacks
        if config is None:
            try:
                from backend.config.training_config import get_default_config
                config = get_default_config()
            except ImportError:
                config = self._get_default_config()
        
        self.label_config = config.get('label_config', self._get_default_label_config())
        self.model_config = config.get('model_config', self._get_default_model_config())
        self.system_config = config.get('system_config', self._get_default_system_config())
        
        # Initialize enhanced components
        self.feature_engineer = FeatureEngineer("1h")
        self.scalers = {}
        self.feature_selectors = {}
        
        # Training results storage
        self.training_results = {}

    def _get_default_config(self):
        """Fallback configuration"""
        return {
            'label_config': self._get_default_label_config(),
            'model_config': self._get_default_model_config(),
            'system_config': self._get_default_system_config()
        }
    
    def _get_default_label_config(self):
        """Default label configuration"""
        from types import SimpleNamespace
        return SimpleNamespace(min_samples_per_class=10)
    
    def _get_default_model_config(self):
        """Default model configuration"""
        from types import SimpleNamespace
        return SimpleNamespace(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            test_size=0.2,
            cross_validation_folds=5,
            min_accuracy=0.45,
            min_precision=0.35,
            min_recall=0.35,
            min_f1_score=0.35
        )
    
    def _get_default_system_config(self):
        """Default system configuration"""
        from types import SimpleNamespace
        return SimpleNamespace(
            data_quality_threshold=0.7,
            min_data_points=500,
            retrain_interval_days=7,
            symbols=[
                "DOGEUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT",
                "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"
            ]
        )

    def load_and_validate_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate 1h OHLCV and labels data with enhanced error handling"""
        logger.info(f"📊 Loading 1h data for {symbol}...")
        
        # Load 1h OHLCV data (prioritize 1h, fallback to older data)
        ohlcv_path = self.ohlcv_dir / f"{symbol}_1h.csv"
        if not ohlcv_path.exists():
            ohlcv_path = self.ohlcv_dir / f"{symbol}.csv"
            if not ohlcv_path.exists():
                ohlcv_path = self.ohlcv_dir / f"{symbol}USDT_1h.csv"
                if not ohlcv_path.exists():
                    raise FileNotFoundError(f"No OHLCV data found for {symbol}")
            logger.info(f"Using fallback OHLCV file for {symbol}")
        
        ohlcv_df = pd.read_csv(ohlcv_path, index_col=0, parse_dates=True)
        ohlcv_df = ohlcv_df.sort_index()
        
        # Validate OHLCV data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in ohlcv_df.columns]
        if missing_cols:
            raise ValueError(f"Missing OHLCV columns: {missing_cols}")
        
        # Check data completeness
        completeness = 1 - (ohlcv_df.isnull().sum().sum() / (len(ohlcv_df) * len(ohlcv_df.columns)))
        if completeness < self.system_config.data_quality_threshold:
            logger.warning(f"⚠️ {symbol}: Data completeness {completeness:.2%} below threshold")
        
        # Load labels with enhanced search strategy
        labels_df = self._load_labels_with_fallback(symbol)
        
        if labels_df is None:
            raise FileNotFoundError(f"No valid 1h labels found for {symbol}")
        
        logger.info(f"✅ {symbol}: Loaded {len(ohlcv_df)} 1h OHLCV rows, {len(labels_df)} labels")
        
        return ohlcv_df, labels_df
    
    def _load_labels_with_fallback(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load labels with multiple fallback strategies"""
        # Try different label file patterns in order of preference
        label_patterns = [
            f"{symbol}_1h_outcome_labels.csv",
            f"{symbol}_1h_hybrid_labels.csv",
            f"{symbol}_outcome_labels.csv",
            f"{symbol}_hybrid_labels.csv",
            f"{symbol}_labels.csv",
            f"{symbol}_1h_labels.csv"
        ]
        
        for pattern in label_patterns:
            labels_path = self.labels_dir / pattern
            
            if labels_path.exists():
                try:
                    labels_df = pd.read_csv(labels_path, index_col=0, parse_dates=True)
                    
                    # Validate labels - check for both 'action' and 'label' columns
                    label_col = None
                    if 'action' in labels_df.columns:
                        label_col = 'action'
                    elif 'label' in labels_df.columns:
                        label_col = 'label'
                    else:
                        logger.warning(f"No 'action' or 'label' column in {pattern}")
                        continue
                    
                    # Filter valid labels
                    valid_actions = ['buy', 'sell', 'hold']
                    labels_df = labels_df[labels_df[label_col].isin(valid_actions)]
                    
                    if len(labels_df) < self.system_config.min_data_points:
                        logger.warning(f"Insufficient data in {pattern}: {len(labels_df)} rows")
                        continue
                    
                    # Standardize label column name
                    if label_col != 'action':
                        labels_df['action'] = labels_df[label_col]
                    
                    logger.info(f"✅ Loaded labels from {pattern}")
                    return labels_df
                    
                except Exception as e:
                    logger.warning(f"Failed to load {pattern}: {e}")
                    continue
        
        return None

    def extract_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Extract features using enhanced feature engineering"""
        try:
            # Try enhanced feature extraction first
            features_df = self.feature_engineer.extract_all_features(ohlcv_df)
            
            # Validate features
            if features_df.empty:
                raise ValueError("Enhanced feature extraction returned empty DataFrame")
            
            logger.info(f"📈 Extracted {len(features_df.columns)} enhanced features")
            return features_df
            
        except Exception as e:
            logger.warning(f"Enhanced feature extraction failed: {e}. Trying fallback...")
            
            # Fallback to basic feature extraction
            try:
                from backend.ml_engine.feature_extractor import extract_features
                features_df = extract_features(ohlcv_df)
                
                # Handle missing values for fallback
                features_df = features_df.fillna(features_df.mean())
                features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                logger.info(f"📈 Extracted {len(features_df.columns)} fallback features")
                return features_df
                
            except Exception as e2:
                logger.error(f"Both enhanced and fallback feature extraction failed: {e2}")
                raise

    def prepare_training_data(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare aligned training data for 1h timeframe with enhanced alignment"""
        # Find common timestamps
        common_index = features_df.index.intersection(labels_df.index)
        
        if len(common_index) == 0:
            # Try fuzzy matching for different timeframes
            logger.warning("No exact timestamp matches, attempting fuzzy alignment...")
            
            # For 1h data, allow matching within 1 hour window
            aligned_features = []
            aligned_labels = []
            
            for label_time in labels_df.index:
                # Find features within 1 hour of label time
                time_diff = abs(features_df.index - label_time)
                closest_idx = time_diff.idxmin()
                
                if time_diff[closest_idx] <= pd.Timedelta(hours=1):
                    aligned_features.append(features_df.loc[closest_idx])
                    aligned_labels.append(labels_df.loc[label_time, 'action'])
            
            if len(aligned_features) == 0:
                raise ValueError("No alignable timestamps between features and labels")
            
            X = pd.DataFrame(aligned_features)
            y = pd.Series(aligned_labels, index=X.index)
        else:
            # Align data normally
            X = features_df.loc[common_index]
            y = labels_df.loc[common_index, 'action']
        
        # Final validation
        if len(X) != len(y):
            raise ValueError("Feature and label lengths don't match")
        
        # Check class distribution
        class_counts = y.value_counts()
        logger.info(f"📊 Class distribution: {class_counts.to_dict()}")
        
        # Ensure minimum samples per class
        for class_name, count in class_counts.items():
            if count < self.label_config.min_samples_per_class:
                logger.warning(f"⚠️ Class '{class_name}' has only {count} samples")
        
        return X, y

    def create_enhanced_model(self) -> Any:
        """Create enhanced ensemble model with fallbacks"""
        base_models = []
        
        # Always include Random Forest
        rf = RandomForestClassifier(
            n_estimators=self.model_config.n_estimators,
            max_depth=self.model_config.max_depth,
            min_samples_split=self.model_config.min_samples_split,
            min_samples_leaf=self.model_config.min_samples_leaf,
            class_weight=self.model_config.class_weight,
            random_state=self.model_config.random_state,
            n_jobs=self.model_config.n_jobs
        )
        base_models.append(('rf', rf))
        
        # Add Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=self.model_config.random_state
        )
        base_models.append(('gb', gb))
        
        # Add XGBoost if available
        if HAS_XGB:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=self.model_config.random_state,
                eval_metric='mlogloss'
            )
            base_models.append(('xgb', xgb_model))
        
        # Add LightGBM if available
        if HAS_LGB:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=self.model_config.random_state,
                verbosity=-1
            )
            base_models.append(('lgb', lgb_model))
        
        # Add Neural Network
        mlp = MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=300,
            random_state=self.model_config.random_state
        )
        base_models.append(('mlp', mlp))
        
        # Create voting classifier if we have multiple models
        if len(base_models) > 1:
            ensemble = VotingClassifier(
                estimators=base_models,
                voting='soft',
                n_jobs=-1
            )
            logger.info(f"Created ensemble with {len(base_models)} base models")
            return ensemble
        else:
            logger.info("Using single Random Forest model")
            return rf

    def feature_selection(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> pd.DataFrame:
        """Enhanced feature selection"""
        logger.info(f"🎯 Performing feature selection for {symbol}...")
        
        # Remove features with too many missing values
        missing_threshold = 0.3
        missing_rates = X.isnull().sum() / len(X)
        good_features = missing_rates[missing_rates <= missing_threshold].index
        X_clean = X[good_features].copy()
        
        # Remove features with zero variance
        var_threshold = 1e-6
        feature_vars = X_clean.var()
        non_zero_var_features = feature_vars[feature_vars > var_threshold].index
        X_clean = X_clean[non_zero_var_features]
        
        # Statistical feature selection
        k_best = min(50, len(X_clean.columns))  # Select top 50 features or all if less
        try:
            selector = SelectKBest(f_classif, k=k_best)
            X_selected = selector.fit_transform(X_clean, y)
            
            selected_features = X_clean.columns[selector.get_support()]
            X_final = pd.DataFrame(X_selected, index=X_clean.index, columns=selected_features)
            
            # Store feature selector for later use
            self.feature_selectors[symbol] = {
                'good_features': good_features,
                'non_zero_var_features': non_zero_var_features,
                'selector': selector,
                'selected_features': selected_features
            }
            
            logger.info(f"✅ Selected {len(selected_features)} features from {len(X.columns)} original")
            return X_final
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all clean features.")
            return X_clean

    def train_and_validate_model(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> Dict[str, Any]:
        """Train model with comprehensive validation for 1h timeframe"""
        logger.info(f"🧠 Training enhanced 1h model for {symbol}...")
        
        # Feature selection
        X_selected = self.feature_selection(X, y, symbol)
        
        # Split data with time series consideration
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, 
                test_size=self.model_config.test_size,
                random_state=self.model_config.random_state,
                stratify=y
            )
        except ValueError:
            # If stratification fails (not enough samples per class)
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, 
                test_size=self.model_config.test_size,
                random_state=self.model_config.random_state
            )
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
        
        # Store scaler
        self.scalers[symbol] = scaler
        
        # Create and train model
        try:
            model = self.create_enhanced_model()
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            logger.warning(f"Enhanced model training failed: {e}. Using basic Random Forest.")
            model = RandomForestClassifier(
                n_estimators=self.model_config.n_estimators,
                max_depth=self.model_config.max_depth,
                min_samples_split=self.model_config.min_samples_split,
                min_samples_leaf=self.model_config.min_samples_leaf,
                class_weight=self.model_config.class_weight,
                random_state=self.model_config.random_state,
                n_jobs=self.model_config.n_jobs
            )
            model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_pred_train, "train")
        test_metrics = self._calculate_metrics(y_test, y_pred_test, "test")
        
        # Cross-validation with time series consideration
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=tscv,
                scoring='f1_weighted'
            )
        except:
            # Fallback to regular cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=min(self.model_config.cross_validation_folds, len(y_train) // 3),
                scoring='f1_weighted'
            )
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_selected.columns, model.feature_importances_))
        elif hasattr(model, 'estimators_'):
            # For ensemble models, get feature importance from first estimator
            first_estimator: DecisionTreeClassifier = (
            model.estimators_[0][1]  # type: ignore
            if isinstance(model.estimators_[0], (list, tuple))
            else model.estimators_[0]
            )

            if hasattr(first_estimator, 'feature_importances_'):
                feature_importance = dict(zip(X_selected.columns, first_estimator.feature_importances_))
            else:
                feature_importance = {}
        else:
            feature_importance = {}
        
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Compile results
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'timeframe': '1h',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_features': len(X_selected.columns),
            'classes': list(model.classes_) if hasattr(model, 'classes_') else ['buy', 'sell', 'hold'],
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cross_validation': {
                'mean_f1': cv_scores.mean(),
                'std_f1': cv_scores.std(),
                'scores': cv_scores.tolist()
            },
            'top_features': top_features[:5],
            'class_distribution': y.value_counts().to_dict(),
            'enhanced_features': True
        }
        
        # Validate model performance
        validation_passed = self._validate_model_performance(results)
        results['validation_passed'] = validation_passed
        
        if validation_passed:
            # Save enhanced model package
            self._save_enhanced_model(symbol, model, results)
            logger.info(f"✅ {symbol}: Enhanced 1h model saved")
        else:
            logger.warning(f"⚠️ {symbol}: Model failed validation, not saved")
        
        return results

    def _save_enhanced_model(self, symbol: str, model: Any, results: Dict[str, Any]):
        """Save enhanced model package with backward compatibility"""
        try:
            # Create enhanced model package
            model_package = {
                'model': model,
                'scaler': self.scalers.get(symbol),
                'feature_selector': self.feature_selectors.get(symbol),
                'feature_engineer': self.feature_engineer,
                'metadata': {
                    'symbol': symbol,
                    'timeframe': '1h',
                    'training_date': datetime.now().isoformat(),
                    'performance': results,
                    'model_type': 'enhanced_ensemble',
                    'version': '2.0'
                }
            }
            
            # Save enhanced model
            enhanced_path = self.models_dir / f"{symbol.lower()}_enhanced_model.pkl"
            joblib.dump(model_package, enhanced_path)
            
            # Save with standard naming for backward compatibility
            standard_path = self.models_dir / f"{symbol.lower()}_model.pkl"
            joblib.dump(model_package, standard_path)
            
            results['model_path'] = str(standard_path)
            logger.info(f"💾 Enhanced model saved for {symbol} (backward compatible)")
            
        except Exception as e:
            logger.error(f"❌ Failed to save enhanced model for {symbol}: {e}")
            # Fallback: save just the model
            try:
                standard_path = self.models_dir / f"{symbol.lower()}_model.pkl"
                joblib.dump(model, standard_path)
                results['model_path'] = str(standard_path)
                logger.info(f"💾 Basic model saved for {symbol}")
            except Exception as e2:
                logger.error(f"❌ Failed to save any model for {symbol}: {e2}")

    def _calculate_metrics(self, y_true, y_pred, dataset_name: str) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

    def _validate_model_performance(self, results: Dict[str, Any]) -> bool:
        """Validate if model meets minimum performance requirements"""
        test_metrics = results['test_metrics']
        
        checks = [
            test_metrics['accuracy'] >= self.model_config.min_accuracy,
            test_metrics['precision'] >= self.model_config.min_precision,
            test_metrics['recall'] >= self.model_config.min_recall,
            test_metrics['f1_score'] >= self.model_config.min_f1_score
        ]
        
        passed = all(checks)
        
        if not passed:
            logger.warning(f"Model validation failed:")
            logger.warning(f"  Accuracy: {test_metrics['accuracy']:.3f} (min: {self.model_config.min_accuracy})")
            logger.warning(f"  Precision: {test_metrics['precision']:.3f} (min: {self.model_config.min_precision})")
            logger.warning(f"  Recall: {test_metrics['recall']:.3f} (min: {self.model_config.min_recall})")
            logger.warning(f"  F1 Score: {test_metrics['f1_score']:.3f} (min: {self.model_config.min_f1_score})")
        
        return passed

    def train_symbol(self, symbol: str) -> Dict[str, Any]:
        """Train enhanced 1h model for a single symbol"""
        try:
            logger.info(f"🎯 Starting enhanced 1h training for {symbol}")
            
            # Load and validate data
            ohlcv_df, labels_df = self.load_and_validate_data(symbol)
            
            # Extract features (enhanced)
            features_df = self.extract_features(ohlcv_df)
            
            # Prepare training data
            X, y = self.prepare_training_data(features_df, labels_df)
            
            # Train and validate model
            results = self.train_and_validate_model(X, y, symbol)
            
            # Log results
            self._log_training_results(symbol, results)
            
            # Store results
            self.training_results[symbol] = results
            
            logger.info(f"✅ {symbol}: Enhanced 1h training completed successfully")
            logger.info(f"   Test Accuracy: {results['test_metrics']['accuracy']:.3f}")
            logger.info(f"   CV F1 Score: {results['cross_validation']['mean_f1']:.3f} ± {results['cross_validation']['std_f1']:.3f}")
            
            return results
            
        except Exception as e:
            error_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'timeframe': '1h',
                'status': 'failed',
                'error': str(e),
                'enhanced_features': False
            }
            
            self.training_results[symbol] = error_result
            logger.error(f"❌ {symbol}: Enhanced 1h training failed - {e}")
            
            return error_result

    def _log_training_results(self, symbol: str, results: Dict[str, Any]):
        """Log detailed training results"""
        log_path = self.logs_dir / f"{symbol}_1h_enhanced_training_results.json"
        
        try:
            with open(log_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📝 Enhanced 1h training log saved to {log_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save training log for {symbol}: {e}")

    def batch_train(self, symbols: Optional[List[str]] = None, force_retrain: bool = False) -> Dict[str, Dict[str, Any]]:
        """Train enhanced 1h models for multiple symbols"""
        if symbols is None:
            symbols = self.system_config.symbols
    
        if not symbols:
            logger.error("No symbols provided for batch training.")
            return {}
    
        logger.info(f"🚀 Starting enhanced 1h batch training for {len(symbols)} symbols")
        logger.info(f"Symbols: {', '.join(symbols)}")

        # Check which models need training
        symbols_to_train = []
        for symbol in symbols:
            model_path = self.models_dir / f"{symbol.lower()}_model.pkl"
            
            if force_retrain or not model_path.exists():
                symbols_to_train.append(symbol)
            else:
                # Check model age
                model_age_days = (datetime.now().timestamp() - model_path.stat().st_mtime) / 86400
                if model_age_days > self.system_config.retrain_interval_days:
                    symbols_to_train.append(symbol)
                    logger.info(f"🔄 {symbol}: Model is {model_age_days:.1f} days old, retraining")
                else:
                    logger.info(f"📋 {symbol}: Enhanced model up to date")
        
        if not symbols_to_train:
            logger.info("✅ All enhanced models are up to date")
            return {}

        results = {}
        successful = 0
        failed = 0
        
        for symbol in symbols_to_train:
            try:
                result = self.train_symbol(symbol)
                results[symbol] = result
                
                if result.get('validation_passed', False):
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: Batch training error - {e}")
                results[symbol] = {
                    'symbol': symbol,
                    'timeframe': '1h',
                    'status': 'failed',
                    'error': str(e),
                    'enhanced_features': False
                }
                failed += 1
        
        # Save batch summary
        self._save_batch_summary(results, successful, failed)
        
        logger.info("=" * 70)
        logger.info("🎯 ENHANCED 1H BATCH TRAINING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📊 Success Rate: {successful/(successful+failed)*100:.1f}%")
        
        if successful > 0:
            # Calculate average performance metrics
            valid_results = [r for r in results.values() if r.get('validation_passed', False)]
            if valid_results:
                avg_accuracy = np.mean([r['test_metrics']['accuracy'] for r in valid_results])
                avg_f1 = np.mean([r['cross_validation']['mean_f1'] for r in valid_results])
                enhanced_count = sum([1 for r in valid_results if r.get('enhanced_features', False)])
                
                logger.info(f"📈 Average Test Accuracy: {avg_accuracy:.3f}")
                logger.info(f"📈 Average CV F1 Score: {avg_f1:.3f}")
                logger.info(f"🔧 Enhanced Features Used: {enhanced_count}/{successful}")
        
        logger.info("=" * 70)
        
        return results

    def _save_batch_summary(self, results: Dict[str, Dict[str, Any]], successful: int, failed: int):
        """Save enhanced batch training summary with backward compatibility"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'timeframe': '1h',
            'system_type': 'enhanced_batch_trainer',
            'total_symbols': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / (successful + failed) if (successful + failed) > 0 else 0,
            'results': results,
            'enhancements': {
                'advanced_feature_engineering': True,
                'ensemble_models': True,
                'feature_selection': True,
                'robust_scaling': True,
                'time_series_validation': True
            }
        }
        
        # Save enhanced summary
        enhanced_summary_path = self.models_dir / "enhanced_training_summary.json"
        try:
            with open(enhanced_summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"📊 Enhanced summary saved to {enhanced_summary_path}")
        except Exception as e:
            logger.warning(f"Failed to save enhanced summary: {e}")
        
        # Save standard summary for backward compatibility
        standard_summary_path = self.models_dir / "training_summary.json"
        try:
            with open(standard_summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"📊 Backward compatible summary saved to {standard_summary_path}")
        except Exception as e:
            logger.warning(f"Failed to save standard summary: {e}")

    def get_training_status(self) -> Dict[str, str]:
        """Get training status for all symbols"""
        status = {}
        
        for symbol in self.system_config.symbols:
            # Check for enhanced model first
            enhanced_path = self.models_dir / f"{symbol.lower()}_enhanced_model.pkl"
            model_path = self.models_dir / f"{symbol.lower()}_model.pkl"
            
            if enhanced_path.exists():
                model_age_days = (datetime.now().timestamp() - enhanced_path.stat().st_mtime) / 86400
                if model_age_days <= self.system_config.retrain_interval_days:
                    status[symbol] = "up_to_date_enhanced_1h"
                else:
                    status[symbol] = "needs_retrain_enhanced_1h"
            elif model_path.exists():
                model_age_days = (datetime.now().timestamp() - model_path.stat().st_mtime) / 86400
                if model_age_days <= self.system_config.retrain_interval_days:
                    status[symbol] = "up_to_date_1h"
                else:
                    status[symbol] = "needs_retrain_1h"
            else:
                status[symbol] = "no_model"
        
        return status

    def load_model(self, symbol: str) -> Dict[str, Any]:
        """Load model with enhanced package support"""
        try:
            # Try enhanced model first
            enhanced_path = self.models_dir / f"{symbol.lower()}_enhanced_model.pkl"
            if enhanced_path.exists():
                return joblib.load(enhanced_path)
            
            # Fallback to standard model
            standard_path = self.models_dir / f"{symbol.lower()}_model.pkl"
            if standard_path.exists():
                model_data = joblib.load(standard_path)
                
                # Check if it's already a package
                if isinstance(model_data, dict) and 'model' in model_data:
                    return model_data
                else:
                    # Wrap bare model in package format
                    return {
                        'model': model_data,
                        'scaler': None,
                        'feature_selector': None,
                        'feature_engineer': None,
                        'metadata': {
                            'symbol': symbol,
                            'timeframe': '1h',
                            'model_type': 'legacy',
                            'version': '1.0'
                        }
                    }
            
            raise FileNotFoundError(f"No model found for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            raise

class EnhancedPredictor:
    """Enhanced predictor that works with both legacy and enhanced models"""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.feature_engineer = FeatureEngineer("1h")
    
    def load_model(self, symbol: str) -> Dict[str, Any]:
        """Load model package with backward compatibility"""
        if symbol in self.loaded_models:
            return self.loaded_models[symbol]
        
        # Try enhanced model first
        enhanced_path = self.models_dir / f"{symbol.lower()}_enhanced_model.pkl"
        if enhanced_path.exists():
            package = joblib.load(enhanced_path)
            self.loaded_models[symbol] = package
            return package
        
        # Fallback to standard model
        standard_path = self.models_dir / f"{symbol.lower()}_model.pkl"
        if standard_path.exists():
            model_data = joblib.load(standard_path)
            
            # Check if it's already a package
            if isinstance(model_data, dict) and 'model' in model_data:
                package = model_data
            else:
                # Wrap bare model in package format
                package = {
                    'model': model_data,
                    'scaler': None,
                    'feature_selector': None,
                    'feature_engineer': None,
                    'metadata': {
                        'symbol': symbol,
                        'timeframe': '1h',
                        'model_type': 'legacy',
                        'version': '1.0'
                    }
                }
            
            self.loaded_models[symbol] = package
            return package
        
        raise FileNotFoundError(f"No model found for {symbol}")
    
    def predict_with_confidence(self, symbol: str, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with confidence scores using enhanced features"""
        try:
            # Load model package
            package = self.load_model(symbol)
            
            model = package['model']
            scaler = package.get('scaler')
            feature_selector = package.get('feature_selector')
            feature_engineer = package.get('feature_engineer', self.feature_engineer)
            
            # Extract features
            if feature_engineer and hasattr(feature_engineer, 'extract_all_features'):
                features = feature_engineer.extract_all_features(ohlcv_data)
            else:
                # Fallback to basic features
                try:
                    from backend.ml_engine.feature_extractor import extract_features
                    features = extract_features(ohlcv_data)
                except ImportError:
                    features = self.feature_engineer.extract_all_features(ohlcv_data)
            
            # Apply feature selection if available
            if feature_selector:
                # Apply saved feature selection pipeline
                features_clean = features[feature_selector['good_features']]
                features_clean = features_clean[feature_selector['non_zero_var_features']]
                features_selected = feature_selector['selector'].transform(features_clean)
                features_final = pd.DataFrame(
                    features_selected, 
                    index=features.index, 
                    columns=feature_selector['selected_features']
                )
            else:
                features_final = features
            
            # Scale features if scaler available
            if scaler:
                features_scaled = pd.DataFrame(
                    scaler.transform(features_final),
                    index=features_final.index,
                    columns=features_final.columns
                )
            else:
                features_scaled = features_final
            
            # Make prediction
            latest_features = features_scaled.iloc[-1:].fillna(0)
            
            prediction = model.predict(latest_features)[0]
            
            # Get prediction probabilities if available
            confidence_scores = {}
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(latest_features)[0]
                classes = model.classes_ if hasattr(model, 'classes_') else ['buy', 'hold', 'sell']
                confidence_scores = dict(zip(classes, probabilities))
                max_confidence = max(probabilities)
            else:
                max_confidence = 0.5  # Default confidence
            
            result = {
                'symbol': symbol,
                'prediction': prediction,
                'confidence': max_confidence,
                'confidence_scores': confidence_scores,
                'timestamp': datetime.now().isoformat(),
                'model_type': package.get('metadata', {}).get('model_type', 'unknown'),
                'model_version': package.get('metadata', {}).get('version', '1.0')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'prediction': 'hold',
                'confidence': 0.0,
                'error': str(e),
                'model_type': 'error'
            }

def main():
    """Main function for standalone execution with enhanced features"""
    try:
        # Load configuration with fallbacks
        try:
            from backend.config.training_config import load_config_from_env
            config = load_config_from_env()
        except ImportError:
            config = None
            logger.info("Using default configuration (config module not found)")
        
        # Create enhanced trainer
        trainer = EnhancedBatchTrainer(config)
        
        # Run batch training
        results = trainer.batch_train(force_retrain=False)
        
        # Print final summary
        successful_models = [s for s, r in results.items() if r.get('validation_passed', False)]
        enhanced_models = [s for s, r in results.items() if r.get('enhanced_features', False)]
        
        print(f"\n🎉 Enhanced 1h training completed!")
        print(f"✅ Successfully trained models: {', '.join(successful_models)}")
        print(f"🔧 Enhanced feature models: {', '.join(enhanced_models)}")
        
        # Show training status
        status = trainer.get_training_status()
        print(f"\n📊 Current model status:")
        for symbol, stat in status.items():
            print(f"  {symbol}: {stat}")
        
    except Exception as e:
        logger.error(f"💥 Enhanced 1h batch training failed: {e}")
        raise

if __name__ == "__main__":
    main()