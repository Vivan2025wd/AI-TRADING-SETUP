"""
Training configuration for the automated ML system
"""
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LabelGenerationConfig:
    """Configuration for label generation"""
    forward_window: int = 24  # Hours to look ahead
    min_return_threshold: float = 0.025  # 2.5% minimum return for buy signals
    stop_loss_threshold: float = -0.04  # -4% for sell signals
    min_samples_per_class: int = 100  # Minimum samples per class for balanced training
    volatility_adjustment: bool = True  # Use dynamic thresholds based on volatility


@dataclass
class ModelTrainingConfig:
    """Configuration for model training"""
    model_type: str = "RandomForest"
    n_estimators: int = 100
    max_depth: int = 6
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1
    
    # Validation settings
    test_size: float = 0.2
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    # Model performance thresholds
    min_accuracy: float = 0.6
    min_precision: float = 0.6
    min_recall: float = 0.6
    min_f1_score: float = 0.6


@dataclass
class SystemConfig:
    """Overall system configuration"""
    # Symbols to train
    symbols: Optional[List[str]] = None
    
    # Parallel processing
    max_workers: int = 3  # Maximum parallel training jobs
    
    # Model management
    model_retention_days: int = 30  # Keep old models for 30 days
    retrain_interval_days: int = 7  # Retrain models every 7 days
    performance_check_interval_hours: int = 24  # Check model performance every 24h
    
    # Data requirements
    min_data_points: int = 1000  # Minimum OHLCV points needed for training
    data_quality_threshold: float = 0.95  # Minimum data completeness ratio
    
    # Feature extraction
    feature_lookback_periods: Optional[List[int]] = None
    include_volume_features: bool = True
    include_momentum_features: bool = True
    include_volatility_features: bool = True
    include_trend_features: bool = True
    
    # Safety settings
    enable_model_validation: bool = True
    enable_backtesting: bool = True
    enable_performance_monitoring: bool = True
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = [
                "ATOMUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT",
                "ADAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT"
            ]
        
        if self.feature_lookback_periods is None:
            self.feature_lookback_periods = [5, 10, 20, 50, 100]


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    config = {}
    
    # Label generation settings
    config['label_config'] = LabelGenerationConfig(
        forward_window=int(os.getenv('LABEL_FORWARD_WINDOW', '24')),
        min_return_threshold=float(os.getenv('LABEL_MIN_RETURN', '0.025')),
        stop_loss_threshold=float(os.getenv('LABEL_STOP_LOSS', '-0.04')),
        min_samples_per_class=int(os.getenv('LABEL_MIN_SAMPLES', '100')),
        volatility_adjustment=os.getenv('LABEL_VOL_ADJUST', 'true').lower() == 'true'
    )
    
    # Model training settings
    config['model_config'] = ModelTrainingConfig(
        n_estimators=int(os.getenv('MODEL_N_ESTIMATORS', '100')),
        max_depth=int(os.getenv('MODEL_MAX_DEPTH', '6')),
        min_samples_split=int(os.getenv('MODEL_MIN_SAMPLES_SPLIT', '5')),
        min_samples_leaf=int(os.getenv('MODEL_MIN_SAMPLES_LEAF', '2')),
        min_accuracy=float(os.getenv('MODEL_MIN_ACCURACY', '0.6'))
    )
    
    # System settings
    symbols_env = os.getenv('TRAINING_SYMBOLS', '')
    symbols = symbols_env.split(',') if symbols_env else None
    
    config['system_config'] = SystemConfig(
        symbols=symbols,
        max_workers=int(os.getenv('TRAINING_MAX_WORKERS', '3')),
        retrain_interval_days=int(os.getenv('RETRAIN_INTERVAL_DAYS', '7')),
        min_data_points=int(os.getenv('MIN_DATA_POINTS', '1000'))
    )
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'label_config': LabelGenerationConfig(),
        'model_config': ModelTrainingConfig(),
        'system_config': SystemConfig()
    }


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return any errors"""
    errors = []
    
    # Validate label config
    label_config = config.get('label_config')
    if label_config:
        if label_config.forward_window <= 0:
            errors.append("forward_window must be positive")
        if label_config.min_return_threshold <= 0:
            errors.append("min_return_threshold must be positive")
        if label_config.stop_loss_threshold >= 0:
            errors.append("stop_loss_threshold must be negative")
        if label_config.min_samples_per_class < 10:
            errors.append("min_samples_per_class should be at least 10")
    
    # Validate model config
    model_config = config.get('model_config')
    if model_config:
        if model_config.n_estimators <= 0:
            errors.append("n_estimators must be positive")
        if model_config.max_depth <= 0:
            errors.append("max_depth must be positive")
        if not 0 < model_config.test_size < 1:
            errors.append("test_size must be between 0 and 1")
        if not 0 < model_config.min_accuracy <= 1:
            errors.append("min_accuracy must be between 0 and 1")
    
    # Validate system config
    system_config = config.get('system_config')
    if system_config:
        if system_config.max_workers <= 0:
            errors.append("max_workers must be positive")
        if system_config.retrain_interval_days <= 0:
            errors.append("retrain_interval_days must be positive")
        if not system_config.symbols:
            errors.append("symbols list cannot be empty")
    
    return errors


# Default configuration instance
DEFAULT_CONFIG = get_default_config()