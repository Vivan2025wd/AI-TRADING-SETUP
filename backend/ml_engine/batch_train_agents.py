import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBatchTrainer:
    """Enhanced batch trainer with better validation and error handling"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.ohlcv_dir = self.base_dir / "data" / "ohlcv"
        self.labels_dir = self.base_dir / "data" / "labels"
        self.models_dir = self.base_dir / "backend" / "agents" / "models"
        self.logs_dir = self.base_dir / "backend" / "storage" / "training_logs"
        
        # Create directories
        for dir_path in [self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        if config is None:
            from backend.config.training_config import get_default_config
            config = get_default_config()
        
        self.label_config = config['label_config']
        self.model_config = config['model_config']
        self.system_config = config['system_config']
        
        # Training results storage
        self.training_results = {}

    def load_and_validate_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate OHLCV and labels data"""
        logger.info(f"📊 Loading data for {symbol}...")
        
        # Load OHLCV data
        ohlcv_path = self.ohlcv_dir / f"{symbol}_1h.csv"
        if not ohlcv_path.exists():
            raise FileNotFoundError(f"OHLCV data not found: {ohlcv_path}")
        
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
        
        # Load labels
        labels_path = self.labels_dir / f"{symbol}_labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")
        
        labels_df = pd.read_csv(labels_path, index_col=0, parse_dates=True)
        
        # Validate labels
        if 'action' not in labels_df.columns:
            raise ValueError(f"Labels must have 'action' column")
        
        # Filter valid labels
        valid_actions = ['buy', 'sell', 'hold']
        labels_df = labels_df[labels_df['action'].isin(valid_actions)]
        
        if len(labels_df) < self.system_config.min_data_points:
            raise ValueError(f"Insufficient labeled data: {len(labels_df)} < {self.system_config.min_data_points}")
        
        logger.info(f"✅ {symbol}: Loaded {len(ohlcv_df)} OHLCV rows, {len(labels_df)} labels")
        
        return ohlcv_df, labels_df

    def extract_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Extract features using the existing feature extractor"""
        try:
            from backend.ml_engine.feature_extractor import extract_features
            features_df = extract_features(ohlcv_df)
            
            # Validate features
            if features_df.empty:
                raise ValueError("Feature extraction returned empty DataFrame")
            
            # Handle missing values
            features_df = features_df.fillna(features_df.mean())
            
            # Remove infinite values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(0)
            
            logger.info(f"📈 Extracted {len(features_df.columns)} features for {len(features_df)} samples")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            raise

    def prepare_training_data(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare aligned training data"""
        # Find common timestamps
        common_index = features_df.index.intersection(labels_df.index)
        
        if len(common_index) == 0:
            raise ValueError("No common timestamps between features and labels")
        
        # Align data
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

    def create_model(self) -> RandomForestClassifier:
        """Create model with configured parameters"""
        return RandomForestClassifier(
            n_estimators=self.model_config.n_estimators,
            max_depth=self.model_config.max_depth,
            min_samples_split=self.model_config.min_samples_split,
            min_samples_leaf=self.model_config.min_samples_leaf,
            class_weight=self.model_config.class_weight,
            random_state=self.model_config.random_state,
            n_jobs=self.model_config.n_jobs
        )

    def train_and_validate_model(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> Dict[str, Any]:
        """Train model with comprehensive validation"""
        logger.info(f"🧠 Training model for {symbol}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.model_config.test_size,
            random_state=self.model_config.random_state,
            stratify=y
        )
        
        # Create and train model
        model = self.create_model()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_pred_train, "train")
        test_metrics = self._calculate_metrics(y_test, y_pred_test, "test")
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.model_config.cross_validation_folds,
            scoring='accuracy'
        )
        
        # Feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Compile results
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_features': len(X.columns),
            'classes': list(model.classes_),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cross_validation': {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'scores': cv_scores.tolist()
            },
            'top_features': top_features[:5],  # Top 5 features
            'class_distribution': y.value_counts().to_dict()
        }
        
        # Validate model performance
        validation_passed = self._validate_model_performance(results)
        results['validation_passed'] = validation_passed
        
        if validation_passed:
            # Save model
            model_path = self.models_dir / f"{symbol.lower()}_model.pkl"
            joblib.dump(model, model_path)
            results['model_path'] = str(model_path)
            logger.info(f"✅ {symbol}: Model saved to {model_path}")
        else:
            logger.warning(f"⚠️ {symbol}: Model failed validation, not saved")
        
        return results

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
        """Train model for a single symbol"""
        try:
            logger.info(f"🎯 Starting training for {symbol}")
            
            # Load and validate data
            ohlcv_df, labels_df = self.load_and_validate_data(symbol)
            
            # Extract features
            features_df = self.extract_features(ohlcv_df)
            
            # Prepare training data
            X, y = self.prepare_training_data(features_df, labels_df)
            
            # Train and validate model
            results = self.train_and_validate_model(X, y, symbol)
            
            # Log results
            self._log_training_results(symbol, results)
            
            # Store results
            self.training_results[symbol] = results
            
            logger.info(f"✅ {symbol}: Training completed successfully")
            logger.info(f"   Test Accuracy: {results['test_metrics']['accuracy']:.3f}")
            logger.info(f"   CV Score: {results['cross_validation']['mean_accuracy']:.3f} ± {results['cross_validation']['std_accuracy']:.3f}")
            
            return results
            
        except Exception as e:
            error_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }
            
            self.training_results[symbol] = error_result
            logger.error(f"❌ {symbol}: Training failed - {e}")
            
            return error_result

    def _log_training_results(self, symbol: str, results: Dict[str, Any]):
        """Log detailed training results"""
        log_path = self.logs_dir / f"{symbol}_training_results.json"
        
        try:
            with open(log_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📝 Training log saved to {log_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save training log for {symbol}: {e}")

    def batch_train(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Train models for multiple symbols"""
        if symbols is None:
            symbols = self.system_config.symbols
    
        if not symbols:
            logger.error("No symbols provided for batch training.")
            return {}
    
        logger.info(f"🚀 Starting batch training for {len(symbols)} symbols")
        logger.info(f"Symbols: {', '.join(symbols)}")

        results = {}
        successful = 0
        failed = 0
        
        for symbol in symbols:
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
                    'status': 'failed',
                    'error': str(e)
                }
                failed += 1
        
        # Save batch summary
        self._save_batch_summary(results, successful, failed)
        
        logger.info("=" * 60)
        logger.info("🎯 BATCH TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📊 Success Rate: {successful/(successful+failed)*100:.1f}%")
        logger.info("=" * 60)
        
        return results

    def _save_batch_summary(self, results: Dict[str, Dict[str, Any]], successful: int, failed: int):
        """Save batch training summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / (successful + failed) if (successful + failed) > 0 else 0,
            'results': results
        }
        
        # Save training summary
        summary_path = self.models_dir / "training_summary.json"
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"📊 Batch summary saved to {summary_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save batch summary: {e}")

    def get_training_status(self) -> Dict[str, str]:
        """Get training status for all symbols"""
        status = {}
        
        for symbol in self.system_config.symbols:
            model_path = self.models_dir / f"{symbol.lower()}_model.pkl"
            
            if model_path.exists():
                # Check if model is recent
                model_age_days = (datetime.now().timestamp() - model_path.stat().st_mtime) / 86400
                
                if model_age_days <= self.system_config.retrain_interval_days:
                    status[symbol] = "up_to_date"
                else:
                    status[symbol] = "needs_retrain"
            else:
                status[symbol] = "no_model"
        
        return status


def main():
    """Main function for standalone execution"""
    try:
        # Load configuration
        from backend.config.training_config import load_config_from_env
        config = load_config_from_env()
        
        # Create trainer
        trainer = EnhancedBatchTrainer(config)
        
        # Run batch training
        results = trainer.batch_train()
        
        # Print final summary
        successful_models = [s for s, r in results.items() if r.get('validation_passed', False)]
        print(f"\n🎉 Training completed!")
        print(f"✅ Successfully trained models: {', '.join(successful_models)}")
        
    except Exception as e:
        logger.error(f"💥 Batch training failed: {e}")
        raise


if __name__ == "__main__":
    main()