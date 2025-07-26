#!/usr/bin/env python3
"""
AI Trading Dashboard - Management Script
Simple CLI for common operations
"""
import sys
import os
import asyncio
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup basic logging"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class TradingSystemManager:
    """Management interface for the trading system"""
    
    def __init__(self):
        self.project_root = project_root
        self.data_dir = project_root / "data"
        self.models_dir = project_root / "backend" / "agents" / "models"
        self.logs_dir = project_root / "backend" / "storage"

    def status(self):
        """Show system status"""
        print("🔍 AI Trading Dashboard - System Status")
        print("=" * 50)
        
        # Check directories
        dirs_to_check = [
            ("Data Directory", self.data_dir / "ohlcv"),
            ("Labels Directory", self.data_dir / "labels"),
            ("Models Directory", self.models_dir),
            ("Logs Directory", self.logs_dir)
        ]
        
        for name, path in dirs_to_check:
            status = "✅ EXISTS" if path.exists() else "❌ MISSING"
            print(f"{name:20}: {status}")
        
        print()
        
        # Check available data
        ohlcv_dir = self.data_dir / "ohlcv"
        if ohlcv_dir.exists():
            ohlcv_files = list(ohlcv_dir.glob("*_1h.csv"))
            print(f"📊 OHLCV Data Files: {len(ohlcv_files)}")
            for file in ohlcv_files[:5]:  # Show first 5
                symbol = file.stem.replace('_1h', '')
                print(f"   - {symbol}")
            if len(ohlcv_files) > 5:
                print(f"   ... and {len(ohlcv_files) - 5} more")
        else:
            print("📊 OHLCV Data Files: 0 (directory not found)")
        
        print()
        
        # Check trained models
        if self.models_dir.exists():
            model_files = list(self.models_dir.glob("*_model.pkl"))
            print(f"🧠 Trained Models: {len(model_files)}")
            for file in model_files[:5]:  # Show first 5
                symbol = file.stem.replace('_model', '').upper()
                modified = datetime.fromtimestamp(file.stat().st_mtime)
                print(f"   - {symbol} (modified: {modified.strftime('%Y-%m-%d %H:%M')})")
            if len(model_files) > 5:
                print(f"   ... and {len(model_files) - 5} more")
        else:
            print("🧠 Trained Models: 0 (directory not found)")
        
        print()
        
        # Check training summary
        summary_file = self.models_dir / "training_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                print(f"📈 Last Training: {summary.get('timestamp', 'Unknown')}")
                print(f"✅ Success Rate: {summary.get('success_rate', 0):.1%}")
                print(f"📊 Total Symbols: {summary.get('total_symbols', 0)}")
            except Exception as e:
                print(f"📈 Training Summary: Error reading file - {e}")
        else:
            print("📈 Training Summary: Not available")

    async def train(self, symbols=None, force=False):
        """Run model training"""
        print("🚀 Starting model training...")
        
        try:
            from backend.startup_trainer import run_startup_training
            await run_startup_training(force_retrain=force)
            print("✅ Training completed successfully")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
        
        return True

    async def generate_labels(self, symbols=None):
        """Generate training labels"""
        print("🏷️ Generating training labels...")
        
        try:
            from backend.ml_engine.generate_labels import main as generate_labels_main
            generate_labels_main()
            print("✅ Label generation completed")
        except Exception as e:
            print(f"❌ Label generation failed: {e}")
            return False
        
        return True

    def clean(self, what="all"):
        """Clean generated files"""
        print(f"🧹 Cleaning {what}...")
        
        cleaned = 0
        
        if what in ["all", "models"]:
            if self.models_dir.exists():
                for file in self.models_dir.glob("*.pkl"):
                    file.unlink()
                    cleaned += 1
                    print(f"   Removed model: {file.name}")
        
        if what in ["all", "labels"]:
            labels_dir = self.data_dir / "labels"
            if labels_dir.exists():
                for file in labels_dir.glob("*.csv"):
                    file.unlink()
                    cleaned += 1
                    print(f"   Removed labels: {file.name}")
        
        if what in ["all", "logs"]:
            logs_dir = self.logs_dir / "training_logs"
            if logs_dir.exists():
                for file in logs_dir.glob("*.json"):
                    file.unlink()
                    cleaned += 1
                    print(f"   Removed log: {file.name}")
        
        print(f"✅ Cleaned {cleaned} files")

    def test_system(self):
        """Run system tests"""
        print("🧪 Running system tests...")
        
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Import core modules
        tests_total += 1
        try:
            from backend.agents.generic_agent import GenericAgent
            from backend.startup_trainer import auto_trainer
            print("✅ Core modules import successfully")
            tests_passed += 1
        except ImportError as e:
            print(f"❌ Core module import failed: {e}")
        
        # Test 2: Check configuration
        tests_total += 1
        try:
            from backend.config.training_config import get_default_config
            config = get_default_config()
            assert len(config['system_config'].symbols) > 0
            print("✅ Configuration loaded successfully")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
        
        # Test 3: Check data directories
        tests_total += 1
        required_dirs = [self.data_dir, self.models_dir, self.logs_dir]
        if all(d.exists() for d in required_dirs):
            print("✅ Required directories exist")
            tests_passed += 1
        else:
            print("❌ Some required directories missing")
        
        # Test 4: Feature extraction
        tests_total += 1
        try:
            import pandas as pd
            import numpy as np
            from backend.ml_engine.feature_extractor import extract_features
            
            # Create dummy OHLCV data
            dates = pd.date_range('2024-01-01', periods=100, freq='H')
            dummy_data = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 102,
                'low': np.random.randn(100).cumsum() + 98,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randn(100) * 1000 + 5000
            }, index=dates)
            
            features = extract_features(dummy_data)
            assert not features.empty
            print("✅ Feature extraction works")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Feature extraction test failed: {e}")
        
        print(f"\n📊 Tests Results: {tests_passed}/{tests_total} passed")
        return tests_passed == tests_total

    def backup(self, target_dir=None):
        """Backup models and configurations"""
        if target_dir is None:
            target_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = Path(target_dir)
        backup_path.mkdir(exist_ok=True)
        
        print(f"💾 Creating backup in {backup_path}")
        
        import shutil
        backed_up = 0
        
        # Backup models
        if self.models_dir.exists():
            models_backup = backup_path / "models"
            shutil.copytree(self.models_dir, models_backup, dirs_exist_ok=True)
            backed_up += len(list(models_backup.glob("*.pkl")))
            print(f"   Backed up models directory")
        
        # Backup labels
        labels_dir = self.data_dir / "labels"
        if labels_dir.exists():
            labels_backup = backup_path / "labels"
            shutil.copytree(labels_dir, labels_backup, dirs_exist_ok=True)
            print(f"   Backed up labels directory")
        
        # Backup configurations
        config_files = [".env", "config/training_config.py"]
        for config_file in config_files:
            src = self.project_root / config_file
            if src.exists():
                dst = backup_path / config_file
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"   Backed up {config_file}")
        
        print(f"✅ Backup completed: {backed_up} files in {backup_path}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="AI Trading Dashboard Management Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage.py status                    # Show system status
  python manage.py train                     # Train all models
  python manage.py train --force             # Force retrain all models
  python manage.py generate-labels           # Generate training labels
  python manage.py clean models              # Clean trained models
  python manage.py test                      # Run system tests
  python manage.py backup                    # Backup models and config
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--force', action='store_true', help='Force retrain existing models')
    train_parser.add_argument('--symbols', nargs='+', help='Specific symbols to train')
    
    # Generate labels command
    labels_parser = subparsers.add_parser('generate-labels', help='Generate training labels')
    labels_parser.add_argument('--symbols', nargs='+', help='Specific symbols to process')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean generated files')
    clean_parser.add_argument('target', choices=['all', 'models', 'labels', 'logs'], 
                             default='all', nargs='?', help='What to clean')
    
    # Test command
    subparsers.add_parser('test', help='Run system tests')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup models and configuration')
    backup_parser.add_argument('--target', help='Backup directory path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = TradingSystemManager()
    
    try:
        if args.command == 'status':
            manager.status()
        
        elif args.command == 'train':
            asyncio.run(manager.train(
                symbols=args.symbols,
                force=args.force
            ))
        
        elif args.command == 'generate-labels':
            asyncio.run(manager.generate_labels(symbols=args.symbols))
        
        elif args.command == 'clean':
            manager.clean(args.target)
        
        elif args.command == 'test':
            success = manager.test_system()
            sys.exit(0 if success else 1)
        
        elif args.command == 'backup':
            manager.backup(args.target)
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"💥 Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()