# 🚀 AI Trading Dashboard - Automated Startup Training

This system automatically generates training labels and trains ML models when the server starts up. Here's everything you need to know!

## 🎯 Quick Start

### 1. Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your preferences
nano .env
```

### 2. Prepare Data
Place your OHLCV data files in `data/ohlcv/`:
```
data/ohlcv/
├── BTCUSDT_1h.csv
├── ETHUSDT_1h.csv
├── DOGEUSDT_1h.csv
└── ...
```

**Required CSV format:**
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42000.0,42500.0,41800.0,42200.0,1250.5
2024-01-01 01:00:00,42200.0,42800.0,42000.0,42600.0,1180.2
...
```

### 3. Start the Server
```bash
# Method 1: Using the startup script (recommended)
python run_server.py

# Method 2: Direct FastAPI
python main.py

# Method 3: With uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_STARTUP_TRAINING` | `false` | Skip training on startup |
| `FORCE_RETRAIN` | `false` | Force retrain all models |
| `TRAINING_MAX_WORKERS` | `3` | Parallel training jobs |
| `LABEL_FORWARD_WINDOW` | `24` | Hours to look ahead |
| `LABEL_MIN_RETURN` | `0.025` | Min return for buy signals (2.5%) |
| `LABEL_STOP_LOSS` | `-0.04` | Stop loss threshold (-4%) |
| `MODEL_MIN_ACCURACY` | `0.6` | Minimum model accuracy |
| `RETRAIN_INTERVAL_DAYS` | `7` | Days before retraining |

### Training Symbols
```bash
# Train specific symbols only
TRAINING_SYMBOLS=BTCUSDT,ETHUSDT,DOGEUSDT

# Leave empty for default list (10 major coins)
TRAINING_SYMBOLS=
```

## 🛠️ Management Commands

Use the `manage.py` script for system operations:

```bash
# Check system status
python manage.py status

# Train all models
python manage.py train

# Force retrain existing models
python manage.py train --force

# Train specific symbols
python manage.py train --symbols BTCUSDT ETHUSDT

# Generate labels only
python manage.py generate-labels

# Clean old files
python manage.py clean models    # Remove models
python manage.py clean labels    # Remove labels
python manage.py clean all       # Remove everything

# Run system tests
python manage.py test

# Backup models and config
python manage.py backup
```

## 📊 Training Process

### 1. Label Generation
- Analyzes future price movements
- Creates `buy`, `sell`, and `hold` labels
- Uses volatility-adjusted thresholds
- Balances classes for better training

### 2. Feature Extraction
- Technical indicators (SMA, RSI, MACD, etc.)
- Price momentum and volatility
- Volume analysis
- Trend indicators

### 3. Model Training
- Random Forest classifier
- Cross-validation
- Performance metrics
- Model validation

### 4. Model Validation
Models must meet minimum requirements:
- Accuracy ≥ 60%
- Precision ≥ 60%
- Recall ≥ 60%
- F1 Score ≥ 60%

## 🔄 Automatic Retraining

Models are automatically retrained when:
- Model is older than 7 days (configurable)
- Model performance drops below threshold
- No model exists for a symbol

## 📈 Monitoring Training

### API Endpoints
```bash
# Check training status
GET /api/training/status

# Trigger manual retrain
POST /api/training/retrain?force=true
```

### Log Files
- Server logs: `backend/storage/logs/server.log`
- Training logs: `backend/storage/training_logs/`
- Model metadata: `backend/agents/models/training_summary.json`

## 🚨 Troubleshooting

### Common Issues

**1. No OHLCV data found**
```bash
# Check data directory
ls -la data/ohlcv/

# Validate file format
head -5 data/ohlcv/BTCUSDT_1h.csv
```

**2. Training fails**
```bash
# Check logs
tail -50 backend/storage/logs/server.log

# Run system tests
python manage.py test

# Check training status
python manage.py status
```

**3. Models not loading**
```bash
# Check model files
ls -la backend/agents/models/

# Retrain specific symbol
python manage.py train --symbols BTCUSDT --force
```

**4. Out of memory during training**
```bash
# Reduce parallel workers
export TRAINING_MAX_WORKERS=1

# Or in .env file
TRAINING_MAX_WORKERS=1
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Skip training for development
export SKIP_STARTUP_TRAINING=true
```

## 📁 Directory Structure

```
project/
├── data/
│   ├── ohlcv/              # Raw OHLCV data
│   └── labels/             # Generated training labels
├── backend/
│   ├── agents/
│   │   └── models/         # Trained ML models
│   ├── storage/
│   │   ├── logs/           # Server logs
│   │   └── training_logs/  # Training logs
│   ├── ml_engine/          # Training code
│   └── startup_trainer.py  # Main training system
├── config/
│   └── training_config.py  # Training configuration
├── .env                    # Environment variables
├── run_server.py           # Server startup script
└── manage.py               # Management CLI
```

## 🎉 Success Indicators

When everything works correctly, you'll see:

```
🎯 Starting parallel training for 10 symbols...
✅ BTCUSDT: Generated 2500 labels, balance ratio: 0.85
🧠 Training model for BTCUSDT...
✅ BTCUSDT: Model training completed successfully
   Test Accuracy: 0.672
   CV Score: 0.658 ± 0.021
...
📊 Training completed: 10 successful, 0 failed
🎉 Startup training completed successfully
🚀 Starting server on http://0.0.0.0:8000
```

## 🔗 Integration with Frontend

The trained models automatically integrate with your existing routes:
- `/api/agent/{symbol}/predict` - Uses trained models
- `/api/agents/predictions` - Batch predictions
- `/api/training/status` - Training status

Models seamlessly blend with rule-based strategies for optimal performance!

## 💡 Best Practices

1. **Data Quality**: Ensure clean, complete OHLCV data
2. **Regular Updates**: Keep data current for best results
3. **Monitor Performance**: Check training logs regularly
4. **Backup Models**: Use `manage.py backup` before major changes
5. **Test Changes**: Run `manage.py test` after configuration changes

## 🆘 Getting Help

1. Check the logs: `tail -f backend/storage/logs/server.log`
2. Run diagnostics: `python manage.py status`
3. Test system: `python manage.py test`
4. Review configuration: Check `.env` file settings

Happy trading! 🚀📈