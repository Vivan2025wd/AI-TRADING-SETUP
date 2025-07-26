from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
from contextlib import asynccontextmanager
import os

# --- Import route modules ---
from backend.routes.agent import router as agent_router                   # Core agent routes + predictions
from backend.routes.strategy import router as strategy_router             # Strategy CRUD (save/load/delete)
from backend.routes.mother_ai import router as mother_ai_router           # Mother AI logic (decision, execute, profits)
from backend.routes.backtest import router as backtest_router             # Backtest strategies
from backend.routes.binance import router as binance_router               # Binance OHLCV utils
from backend.routes.agent_registry import router as agents_router         # Lists all agents
from backend.routes.strategy_file_loader import router as strategy_file_loader_router  # Load saved .json strategies

# --- Import startup training system ---
from backend.startup_trainer import run_startup_training, get_training_status

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting AI Trading Dashboard...")
    
    # Check if we should skip training (useful for development)
    skip_training = os.getenv("SKIP_STARTUP_TRAINING", "false").lower() == "true"
    force_retrain = os.getenv("FORCE_RETRAIN", "false").lower() == "true"
    
    if skip_training:
        logger.info("⏭️ Skipping startup training (SKIP_STARTUP_TRAINING=true)")
    else:
        try:
            # Run startup training in background
            logger.info("🎯 Initializing model training system...")
            await run_startup_training(force_retrain=force_retrain)
            logger.info("✅ Startup training initialization completed")
        except Exception as e:
            logger.error(f"❌ Startup training failed: {e}")
            # Don't crash the server, but log the error
            logger.warning("⚠️ Server will continue without trained models")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI Trading Dashboard...")


# --- Initialize FastAPI app with lifespan ---
app = FastAPI(
    title="AI Trading Dashboard API",
    description="Backend API for AI-driven trading system with multi-agent architecture.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS config ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite frontend
        "http://localhost:5174",  # Optional second frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Root health check ---
@app.get("/")
async def read_root():
    return {"message": "Backend is running!"}


# --- Training status endpoint ---
@app.get("/api/training/status")
async def get_system_training_status():
    """Get current model training status"""
    try:
        status = get_training_status()
        return {
            "status": "success",
            "training_status": status,
            "total_symbols": len(status),
            "trained_count": sum(1 for s in status.values() if s in ["trained", "up_to_date"]),
            "failed_count": sum(1 for s in status.values() if "failed" in s),
            "pending_count": sum(1 for s in status.values() if s == "pending")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# --- Manual training trigger ---
@app.post("/api/training/retrain")
async def trigger_manual_retrain(force: bool = False):
    """Manually trigger model retraining"""
    try:
        logger.info(f"🔄 Manual retrain triggered (force={force})")
        
        # Run training in background task
        asyncio.create_task(run_startup_training(force_retrain=force))
        
        return {
            "status": "success",
            "message": "Model retraining started in background",
            "force_retrain": force
        }
    except Exception as e:
        logger.error(f"Failed to start manual retrain: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# --- Register routers ---
app.include_router(agent_router, prefix="/api/agent", tags=["Agent"])                         # Symbol agents
app.include_router(strategy_router, prefix="/api/strategy", tags=["Strategy"])                # JSON strategy management
app.include_router(mother_ai_router, prefix="/api/mother-ai", tags=["Mother AI"])             # AI logic, decisions, profits
app.include_router(backtest_router, prefix="/api/backtest", tags=["Backtesting"])             # Run strategy tests
app.include_router(binance_router, prefix="/api/binance", tags=["Binance"])                   # OHLCV fetcher
app.include_router(agents_router, prefix="/api/agents", tags=["Agent Registry"])              # Registry for active agents
app.include_router(strategy_file_loader_router, prefix="/api/strategies", tags=["Strategy Files"])  # Load JSON strategies


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)