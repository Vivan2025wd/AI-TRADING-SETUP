from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Import route modules ---
from backend.routes.agent import router as agent_router                   # Core agent routes + predictions
from backend.routes.strategy import router as strategy_router             # Strategy CRUD (save/load/delete)
from backend.routes.mother_ai import router as mother_ai_router           # Mother AI logic (decision, execute, profits)
from backend.routes.backtest import router as backtest_router             # Backtest strategies
from backend.routes.binance import router as binance_router               # Binance OHLCV utils
from backend.routes.agent_registry import router as agents_router         # Lists all agents
from backend.routes.strategy_file_loader import router as strategy_file_loader_router  # Load saved .json strategies

# --- Initialize FastAPI app ---
app = FastAPI(
    title="AI Trading Dashboard API",
    description="Backend API for AI-driven trading system with multi-agent architecture.",
    version="1.0.0"
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

# --- Register routers ---
app.include_router(agent_router, prefix="/api/agent", tags=["Agent"])                         # Symbol agents
app.include_router(strategy_router, prefix="/api/strategy", tags=["Strategy"])                # JSON strategy management
app.include_router(mother_ai_router, prefix="/api/mother-ai", tags=["Mother AI"])             # AI logic, decisions, profits
app.include_router(backtest_router, prefix="/api/backtest", tags=["Backtesting"])             # Run strategy tests
app.include_router(binance_router, prefix="/api/binance", tags=["Binance"])                   # OHLCV fetcher
app.include_router(agents_router, prefix="/api/agents", tags=["Agent Registry"])              # Registry for active agents
app.include_router(strategy_file_loader_router, prefix="/api/strategies", tags=["Strategy Files"])  # Load JSON strategies

# --- Custom events or startup hooks can go here ---
