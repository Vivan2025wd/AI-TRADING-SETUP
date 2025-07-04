from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Import route modules ---
from backend.routes.agent import router as agent_router                   # Core agent routes + predictions
from backend.routes.strategy import router as strategy_router             # Strategy CRUD
from backend.routes.mother_ai import router as mother_ai_router           # Mother AI logic
from backend.routes.backtest import router as backtest_router             # Backtest strategies
from backend.routes.binance import router as binance_router               # Binance data utils
from backend.routes.agent_registry import router as agents_router         # Lists agents in /agents/*.py
from backend.routes.strategy_file_loader import router as strategy_file_loader_router  # Loads .json strategies
from backend.routes import binance

# --- Initialize FastAPI app ---
app = FastAPI(
    title="AI Trading Dashboard API",
    description="Backend API for AI-driven trading system with multi-agent architecture.",
    version="1.0.0"
)

# --- CORS (frontend access config) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health check route ---
@app.get("/")
async def read_root():
    return {"message": "Backend is running!"}

# --- Register API routers ---
app.include_router(agent_router, prefix="/api/agent", tags=["Agent"])                  # agent prediction + symbol
app.include_router(strategy_router, prefix="/api/strategy", tags=["Strategy"])         # save/load/delete
app.include_router(mother_ai_router, prefix="/api/mother-ai", tags=["Mother AI"])      # mother decision logic
app.include_router(backtest_router, prefix="/api/backtest", tags=["Backtesting"])      # strategy test engine
app.include_router(binance_router, prefix="/api/binance", tags=["Binance"])            # OHLCV fetch
app.include_router(agents_router, prefix="/api/agents", tags=["Agent Registry"])       # shows real agents
app.include_router(strategy_file_loader_router, prefix="/api/strategies", tags=["Strategy Files"])  # load .json
app.include_router(binance.router, prefix="/api/binance")
# --- Add more routes or events here if needed ---
