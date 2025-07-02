from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Import all route modules ---
from backend.routes.agent import router as agent_router
from backend.routes.strategy import router as strategy_router
from backend.routes.mother_ai import router as mother_ai_router
from backend.routes.backtest import router as backtest_router
from backend.routes.binance import router as binance_router
from backend.routes.agent_registry import router as agents_router
from backend.routes.strategy_file_loader import router as strategy_file_loader_router  # ✅ fixed
from backend.routes.strategy import router as strategy_router

# --- Initialize FastAPI app ---
app = FastAPI(
    title="AI Trading Dashboard API",
    description="Backend API for AI-driven trading system with multi-agent architecture.",
    version="1.0.0"
)

# --- CORS config for frontend (localhost:5173 = Vite) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Root health check route ---
@app.get("/")
async def read_root():
    return {"message": "Backend is running!"}

# --- Register all routers ---
app.include_router(agent_router, prefix="/api/agent", tags=["Agent"])
app.include_router(strategy_router, prefix="/api/strategy", tags=["Strategy"])
app.include_router(mother_ai_router, prefix="/api/mother-ai", tags=["Mother AI"])
app.include_router(backtest_router, prefix="/api/backtest", tags=["Backtesting"])
app.include_router(binance_router, prefix="/api/binance", tags=["Binance"])
app.include_router(agents_router, prefix="/api/agents", tags=["Agent Registry"])
app.include_router(strategy_file_loader_router, prefix="/api/strategies", tags=["Strategy Files"])  # ✅ JSON list endpoint
# app.include_router(strategy_router, prefix="/api", tags=["Strategy"]) # Removed duplicate