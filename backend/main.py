from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.agent import router as agent_router
from backend.routes.strategy import router as strategy_router
from backend.routes.mother_ai import router as mother_ai_router
from backend.routes.backtest import router as backtest_router


app = FastAPI(
    title="AI Trading Dashboard API",
    description="Backend API for AI-driven trading system with multi-agent architecture.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Backend is running!"}

app.include_router(agent_router, prefix="/api/agent", tags=["Agent"])
app.include_router(strategy_router, prefix="/api/strategy", tags=["Strategy"])
app.include_router(mother_ai_router, prefix="/api/mother-ai", tags=["Mother AI"])
app.include_router(backtest_router, prefix="/api/backtest", tags=["Backtest"])
