from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


# ========== Trade Log ==========

class TradeLog(BaseModel):
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position: Literal["long", "short"]
    result: Literal["win", "loss", "breakeven"]
    strategy_name: str
    indicators_used: List[str]
    confidence_score: Optional[float]
    roi: Optional[float] = Field(None, description="Return on Investment %")
    drawdown: Optional[float]
    duration: Optional[float] = Field(None, description="Trade duration in minutes")
    notes: Optional[str]


# ========== Strategy Config ==========

class IndicatorCondition(BaseModel):
    indicator: Literal["rsi", "macd", "ema", "sma"]
    condition: str  # Example: "rsi < 30"
    timeframe: str = "1m"  # Default 1 minute

class StrategyConfig(BaseModel):
    name: str
    symbol: str
    entry_conditions: List[IndicatorCondition]
    exit_conditions: List[IndicatorCondition]
    created_at: datetime
    notes: Optional[str]


# ========== Performance Summary ==========

class StrategyPerformance(BaseModel):
    strategy_name: str
    symbol: str
    total_trades: int
    win_rate: float
    avg_roi: float
    max_drawdown: float
    sharpe_ratio: Optional[float]
    last_tested: datetime


# ========== (Optional) ML Features Schema ==========

class FeatureVector(BaseModel):
    timestamp: datetime
    symbol: str
    rsi: float
    macd: float
    ema: float
    label: Optional[Literal["buy", "sell", "hold"]]
