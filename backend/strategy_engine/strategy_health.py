import math
from typing import List, Dict, Any, Union, Optional

# Handle numpy import safely
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

class StrategyHealth:
    def __init__(self, performance_log: List[Dict[str, Any]]):
        """
        Initializes with a list of trade dictionaries.
        Each trade dict should include keys like 'result' ('win'/'loss') and 'roi' (float).
        """
        self.performance_log = performance_log

    def _safe_float(self, value: Any, default: float = 0.0, max_value: float = 999999.0) -> float:
        """
        Convert potentially problematic values to JSON-safe float values.
        Handles inf, -inf, NaN values, and numpy types.
        """
        try:
            # Convert to Python float first
            if HAS_NUMPY and hasattr(value, 'item'):
                # Handle numpy scalars
                float_val = float(value.item())
            else:
                float_val = float(value)
            
            # Check for problematic values
            if math.isnan(float_val) or math.isinf(float_val):
                return default
            
            # Cap extremely large values
            if abs(float_val) > max_value:
                return max_value if float_val > 0 else -max_value
                
            return float_val
            
        except (ValueError, TypeError, AttributeError):
            return default

    def win_rate(self, lookback: int = 20) -> float:
        """
        Calculate the win rate over the last `lookback` trades.
        Win rate = number of wins / total trades.
        """
        trades = self.performance_log[-lookback:] if lookback > 0 else self.performance_log
        if not trades:
            return 0.0
        
        wins = sum(1 for trade in trades if trade.get("result") == "win")
        rate = wins / len(trades) if len(trades) > 0 else 0.0
        return self._safe_float(rate)

    def avg_profit(self, lookback: int = 20) -> float:
        """
        Calculate average ROI (return on investment) over the last `lookback` trades.
        """
        trades = self.performance_log[-lookback:] if lookback > 0 else self.performance_log
        if not trades:
            return 0.0
        
        profits = [trade.get("roi", 0.0) for trade in trades if "roi" in trade]
        if not profits:
            return 0.0
        
        if HAS_NUMPY and np is not None:
            avg = np.mean(profits)
        else:
            avg = sum(profits) / len(profits)
        
        return self._safe_float(avg)

    def recent_drawdown(self, lookback: int = 20) -> float:
        """
        Calculate the maximum drawdown (largest absolute loss) in the last `lookback` trades.
        """
        trades = self.performance_log[-lookback:] if lookback > 0 else self.performance_log
        losses = [abs(trade.get("roi", 0.0)) for trade in trades if trade.get("result") == "loss"]
        if not losses:
            return 0.0
        
        max_loss = max(losses)
        return self._safe_float(max_loss)

    def profit_factor(self, lookback: int = 20) -> float:
        """
        Calculate profit factor: total profits / total losses.
        Values > 1.0 indicate profitable strategy.
        Returns a capped value to avoid JSON serialization issues.
        """
        trades = self.performance_log[-lookback:] if lookback > 0 else self.performance_log
        if not trades:
            return 0.0
        
        total_profits = sum(trade.get("roi", 0.0) for trade in trades if trade.get("roi", 0.0) > 0)
        total_losses = abs(sum(trade.get("roi", 0.0) for trade in trades if trade.get("roi", 0.0) < 0))
        
        if total_losses == 0:
            # If no losses, return a high but finite value
            return 999.0 if total_profits > 0 else 0.0
        
        profit_factor_val = total_profits / total_losses
        return self._safe_float(profit_factor_val, default=0.0, max_value=999.0)

    def max_consecutive_losses(self, lookback: int = 20) -> int:
        """
        Calculate maximum consecutive losses in recent trades.
        """
        trades = self.performance_log[-lookback:] if lookback > 0 else self.performance_log
        if not trades:
            return 0
        
        max_losses = current_losses = 0
        for trade in trades:
            if trade.get("result") == "loss":
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses

    def max_consecutive_wins(self, lookback: int = 20) -> int:
        """
        Calculate maximum consecutive wins in recent trades.
        """
        trades = self.performance_log[-lookback:] if lookback > 0 else self.performance_log
        if not trades:
            return 0
        
        max_wins = current_wins = 0
        for trade in trades:
            if trade.get("result") == "win":
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0
        
        return max_wins

    def volatility(self, lookback: int = 20) -> float:
        """
        Calculate volatility (standard deviation) of returns.
        """
        trades = self.performance_log[-lookback:] if lookback > 0 else self.performance_log
        if not trades:
            return 0.0
        
        returns = [trade.get("roi", 0.0) for trade in trades if "roi" in trade]
        if len(returns) <= 1:
            return 0.0
        
        if HAS_NUMPY and np is not None:
            vol = np.std(returns)
        else:
            # Manual standard deviation calculation
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            vol = math.sqrt(variance)
        
        return self._safe_float(vol)

    def sharpe_ratio(self, lookback: int = 20, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted returns).
        """
        avg_return = self.avg_profit(lookback)
        vol = self.volatility(lookback)
        
        if vol == 0:
            return 0.0
        
        sharpe = (avg_return - risk_free_rate) / vol
        return self._safe_float(sharpe, default=0.0, max_value=10.0)

    def recovery_factor(self, lookback: int = 20) -> float:
        """
        Calculate recovery factor: total return / max drawdown.
        Higher values indicate better recovery from losses.
        """
        trades = self.performance_log[-lookback:] if lookback > 0 else self.performance_log
        if not trades:
            return 0.0
        
        total_return = sum(trade.get("roi", 0.0) for trade in trades)
        max_dd = self.recent_drawdown(lookback)
        
        if max_dd == 0:
            return 999.0 if total_return > 0 else 0.0
        
        recovery = total_return / max_dd
        return self._safe_float(recovery, default=0.0, max_value=999.0)

    def total_return(self, lookback: int = 20) -> float:
        """
        Calculate total return over the lookback period.
        """
        trades = self.performance_log[-lookback:] if lookback > 0 else self.performance_log
        if not trades:
            return 0.0
        
        total = sum(trade.get("roi", 0.0) for trade in trades)
        return self._safe_float(total)

    def summary(self, lookback: int = 20) -> Dict[str, Any]:
        """
        Returns a comprehensive summary dict with all key metrics.
        All values are JSON-serializable.
        """
        recent_trades = self.performance_log[-lookback:] if lookback > 0 else self.performance_log
        
        return {
            "win_rate": round(self.win_rate(lookback), 3),
            "avg_profit": round(self.avg_profit(lookback), 4),
            "total_return": round(self.total_return(lookback), 4),
            "max_drawdown": round(self.recent_drawdown(lookback), 4),
            "profit_factor": round(self.profit_factor(lookback), 3),
            "max_consecutive_wins": self.max_consecutive_wins(lookback),
            "max_consecutive_losses": self.max_consecutive_losses(lookback),
            "volatility": round(self.volatility(lookback), 4),
            "sharpe_ratio": round(self.sharpe_ratio(lookback), 3),
            "recovery_factor": round(self.recovery_factor(lookback), 3),
            "trades_analyzed": len(recent_trades)
        }

    def health_grade(self, lookback: int = 20) -> Dict[str, Any]:
        """
        Calculate an overall health grade (A-F) based on multiple metrics.
        All values are JSON-serializable.
        """
        metrics = self.summary(lookback)
        
        # Scoring components (0-100 each) with safe calculations
        win_rate_score = min(100.0, max(0.0, float(metrics["win_rate"]) * 100))
        
        # Safe profit factor scoring
        pf = float(metrics["profit_factor"])
        profit_factor_score = min(100.0, max(0.0, (pf - 1) * 50)) if pf != 999.0 else 100.0
        
        # Safe Sharpe ratio scoring
        sharpe = float(metrics["sharpe_ratio"])
        sharpe_score = min(100.0, max(0.0, (sharpe + 1) * 25)) if abs(sharpe) != 10.0 else 50.0
        
        # Drawdown scoring
        drawdown_score = max(0.0, 100.0 - (float(metrics["max_drawdown"]) * 1000))
        
        # Weighted total score
        total_score = (
            win_rate_score * 0.3 +
            profit_factor_score * 0.3 +
            sharpe_score * 0.2 +
            drawdown_score * 0.2
        )
        
        # Assign letter grade
        if total_score >= 90:
            grade = "A+"
        elif total_score >= 85:
            grade = "A"
        elif total_score >= 80:
            grade = "B+"
        elif total_score >= 75:
            grade = "B"  
        elif total_score >= 70:
            grade = "C+"
        elif total_score >= 65:
            grade = "C"
        elif total_score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "grade": grade,
            "score": round(total_score, 1),
            "components": {
                "win_rate_score": round(win_rate_score, 1),
                "profit_factor_score": round(profit_factor_score, 1),
                "sharpe_score": round(sharpe_score, 1),
                "drawdown_score": round(drawdown_score, 1)
            }
        }

    def is_healthy(self, 
                   min_win_rate: float = 0.5,
                   min_profit_factor: float = 1.2,
                   max_drawdown: float = 0.1,
                   min_trades: int = 10,
                   lookback: int = 20) -> Dict[str, Any]:
        """
        Determine if strategy is healthy based on configurable thresholds.
        All values are JSON-serializable.
        """
        metrics = self.summary(lookback)
        issues: List[str] = []
        
        # Check each criterion
        trades_analyzed = int(metrics["trades_analyzed"])
        if trades_analyzed < min_trades:
            issues.append(f"Insufficient trades: {trades_analyzed} < {min_trades}")
        
        win_rate = float(metrics["win_rate"])
        if win_rate < min_win_rate:
            issues.append(f"Low win rate: {win_rate:.1%} < {min_win_rate:.1%}")
        
        # Handle capped profit factor values
        pf = float(metrics["profit_factor"])
        if pf < min_profit_factor and pf != 999.0:
            issues.append(f"Low profit factor: {pf:.2f} < {min_profit_factor}")
        
        max_dd = float(metrics["max_drawdown"])
        if max_dd > max_drawdown:
            issues.append(f"High drawdown: {max_dd:.1%} > {max_drawdown:.1%}")
        
        # Determine overall health
        is_healthy_bool = len(issues) == 0
        status = "HEALTHY" if is_healthy_bool else "UNHEALTHY"
        
        return {
            "is_healthy": is_healthy_bool,
            "status": status,
            "issues": issues,
            "metrics_analyzed": dict(metrics)
        }

    def quick_health_check(self, lookback: int = 20) -> Dict[str, Any]:
        """
        Quick health check with essential metrics only.
        Optimized for API responses.
        """
        if not self.performance_log:
            return {
                "status": "NO_DATA",
                "message": "No trading history available",
                "metrics": {}
            }
        
        recent_trades = self.performance_log[-lookback:] if lookback > 0 else self.performance_log
        
        if len(recent_trades) < 5:
            return {
                "status": "INSUFFICIENT_DATA", 
                "message": f"Only {len(recent_trades)} trades available, need at least 5",
                "metrics": {
                    "trades": len(recent_trades),
                    "win_rate": self.win_rate(lookback),
                    "total_return": self.total_return(lookback)
                }
            }
        
        # Calculate key metrics
        win_rate = self.win_rate(lookback)
        profit_factor = self.profit_factor(lookback)
        total_return = self.total_return(lookback)
        max_drawdown = self.recent_drawdown(lookback)
        
        # Determine status
        if win_rate >= 0.6 and profit_factor >= 1.5 and total_return > 0:
            status = "EXCELLENT"
        elif win_rate >= 0.5 and profit_factor >= 1.2 and total_return > 0:
            status = "GOOD"
        elif win_rate >= 0.4 and profit_factor >= 1.0:
            status = "FAIR"
        else:
            status = "POOR"
        
        return {
            "status": status,
            "message": f"Strategy performance is {status.lower()}",
            "metrics": {
                "trades": len(recent_trades),
                "win_rate": round(win_rate, 3),
                "profit_factor": round(profit_factor, 3),
                "total_return": round(total_return, 4),
                "max_drawdown": round(max_drawdown, 4)
            }
        }

# Example usage and JSON serialization test
if __name__ == "__main__":
    import json
    
    # Test with edge case data
    test_trades = [
        {"result": "win", "roi": 0.05},
        {"result": "win", "roi": 0.08},
        {"result": "win", "roi": 0.02},
        # No losses - would normally cause inf in profit_factor
    ]
    
    health = StrategyHealth(test_trades)
    
    # Test all methods for JSON compatibility
    methods_to_test = [
        ("summary", health.summary()),
        ("health_grade", health.health_grade()),
        ("is_healthy", health.is_healthy()),
        ("quick_health_check", health.quick_health_check())
    ]
    
    for method_name, result in methods_to_test:
        try:
            json_str = json.dumps(result, indent=2)
            print(f"✅ {method_name} JSON serialization successful")
        except (ValueError, TypeError) as e:
            print(f"❌ {method_name} JSON serialization failed: {e}")
            print(f"   Result type: {type(result)}")
            print(f"   Result: {result}")