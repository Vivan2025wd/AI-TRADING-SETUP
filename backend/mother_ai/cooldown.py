import time
from typing import Dict, Optional


class CooldownManager:
    """Manages trading cooldowns for symbols"""
    
    def __init__(self, default_cooldown_seconds: int = 120):
        self.cooldown_tracker: Dict[str, float] = {}
        self.default_cooldown_seconds = default_cooldown_seconds
        self.symbol_specific_cooldowns: Dict[str, int] = {}
    
    def set_cooldown(self, symbol: str, cooldown_seconds: Optional[int] = None):
        """Set cooldown for a specific symbol"""
        if cooldown_seconds is None:
            cooldown_seconds = self.symbol_specific_cooldowns.get(symbol, self.default_cooldown_seconds)
        
        self.cooldown_tracker[symbol] = time.time()
        print(f"⏰ Cooldown set for {symbol}: {cooldown_seconds}s")
    
    def is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is currently in cooldown"""
        if symbol not in self.cooldown_tracker:
            return False
        
        cooldown_seconds = self.symbol_specific_cooldowns.get(symbol, self.default_cooldown_seconds)
        elapsed_time = time.time() - self.cooldown_tracker[symbol]
        
        return elapsed_time < cooldown_seconds
    
    def get_cooldown_remaining(self, symbol: str) -> float:
        """Get remaining cooldown time in seconds"""
        if symbol not in self.cooldown_tracker:
            return 0.0
        
        cooldown_seconds = self.symbol_specific_cooldowns.get(symbol, self.default_cooldown_seconds)
        elapsed_time = time.time() - self.cooldown_tracker[symbol]
        remaining = cooldown_seconds - elapsed_time
        
        return max(0.0, remaining)
    
    def set_symbol_cooldown(self, symbol: str, cooldown_seconds: int):
        """Set specific cooldown duration for a symbol"""
        self.symbol_specific_cooldowns[symbol] = cooldown_seconds
        print(f"🔧 Custom cooldown set for {symbol}: {cooldown_seconds}s")
    
    def clear_cooldown(self, symbol: str):
        """Manually clear cooldown for a symbol"""
        if symbol in self.cooldown_tracker:
            del self.cooldown_tracker[symbol]
            print(f"🚫 Cooldown cleared for {symbol}")
    
    def clear_all_cooldowns(self):
        """Clear all cooldowns"""
        self.cooldown_tracker.clear()
        print("🚫 All cooldowns cleared")
    
    def get_cooldown_status(self) -> Dict[str, Dict]:
        """Get status of all cooldowns"""
        status = {}
        current_time = time.time()
        
        for symbol, start_time in self.cooldown_tracker.items():
            cooldown_seconds = self.symbol_specific_cooldowns.get(symbol, self.default_cooldown_seconds)
            elapsed = current_time - start_time
            remaining = max(0.0, cooldown_seconds - elapsed)
            
            status[symbol] = {
                "in_cooldown": remaining > 0,
                "remaining_seconds": remaining,
                "total_cooldown": cooldown_seconds,
                "elapsed_seconds": elapsed
            }
        
        return status
    
    def cleanup_expired_cooldowns(self):
        """Remove expired cooldowns from tracker"""
        current_time = time.time()
        expired_symbols = []
        
        for symbol, start_time in self.cooldown_tracker.items():
            cooldown_seconds = self.symbol_specific_cooldowns.get(symbol, self.default_cooldown_seconds)
            if current_time - start_time >= cooldown_seconds:
                expired_symbols.append(symbol)
        
        for symbol in expired_symbols:
            del self.cooldown_tracker[symbol]
        
        if expired_symbols:
            print(f"🧹 Cleaned up expired cooldowns for: {expired_symbols}")
    
    def update_default_cooldown(self, cooldown_seconds: int):
        """Update default cooldown duration"""
        self.default_cooldown_seconds = cooldown_seconds
        print(f"⚙️ Default cooldown updated to {cooldown_seconds}s")