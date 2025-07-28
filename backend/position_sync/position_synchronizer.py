import ccxt
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from backend.utils.binance_api import get_binance_client, get_trading_mode

class PositionSynchronizer:
    """Synchronizes internal agent positions with actual Binance account balances"""
    
    def __init__(self, mother_ai_instance):
        self.mother_ai = mother_ai_instance
        self.last_sync_file = "storage/last_position_sync.json"
        self.position_changes_log = "storage/position_changes.json"
        
    def get_actual_binance_positions(self) -> Dict[str, Dict[str, float]]:
        """Get actual asset balances from Binance account"""
        try:
            if get_trading_mode() == "mock":
                print("🧪 Mock mode: Skipping real balance check")
                return {}
                
            client = get_binance_client()
            account_info = client.get_account()
            
            positions: Dict[str, Dict[str, float]] = {}
            for balance in account_info['balances']:
                asset = balance['asset']
                free_balance = float(balance['free'])
                locked_balance = float(balance['locked'])
                total_balance = free_balance + locked_balance
                
                # Only track non-USDT assets with meaningful balances
                if asset != 'USDT' and total_balance > 0.00001:  # Ignore dust
                    positions[asset] = {
                        'free': free_balance,
                        'locked': locked_balance,
                        'total': total_balance
                    }
                    
            return positions
            
        except Exception as e:
            print(f"⚠️ Failed to fetch Binance positions: {e}")
            return {}
    
    def get_agent_expected_positions(self) -> Dict[str, str]:
        """Get what positions agents think they have"""
        expected_positions: Dict[str, str] = {}
        
        for symbol, agent in self.mother_ai.loaded_agents.items():
            if hasattr(agent, 'position_state') and agent.position_state == 'long':
                # Convert symbol to base asset (e.g., BTCUSDT -> BTC)
                base_asset = symbol.replace('USDT', '').replace('/', '')
                expected_positions[base_asset] = symbol
                
        return expected_positions
    
    def detect_position_discrepancies(self) -> List[Dict]:
        """Detect discrepancies between expected and actual positions"""
        actual_positions = self.get_actual_binance_positions()
        expected_positions = self.get_agent_expected_positions()
        
        discrepancies: List[Dict] = []
        
        # Check for positions agents think they have but don't exist in Binance
        for base_asset, symbol in expected_positions.items():
            if base_asset not in actual_positions:
                discrepancies.append({
                    'type': 'force_sell_detected',
                    'symbol': symbol,
                    'base_asset': base_asset,
                    'agent_state': 'long',
                    'actual_balance': 0.0,
                    'action_needed': 'update_agent_to_flat'
                })
                
        # Check for positions that exist in Binance but agents don't know about
        for base_asset, balance_info in actual_positions.items():
            symbol = f"{base_asset}USDT"
            if base_asset not in expected_positions and symbol in self.mother_ai.loaded_agents:
                agent = self.mother_ai.loaded_agents[symbol]
                agent_state = getattr(agent, 'position_state', None)
                
                if agent_state != 'long':
                    discrepancies.append({
                        'type': 'untracked_position_detected',
                        'symbol': symbol,
                        'base_asset': base_asset,
                        'agent_state': agent_state,
                        'actual_balance': balance_info['total'],  # This should now work correctly
                        'action_needed': 'update_agent_to_long'
                    })
        
        return discrepancies
    
    def sync_positions(self, auto_fix: bool = True) -> Dict:
        """Synchronize agent positions with actual Binance account"""
        print("🔄 Synchronizing positions with Binance account...")
        
        discrepancies = self.detect_position_discrepancies()
        sync_results = {
            'timestamp': datetime.now().isoformat(),
            'discrepancies_found': len(discrepancies),
            'fixes_applied': 0,
            'details': []
        }
        
        if not discrepancies:
            print("✅ All positions are synchronized")
            return sync_results
        
        print(f"⚠️ Found {len(discrepancies)} position discrepancies:")
        
        for discrepancy in discrepancies:
            symbol = discrepancy['symbol']
            discrepancy_type = discrepancy['type']
            
            print(f"\n📊 {symbol}:")
            print(f"   Type: {discrepancy_type}")
            print(f"   Agent State: {discrepancy['agent_state']}")
            print(f"   Actual Balance: {discrepancy['actual_balance']}")
            print(f"   Action Needed: {discrepancy['action_needed']}")
            
            if auto_fix:
                fix_result = self._apply_position_fix(discrepancy)
                discrepancy['fix_result'] = fix_result
                if fix_result['success']:
                    sync_results['fixes_applied'] += 1
                    
            sync_results['details'].append(discrepancy)
        
        # Log the synchronization
        self._log_sync_results(sync_results)
        
        print(f"\n📈 Sync Summary:")
        print(f"   Discrepancies found: {sync_results['discrepancies_found']}")
        print(f"   Fixes applied: {sync_results['fixes_applied']}")
        
        return sync_results
    
    def _apply_position_fix(self, discrepancy: Dict) -> Dict:
        """Apply fix for a position discrepancy"""
        symbol = discrepancy['symbol']
        action = discrepancy['action_needed']
        
        try:
            agent = self.mother_ai.get_agent_by_symbol(symbol)
            if not agent:
                return {'success': False, 'error': 'Agent not found'}
            
            old_state = getattr(agent, 'position_state', None)
            
            if action == 'update_agent_to_flat':
                # Force sell was detected - update agent to flat
                agent.position_state = None
                
                # Log the forced exit
                self._log_forced_exit(symbol, discrepancy)
                
                print(f"   ✅ Updated {symbol} agent: {old_state} → None (force sell detected)")
                
            elif action == 'update_agent_to_long':
                # Untracked position detected - update agent to long
                agent.position_state = 'long'
                
                print(f"   ✅ Updated {symbol} agent: {old_state} → long (untracked position)")
            
            return {'success': True, 'old_state': old_state, 'new_state': agent.position_state}
            
        except Exception as e:
            print(f"   ❌ Failed to fix {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _log_forced_exit(self, symbol: str, discrepancy: Dict):
        """Log a forced exit event to the performance logs"""
        try:
            # Create a forced exit entry in the performance logs
            performance_log_path = f"backend/storage/performance_logs/{symbol}_trades.json"
            
            # Load existing trades
            if os.path.exists(performance_log_path):
                with open(performance_log_path, 'r') as f:
                    trades = json.load(f)
            else:
                trades = []
            
            # Add forced exit entry
            forced_exit_entry = {
                "symbol": symbol,
                "signal": "sell",
                "confidence": 1.0,
                "score": 0.0,
                "last_price": 0.0,  # Unknown price for forced exit
                "price": 0.0,
                "qty": None,
                "timestamp": datetime.now().isoformat(),
                "source": "forced_exit_detected",
                "note": "Position closed externally via Binance interface"
            }
            
            trades.append(forced_exit_entry)
            
            # Save updated trades
            with open(performance_log_path, 'w') as f:
                json.dump(trades, f, indent=2)
                
            print(f"   📝 Logged forced exit for {symbol}")
            
        except Exception as e:
            print(f"   ⚠️ Failed to log forced exit for {symbol}: {e}")
    
    def _log_sync_results(self, results: Dict):
        """Log synchronization results"""
        try:
            # Load existing log
            if os.path.exists(self.position_changes_log):
                with open(self.position_changes_log, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
            
            # Add new results
            log_data.append(results)
            
            # Keep only last 100 sync results
            if len(log_data) > 100:
                log_data = log_data[-100:]
            
            # Save log
            with open(self.position_changes_log, 'w') as f:
                json.dump(log_data, f, indent=2)
                
            # Update last sync timestamp
            with open(self.last_sync_file, 'w') as f:
                json.dump({
                    'last_sync': results['timestamp'],
                    'discrepancies_found': results['discrepancies_found']
                }, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to log sync results: {e}")
    
    def get_sync_status(self) -> Dict:
        """Get last synchronization status"""
        try:
            if os.path.exists(self.last_sync_file):
                with open(self.last_sync_file, 'r') as f:
                    return json.load(f)
            else:
                return {'last_sync': 'never', 'discrepancies_found': 0}
        except:
            return {'last_sync': 'error', 'discrepancies_found': 0}


# Enhanced MotherAI methods to integrate position synchronization
class MotherAIEnhanced:
    """Enhanced MotherAI methods for position synchronization"""
    
    def __init__(self, mother_ai_instance):
        self.mother_ai = mother_ai_instance
        self.position_sync = PositionSynchronizer(mother_ai_instance)
    
    def sync_with_binance(self, auto_fix: bool = True) -> Dict:
        """Manually trigger position synchronization with Binance"""
        return self.position_sync.sync_positions(auto_fix=auto_fix)
    
    def enhanced_portfolio_decision(self, min_score=0.5, sync_positions=True):
        """Enhanced portfolio decision with automatic position synchronization"""
        if sync_positions:
            # Sync positions before making decisions
            sync_results = self.position_sync.sync_positions(auto_fix=True)
            if sync_results['fixes_applied'] > 0:
                print(f"🔄 Applied {sync_results['fixes_applied']} position fixes before decision making")
        
        # Proceed with normal decision making
        return self.mother_ai.make_portfolio_decision(min_score=min_score)
    
    def get_position_report(self) -> Dict:
        """Get comprehensive position report including Binance sync status"""
        # Get current sync status
        sync_status = self.position_sync.get_sync_status()
        
        # Get actual vs expected positions
        actual_positions = self.position_sync.get_actual_binance_positions()
        expected_positions = self.position_sync.get_agent_expected_positions()
        discrepancies = self.position_sync.detect_position_discrepancies()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'sync_status': sync_status,
            'actual_binance_positions': actual_positions,
            'expected_agent_positions': expected_positions,
            'discrepancies': discrepancies,
            'sync_needed': len(discrepancies) > 0
        }


# Usage example functions
def setup_position_monitoring(mother_ai_instance):
    """Setup position monitoring for a MotherAI instance"""
    return MotherAIEnhanced(mother_ai_instance)

def periodic_position_sync(mother_ai_instance, interval_minutes: int = 30):
    """
    Run periodic position synchronization
    This could be called from a scheduler or timer
    """
    enhanced_ai = MotherAIEnhanced(mother_ai_instance)
    
    print(f"🔄 Running periodic position sync (every {interval_minutes} minutes)")
    sync_results = enhanced_ai.sync_with_binance(auto_fix=True)
    
    if sync_results['discrepancies_found'] > 0:
        print(f"⚠️ ALERT: Found {sync_results['discrepancies_found']} position discrepancies!")
        # You could add notification logic here (email, Slack, etc.)
    
    return sync_results