# Main MotherAI file - backward compatibility wrapper
from .mother_ai_exit_manager import MotherAIExitManager
import time
from typing import Dict


class MotherAI(MotherAIExitManager):
    """
    Main MotherAI class - provides complete functionality with backward compatibility
    
    This is the main entry point that users should import. It inherits from:
    - MotherAICore: Basic agent loading and evaluation
    - MotherAITrader: Trade execution and position management  
    - MotherAIExitManager: Exit conditions and portfolio decisions
    
    Usage:
        # Backward compatible - existing code should work unchanged
        mother_ai = MotherAI(
            agent_symbols=["BTCUSDT", "ETHUSDT"], 
            data_interval="1m"
        )
        
        # All existing methods work the same way
        decision = mother_ai.make_portfolio_decision(min_score=0.7)
        status = mother_ai.get_agent_status_summary()
        predictions = mother_ai.load_all_predictions()
    """
    
    def __init__(self, **kwargs):
        """Initialize MotherAI with full functionality"""
        super().__init__(**kwargs)
        print(f"✅ MotherAI fully initialized with complete functionality")
        print(f"📊 Components loaded: Core + Trader + ExitManager")


# Enhanced utility functions for backward compatibility
def enhanced_trading_loop_with_smart_exits():
    """Enhanced trading loop with intelligent exit timing"""
    mother_ai = MotherAI(
        agent_symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
        data_interval="1m"
    )
    
    # Configure timing
    mother_ai.set_minimum_hold_time(900)  # 15 minutes
    mother_ai.set_exit_check_interval(300)  # 5 minutes
    
    decision_count = 0
    
    while True:
        try:
            decision_count += 1
            print(f"\n{'='*80}")
            print(f"DECISION CYCLE #{decision_count}")
            print(f"{'='*80}")
            
            # Show current hold times
            hold_times = mother_ai.get_position_hold_times()
            if hold_times:
                print(f"📊 Current Position Hold Times:")
                for symbol, info in hold_times.items():
                    print(f"   {symbol}: {info['hold_duration_minutes']:.1f}min "
                          f"(Min hold: {'✅' if info['meets_minimum_hold'] else '❌'})")
            
            # Make decision (with smart exit logic)
            decision_result = mother_ai.make_portfolio_decision(min_score=0.6)
            
            # Every 5th cycle, run strategic exits
            if decision_count % 5 == 0:
                print(f"\n🎯 Running strategic exit analysis (cycle #{decision_count})")
                mother_ai.check_strategic_exits()
            
            print(f"📊 Decision result: {decision_result}")
            time.sleep(180)  # 3 minutes between decisions
            
        except KeyboardInterrupt:
            print("🛑 Trading loop stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in trading loop: {e}")
            time.sleep(60)


def get_position_status_report(): 
    """Get detailed position status including hold times"""
    mother_ai = MotherAI()
    
    print("\n📊 POSITION STATUS REPORT")
    print("=" * 60)
    
    status = mother_ai.get_agent_status_summary()
    hold_times = mother_ai.get_position_hold_times()
    
    print("Configuration:")
    print(f"  Minimum hold time: {status['minimum_hold_time']}s ({status['minimum_hold_time']/60:.1f}min)")
    print(f"  Exit check interval: {status['exit_check_interval']}s ({status['exit_check_interval']/60:.1f}min)")
    
    active_positions = [s for s, info in status['agents'].items() 
                        if info['position_state'] == 'long']
    
    print(f"\nActive Positions: {len(active_positions)}")
    
    for symbol in active_positions:
        info = status['agents'][symbol]
        hold_info = hold_times.get(symbol, {})
        
        print(f"\n{symbol}:")
        print(f"  Position: {info['position_state']}")
        print(f"  Hold time: {hold_info.get('hold_duration_minutes', 0):.1f} minutes")
        print(f"  Meets min hold: {'✅' if hold_info.get('meets_minimum_hold', False) else '❌'}")
        print(f"  Remaining hold: {hold_info.get('remaining_hold_time', 0):.0f}s")
        print(f"  Next exit check: {info.get('next_exit_check', 0):.0f}s")
        print(f"  In cooldown: {'Yes' if info['in_cooldown'] else 'No'}")
    
    return status, hold_times


# Quick testing functions
def test_mother_ai_initialization():
    """Test that all components initialize correctly"""
    print("🧪 Testing MotherAI initialization...")
    
    try:
        mother_ai = MotherAI(
            agent_symbols=["BTCUSDT"], 
            data_interval="5m"
        )
        
        print("✅ MotherAI initialization successful")
        print(f"📊 Loaded agents: {len(mother_ai.loaded_agents)}")
        print(f"📊 Data interval: {mother_ai.data_interval}")
        print(f"📊 Min hold time: {mother_ai.minimum_hold_time}s")
        print(f"📊 Exit check interval: {mother_ai.exit_check_interval}s")
        
        # Test basic functionality
        positions = mother_ai.get_current_positions()
        print(f"📊 Current positions: {len(positions)}")
        
        status = mother_ai.get_agent_status_summary()
        print(f"📊 Agent status summary generated: {len(status['agents'])} agents")
        
        return True
        
    except Exception as e:
        print(f"❌ MotherAI initialization failed: {e}")
        return False


def test_backward_compatibility():
    """Test that existing code patterns still work"""
    print("🧪 Testing backward compatibility...")
    
    try:
        # Test old initialization pattern
        mother_ai = MotherAI(
            agents_dir="backend/agents",
            strategy_dir="backend/storage/strategies", 
            agent_symbols=["BTCUSDT"],
            data_interval="30m"
        )
        
        # Test old method calls
        agents = mother_ai.load_agents()
        print(f"✅ load_agents() works: {len(agents)} agents loaded")
        
        predictions = mother_ai.load_all_predictions()
        print(f"✅ load_all_predictions() works: {len(predictions)} predictions")
        
        decision = mother_ai.make_portfolio_decision(min_score=0.5)
        print(f"✅ make_portfolio_decision() works: {type(decision)}")
        
        trades = mother_ai.decide_trades(top_n=3, min_score=0.5, min_confidence=0.6)
        print(f"✅ decide_trades() works: {len(trades)} trades")
        
        print("✅ All backward compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests if this file is executed directly
    print("🚀 Running MotherAI tests...")
    
    init_success = test_mother_ai_initialization()
    compat_success = test_backward_compatibility()
    
    if init_success and compat_success:
        print("\n🎉 All tests passed! MotherAI is ready to use.")
        print("\nUsage examples:")
        print("from backend.mother_ai.mother_ai import MotherAI")
        print("mother_ai = MotherAI(agent_symbols=['BTCUSDT'], data_interval='5m')")
        print("decision = mother_ai.make_portfolio_decision()")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")