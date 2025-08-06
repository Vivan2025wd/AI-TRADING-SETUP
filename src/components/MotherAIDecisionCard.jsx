import React, { useState, useEffect, useRef } from "react";
import { 
  ArrowUpRight, 
  AlertCircle, 
  Clock, 
  RefreshCw, 
  Loader2, 
  Info,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  Zap,
  Eye,
  EyeOff,
  Signal,
  Wifi,
  WifiOff,
  CheckCircle,
  XCircle,
  Pause,
  Play
} from "lucide-react";

const POLL_INTERVAL_MS = 30 * 60 * 1000; // 30 minutes

export default function MotherAIDecisionCard({ isLive, decisionData, loading, error, refreshDecision }) {
  const [isTabVisible, setIsTabVisible] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [timeUntilRefresh, setTimeUntilRefresh] = useState(0);
  const [lastRefreshTime, setLastRefreshTime] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('online');
  const [refreshHistory, setRefreshHistory] = useState([]);
  const [expandedRationale, setExpandedRationale] = useState(false);
  
  const intervalRef = useRef(null);
  const countdownRef = useRef(null);

  // Handle tab visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      const isVisible = !document.hidden;
      setIsTabVisible(isVisible);
      
      if (!isVisible && autoRefresh) {
        // Pause auto-refresh when tab is hidden
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
        if (countdownRef.current) {
          clearInterval(countdownRef.current);
          countdownRef.current = null;
        }
      } else if (isVisible && autoRefresh) {
        // Resume auto-refresh when tab becomes visible
        startAutoRefresh();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    setIsTabVisible(!document.hidden);
    
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (countdownRef.current) clearInterval(countdownRef.current);
    };
  }, [autoRefresh]);

  // Monitor connection status
  useEffect(() => {
    const handleOnline = () => setConnectionStatus('online');
    const handleOffline = () => setConnectionStatus('offline');
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    setConnectionStatus(navigator.onLine ? 'online' : 'offline');
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Start auto-refresh functionality
  const startAutoRefresh = () => {
    if (!isTabVisible || connectionStatus === 'offline') return;
    
    // Clear existing intervals
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (countdownRef.current) clearInterval(countdownRef.current);
    
    setTimeUntilRefresh(POLL_INTERVAL_MS / 1000);
    
    // Countdown timer
    countdownRef.current = setInterval(() => {
      setTimeUntilRefresh(prev => {
        if (prev <= 1) {
          return POLL_INTERVAL_MS / 1000;
        }
        return prev - 1;
      });
    }, 1000);
    
    // Auto-refresh interval
    intervalRef.current = setInterval(() => {
      if (isTabVisible && connectionStatus === 'online') {
        handleRefresh();
      }
    }, POLL_INTERVAL_MS);
  };

  // Enhanced refresh handler
  const handleRefresh = () => {
    const refreshTime = new Date();
    setLastRefreshTime(refreshTime);
    
    // Add to refresh history
    setRefreshHistory(prev => [
      {
        time: refreshTime,
        success: !error,
        status: decisionData?.status || 'Unknown'
      },
      ...prev.slice(0, 4) // Keep last 5 entries
    ]);
    
    refreshDecision();
  };

  // Auto-refresh effect
  useEffect(() => {
    if (autoRefresh && isTabVisible && connectionStatus === 'online') {
      startAutoRefresh();
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (countdownRef.current) clearInterval(countdownRef.current);
      intervalRef.current = null;
      countdownRef.current = null;
    }
    
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (countdownRef.current) clearInterval(countdownRef.current);
    };
  }, [autoRefresh, isTabVisible, connectionStatus]);

  // Format countdown time
  const formatCountdown = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Get trade signal icon
  const getTradeIcon = (tradePick) => {
    if (!tradePick) return <Minus className="w-5 h-5" />;
    const signal = tradePick.toLowerCase();
    if (signal.includes('buy') || signal.includes('long')) return <TrendingUp className="w-5 h-5" />;
    if (signal.includes('sell') || signal.includes('short')) return <TrendingDown className="w-5 h-5" />;
    return <Minus className="w-5 h-5" />;
  };

  // Get trade signal color
  const getTradeColor = (tradePick) => {
    if (!tradePick) return "text-gray-400";
    const signal = tradePick.toLowerCase();
    if (signal.includes('buy') || signal.includes('long')) return "text-green-400";
    if (signal.includes('sell') || signal.includes('short')) return "text-red-400";
    return "text-yellow-400";
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case "Active":
        return "bg-green-700 text-green-100 animate-pulse";
      case "Inactive":
        return "bg-gray-600 text-gray-200";
      case "Waiting":
        return "bg-yellow-600 text-yellow-100";
      case "Error":
        return "bg-red-700 text-red-200";
      default:
        return "bg-gray-700 text-gray-300";
    }
  };

  const confidenceGlow = decisionData?.confidence >= 90 ? "text-green-400 animate-pulse" : "text-green-300";

  return (
    <div className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white shadow-lg rounded-2xl p-6 max-w-xl mx-auto space-y-6 border border-gray-700 transition-all duration-300 relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 25% 25%, #3b82f6 0%, transparent 50%), 
                           radial-gradient(circle at 75% 75%, #10b981 0%, transparent 50%)`
        }} />
      </div>

      {/* Header with Enhanced Status */}
      <div className="flex items-center justify-between relative z-10">
        <div className="flex items-center gap-3">
          <div className="relative">
            <ArrowUpRight className="text-blue-400 w-6 h-6" />
            {isLive && (
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            )}
          </div>
          <div>
            <h2 className="text-xl font-bold">Mother AI Signal</h2>
            <div className="flex items-center gap-2 text-xs text-gray-400">
              {connectionStatus === 'online' ? (
                <Wifi className="w-3 h-3 text-green-400" />
              ) : (
                <WifiOff className="w-3 h-3 text-red-400" />
              )}
              <span>{connectionStatus}</span>
              {!isTabVisible && (
                <>
                  <EyeOff className="w-3 h-3 text-orange-400" />
                  <span className="text-orange-400">Hidden</span>
                </>
              )}
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Auto-refresh toggle */}
          <div className="flex items-center gap-2 bg-gray-800/50 px-2 py-1 rounded-lg backdrop-blur-sm">
            <Clock className="w-3 h-3 text-gray-400" />
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`relative inline-flex h-4 w-7 items-center rounded-full transition-colors ${
                autoRefresh ? 'bg-blue-600' : 'bg-gray-600'
              }`}
              title={`Auto-refresh: ${autoRefresh ? 'ON' : 'OFF'}`}
            >
              <span
                className={`inline-block h-2 w-2 transform rounded-full bg-white transition-transform ${
                  autoRefresh ? 'translate-x-4' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          <span
            className={`px-3 py-1 text-sm font-semibold rounded-full flex items-center gap-1 ${getStatusBadge(decisionData?.status)} backdrop-blur-sm`}
            title={`Status: ${decisionData?.status || "Unknown"}`}
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin w-4 h-4" /> Loading...
              </>
            ) : error ? (
              <>
                <XCircle className="w-4 h-4" /> Error
              </>
            ) : (
              <>
                <CheckCircle className="w-4 h-4" />
                {decisionData?.status || "Unknown"}
              </>
            )}
          </span>
        </div>
      </div>

      {/* Auto-refresh Status */}
      {autoRefresh && (
        <div className="bg-gray-800/30 border border-gray-700/50 rounded-lg p-3 backdrop-blur-sm">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              {!isTabVisible || connectionStatus === 'offline' ? (
                <>
                  <Pause className="w-4 h-4 text-orange-400" />
                  <span className="text-orange-400">
                    Auto-refresh paused ({!isTabVisible ? 'tab hidden' : 'offline'})
                  </span>
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />
                  <span className="text-blue-400">Auto-refresh active</span>
                </>
              )}
            </div>
            {autoRefresh && isTabVisible && connectionStatus === 'online' && timeUntilRefresh > 0 && (
              <div className="text-gray-400">
                Next: {formatCountdown(timeUntilRefresh)}
              </div>
            )}
          </div>
        </div>
      )}

      {error ? (
        <div className="bg-red-900/50 text-white p-4 rounded-lg text-center border border-red-600 shadow-md backdrop-blur-sm">
          <div className="flex items-center justify-center gap-2 mb-2">
            <AlertCircle className="w-5 h-5" />
            <p className="font-bold">Error Fetching Decision</p>
          </div>
          <p className="text-sm text-red-200">{error}</p>
        </div>
      ) : (
        <>
          {/* Main Signal Display */}
          <div className="space-y-3 relative z-10">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg bg-gray-800/50 backdrop-blur-sm ${getTradeColor(decisionData?.tradePick)}`}>
                {getTradeIcon(decisionData?.tradePick)}
              </div>
              <div>
                <p className={`text-2xl font-bold tracking-wide ${getTradeColor(decisionData?.tradePick)}`}>
                  {decisionData?.tradePick || "No Signal"}
                </p>
                <p className="text-sm text-gray-400 flex items-center gap-1">
                  <Clock className="w-4 h-4" /> 
                  {decisionData?.lastUpdated || "N/A"}
                </p>
              </div>
            </div>
          </div>

          {/* Enhanced Rationale Section */}
          <div className="bg-gray-800/30 p-4 rounded-lg border border-gray-700/50 text-sm text-gray-300 transition-all backdrop-blur-sm">
            <div className="flex items-center justify-between mb-2">
              <p className="font-semibold text-white flex items-center gap-2">
                <Info className="w-4 h-4 text-blue-400" />
                Rationale:
              </p>
              {decisionData?.rationale && decisionData.rationale.length > 100 && (
                <button
                  onClick={() => setExpandedRationale(!expandedRationale)}
                  className="text-blue-400 hover:text-blue-300 text-xs flex items-center gap-1"
                >
                  {expandedRationale ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                  {expandedRationale ? 'Less' : 'More'}
                </button>
              )}
            </div>
            <p className="italic flex gap-2 items-start leading-relaxed">
              <AlertCircle className="w-4 h-4 text-yellow-400 mt-1 flex-shrink-0" />
              <span className={expandedRationale ? '' : 'line-clamp-3'}>
                {decisionData?.rationale || "No rationale provided."}
              </span>
            </p>
          </div>

          {/* Enhanced Confidence Display */}
          <div className="bg-gray-800/20 p-4 rounded-lg border border-gray-700/30 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold text-white flex items-center gap-2">
                <Activity className="w-4 h-4 text-blue-400" />
                Confidence:
              </span>
              <span className={`${confidenceGlow} font-bold text-lg`} title="Confidence Score">
                {decisionData?.confidence !== undefined ? `${decisionData.confidence}%` : "N/A"}
              </span>
            </div>
            
            {/* Confidence Bar */}
            {decisionData?.confidence !== undefined && (
              <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                <div 
                  className={`h-2 rounded-full transition-all duration-500 ${
                    decisionData.confidence >= 90 ? 'bg-green-400' :
                    decisionData.confidence >= 70 ? 'bg-yellow-400' :
                    decisionData.confidence >= 50 ? 'bg-orange-400' : 'bg-red-400'
                  }`}
                  style={{ width: `${decisionData.confidence}%` }}
                />
              </div>
            )}
            
            {decisionData?.confidence >= 90 && (
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-green-400" />
                <span className="text-green-400 text-xs bg-green-900/30 px-2 py-0.5 rounded-md animate-pulse">
                  High Confidence Signal
                </span>
              </div>
            )}
          </div>

          {/* Refresh History */}
          {refreshHistory.length > 0 && (
            <div className="bg-gray-800/20 p-3 rounded-lg border border-gray-700/30 backdrop-blur-sm">
              <p className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
                <Signal className="w-4 h-4 text-blue-400" />
                Recent Updates
              </p>
              <div className="space-y-1">
                {refreshHistory.slice(0, 3).map((entry, index) => (
                  <div key={index} className="flex items-center justify-between text-xs">
                    <span className="text-gray-400">
                      {entry.time.toLocaleTimeString()}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-gray-300">{entry.status}</span>
                      {entry.success ? (
                        <CheckCircle className="w-3 h-3 text-green-400" />
                      ) : (
                        <XCircle className="w-3 h-3 text-red-400" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!loading && decisionData?.status === "Active" && !autoRefresh && (
            <p className="text-sm text-gray-400 italic mt-2 text-center">
              Next manual update available in {Math.ceil(POLL_INTERVAL_MS / 60000)} minutes...
            </p>
          )}
        </>
      )}

      {/* Enhanced Refresh Button */}
      <button
        onClick={handleRefresh}
        disabled={loading || connectionStatus === 'offline'}
        className={`mt-4 px-6 py-3 rounded-lg flex items-center gap-2 mx-auto transition-all duration-300 relative overflow-hidden ${
          loading || connectionStatus === 'offline'
            ? "bg-gray-600 text-gray-300 cursor-not-allowed"
            : "bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white shadow-lg hover:shadow-blue-500/25"
        }`}
      >
        <div className="absolute inset-0 bg-white/10 opacity-0 hover:opacity-100 transition-opacity" />
        <RefreshCw className={`w-4 h-4 relative z-10 ${loading ? "animate-spin" : ""}`} />
        <span className="relative z-10">
          {loading ? "Refreshing..." : connectionStatus === 'offline' ? "Offline" : "Refresh Now"}
        </span>
        {lastRefreshTime && (
          <span className="text-xs opacity-75 relative z-10">
            • {lastRefreshTime.toLocaleTimeString()}
          </span>
        )}
      </button>
    </div>
  );
}