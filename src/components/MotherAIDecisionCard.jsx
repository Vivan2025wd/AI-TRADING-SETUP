import React, { useState, useEffect } from "react";
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
  Play,
  Shield
} from "lucide-react";

const DECISION_REFRESH_INTERVAL = 10 * 60 * 1000; // 15 minutes

export default function MotherAIDecisionCard({ 
  isLive, 
  decisionData, 
  loading, 
  error, 
  refreshDecision,
  // New props from App.jsx
  persistedDecision,
  decisionTimestamp,
  timeUntilNextFetch,
  backgroundTimer,
  setBackgroundTimer,
  refreshHistory,
  connectionStatus
}) {
  const [isTabVisible, setIsTabVisible] = useState(true);
  const [expandedRationale, setExpandedRationale] = useState(false);

  // Handle tab visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      setIsTabVisible(!document.hidden);
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    setIsTabVisible(!document.hidden);
    
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

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

  const currentDecision = persistedDecision || decisionData;
  const confidenceGlow = currentDecision?.confidence >= 90 ? "text-green-400 animate-pulse" : "text-green-300";
  const isTimerActive = decisionTimestamp && timeUntilNextFetch > 0;

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
          {/* Background Timer Toggle */}
          <div className="flex items-center gap-2 bg-gray-800/50 px-2 py-1 rounded-lg backdrop-blur-sm">
            <Clock className="w-3 h-3 text-gray-400" />
            <button
              onClick={() => setBackgroundTimer(!backgroundTimer)}
              className={`relative inline-flex h-4 w-7 items-center rounded-full transition-colors ${
                backgroundTimer ? 'bg-green-600' : 'bg-gray-600'
              }`}
              title={`30m Timer: ${backgroundTimer ? 'ON' : 'OFF'}`}
            >
              <span
                className={`inline-block h-2 w-2 transform rounded-full bg-white transition-transform ${
                  backgroundTimer ? 'translate-x-4' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          <span
            className={`px-3 py-1 text-sm font-semibold rounded-full flex items-center gap-1 ${getStatusBadge(currentDecision?.status)} backdrop-blur-sm`}
            title={`Status: ${currentDecision?.status || "Unknown"}`}
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
                {currentDecision?.status || "Unknown"}
              </>
            )}
          </span>
        </div>
      </div>

      {/* Persistent Timer Status */}
      {isTimerActive && backgroundTimer && (
        <div className="bg-green-900/30 border border-green-700/50 rounded-lg p-3 backdrop-blur-sm">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-green-400" />
              <span className="text-green-400">
                {connectionStatus === 'online' ? 'Next auto-fetch in:' : 'Next auto-fetch (when online):'}
              </span>
            </div>
            <div className="text-green-300 font-mono">
              {formatCountdown(timeUntilNextFetch)}
            </div>
          </div>
          <div className="w-full bg-green-900/50 rounded-full h-1 mt-2">
            <div 
              className="h-1 rounded-full bg-green-400 transition-all duration-1000"
              style={{ 
                width: `${((DECISION_REFRESH_INTERVAL / 1000 - timeUntilNextFetch) / (DECISION_REFRESH_INTERVAL / 1000)) * 100}%` 
              }}
            />
          </div>
          <div className="flex items-center gap-2 mt-2 text-xs text-green-300">
            <Shield className="w-3 h-3" />
            <span>Timer persists across all navigation</span>
          </div>
        </div>
      )}

      {/* Connection Status */}
      {connectionStatus === 'offline' && (
        <div className="bg-orange-900/30 border border-orange-700/50 rounded-lg p-3 backdrop-blur-sm">
          <div className="flex items-center gap-2 text-sm">
            <WifiOff className="w-4 h-4 text-orange-400" />
            <span className="text-orange-400">
              Offline - Timer continues, will fetch when reconnected
            </span>
          </div>
        </div>
      )}

      {/* Tab Visibility Status */}
      {!isTabVisible && (
        <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg p-3 backdrop-blur-sm">
          <div className="flex items-center gap-2 text-sm">
            <EyeOff className="w-4 h-4 text-blue-400" />
            <span className="text-blue-400">
              Tab hidden - Background timer active, showing persisted decision
            </span>
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
              <div className={`p-2 rounded-lg bg-gray-800/50 backdrop-blur-sm ${getTradeColor(currentDecision?.tradePick)}`}>
                {getTradeIcon(currentDecision?.tradePick)}
              </div>
              <div className="flex-1">
                <p className={`text-2xl font-bold tracking-wide ${getTradeColor(currentDecision?.tradePick)}`}>
                  {currentDecision?.tradePick || "No Signal"}
                </p>
                <p className="text-sm text-gray-400 flex items-center gap-1">
                  <Clock className="w-4 h-4" /> 
                  {currentDecision?.lastUpdated || "N/A"}
                  {persistedDecision && decisionTimestamp && (
                    <span className="ml-2 px-2 py-0.5 bg-green-900/50 text-green-300 rounded text-xs">
                      PERSISTENT
                    </span>
                  )}
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
              {currentDecision?.rationale && currentDecision.rationale.length > 100 && (
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
                {currentDecision?.rationale || "No rationale provided."}
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
                {currentDecision?.confidence !== undefined ? `${currentDecision.confidence}%` : "N/A"}
              </span>
            </div>
            
            {/* Confidence Bar */}
            {currentDecision?.confidence !== undefined && (
              <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                <div 
                  className={`h-2 rounded-full transition-all duration-500 ${
                    currentDecision.confidence >= 90 ? 'bg-green-400' :
                    currentDecision.confidence >= 70 ? 'bg-yellow-400' :
                    currentDecision.confidence >= 50 ? 'bg-orange-400' : 'bg-red-400'
                  }`}
                  style={{ width: `${currentDecision.confidence}%` }}
                />
              </div>
            )}
            
            {currentDecision?.confidence >= 90 && (
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-green-400" />
                <span className="text-green-400 text-xs bg-green-900/30 px-2 py-0.5 rounded-md animate-pulse">
                  High Confidence Signal
                </span>
              </div>
            )}
          </div>

          {/* Refresh History */}
          {refreshHistory && refreshHistory.length > 0 && (
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

          {!loading && currentDecision?.status === "Active" && !isTimerActive && (
            <p className="text-sm text-gray-400 italic mt-2 text-center">
              Background timer disabled. Use manual refresh to get new decisions.
            </p>
          )}
        </>
      )}

      {/* Manual Refresh Button */}
      <div className="flex gap-2">
        <button
          onClick={refreshDecision}
          disabled={loading || connectionStatus === 'offline'}
          className={`flex-1 px-6 py-3 rounded-lg flex items-center gap-2 justify-center transition-all duration-300 relative overflow-hidden ${
            loading || connectionStatus === 'offline'
              ? "bg-gray-600 text-gray-300 cursor-not-allowed"
              : "bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white shadow-lg hover:shadow-blue-500/25"
          }`}
        >
          <div className="absolute inset-0 bg-white/10 opacity-0 hover:opacity-100 transition-opacity" />
          <RefreshCw className={`w-4 h-4 relative z-10 ${loading ? "animate-spin" : ""}`} />
          <span className="relative z-10">
            {loading ? "Refreshing..." : 
             connectionStatus === 'offline' ? "Offline" : "Manual Refresh"}
          </span>
        </button>
      </div>

      {/* Last Update Info */}
      {decisionTimestamp && (
        <div className="flex items-center justify-between text-xs text-gray-400">
          <span>Decision from: {decisionTimestamp.toLocaleString()}</span>
          {refreshHistory && refreshHistory.length > 0 && refreshHistory[0].type === 'manual' && (
            <span className="text-blue-400">• Manual refresh</span>
          )}
        </div>
      )}
    </div>
  );
}