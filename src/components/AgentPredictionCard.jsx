import React, { useState, useEffect, useMemo, useRef } from "react";
import {
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  Shield,
  DollarSign,
  Activity,
  Filter,
  Grid,
  List,
  Search,
  Clock,
  AlertCircle,
  CheckCircle,
  Eye,
  BarChart3,
  Zap,
  Pause,
  Play
} from "lucide-react";

const getEmoji = (prediction) => {
  const p = prediction.toLowerCase();
  if (p === "buy") return "🟢";
  if (p === "sell") return "🔴";
  return "🟡"; // hold or others
};

const getPredictionIcon = (prediction) => {
  const p = prediction.toLowerCase();
  if (p === "buy") return <TrendingUp className="w-4 h-4" />;
  if (p === "sell") return <TrendingDown className="w-4 h-4" />;
  return <Minus className="w-4 h-4" />;
};

const getPredictionColor = (prediction) => {
  const p = prediction.toLowerCase();
  if (p === "buy") return "text-green-400 bg-green-400/10 border-green-400/20";
  if (p === "sell") return "text-red-400 bg-red-400/10 border-red-400/20";
  return "text-yellow-400 bg-yellow-400/10 border-yellow-400/20";
};

const getConfidenceColor = (confidence) => {
  const conf = parseFloat(confidence);
  if (conf >= 80) return "text-green-400";
  if (conf >= 60) return "text-yellow-400";
  if (conf >= 40) return "text-orange-400";
  return "text-red-400";
};

const fetchWithTimeout = (url, options = {}, timeout = 100000) =>
  Promise.race([
    fetch(url, options),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Request timed out")), timeout)
    ),
  ]);

export default function AgentPredictionCard() {
  const [allAgents, setAllAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retry, setRetry] = useState(0);
  const [viewMode, setViewMode] = useState("grid"); // grid or list
  const [searchTerm, setSearchTerm] = useState("");
  const [filterPrediction, setFilterPrediction] = useState("all");
  const [sortBy, setSortBy] = useState("confidence"); // confidence, symbol, prediction
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [isTabVisible, setIsTabVisible] = useState(true);
  const [refreshPaused, setRefreshPaused] = useState(false);

  // Refs to store interval and current fetch controller
  const intervalRef = useRef(null);
  const fetchControllerRef = useRef(null);

  // Handle tab visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      const isVisible = !document.hidden;
      setIsTabVisible(isVisible);
      
      if (!isVisible) {
        // Tab is not visible - pause refresh and cancel ongoing requests
        setRefreshPaused(true);
        
        // Cancel any ongoing fetch request
        if (fetchControllerRef.current) {
          fetchControllerRef.current.abort();
          fetchControllerRef.current = null;
        }
        
        // Clear the auto-refresh interval
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      } else {
        // Tab is visible again - resume refresh if auto-refresh is enabled
        setRefreshPaused(false);
        
        // If auto-refresh was enabled, restart the interval
        if (autoRefresh) {
          startAutoRefreshInterval();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Set initial state
    setIsTabVisible(!document.hidden);
    
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      
      // Clean up on unmount
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (fetchControllerRef.current) {
        fetchControllerRef.current.abort();
      }
    };
  }, []);

  // Function to start auto-refresh interval
  const startAutoRefreshInterval = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    if (isTabVisible && autoRefresh) {
      intervalRef.current = setInterval(() => {
        setRetry(r => r + 1);
      }, 30000);
    }
  };

  // Main data fetching function
  useEffect(() => {
    async function fetchData() {
      // Don't fetch if tab is not visible
      if (!isTabVisible) {
        return;
      }

      setLoading(true);
      setError(null);

      // Create new AbortController for this request
      fetchControllerRef.current = new AbortController();
      const signal = fetchControllerRef.current.signal;

      try {
        // 1. Fetch all agents
        const resAgents = await fetchWithTimeout("/api/agents", { signal });
        if (!resAgents.ok) {
          throw new Error(`Agents fetch failed (${resAgents.status})`);
        }
        const agentsList = await resAgents.json();

        // Check if request was aborted
        if (signal.aborted) return;

        // 2. Fetch predictions
        const resPred = await fetchWithTimeout(
          "/api/agent/predictions?page=1&limit=100",
          { signal }
        );
        if (!resPred.ok) {
          throw new Error(`Predictions fetch failed (${resPred.status})`);
        }
        const predData = await resPred.json();

        // Check if request was aborted
        if (signal.aborted) return;

        const predictions = Array.isArray(predData.data) ? predData.data : [];

        // 3. Map predictions by normalized agentName
        const predMap = {};
        for (const pred of predictions) {
          if (pred.agentName) {
            const normalizedName = pred.agentName.replace(/\s+Agent$/i, "").toUpperCase();
            predMap[normalizedName] = pred;
          }
        }

        // 4. Merge agents with predictions
        const merged = agentsList.map((agentSymbol) => {
          const key = agentSymbol.toUpperCase();
          const pred = predMap[key];

          const predictionText =
            pred && typeof pred.prediction === "string"
              ? pred.prediction.charAt(0).toUpperCase() +
                pred.prediction.slice(1).toLowerCase()
              : "Hold";

          return {
            agentName: agentSymbol,
            prediction: predictionText,
            confidence:
              pred && typeof pred.confidence === "number"
                ? pred.confidence.toFixed(2)
                : "0.00",
            tradeDetails: pred
              ? pred.tradeDetails || {
                  symbol: "N/A",
                  entryPrice: 0,
                  targetPrice: 0,
                  stopLoss: 0,
                }
              : {
                  symbol: "N/A",
                  entryPrice: 0,
                  targetPrice: 0,
                  stopLoss: 0,
                },
            timestamp: pred?.timestamp || new Date().toISOString(),
            status: pred ? "active" : "inactive"
          };
        });

        // Only update state if request wasn't aborted
        if (!signal.aborted) {
          setAllAgents(merged);
          setLastUpdate(new Date().toLocaleString());
        }
      } catch (err) {
        // Don't show error if request was aborted (normal behavior when tab changes)
        if (!signal.aborted) {
          setError(err.message || "Unknown error occurred");
        }
      } finally {
        if (!signal.aborted) {
          setLoading(false);
        }
        fetchControllerRef.current = null;
      }
    }

    fetchData();
  }, [retry, isTabVisible]);

  // Auto-refresh functionality - only run when tab is visible
  useEffect(() => {
    // Clear existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Start interval only if auto-refresh is enabled and tab is visible
    if (autoRefresh && isTabVisible && !refreshPaused) {
      startAutoRefreshInterval();
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [autoRefresh, isTabVisible, refreshPaused]);

  // Filter and sort agents
  const filteredAndSortedAgents = useMemo(() => {
    let filtered = allAgents.filter(agent => {
      const matchesSearch = agent.agentName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           agent.tradeDetails.symbol.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesFilter = filterPrediction === "all" || 
                           agent.prediction.toLowerCase() === filterPrediction.toLowerCase();
      return matchesSearch && matchesFilter;
    });

    // Sort agents
    filtered.sort((a, b) => {
      switch (sortBy) {
        case "confidence":
          return parseFloat(b.confidence) - parseFloat(a.confidence);
        case "symbol":
          return a.agentName.localeCompare(b.agentName);
        case "prediction":
          return a.prediction.localeCompare(b.prediction);
        default:
          return 0;
      }
    });

    return filtered;
  }, [allAgents, searchTerm, filterPrediction, sortBy]);

  // Calculate summary stats
  const stats = useMemo(() => {
    const total = allAgents.length;
    const active = allAgents.filter(a => a.status === "active").length;
    const predictions = {
      buy: allAgents.filter(a => a.prediction.toLowerCase() === "buy").length,
      sell: allAgents.filter(a => a.prediction.toLowerCase() === "sell").length,
      hold: allAgents.filter(a => a.prediction.toLowerCase() === "hold").length
    };
    const avgConfidence = allAgents.length > 0 
      ? (allAgents.reduce((sum, a) => sum + parseFloat(a.confidence), 0) / allAgents.length).toFixed(1)
      : "0.0";

    return { total, active, predictions, avgConfidence };
  }, [allAgents]);

  const formatPrice = (price) => {
    const num = Number(price);
    if (num === 0) return "N/A";
    return `$${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8 bg-gray-900 rounded-2xl">
        <RefreshCw className="animate-spin w-8 h-8 text-blue-400 mr-3" />
        <span className="text-white">Loading agent predictions...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/50 border border-red-500 text-red-100 p-6 rounded-2xl max-w-md mx-auto">
        <div className="flex items-center gap-3 mb-4">
          <AlertCircle className="w-6 h-6 text-red-400" />
          <h3 className="font-semibold">Error Loading Predictions</h3>
        </div>
        <p className="text-sm mb-4">{error}</p>
        <button
          onClick={() => setRetry((r) => r + 1)}
          className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg text-white transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto bg-gray-900 p-6 rounded-2xl">
      {/* Header */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
        <div className="flex items-center gap-3">
          <Activity className="w-8 h-8 text-blue-400" />
          <div>
            <h3 className="text-3xl font-bold text-white">
              Agent Predictions
            </h3>
            <p className="text-gray-400 text-sm">
              {stats.active} active agents • {filteredAndSortedAgents.length} shown
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Auto-refresh toggle */}
          <div className="flex items-center gap-2 bg-gray-800 px-3 py-2 rounded-lg">
            <Clock className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-300">Auto</span>
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                autoRefresh ? 'bg-blue-600' : 'bg-gray-600'
              }`}
            >
              <span
                className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                  autoRefresh ? 'translate-x-5' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          
          <button
            onClick={() => setRetry(r => r + 1)}
            disabled={loading}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <BarChart3 className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-gray-400">Total Agents</span>
          </div>
          <div className="text-2xl font-bold text-white">{stats.total}</div>
          <div className="text-xs text-gray-500">{stats.active} active</div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-green-400" />
            <span className="text-sm text-gray-400">Buy Signals</span>
          </div>
          <div className="text-2xl font-bold text-green-400">{stats.predictions.buy}</div>
          <div className="text-xs text-gray-500">
            {stats.total > 0 ? ((stats.predictions.buy / stats.total) * 100).toFixed(0) : 0}%
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-4 h-4 text-red-400" />
            <span className="text-sm text-gray-400">Sell Signals</span>
          </div>
          <div className="text-2xl font-bold text-red-400">{stats.predictions.sell}</div>
          <div className="text-xs text-gray-500">
            {stats.total > 0 ? ((stats.predictions.sell / stats.total) * 100).toFixed(0) : 0}%
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-yellow-400" />
            <span className="text-sm text-gray-400">Avg Confidence</span>
          </div>
          <div className="text-2xl font-bold text-white">{stats.avgConfidence}%</div>
          <div className="text-xs text-gray-500">Overall score</div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-col lg:flex-row gap-4 bg-gray-800 p-4 rounded-lg border border-gray-700">
        {/* Search */}
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search agents or symbols..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none"
          />
        </div>

        {/* Filters */}
        <div className="flex gap-3">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <select
              value={filterPrediction}
              onChange={(e) => setFilterPrediction(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
            >
              <option value="all">All Predictions</option>
              <option value="buy">Buy Only</option>
              <option value="sell">Sell Only</option>
              <option value="hold">Hold Only</option>
            </select>
          </div>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
          >
            <option value="confidence">Sort by Confidence</option>
            <option value="symbol">Sort by Symbol</option>
            <option value="prediction">Sort by Prediction</option>
          </select>

          {/* View Mode Toggle */}
          <div className="flex bg-gray-700 rounded-lg p-1">
            <button
              onClick={() => setViewMode("grid")}
              className={`p-2 rounded transition-all ${
                viewMode === "grid"
                  ? "bg-blue-600 text-white"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              <Grid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode("list")}
              className={`p-2 rounded transition-all ${
                viewMode === "list"
                  ? "bg-blue-600 text-white"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Status Bar */}
      {lastUpdate && (
        <div className="flex items-center justify-between text-sm text-gray-400 bg-gray-800 px-4 py-2 rounded-lg border border-gray-700">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-green-400" />
            <span>Data loaded successfully</span>
          </div>
          <div className="flex items-center gap-4">
            <span>Last updated: {lastUpdate}</span>
            {autoRefresh && (
              <div className="flex items-center gap-1">
                {!isTabVisible || refreshPaused ? (
                  <div className="flex items-center gap-1 text-orange-400">
                    <Pause className="w-3 h-3" />
                    <span>Paused (tab hidden)</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-1 text-blue-400">
                    <RefreshCw className="w-3 h-3 animate-spin" />
                    <span>Auto-refresh active</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab Visibility Indicator */}
      {!isTabVisible && (
        <div className="bg-orange-900/50 border border-orange-500 text-orange-100 p-3 rounded-lg">
          <div className="flex items-center gap-2">
            <Pause className="w-4 h-4" />
            <span className="text-sm">
              Auto-refresh paused while tab is hidden. Refresh will resume when you return to this tab.
            </span>
          </div>
        </div>
      )}

      {/* Agents Display */}
      {filteredAndSortedAgents.length === 0 ? (
        <div className="text-center py-12 bg-gray-800 rounded-lg border border-gray-700">
          <Eye className="w-12 h-12 text-gray-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">No agents found</h3>
          <p className="text-gray-400">Try adjusting your search or filter criteria</p>
        </div>
      ) : (
        <div className={
          viewMode === "grid" 
            ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            : "space-y-4"
        }>
          {filteredAndSortedAgents.map((agent) => (
            <div
              key={agent.agentName}
              className={`bg-gray-800 border border-gray-700 rounded-xl transition-all duration-300 hover:border-gray-600 hover:shadow-lg ${
                viewMode === "list" ? "p-4" : "p-6"
              }`}
            >
              {viewMode === "grid" ? (
                /* Grid View */
                <>
                  {/* Header */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${agent.status === 'active' ? 'bg-green-400/10' : 'bg-gray-700'}`}>
                        <Activity className={`w-5 h-5 ${agent.status === 'active' ? 'text-green-400' : 'text-gray-400'}`} />
                      </div>
                      <div>
                        <h4 className="text-lg font-semibold text-white">{agent.agentName}</h4>
                        <p className="text-xs text-gray-400">{agent.tradeDetails.symbol}</p>
                      </div>
                    </div>
                    <div className={`flex items-center gap-2 px-3 py-1 rounded-full border ${getPredictionColor(agent.prediction)}`}>
                      {getPredictionIcon(agent.prediction)}
                      <span className="font-semibold text-sm">{agent.prediction}</span>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div className="mb-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-400">Confidence</span>
                      <span className={`font-bold ${getConfidenceColor(agent.confidence)}`}>
                        {agent.confidence}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-300 ${
                          parseFloat(agent.confidence) >= 80 ? 'bg-green-400' :
                          parseFloat(agent.confidence) >= 60 ? 'bg-yellow-400' :
                          parseFloat(agent.confidence) >= 40 ? 'bg-orange-400' : 'bg-red-400'
                        }`}
                        style={{ width: `${agent.confidence}%` }}
                      />
                    </div>
                  </div>

                  {/* Trade Details */}
                  <div className="space-y-3 pt-3 border-t border-gray-700">
                    <h5 className="text-sm font-medium text-white flex items-center gap-2">
                      <DollarSign className="w-4 h-4" />
                      Trade Details
                    </h5>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <div className="text-gray-400 mb-1">Entry Price</div>
                        <div className="text-white font-mono">{formatPrice(agent.tradeDetails.entryPrice)}</div>
                      </div>
                      <div>
                        <div className="text-gray-400 mb-1">Target</div>
                        <div className="text-green-400 font-mono">{formatPrice(agent.tradeDetails.targetPrice)}</div>
                      </div>
                    </div>
                    <div className="text-sm">
                      <div className="text-gray-400 mb-1">Stop Loss</div>
                      <div className="text-red-400 font-mono">{formatPrice(agent.tradeDetails.stopLoss)}</div>
                    </div>
                  </div>
                </>
              ) : (
                /* List View */
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`p-2 rounded-lg ${agent.status === 'active' ? 'bg-green-400/10' : 'bg-gray-700'}`}>
                      <Activity className={`w-4 h-4 ${agent.status === 'active' ? 'text-green-400' : 'text-gray-400'}`} />
                    </div>
                    <div>
                      <h4 className="font-semibold text-white">{agent.agentName}</h4>
                      <p className="text-sm text-gray-400">{agent.tradeDetails.symbol}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-6">
                    <div className={`flex items-center gap-2 px-2 py-1 rounded-full border ${getPredictionColor(agent.prediction)}`}>
                      {getPredictionIcon(agent.prediction)}
                      <span className="font-semibold text-sm">{agent.prediction}</span>
                    </div>
                    
                    <div className="text-right">
                      <div className={`font-bold ${getConfidenceColor(agent.confidence)}`}>
                        {agent.confidence}%
                      </div>
                      <div className="text-xs text-gray-400">confidence</div>
                    </div>
                    
                    <div className="text-right font-mono text-sm">
                      <div className="text-white">{formatPrice(agent.tradeDetails.entryPrice)}</div>
                      <div className="text-xs text-gray-400">entry</div>
                    </div>
                    
                    <div className="text-right font-mono text-sm">
                      <div className="text-green-400">{formatPrice(agent.tradeDetails.targetPrice)}</div>
                      <div className="text-xs text-gray-400">target</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="text-center text-sm text-gray-400 pt-4 border-t border-gray-700">
        <p>
          Showing {filteredAndSortedAgents.length} of {allAgents.length} agents • 
          {stats.predictions.buy} Buy • {stats.predictions.sell} Sell • {stats.predictions.hold} Hold
        </p>
        {autoRefresh && (
          <p className="mt-2 text-blue-400">
            {!isTabVisible ? "Auto-refresh paused (tab hidden)" : "Auto-refreshing every 30 seconds"}
          </p>
        )}
      </div>
    </div>
  );
}