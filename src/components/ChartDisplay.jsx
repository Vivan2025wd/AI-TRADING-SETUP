import React, { useEffect, useState, useMemo } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  BarElement,
} from 'chart.js';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  BarChart3,
  RefreshCw,
  Calendar,
  Activity,
  Target,
  Zap,
  Filter,
  Eye,
  EyeOff,
  Settings,
  ArrowUp,
  ArrowDown,
  Minus,
  AlertCircle,
  Download
} from 'lucide-react';

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  BarElement
);

const fetchWithTimeout = (url, options = {}, timeout = 300000) =>
  Promise.race([
    fetch(url, options),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Request timed out")), timeout)
    ),
  ]);

export default function EnhancedBacktestResults() {
  // State Management
  const [capitalCurve, setCapitalCurve] = useState([]);
  const [allTrades, setAllTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retry, setRetry] = useState(0);
  const [initialCapital, setInitialCapital] = useState(100);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [chartType, setChartType] = useState("balance");
  const [timeRange, setTimeRange] = useState("all");
  const [showTable, setShowTable] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState("ALL");

  // Theme
  const theme = {
    text: "#ffffff",
    grid: "#374151",
    background: "rgba(17, 24, 39, 0.8)"
  };

  // Fetch backtest data
  useEffect(() => {
    async function fetchBacktestResults() {
      setLoading(true);
      setError(null);
      try {
        const timestamp = new Date().getTime();
        const res = await fetchWithTimeout(
          `http://localhost:8000/api/backtest/results?limit=5000&_t=${timestamp}`,
          {
            headers: {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          }
        );
        
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(`HTTP ${res.status}: ${errorText}`);
        }
        
        const data = await res.json();
        console.log("Fetched data:", data);

        if (!data.capital_curve) {
          throw new Error("Invalid data format: missing capital_curve");
        }

        setInitialCapital(data.initial_capital || 100);
        setCapitalCurve(data.capital_curve);
        setAllTrades(data.capital_curve); // Store all trades for filtering
        setLastUpdate(new Date().toLocaleString());
      } catch (err) {
        console.error("Fetch error:", err);
        setError(err.message || "Unknown error occurred");
      } finally {
        setLoading(false);
      }
    }

    fetchBacktestResults();
  }, [retry]);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      setRetry(r => r + 1);
    }, 30000);
    
    return () => clearInterval(interval);
  }, [autoRefresh]);

  // Filter trades based on selected filters
  const filteredTrades = useMemo(() => {
    let trades = [...allTrades];
    
    // Filter by symbol
    if (selectedSymbol !== "ALL") {
      trades = trades.filter(trade => trade.symbol === selectedSymbol);
    }
    
    // Filter by time range
    if (timeRange !== "all") {
      const now = new Date();
      trades = trades.filter(trade => {
        if (!trade.timestamp) return true;
        const tradeDate = new Date(trade.timestamp);
        const daysAgo = (now - tradeDate) / (1000 * 60 * 60 * 24);
        
        switch (timeRange) {
          case "7d": return daysAgo <= 7;
          case "30d": return daysAgo <= 30;
          case "90d": return daysAgo <= 90;
          default: return true;
        }
      });
    }
    
    return trades.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  }, [allTrades, selectedSymbol, timeRange]);

  // Get unique symbols for filter dropdown
  const availableSymbols = useMemo(() => {
    const symbols = [...new Set(allTrades.map(trade => trade.symbol).filter(Boolean))];
    return ["ALL", ...symbols.sort()];
  }, [allTrades]);

  // Calculate analytics
  const analytics = useMemo(() => {
    if (filteredTrades.length === 0) return null;

    const finalBalance = filteredTrades[filteredTrades.length - 1]?.balance || initialCapital;
    const totalReturn = finalBalance - initialCapital;
    const totalReturnPercent = (totalReturn / initialCapital) * 100;
    
    const profitableTrades = filteredTrades.filter(t => t.pnl_dollars > 0);
    const losingTrades = filteredTrades.filter(t => t.pnl_dollars < 0);
    const winRate = filteredTrades.length > 0 ? (profitableTrades.length / filteredTrades.length) * 100 : 0;
    
    const maxBalance = Math.max(...filteredTrades.map(t => t.balance));
    const maxDrawdown = filteredTrades.reduce((max, curr, index) => {
      const peak = Math.max(...filteredTrades.slice(0, index + 1).map(t => t.balance));
      const drawdown = ((peak - curr.balance) / peak) * 100;
      return Math.max(max, drawdown);
    }, 0);

    const avgPnL = filteredTrades.length > 0 ? 
      filteredTrades.reduce((sum, t) => sum + (t.pnl_dollars || 0), 0) / filteredTrades.length : 0;
    
    const avgWin = profitableTrades.length > 0 ?
      profitableTrades.reduce((sum, t) => sum + t.pnl_dollars, 0) / profitableTrades.length : 0;
    
    const avgLoss = losingTrades.length > 0 ?
      Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl_dollars, 0) / losingTrades.length) : 0;
    
    const profitFactor = avgLoss > 0 ? avgWin / avgLoss : 0;

    return {
      totalReturn,
      totalReturnPercent,
      finalBalance,
      winRate,
      totalTrades: filteredTrades.length,
      wins: profitableTrades.length,
      losses: losingTrades.length,
      maxBalance,
      maxDrawdown,
      avgPnL,
      avgWin,
      avgLoss,
      profitFactor: profitFactor.toFixed(2)
    };
  }, [filteredTrades, initialCapital]);

  // Chart data preparation
  const getChartData = () => {
    const labels = filteredTrades.map((trade, index) => {
      if (trade.timestamp) {
        return new Date(trade.timestamp).toLocaleDateString();
      }
      return `Trade ${index + 1}`;
    });
    
    switch (chartType) {
      case "balance":
        return {
          labels,
          datasets: [{
            label: "Portfolio Balance ($)",
            data: filteredTrades.map(trade => trade.balance),
            borderColor: "rgb(34, 197, 94)",
            backgroundColor: "rgba(34, 197, 94, 0.1)",
            tension: 0.4,
            fill: true,
            pointRadius: 1,
            pointHoverRadius: 4,
          }]
        };
        
      case "pnl":
        return {
          labels,
          datasets: [{
            label: "Cumulative P&L ($)",
            data: filteredTrades.map(trade => (trade.balance || initialCapital) - initialCapital),
            borderColor: "rgb(59, 130, 246)",
            backgroundColor: "rgba(59, 130, 246, 0.1)",
            tension: 0.4,
            fill: true,
            pointRadius: 1,
            pointHoverRadius: 4,
          }]
        };
        
      case "trades":
        return {
          labels,
          datasets: [{
            label: "Trade P&L ($)",
            data: filteredTrades.map(trade => trade.pnl_dollars || 0),
            backgroundColor: filteredTrades.map(trade => 
              (trade.pnl_dollars || 0) >= 0 ? "rgba(34, 197, 94, 0.7)" : "rgba(239, 68, 68, 0.7)"
            ),
            borderColor: filteredTrades.map(trade => 
              (trade.pnl_dollars || 0) >= 0 ? "rgb(34, 197, 94)" : "rgb(239, 68, 68)"
            ),
            borderWidth: 1,
          }]
        };
        
      default:
        return { labels: [], datasets: [] };
    }
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "nearest", intersect: false },
    plugins: {
      legend: { 
        labels: { color: theme.text },
        position: 'top'
      },
      tooltip: {
        backgroundColor: theme.background,
        titleColor: theme.text,
        bodyColor: theme.text,
        borderColor: theme.grid,
        borderWidth: 1,
        callbacks: {
          label: (ctx) => {
            const trade = filteredTrades[ctx.dataIndex];
            if (!trade) return "";
            
            const lines = [
              `P&L: $${(trade.pnl_dollars || 0).toFixed(4)}`,
              `Balance: $${(trade.balance || 0).toFixed(2)}`,
            ];
            
            if (trade.symbol) lines.push(`Symbol: ${trade.symbol}`);
            if (trade.exit_price) lines.push(`Exit Price: $${trade.exit_price.toLocaleString()}`);
            if (trade.timestamp) lines.push(`Time: ${new Date(trade.timestamp).toLocaleString()}`);
            
            return lines;
          },
        },
      },
    },
    scales: chartType !== "winloss" ? {
      x: { 
        ticks: { 
          color: theme.text,
          maxTicksLimit: 10 
        }, 
        grid: { color: theme.grid },
        title: { display: true, text: "Trades", color: theme.text }
      },
      y: { 
        ticks: { color: theme.text }, 
        grid: { color: theme.grid },
        title: { 
          display: true, 
          text: chartType === "trades" ? "P&L ($)" : "Value ($)", 
          color: theme.text 
        }
      },
    } : {},
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  const getReturnColor = (value) => {
    if (value > 0) return "text-green-400";
    if (value < 0) return "text-red-400";
    return "text-gray-400";
  };

  const getReturnIcon = (value) => {
    if (value > 0) return <ArrowUp className="w-4 h-4" />;
    if (value < 0) return <ArrowDown className="w-4 h-4" />;
    return <Minus className="w-4 h-4" />;
  };

  return (
    <div className="space-y-6 bg-gray-900 text-white p-6 rounded-2xl max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-8 h-8 text-blue-400" />
          <div>
            <h2 className="text-3xl font-bold text-white">Backtest Results</h2>
            <p className="text-gray-400 text-sm">Capital Curve & Performance Analytics</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
          >
            <Settings className="w-4 h-4" />
          </button>
          <button
            onClick={() => setRetry(r => r + 1)}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="bg-gray-800 rounded-lg p-4 space-y-4 border border-gray-600">
          <h3 className="font-semibold text-white flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Settings
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-300">Auto Refresh (30s)</span>
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  autoRefresh ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    autoRefresh ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
            
            {lastUpdate && (
              <div className="text-sm text-gray-400">
                Last updated: {lastUpdate}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
        <div className="flex flex-wrap gap-3">
          {/* Symbol Selection */}
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <select
              className="bg-gray-700 text-white rounded-lg px-3 py-2 border border-gray-600 focus:border-blue-500 focus:outline-none"
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
            >
              {availableSymbols.map((sym) => (
                <option key={sym} value={sym}>
                  {sym === "ALL" ? "All Symbols" : sym}
                </option>
              ))}
            </select>
          </div>

          {/* Time Range */}
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-gray-400" />
            <select
              className="bg-gray-700 text-white rounded-lg px-3 py-2 border border-gray-600 focus:border-blue-500 focus:outline-none"
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <option value="all">All Time</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
              <option value="90d">Last 90 Days</option>
            </select>
          </div>
        </div>

        {/* Chart Type Selection */}
        <div className="flex bg-gray-700 rounded-lg p-1">
          {[
            { value: "balance", icon: TrendingUp, label: "Balance" },
            { value: "pnl", icon: DollarSign, label: "P&L" },
            { value: "trades", icon: BarChart3, label: "Trades" },
          ].map(({ value, icon: Icon, label }) => (
            <button
              key={value}
              onClick={() => setChartType(value)}
              className={`flex items-center gap-2 px-3 py-2 rounded-md transition-all ${
                chartType === value
                  ? "bg-blue-600 text-white"
                  : "text-gray-300 hover:text-white hover:bg-gray-600"
              }`}
            >
              <Icon className="w-4 h-4" />
              <span className="text-sm">{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Analytics Cards */}
      {analytics && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-600">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign className="w-4 h-4 text-green-400" />
              <span className="text-xs text-gray-400">Total Return</span>
            </div>
            <div className={`font-bold text-lg ${getReturnColor(analytics.totalReturn)}`}>
              {formatCurrency(analytics.totalReturn)}
            </div>
            <div className={`text-xs flex items-center gap-1 ${getReturnColor(analytics.totalReturnPercent)}`}>
              {getReturnIcon(analytics.totalReturnPercent)}
              {analytics.totalReturnPercent.toFixed(2)}%
            </div>
          </div>

          <div className="bg-gray-800 p-4 rounded-lg border border-gray-600">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-blue-400" />
              <span className="text-xs text-gray-400">Win Rate</span>
            </div>
            <div className="font-bold text-lg text-white">
              {analytics.winRate.toFixed(1)}%
            </div>
            <div className="text-xs text-gray-400">
              {analytics.wins}W / {analytics.losses}L
            </div>
          </div>

          <div className="bg-gray-800 p-4 rounded-lg border border-gray-600">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-purple-400" />
              <span className="text-xs text-gray-400">Total Trades</span>
            </div>
            <div className="font-bold text-lg text-white">
              {analytics.totalTrades}
            </div>
            <div className="text-xs text-gray-400">
              {filteredTrades.length !== capitalCurve.length ? `of ${capitalCurve.length}` : 'Executed'}
            </div>
          </div>

          <div className="bg-gray-800 p-4 rounded-lg border border-gray-600">
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="w-4 h-4 text-red-400" />
              <span className="text-xs text-gray-400">Max Drawdown</span>
            </div>
            <div className="font-bold text-lg text-red-400">
              {analytics.maxDrawdown.toFixed(2)}%
            </div>
            <div className="text-xs text-gray-400">
              Peak to trough
            </div>
          </div>

          <div className="bg-gray-800 p-4 rounded-lg border border-gray-600">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-yellow-400" />
              <span className="text-xs text-gray-400">Avg P&L</span>
            </div>
            <div className={`font-bold text-lg ${getReturnColor(analytics.avgPnL)}`}>
              {formatCurrency(analytics.avgPnL)}
            </div>
            <div className="text-xs text-gray-400">
              Per trade
            </div>
          </div>

          <div className="bg-gray-800 p-4 rounded-lg border border-gray-600">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-orange-400" />
              <span className="text-xs text-gray-400">Profit Factor</span>
            </div>
            <div className="font-bold text-lg text-white">
              {analytics.profitFactor}
            </div>
            <div className="text-xs text-gray-400">
              Win/Loss ratio
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/50 border border-red-500 text-red-100 p-4 rounded-lg flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <div>
            <p className="font-semibold">Error loading backtest data:</p>
            <p className="text-sm">{error}</p>
          </div>
          <button
            onClick={() => setRetry(r => r + 1)}
            className="ml-auto bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm"
          >
            Retry
          </button>
        </div>
      )}

      {/* Chart Display */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-600">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            {chartType === "balance" && <TrendingUp className="w-5 h-5 text-green-400" />}
            {chartType === "pnl" && <DollarSign className="w-5 h-5 text-blue-400" />}
            {chartType === "trades" && <BarChart3 className="w-5 h-5 text-purple-400" />}
            {chartType === "balance" && "Portfolio Balance Over Time"}
            {chartType === "pnl" && "Cumulative P&L"}
            {chartType === "trades" && "Individual Trade Performance"}
          </h3>
          
          <div className="flex items-center gap-4 text-sm text-gray-400">
            {filteredTrades.length !== capitalCurve.length && (
              <span>Showing {filteredTrades.length} of {capitalCurve.length} trades</span>
            )}
            {autoRefresh && (
              <div className="flex items-center gap-1 text-blue-400">
                <RefreshCw className="w-3 h-3 animate-spin" />
                Auto-refresh
              </div>
            )}
          </div>
        </div>
        
        <div style={{ height: 450 }}>
          {loading && (
            <div className="flex items-center justify-center h-full">
              <RefreshCw className="animate-spin w-8 h-8 text-blue-400" />
              <span className="ml-3 text-blue-400">Loading chart data...</span>
            </div>
          )}
          
          {!loading && !error && filteredTrades.length > 0 && (
            <>
              {chartType === "trades" ? (
                <Bar data={getChartData()} options={chartOptions} />
              ) : (
                <Line data={getChartData()} options={chartOptions} />
              )}
            </>
          )}
          
          {!loading && !error && filteredTrades.length === 0 && (
            <div className="flex items-center justify-center h-full text-gray-400">
              <div className="text-center">
                <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-lg font-medium">No backtest data available</p>
                <p className="text-sm">Run a backtest to see your performance</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Trade History Table */}
      <div className="bg-gray-800 rounded-lg border border-gray-600">
        <div className="flex justify-between items-center p-4 border-b border-gray-600">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            Trade History
            {filteredTrades.length > 0 && (
              <span className="text-sm font-normal text-gray-400">
                ({filteredTrades.length} trades)
              </span>
            )}
          </h3>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowTable(!showTable)}
              className="flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm"
            >
              {showTable ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              {showTable ? "Hide" : "Show"} Table
            </button>
          </div>
        </div>
        
        {showTable && (
          <div className="overflow-x-auto">
            {loading ? (
              <div className="text-center py-8">
                <RefreshCw className="animate-spin w-6 h-6 text-blue-400 mx-auto mb-2" />
                <p className="text-gray-400">Loading trades...</p>
              </div>
            ) : filteredTrades.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No trades to display</p>
                <p className="text-sm">Adjust your filters or run a backtest</p>
              </div>
            ) : (
              <table className="min-w-full text-white">
                <thead className="bg-gray-700">
                  <tr>
                    <th className="py-3 px-4 text-left">#</th>
                    <th className="py-3 px-4 text-left">Timestamp</th>
                    <th className="py-3 px-4 text-left">Symbol</th>
                    <th className="py-3 px-4 text-right">Exit Price</th>
                    <th className="py-3 px-4 text-right">P&L ($)</th>
                    <th className="py-3 px-4 text-right">Balance</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTrades.map((trade, index) => (
                    <tr key={index} className="hover:bg-gray-700 border-b border-gray-600 transition-colors">
                      <td className="py-3 px-4 text-gray-400 text-sm">{index + 1}</td>
                      <td className="py-3 px-4 text-gray-300 text-sm">
                        {trade.timestamp
                          ? new Date(trade.timestamp).toLocaleString()
                          : "-"}
                      </td>
                      <td className="py-3 px-4 font-medium">
                        {trade.symbol || "-"}
                      </td>
                      <td className="py-3 px-4 text-right text-gray-300">
                        {trade.exit_price != null
                          ? `$${trade.exit_price.toLocaleString()}`
                          : "-"}
                      </td>
                      <td className={`py-3 px-4 text-right font-semibold ${
                        (trade.pnl_dollars || 0) > 0
                          ? "text-green-400"
                          : (trade.pnl_dollars || 0) < 0
                          ? "text-red-400"
                          : "text-gray-400"
                      }`}>
                        {trade.pnl_dollars != null
                          ? `$${trade.pnl_dollars.toFixed(4)}`
                          : "-"}
                      </td>
                      <td className="py-3 px-4 text-right font-mono text-white">
                        {trade.balance != null
                          ? `$${trade.balance.toFixed(2)}`
                          : "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </div>

      {/* Footer Stats */}
      {analytics && (
        <div className="text-center text-sm text-gray-400 pt-4 border-t border-gray-700">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
            <div>
              <p className="font-medium text-white mb-1">Portfolio Summary</p>
              <p>Started with {formatCurrency(initialCapital)}</p>
              <p>Current value: {formatCurrency(analytics.finalBalance)}</p>
              <p>Net return: {formatCurrency(analytics.totalReturn)} ({analytics.totalReturnPercent.toFixed(2)}%)</p>
            </div>
            <div>
              <p className="font-medium text-white mb-1">Trade Statistics</p>
              <p>Total trades: {analytics.totalTrades}</p>
              <p>Win rate: {analytics.winRate.toFixed(1)}% ({analytics.wins}W/{analytics.losses}L)</p>
              <p>Average P&L: {formatCurrency(analytics.avgPnL)}</p>
            </div>
            <div>
              <p className="font-medium text-white mb-1">Risk Metrics</p>
              <p>Max drawdown: {analytics.maxDrawdown.toFixed(2)}%</p>
              <p>Profit factor: {analytics.profitFactor}</p>
              {filteredTrades.length > 0 && (
                <p>Last trade: {new Date(filteredTrades[filteredTrades.length - 1]?.timestamp).toLocaleString()}</p>
              )}
            </div>
          </div>
          {autoRefresh && (
            <p className="mt-4 text-blue-400 text-center">
              Auto-refreshing every 30 seconds • Last update: {lastUpdate}
            </p>
          )}
        </div>
      )}
    </div>
  );
}