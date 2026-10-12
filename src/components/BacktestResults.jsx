import React, { useEffect, useState, useMemo } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";
import {
  RefreshCw,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Activity,
  Target,
  BarChart3,
  AlertCircle,
  Clock,
  ArrowUp,
  ArrowDown,
  Minus,
  Settings,
  Eye,
  EyeOff
} from "lucide-react";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

const fetchWithTimeout = (url, options = {}, timeout = 300000) =>
  Promise.race([
    fetch(url, options),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Request timed out")), timeout)
    ),
  ]);

export default function BacktestResults() {
  const [capitalCurve, setCapitalCurve] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retry, setRetry] = useState(0);
  const [initialCapital, setInitialCapital] = useState(0);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showAllTrades, setShowAllTrades] = useState(false);
  const [chartHeight, setChartHeight] = useState(300);

  useEffect(() => {
    async function fetchBacktestResults() {
      setLoading(true);
      setError(null);
      try {
        // Add cache busting with timestamp and get more results
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
        console.log("Fetched data:", data); // Debug log

        if (!data.capital_curve) {
          throw new Error("Invalid data format: missing capital_curve");
        }

        setInitialCapital(data.initial_capital || 100);
        setCapitalCurve(data.capital_curve);
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

  // Auto-refresh every 30 seconds - controllable
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      setRetry(r => r + 1);
    }, 30000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  // Calculate enhanced analytics
  const analytics = useMemo(() => {
    if (capitalCurve.length === 0) return null;

    const finalBalance = capitalCurve[capitalCurve.length - 1]?.balance || initialCapital;
    const totalReturn = finalBalance - initialCapital;
    const totalReturnPercent = (totalReturn / initialCapital) * 100;
    
    const wins = capitalCurve.filter(t => t.pnl_dollars > 0).length;
    const losses = capitalCurve.filter(t => t.pnl_dollars < 0).length;
    const winRate = capitalCurve.length > 0 ? (wins / capitalCurve.length) * 100 : 0;
    
    // Calculate max drawdown
    const balances = capitalCurve.map(t => t.balance);
    let maxBalance = initialCapital;
    let maxDrawdown = 0;
    
    balances.forEach(balance => {
      if (balance > maxBalance) {
        maxBalance = balance;
      }
      const drawdown = ((maxBalance - balance) / maxBalance) * 100;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    });

    // Calculate average P&L
    const avgPnL = capitalCurve.length > 0 ? 
      capitalCurve.reduce((sum, t) => sum + (t.pnl_dollars || 0), 0) / capitalCurve.length : 0;

    // Calculate profit factor
    const grossProfit = capitalCurve.filter(t => t.pnl_dollars > 0)
      .reduce((sum, t) => sum + t.pnl_dollars, 0);
    const grossLoss = Math.abs(capitalCurve.filter(t => t.pnl_dollars < 0)
      .reduce((sum, t) => sum + t.pnl_dollars, 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0;

    return {
      totalReturn,
      totalReturnPercent,
      finalBalance,
      winRate,
      wins,
      losses,
      maxDrawdown,
      avgPnL,
      profitFactor,
      grossProfit,
      grossLoss
    };
  }, [capitalCurve, initialCapital]);

  const chartData = {
    labels: capitalCurve.map((point, index) =>
      point.timestamp ? new Date(point.timestamp).toLocaleDateString() : `Trade ${index + 1}`
    ),
    datasets: [
      {
        label: `Balance Over Time (Start: $${initialCapital})`,
        data: capitalCurve.map((point) => point.balance ?? null),
        fill: true,
        borderColor: "rgb(34,197,94)",
        backgroundColor: "rgba(34,197,94,0.15)",
        tension: 0.4,
        spanGaps: true,
        pointRadius: capitalCurve.length > 100 ? 0 : 2,
        pointHoverRadius: 4,
        borderWidth: 2,
      },
    ],
  };

  const totalPnL = capitalCurve.length > 0 
    ? (capitalCurve[capitalCurve.length - 1].balance - initialCapital).toFixed(2)
    : "0.00";

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

  const tradesToShow = showAllTrades ? capitalCurve : capitalCurve.slice(-20);

  return (
    <div className="space-y-8 max-w-6xl mx-auto bg-gray-900 p-6 rounded-2xl">
      {/* Enhanced Header */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-8 h-8 text-blue-400" />
          <div>
            <h2 className="text-3xl font-bold text-white">
              Backtest Results
            </h2>
            <p className="text-gray-400 text-sm">Capital Curve & Performance Analytics</p>
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
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white transition-all duration-200 disabled:opacity-50"
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {/* Status Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-gray-800 p-4 rounded-lg border border-gray-700">
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${error ? 'bg-red-400' : loading ? 'bg-yellow-400' : 'bg-green-400'}`} />
            <span className="text-gray-300">
              {error ? 'Error' : loading ? 'Loading' : 'Connected'}
            </span>
          </div>
          {capitalCurve.length > 0 && (
            <div className="text-gray-400">
              {capitalCurve.length} trades loaded
            </div>
          )}
          {autoRefresh && !loading && (
            <div className="flex items-center gap-1 text-blue-400">
              <RefreshCw className="w-3 h-3 animate-spin" />
              <span>Auto-refresh active</span>
            </div>
          )}
        </div>
        {lastUpdate && (
          <div className="text-gray-400 text-sm">
            Last updated: {lastUpdate}
          </div>
        )}
      </div>

      {/* Enhanced Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 hover:border-gray-600 transition-colors">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-400">Total Trades</h3>
            <Activity className="w-5 h-5 text-blue-400" />
          </div>
          <p className="text-2xl font-bold text-white mb-1">{capitalCurve.length}</p>
          {analytics && (
            <p className="text-xs text-gray-500">
              {analytics.wins}W / {analytics.losses}L
            </p>
          )}
        </div>

        <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 hover:border-gray-600 transition-colors">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-400">Total P&L</h3>
            <DollarSign className="w-5 h-5 text-green-400" />
          </div>
          <p className={`text-2xl font-bold mb-1 ${getReturnColor(parseFloat(totalPnL))}`}>
            ${totalPnL}
          </p>
          {analytics && (
            <div className={`flex items-center gap-1 text-xs ${getReturnColor(analytics.totalReturnPercent)}`}>
              {getReturnIcon(analytics.totalReturnPercent)}
              {analytics.totalReturnPercent.toFixed(2)}%
            </div>
          )}
        </div>

        <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 hover:border-gray-600 transition-colors">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-400">Current Balance</h3>
            <TrendingUp className="w-5 h-5 text-purple-400" />
          </div>
          <p className="text-2xl font-bold text-white mb-1">
            ${capitalCurve.length > 0 ? capitalCurve[capitalCurve.length - 1].balance.toFixed(2) : initialCapital}
          </p>
          <p className="text-xs text-gray-500">
            Started: ${initialCapital}
          </p>
        </div>

        <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 hover:border-gray-600 transition-colors">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-400">Win Rate</h3>
            <Target className="w-5 h-5 text-orange-400" />
          </div>
          <p className="text-2xl font-bold text-white mb-1">
            {analytics ? `${analytics.winRate.toFixed(1)}%` : '0%'}
          </p>
          {analytics && (
            <p className="text-xs text-gray-500">
              Avg: {formatCurrency(analytics.avgPnL)}
            </p>
          )}
        </div>
      </div>

      {/* Additional Analytics Row */}
      {analytics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="w-4 h-4 text-red-400" />
              <span className="text-sm text-gray-400">Max Drawdown</span>
            </div>
            <div className="text-lg font-bold text-red-400">
              {analytics.maxDrawdown.toFixed(2)}%
            </div>
          </div>

          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-yellow-400" />
              <span className="text-sm text-gray-400">Profit Factor</span>
            </div>
            <div className="text-lg font-bold text-white">
              {analytics.profitFactor.toFixed(2)}
            </div>
          </div>

          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign className="w-4 h-4 text-blue-400" />
              <span className="text-sm text-gray-400">Gross Profit</span>
            </div>
            <div className="text-lg font-bold text-green-400">
              {formatCurrency(analytics.grossProfit)}
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Chart */}
      <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-400" />
            Capital Curve
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setChartHeight(chartHeight === 300 ? 450 : 300)}
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              {chartHeight === 300 ? '↗ Expand' : '↙ Compact'}
            </button>
          </div>
        </div>
        
        <div style={{ height: chartHeight }}>
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <RefreshCw className="animate-spin w-8 h-8 text-blue-400 mr-3" />
              <span className="text-blue-400">Loading chart data...</span>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-full text-red-400 space-y-3">
              <AlertCircle className="w-12 h-12" />
              <div className="text-center">
                <p className="font-semibold">Error loading chart</p>
                <p className="text-sm text-gray-400">{error}</p>
              </div>
              <button
                onClick={() => setRetry((r) => r + 1)}
                className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-white transition-colors"
              >
                Retry
              </button>
            </div>
          ) : capitalCurve.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-400">
              <BarChart3 className="w-12 h-12 mb-3 opacity-50" />
              <p className="text-lg">No backtest data found</p>
              <p className="text-sm">Run a backtest to see your performance</p>
            </div>
          ) : (
            <Line
              data={chartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                  mode: 'nearest',
                  intersect: false,
                },
                plugins: {
                  legend: { 
                    labels: { color: "white" },
                    position: 'top'
                  },
                  tooltip: {
                    backgroundColor: "rgba(17, 24, 39, 0.95)",
                    titleColor: "white",
                    bodyColor: "white",
                    borderColor: "#374151",
                    borderWidth: 1,
                    callbacks: {
                      label: (context) => {
                        const trade = capitalCurve[context.dataIndex];
                        const lines = [
                          `Balance: ${formatCurrency(trade.balance)}`,
                          `P&L: ${formatCurrency(trade.pnl_dollars || 0)}`,
                        ];
                        if (trade.symbol) lines.push(`Symbol: ${trade.symbol}`);
                        if (trade.timestamp) lines.push(`Time: ${new Date(trade.timestamp).toLocaleString()}`);
                        return lines;
                      }
                    }
                  },
                },
                scales: {
                  x: { 
                    ticks: { 
                      color: "white",
                      maxTicksLimit: 8
                    }, 
                    grid: { color: "#374151" },
                    title: {
                      display: true,
                      text: 'Trades',
                      color: 'white'
                    }
                  },
                  y: { 
                    ticks: { 
                      color: "white",
                      callback: function(value) {
                        return '$' + value.toLocaleString();
                      }
                    }, 
                    grid: { color: "#374151" },
                    title: {
                      display: true,
                      text: 'Portfolio Value ($)',
                      color: 'white'
                    }
                  },
                },
                elements: {
                  point: {
                    hoverBackgroundColor: 'rgb(34,197,94)',
                    hoverBorderColor: 'white',
                    hoverBorderWidth: 2
                  }
                }
              }}
            />
          )}
        </div>
      </div>

      {/* Enhanced Trade History Table */}
      <div className="bg-gray-800 rounded-xl shadow-lg border border-gray-700">
        <div className="flex justify-between items-center p-6 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            Recent Trades
            <span className="text-sm font-normal text-gray-400">
              ({showAllTrades ? capitalCurve.length : Math.min(20, capitalCurve.length)} trades)
            </span>
          </h3>
          <button
            onClick={() => setShowAllTrades(!showAllTrades)}
            className="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors"
          >
            {showAllTrades ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            {showAllTrades ? 'Show Recent' : 'Show All'}
          </button>
        </div>
        
        <div className="overflow-x-auto">
          {loading ? (
            <div className="text-center py-8">
              <RefreshCw className="animate-spin w-6 h-6 text-blue-400 mx-auto mb-2" />
              <p className="text-gray-400">Loading trades...</p>
            </div>
          ) : error ? (
            <div className="text-center py-8 space-y-3">
              <AlertCircle className="w-8 h-8 text-red-400 mx-auto" />
              <div>
                <p className="text-red-400 font-semibold">Error loading trades</p>
                <p className="text-gray-400 text-sm">{error}</p>
              </div>
              <button
                onClick={() => setRetry((r) => r + 1)}
                className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-white transition-colors"
              >
                Retry
              </button>
            </div>
          ) : capitalCurve.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No trades to display</p>
            </div>
          ) : (
            <table className="min-w-full text-sm">
              <thead className="bg-gray-700 text-gray-300">
                <tr>
                  <th className="px-6 py-4 text-left">#</th>
                  <th className="px-6 py-4 text-left">Timestamp</th>
                  <th className="px-6 py-4 text-left">Symbol</th>
                  <th className="px-6 py-4 text-right">Exit Price</th>
                  <th className="px-6 py-4 text-right">P&L ($)</th>
                  <th className="px-6 py-4 text-right">Balance</th>
                </tr>
              </thead>
              <tbody>
                {[...tradesToShow]
                  .reverse()
                  .map((trade, i) => {
                    const originalIndex = capitalCurve.length - i;
                    return (
                      <tr
                        key={i}
                        className="border-t border-gray-700 hover:bg-gray-700 transition-colors"
                      >
                        <td className="px-6 py-4 text-gray-400 font-mono text-xs">
                          {originalIndex}
                        </td>
                        <td className="px-6 py-4 text-gray-300">
                          {trade.timestamp
                            ? new Date(trade.timestamp).toLocaleString()
                            : "-"}
                        </td>
                        <td className="px-6 py-4 text-gray-300 font-medium">
                          {trade.symbol || "-"}
                        </td>
                        <td className="px-6 py-4 text-right text-gray-300 font-mono">
                          {trade.exit_price != null
                            ? `$${trade.exit_price.toLocaleString()}`
                            : "-"}
                        </td>
                        <td
                          className={`px-6 py-4 text-right font-semibold ${
                            (trade.pnl_dollars || 0) > 0
                              ? "text-green-400"
                              : (trade.pnl_dollars || 0) < 0
                              ? "text-red-400"
                              : "text-gray-400"
                          }`}
                        >
                          <div className="flex items-center justify-end gap-1">
                            {getReturnIcon(trade.pnl_dollars || 0)}
                            {trade.pnl_dollars != null
                              ? `$${trade.pnl_dollars.toFixed(4)}`
                              : "-"}
                          </div>
                        </td>
                        <td className="px-6 py-4 text-right font-semibold text-white font-mono">
                          {trade.balance != null
                            ? `$${trade.balance.toFixed(2)}`
                            : "-"}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Enhanced Footer */}
      {analytics && (
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <h4 className="font-semibold text-white mb-2">Portfolio Summary</h4>
              <div className="space-y-1 text-gray-400">
                <p>Initial Capital: {formatCurrency(initialCapital)}</p>
                <p>Final Balance: {formatCurrency(analytics.finalBalance)}</p>
                <p className={getReturnColor(analytics.totalReturn)}>
                  Net Return: {formatCurrency(analytics.totalReturn)} ({analytics.totalReturnPercent.toFixed(2)}%)
                </p>
              </div>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Trade Statistics</h4>
              <div className="space-y-1 text-gray-400">
                <p>Total Trades: {capitalCurve.length}</p>
                <p>Win Rate: {analytics.winRate.toFixed(1)}% ({analytics.wins}W/{analytics.losses}L)</p>
                <p>Avg P&L: {formatCurrency(analytics.avgPnL)}</p>
              </div>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Risk Metrics</h4>
              <div className="space-y-1 text-gray-400">
                <p>Max Drawdown: {analytics.maxDrawdown.toFixed(2)}%</p>
                <p>Profit Factor: {analytics.profitFactor.toFixed(2)}</p>
                <p>Gross Profit: {formatCurrency(analytics.grossProfit)}</p>
              </div>
            </div>
          </div>
          {autoRefresh && (
            <div className="mt-4 pt-4 border-t border-gray-700 text-center text-xs text-blue-400">
              Auto-refreshing every 30 seconds • Last update: {lastUpdate}
            </div>
          )}
        </div>
      )}
    </div>
  );
}