import React, { useEffect, useState } from "react"; 
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

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

// Helper: fetch with timeout
const fetchWithTimeout = (url, options = {}, timeout = 10000) =>
  Promise.race([
    fetch(url, options),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Request timed out")), timeout)
    ),
  ]);

export default function BacktestResults() {
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retry, setRetry] = useState(0); // trigger for retry

  useEffect(() => {
    async function fetchBacktestResults() {
      setLoading(true);
      setError(null);
      try {
        // Add pagination query params here as needed
        const res = await fetchWithTimeout("http://localhost:8000/api/backtest/results?page=1&limit=100");
        if (!res.ok) throw new Error("Failed to fetch backtest results");
        const data = await res.json();

        // data object has 'data' property with the trades array
        setTrades(Array.isArray(data.data) ? data.data : []);
      } catch (err) {
        console.error(err);
        setError(err.message || "Unknown error occurred");
      } finally {
        setLoading(false);
      }
    }

    fetchBacktestResults();
  }, [retry]);

  const chartData = {
    labels: trades.map((trade) =>
      trade.timestamp ? new Date(trade.timestamp).toLocaleString() : "-"
    ),
    datasets: [
      {
        label: "Balance Over Time",
        data: trades.map((trade) => trade.balance ?? null),
        fill: true,
        borderColor: "rgb(34,197,94)",
        backgroundColor: "rgba(34,197,94,0.2)",
        tension: 0.3,
        spanGaps: true,
      },
    ],
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold text-white border-b border-gray-700 pb-3 mb-6">
        Backtest Results
      </h2>

      {/* Chart Section */}
      <div className="bg-gray-900 p-6 rounded-xl shadow-lg" style={{ minHeight: 250 }}>
        {loading ? (
          <p className="text-gray-400 text-center">⏳ Loading chart...</p>
        ) : error ? (
          <div className="text-red-500 text-center space-y-3">
            <p>⚠️ Error loading chart: {error}</p>
            <button
              onClick={() => setRetry((r) => r + 1)}
              className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-white"
            >
              Retry
            </button>
          </div>
        ) : trades.length === 0 ? (
          <p className="text-gray-400 text-center">No backtest trades found.</p>
        ) : (
          <Line
            data={chartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { labels: { color: "white" } },
                tooltip: {
                  backgroundColor: "#111827",
                  titleColor: "white",
                  bodyColor: "white",
                },
              },
              scales: {
                x: { ticks: { color: "white" }, grid: { color: "#374151" } },
                y: { ticks: { color: "white" }, grid: { color: "#374151" } },
              },
            }}
            height={220}
          />
        )}
      </div>

      {/* Table Section */}
      <div className="overflow-x-auto bg-gray-900 rounded-xl shadow-lg">
        {loading ? (
          <p className="text-gray-400 text-center p-4">⏳ Loading table...</p>
        ) : error ? (
          <div className="text-red-500 text-center p-4 space-y-3">
            <p>⚠️ Error loading table: {error}</p>
            <button
              onClick={() => setRetry((r) => r + 1)}
              className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-white"
            >
              Retry
            </button>
          </div>
        ) : trades.length === 0 ? (
          <p className="text-gray-400 text-center p-4">No trades to display.</p>
        ) : (
          <table className="min-w-full text-sm text-left">
            <thead className="bg-gray-800 text-gray-300">
              <tr>
                <th className="px-6 py-3">Type</th>
                <th className="px-6 py-3">Timestamp</th>
                <th className="px-6 py-3">Price</th>
                <th className="px-6 py-3">Profit %</th>
                <th className="px-6 py-3">Balance</th>
              </tr>
            </thead>
            <tbody>
              {[...trades]
                .slice(-10)
                .reverse()
                .map((trade, i) => (
                  <tr
                    key={i}
                    className="border-t border-gray-700 hover:bg-gray-800 transition-colors"
                  >
                    <td className="px-6 py-3 font-semibold">
                      <span
                        className={`px-3 py-1 rounded-full text-xs font-semibold ${
                          trade.type === "BUY"
                            ? "bg-blue-700 text-blue-200"
                            : trade.type === "SELL"
                            ? "bg-red-700 text-red-200"
                            : "bg-gray-700 text-gray-300"
                        }`}
                      >
                        {trade.type}
                      </span>
                    </td>
                    <td className="px-6 py-3 text-gray-300">
                      {trade.timestamp
                        ? new Date(trade.timestamp).toLocaleString()
                        : "-"}
                    </td>
                    <td className="px-6 py-3 text-gray-300">
                      {trade.price != null
                        ? `$${trade.price.toLocaleString()}`
                        : "-"}
                    </td>
                    <td
                      className={`px-6 py-3 font-semibold ${
                        typeof trade.profit_percent === "number"
                          ? trade.profit_percent > 0
                            ? "text-green-400"
                            : trade.profit_percent < 0
                            ? "text-red-400"
                            : "text-gray-400"
                          : "text-gray-400"
                      }`}
                    >
                      {typeof trade.profit_percent === "number"
                        ? `${trade.profit_percent.toFixed(2)}%`
                        : "-"}
                    </td>
                    <td className="px-6 py-3 font-semibold text-gray-300">
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
    </div>
  );
}
