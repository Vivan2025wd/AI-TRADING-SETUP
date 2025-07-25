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

const fetchWithTimeout = (url, options = {}, timeout = 10000) =>
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

  useEffect(() => {
    async function fetchBacktestResults() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetchWithTimeout("http://localhost:8000/api/backtest/results");
        if (!res.ok) throw new Error("Failed to fetch backtest results");
        const data = await res.json();

        if (!data.capital_curve) {
          throw new Error("Invalid data format: missing capital_curve");
        }

        setInitialCapital(data.initial_capital || 100);
        setCapitalCurve(data.capital_curve);
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
    labels: capitalCurve.map((point) =>
      point.timestamp ? new Date(point.timestamp).toLocaleString() : "-"
    ),
    datasets: [
      {
        label: `Balance Over Time (Start: $${initialCapital})`,
        data: capitalCurve.map((point) => point.balance ?? null),
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
        Backtest Results - Capital Curve
      </h2>

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
        ) : capitalCurve.length === 0 ? (
          <p className="text-gray-400 text-center">No backtest data found.</p>
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
        ) : capitalCurve.length === 0 ? (
          <p className="text-gray-400 text-center p-4">No trades to display.</p>
        ) : (
          <table className="min-w-full text-sm text-left">
            <thead className="bg-gray-800 text-gray-300">
              <tr>
                <th className="px-6 py-3">Timestamp</th>
                <th className="px-6 py-3">Symbol</th>
                <th className="px-6 py-3">Exit Price</th>
                <th className="px-6 py-3">PnL ($)</th>
                <th className="px-6 py-3">Balance</th>
              </tr>
            </thead>
            <tbody>
              {[...capitalCurve]
                .slice(-10)
                .reverse()
                .map((trade, i) => (
                  <tr
                    key={i}
                    className="border-t border-gray-700 hover:bg-gray-800 transition-colors"
                  >
                    <td className="px-6 py-3 text-gray-300">
                      {trade.timestamp
                        ? new Date(trade.timestamp).toLocaleString()
                        : "-"}
                    </td>
                    <td className="px-6 py-3 text-gray-300">{trade.symbol || "-"}</td>
                    <td className="px-6 py-3 text-gray-300">
                      {trade.exit_price != null
                        ? `$${trade.exit_price.toLocaleString()}`
                        : "-"}
                    </td>
                    <td
                      className={`px-6 py-3 font-semibold ${
                        trade.pnl_dollars > 0
                          ? "text-green-400"
                          : trade.pnl_dollars < 0
                          ? "text-red-400"
                          : "text-gray-400"
                      }`}
                    >
                      {trade.pnl_dollars != null
                        ? `$${trade.pnl_dollars.toFixed(4)}`
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
