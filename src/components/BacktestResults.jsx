import React from "react";
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

const mockTrades = [
  {
    type: "BUY",
    timestamp: "2025-06-01T10:00:00Z",
    price: 20000,
    balance: 1000,
  },
  {
    type: "SELL",
    timestamp: "2025-06-02T14:00:00Z",
    price: 22000,
    profit_percent: 10,
    balance: 1100,
  },
  {
    type: "BUY",
    timestamp: "2025-06-04T09:30:00Z",
    price: 21500,
    balance: 1100,
  },
  {
    type: "SELL",
    timestamp: "2025-06-05T12:00:00Z",
    price: 23000,
    profit_percent: 7,
    balance: 1177,
  },
];

const BacktestResults = () => {
  const chartData = {
    labels: mockTrades.map(trade => new Date(trade.timestamp).toLocaleString()),
    datasets: [
      {
        label: "Balance Over Time",
        data: mockTrades.map(trade => trade.balance),
        fill: true,
        borderColor: "rgb(34,197,94)", // green-500
        backgroundColor: "rgba(34,197,94,0.2)",
        tension: 0.3,
      },
    ],
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold text-white border-b border-gray-700 pb-3 mb-6">
        Backtest Results
      </h2>

      {/* Chart */}
      <div className="bg-gray-900 p-6 rounded-xl shadow-lg">
        <Line
          data={chartData}
          options={{ 
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { labels: { color: "white" } },
              tooltip: { 
                backgroundColor: "#111827", // gray-900
                titleColor: "white",
                bodyColor: "white",
              },
            },
            scales: {
              x: { ticks: { color: "white" }, grid: { color: "#374151" } }, // gray-700 grid
              y: { ticks: { color: "white" }, grid: { color: "#374151" } },
            },
          }}
          height={220}
        />
      </div>

      {/* Table */}
      <div className="overflow-x-auto bg-gray-900 rounded-xl shadow-lg">
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
            {mockTrades.map((trade, i) => (
              <tr
                key={i}
                className="border-t border-gray-700 hover:bg-gray-800 transition-colors"
              >
                <td className="px-6 py-3 font-semibold">
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-semibold ${
                      trade.type === "BUY"
                        ? "bg-blue-700 text-blue-200"
                        : "bg-green-700 text-green-200"
                    }`}
                  >
                    {trade.type}
                  </span>
                </td>
                <td className="px-6 py-3 text-gray-300">
                  {new Date(trade.timestamp).toLocaleString()}
                </td>
                <td className="px-6 py-3 text-gray-300">
                  ${trade.price.toLocaleString()}
                </td>
                <td
                  className={`px-6 py-3 font-semibold ${
                    trade.profit_percent > 0
                      ? "text-green-400"
                      : trade.profit_percent < 0
                      ? "text-red-400"
                      : "text-gray-400"
                  }`}
                >
                  {trade.profit_percent ? `${trade.profit_percent}%` : "-"}
                </td>
                <td className="px-6 py-3 font-semibold text-gray-300">
                  ${trade.balance.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default BacktestResults;
