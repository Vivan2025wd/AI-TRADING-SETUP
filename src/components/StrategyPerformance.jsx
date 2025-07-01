import React, { useState } from "react";
import { Trash2 } from "lucide-react";

const initialStrategies = [
  {
    id: 1,
    agent: "BTC Agent",
    winRate: 72.5,
    rules: [
      { indicator: "RSI", condition: "<", value: 30, action: "BUY" },
      { indicator: "MACD", condition: ">", value: 0, action: "SELL" },
    ],
  },
  {
    id: 2,
    agent: "ETH Agent",
    winRate: 64.0,
    rules: [
      { indicator: "SMA", condition: ">", value: 50, action: "BUY" },
      { indicator: "EMA", condition: "<", value: 20, action: "HOLD" },
    ],
  },
  {
    id: 3,
    agent: "SOL Agent",
    winRate: 80.2,
    rules: [
      { indicator: "RSI", condition: "<", value: 25, action: "BUY" },
      { indicator: "MACD", condition: "<", value: 0, action: "SELL" },
      { indicator: "SMA", condition: ">", value: 100, action: "BUY" },
    ],
  },
];

export default function StrategyPerformance() {
  const [strategies, setStrategies] = useState(initialStrategies);

  const deleteStrategy = (id) => {
    if (window.confirm("Are you sure you want to delete this strategy?")) {
      setStrategies((prev) => prev.filter((strategy) => strategy.id !== id));
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-900 rounded-xl shadow-lg border border-gray-700 text-white">
      <h2 className="text-2xl font-bold mb-6">📈 Strategy Performance</h2>

      {strategies.length === 0 ? (
        <p className="text-gray-400 text-center">No saved strategies found.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse border border-gray-700">
            <thead className="bg-gray-800">
              <tr>
                <th className="py-3 px-4 border border-gray-700 text-left">Agent</th>
                <th className="py-3 px-4 border border-gray-700 text-left">Win Rate</th>
                <th className="py-3 px-4 border border-gray-700 text-left">Rules</th>
                <th className="py-3 px-4 border border-gray-700 text-center">Actions</th>
              </tr>
            </thead>

            <tbody>
              {strategies.map(({ id, agent, winRate, rules }) => (
                <tr
                  key={id}
                  className="hover:bg-gray-800 transition-colors"
                >
                  <td className="py-3 px-4 border border-gray-700">{agent}</td>
                  <td className="py-3 px-4 border border-gray-700">
                    <span
                      className={`font-semibold px-3 py-1 rounded-full inline-block ${
                        winRate >= 70
                          ? "bg-green-700 text-green-300"
                          : winRate >= 50
                          ? "bg-yellow-700 text-yellow-300"
                          : "bg-red-700 text-red-300"
                      }`}
                      title={`Win Rate: ${winRate}%`}
                    >
                      {winRate.toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 border border-gray-700 text-sm whitespace-pre-wrap">
                    {rules
                      .map(
                        ({ indicator, condition, value, action }) =>
                          `IF ${indicator} ${condition} ${value} THEN ${action}`
                      )
                      .join("\n")}
                  </td>
                  <td className="py-3 px-4 border border-gray-700 text-center">
                    <button
                      onClick={() => deleteStrategy(id)}
                      className="text-red-500 hover:text-red-700 transition"
                      title="Delete strategy"
                    >
                      <Trash2 className="w-5 h-5 mx-auto" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
