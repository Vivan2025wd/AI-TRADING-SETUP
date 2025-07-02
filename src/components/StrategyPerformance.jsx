import React, { useEffect, useState } from "react";
import { Trash2 } from "lucide-react";

// --- Utility functions ---
const getOperator = (k) =>
  k.includes("below") ? "<" : k.includes("above") ? ">" : "==";

const parseConditionAction = (key) => {
  if (key.includes("buy")) return [getOperator(key), "BUY"];
  if (key.includes("sell")) return [getOperator(key), "SELL"];
  return ["==", "HOLD"];
};

export default function StrategyPerformance() {
  const [strategies, setStrategies] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/strategies/list");
        const data = await res.json();

        const enriched = await Promise.all(
          data.map(async (strat) => {
            const { symbol = "UNKNOWN", strategy_id, strategy_json } = strat;

            let winRate = 0.0;
            try {
              const perfRes = await fetch(
                `http://localhost:8000/api/strategies/${symbol}/${strategy_id}/performance`
              );
              if (perfRes.ok) {
                const perf = await perfRes.json();
                winRate = perf?.win_rate ? perf.win_rate * 100 : 0;
              } else {
                console.warn(
                  `No performance data for ${symbol}/${strategy_id}`
                );
              }
            } catch (err) {
              console.warn(
                `Error fetching performance for ${symbol}/${strategy_id}:`,
                err
              );
            }

            const rules = strategy_json?.indicators
              ? Object.entries(strategy_json.indicators).flatMap(
                  ([indicator, config]) =>
                    Object.entries(config).map(([cond, val]) => {
                      const [condition, action] = parseConditionAction(cond);
                      return {
                        indicator: indicator.toUpperCase(),
                        condition,
                        value: val,
                        action,
                      };
                    })
                )
              : [];

            return {
              id: strategy_id,
              agent: `${symbol} Agent`,
              winRate,
              rules,
              symbol,
            };
          })
        );

        setStrategies(enriched);
      } catch (err) {
        console.error("Failed to fetch strategies", err);
      } finally {
        setLoading(false);
      }
    };

    fetchStrategies();
  }, []);

  const deleteStrategy = async (symbol, id) => {
    if (!window.confirm("Are you sure you want to delete this strategy?"))
      return;

    try {
      const res = await fetch(
        `http://localhost:8000/api/strategies/${symbol}/${id}`,
        {
          method: "DELETE",
        }
      );

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Failed to delete strategy");
      }

      setStrategies((prev) => prev.filter((s) => s.id !== id));
    } catch (err) {
      alert("Error deleting strategy: " + err.message);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-900 rounded-xl shadow-lg border border-gray-700 text-white">
      <h2 className="text-2xl font-bold mb-6">Strategy Performance</h2>

      {loading ? (
        <p className="text-gray-400 text-center">Loading strategies...</p>
      ) : strategies.length === 0 ? (
        <p className="text-gray-400 text-center">No saved strategies found.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse border border-gray-700">
            <thead className="bg-gray-800">
              <tr>
                <th className="py-3 px-4 border border-gray-700 text-left">
                  Agent
                </th>
                <th className="py-3 px-4 border border-gray-700 text-left">
                  Win Rate
                </th>
                <th className="py-3 px-4 border border-gray-700 text-left">
                  Rules
                </th>
                <th className="py-3 px-4 border border-gray-700 text-center">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {strategies.map(({ id, agent, winRate, rules, symbol }) => (
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
                      title={`Win Rate: ${winRate.toFixed(1)}%`}
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
                      onClick={() => deleteStrategy(symbol, id)}
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
