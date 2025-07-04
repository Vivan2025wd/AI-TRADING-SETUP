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
  const [error, setError] = useState(null); // <-- Added error state

  // Pagination state
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const limit = 10;

  useEffect(() => {
    const fetchStrategies = async () => {
      setLoading(true);
      setError(null); // Reset error on each fetch
      try {
        const res = await fetch(`/api/strategy/list?page=${page}&limit=${limit}`);
        if (!res.ok) throw new Error("Failed to fetch strategies");

        const data = await res.json();
        const strategiesRaw = Array.isArray(data.data) ? data.data : [];

        setTotalPages(data.totalPages || 1);

        // Enrich strategies with performance data and parsed rules
        const enriched = await Promise.all(
          strategiesRaw.map(async (strat) => {
            const { symbol = "UNKNOWN", strategy_id, strategy_json } = strat;

            let winRate = 0.0;
            try {
              const perfRes = await fetch(
                `/api/strategy/${symbol}/${strategy_id}/performance`
              );
              if (perfRes.ok) {
                const perf = await perfRes.json();
                winRate = perf?.win_rate ? perf.win_rate * 100 : 0;
              } else {
                console.warn(`No performance data for ${symbol}/${strategy_id}`);
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
              agent: symbol,
              winRate,
              rules,
              symbol,
            };
          })
        );

        setStrategies(enriched);
      } catch (err) {
        console.error("Failed to fetch strategies", err);
        setError("Failed to load strategies. Please try again later.");
        setStrategies([]);
      } finally {
        setLoading(false);
      }
    };

    fetchStrategies();
  }, [page]);

  const deleteStrategy = async (symbol, id) => {
    if (!window.confirm("Are you sure you want to delete this strategy?")) return;

    try {
      const strategyKey = `${symbol}_strategy_${id}`;
      const res = await fetch(`/api/strategy/${strategyKey}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Failed to delete strategy");
      }

      // Remove deleted strategy from current list
      setStrategies((prev) => prev.filter((s) => s.id !== id));
    } catch (err) {
      alert("Error deleting strategy: " + err.message);
    }
  };

  // Pagination handlers
  const goToNextPage = () => {
    if (page < totalPages) setPage(page + 1);
  };

  const goToPrevPage = () => {
    if (page > 1) setPage(page - 1);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-900 rounded-xl shadow-lg border border-gray-700 text-white">
      <h2 className="text-2xl font-bold mb-6">Strategy Performance</h2>

      {loading ? (
        <p className="text-gray-400 text-center">Loading strategies...</p>
      ) : error ? (
        <p className="text-red-500 text-center font-semibold">{error}</p>
      ) : strategies.length === 0 ? (
        <p className="text-gray-400 text-center">No saved strategies found.</p>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse border border-gray-700">
              <thead className="bg-gray-800">
                <tr>
                  <th className="py-3 px-4 border border-gray-700 text-left">Agent</th>
                  <th className="py-3 px-4 border border-gray-700 text-left">Strategy Name</th>
                  <th className="py-3 px-4 border border-gray-700 text-left">Win Rate</th>
                  <th className="py-3 px-4 border border-gray-700 text-left">Rules</th>
                  <th className="py-3 px-4 border border-gray-700 text-center">Actions</th>
                </tr>
              </thead>
              <tbody>
                {strategies.map(({ id, agent, winRate, rules, symbol }) => (
                  <tr key={id} className="hover:bg-gray-800 transition-colors">
                    <td className="py-3 px-4 border border-gray-700">{agent}</td>
                    <td className="py-3 px-4 border border-gray-700 font-semibold">{id}</td>
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

          {/* Pagination controls */}
          <div className="flex justify-between mt-4">
            <button
              onClick={goToPrevPage}
              disabled={page === 1}
              className={`px-4 py-2 rounded ${
                page === 1 ? "bg-gray-700 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
              } text-white`}
            >
              Previous
            </button>

            <span className="self-center">
              Page {page} of {totalPages}
            </span>

            <button
              onClick={goToNextPage}
              disabled={page === totalPages}
              className={`px-4 py-2 rounded ${
                page === totalPages ? "bg-gray-700 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
              } text-white`}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}
