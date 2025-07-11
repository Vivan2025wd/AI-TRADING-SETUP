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
  const [error, setError] = useState(null);

  // Pagination state
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const limit = 10;

  useEffect(() => {
    const fetchRatedStrategies = async () => {
      setLoading(true);
      setError(null);

      try {
        // Step 1: fetch all saved strategies with symbol & strategy_id
        const resList = await fetch(`/api/strategy/list?page=${page}&limit=${limit}`);
        if (!resList.ok) throw new Error("Failed to fetch strategy list");
        const listData = await resList.json();
        setTotalPages(listData.totalPages || 1);

        // Step 2: For each symbol, fetch rated strategies using new endpoint
        const strategiesRaw = listData.data || [];

        // Group by symbol to minimize requests
        const symbolSet = [...new Set(strategiesRaw.map((s) => s.symbol))];

        let allRatedStrategies = [];

        for (const symbol of symbolSet) {
          const resRated = await fetch(`/api/strategy/${symbol}/rate-strategies`);
          if (!resRated.ok) {
            console.warn(`No rated strategies found for ${symbol}`);
            continue;
          }
          const ratedData = await resRated.json();

          if (ratedData.strategies && ratedData.strategies.length) {
            // Map and add symbol for uniformity
            const mapped = ratedData.strategies.map((strat) => ({
              ...strat,
              symbol,
            }));
            allRatedStrategies = allRatedStrategies.concat(mapped);
          }
        }

        // Filter for current page limit (we could do better server-side paging)
        const pagedStrategies = allRatedStrategies.slice(0, limit);

        // Parse rules for display (attempt to load strategy JSON from STRATEGY_DIR)
        const enriched = await Promise.all(
          pagedStrategies.map(async (strat) => {
            // Fetch strategy JSON for rules display
            let strategy_json = {};
            try {
              const resStratJson = await fetch(
                `/api/strategy/${strat.symbol}/${strat.strategy_id}`
              );
              if (resStratJson.ok) {
                strategy_json = await resStratJson.json();
              }
            } catch {
              // ignore error, rules empty
            }

            const rules = strategy_json.indicators
              ? Object.entries(strategy_json.indicators).flatMap(([indicator, config]) =>
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
              id: strat.strategy_id,
              agent: strat.symbol,
              winRate: strat.win_rate * 100 || 0,
              avgProfit: strat.avg_profit || 0,
              avgConfidence: strat.avg_confidence * 100 || 0,
              totalPredictions: strat.total || 0,
              rules,
              symbol: strat.symbol,
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

    fetchRatedStrategies();
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
                  <th className="py-3 px-4 border border-gray-700 text-left">Avg Profit %</th>
                  <th className="py-3 px-4 border border-gray-700 text-left">Avg Confidence</th>
                  <th className="py-3 px-4 border border-gray-700 text-left">Total Predictions</th>
                  <th className="py-3 px-4 border border-gray-700 text-left">Rules</th>
                  <th className="py-3 px-4 border border-gray-700 text-center">Actions</th>
                </tr>
              </thead>
              <tbody>
                {strategies.map(
                  ({ id, agent, winRate, avgProfit, avgConfidence, totalPredictions, rules, symbol }) => (
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
                      <td className="py-3 px-4 border border-gray-700">{avgProfit.toFixed(2)}</td>
                      <td className="py-3 px-4 border border-gray-700">{avgConfidence.toFixed(1)}%</td>
                      <td className="py-3 px-4 border border-gray-700">{totalPredictions}</td>
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
                  )
                )}
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
