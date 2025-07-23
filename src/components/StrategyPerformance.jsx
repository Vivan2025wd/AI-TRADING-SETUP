import React, { useEffect, useState } from "react";
import { Trash2 } from "lucide-react";

// Helpers
const getOperator = (k) => (k.includes("below") ? "<" : k.includes("above") ? ">" : "==");
const parseConditionAction = (key) => {
  if (key.includes("buy")) return [getOperator(key), "BUY"];
  if (key.includes("sell")) return [getOperator(key), "SELL"];
  return ["==", "HOLD"];
};

export default function StrategyPerformance() {
  const [strategies, setStrategies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const limit = 10;

  const [deleting, setDeleting] = useState(null); // for confirmation modal

  useEffect(() => {
    const fetchRatedStrategies = async () => {
      setLoading(true);
      setError(null);

      try {
        const resList = await fetch(`/api/strategies/list?page=${page}&limit=${limit}`);
        if (!resList.ok) throw new Error("Failed to fetch strategy list");
        const listData = await resList.json();
        setTotalPages(listData.totalPages || 1);

        const strategiesRaw = listData.data || [];
        const symbolSet = [...new Set(strategiesRaw.map((s) => s.symbol))];
        let allRatedStrategies = [];

        for (const symbol of symbolSet) {
          try {
            const resRated = await fetch(`/api/strategies/${symbol}/rate-strategies`);
            if (!resRated.ok) continue;
            const ratedData = await resRated.json();
            if (ratedData.strategies?.length) {
              const mapped = ratedData.strategies.map((s) => ({ ...s, symbol }));
              allRatedStrategies.push(...mapped);
            }
          } catch {}
        }

        const paged = allRatedStrategies.slice(0, limit);

        const enriched = await Promise.all(
          paged.map(async (strat) => {
            let strategy_json = {};
            try {
              const res = await fetch(`/api/strategies/${strat.symbol}/${strat.strategy_id}`);
              if (res.ok) strategy_json = await res.json();
            } catch {}

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
              totalProfit: strat.total_profit || 0,
              wins: strat.wins || 0,
              losses: strat.losses || 0,
              rules,
              symbol: strat.symbol,
            };
          })
        );

        setStrategies(enriched);
      } catch (err) {
        console.error("Fetch error", err);
        setError("Could not load strategies. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    fetchRatedStrategies();
  }, [page]);

  const handleDelete = async () => {
    if (!deleting) return;
    const { symbol, id } = deleting;

    try {
      const res = await fetch(`/api/strategies/${symbol}_strategy_${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to delete");

      setStrategies((prev) => prev.filter((s) => s.id !== id));
      setDeleting(null);
    } catch (err) {
      alert("Delete failed: " + err.message);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gray-900 rounded-xl shadow border border-gray-700 text-white">
      <h2 className="text-2xl font-bold mb-6">Strategy Performance</h2>

      {loading ? (
        <p className="text-gray-400">Loading strategies...</p>
      ) : error ? (
        <p className="text-red-500">{error}</p>
      ) : strategies.length === 0 ? (
        <p className="text-gray-400">No strategies found.</p>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border border-gray-700 rounded">
              <thead className="bg-gray-800 text-left">
                <tr>
                  {[
                    "Agent",
                    "Strategy ID",
                    "Win Rate",
                    "Profit",
                    "Avg Profit",
                    "Confidence",
                    "Trades",
                    "Rules",
                    "Action",
                  ].map((h) => (
                    <th key={h} className="px-4 py-2 border border-gray-700 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {strategies.map((s) => (
                  <tr key={s.id} className="hover:bg-gray-800">
                    <td className="px-4 py-2 border border-gray-700">{s.agent}</td>
                    <td className="px-4 py-2 border border-gray-700 text-gray-400">{s.id}</td>
                    <td className="px-4 py-2 border border-gray-700">
                      <span
                        className={`px-2 py-1 text-xs font-bold rounded-full ${
                          s.winRate >= 70
                            ? "bg-green-800 text-green-300"
                            : s.winRate >= 50
                            ? "bg-yellow-800 text-yellow-300"
                            : "bg-red-800 text-red-300"
                        }`}
                      >
                        {s.winRate.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-4 py-2 border border-gray-700">
                      <span className={s.totalProfit >= 0 ? "text-green-400" : "text-red-400"}>
                        ${s.totalProfit.toFixed(2)}
                      </span>
                    </td>
                    <td className="px-4 py-2 border border-gray-700 text-sm">
                      <span className={s.avgProfit >= 0 ? "text-green-400" : "text-red-400"}>
                        ${s.avgProfit.toFixed(2)}
                      </span>
                    </td>
                    <td className="px-4 py-2 border border-gray-700 text-blue-400">
                      {s.avgConfidence.toFixed(1)}%
                    </td>
                    <td className="px-4 py-2 border border-gray-700 text-sm">
                      <div className="flex flex-col">
                        <span>Total: {s.totalPredictions}</span>
                        <span className="text-green-400">W: {s.wins}</span>
                        <span className="text-red-400">L: {s.losses}</span>
                      </div>
                    </td>
                    <td className="px-4 py-2 border border-gray-700 text-xs max-w-sm">
                      <div className="max-h-24 overflow-y-auto space-y-1">
                        {s.rules.length > 0 ? (
                          s.rules.map((r, i) => (
                            <div key={i}>
                              <span className="text-gray-200">
                                IF {r.indicator} {r.condition} {r.value} →{" "}
                                <span className="font-semibold text-green-400">{r.action}</span>
                              </span>
                            </div>
                          ))
                        ) : (
                          <span className="text-gray-500 italic">No rules</span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-2 border border-gray-700 text-center">
                      <button
                        onClick={() => setDeleting({ symbol: s.symbol, id: s.id })}
                        className="text-red-400 hover:text-red-600 transition"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex justify-between items-center mt-6">
            <button
              onClick={() => setPage((p) => Math.max(p - 1, 1))}
              disabled={page === 1}
              className={`px-4 py-2 rounded ${
                page === 1
                  ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700 text-white"
              }`}
            >
              Previous
            </button>
            <div className="text-sm text-gray-300">
              Page {page} of {totalPages}
            </div>
            <button
              onClick={() => setPage((p) => Math.min(p + 1, totalPages))}
              disabled={page === totalPages}
              className={`px-4 py-2 rounded ${
                page === totalPages
                  ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700 text-white"
              }`}
            >
              Next
            </button>
          </div>
        </>
      )}

      {/* Delete Confirmation Modal */}
      {deleting && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg text-white w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Confirm Delete</h3>
            <p className="text-sm mb-6">
              Are you sure you want to delete strategy <span className="font-bold">{deleting.id}</span> for{" "}
              <span className="text-blue-400">{deleting.symbol}</span>?
            </p>
            <div className="flex justify-end gap-3">
              <button
                className="px-4 py-2 rounded bg-gray-700 hover:bg-gray-600"
                onClick={() => setDeleting(null)}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 rounded bg-red-600 hover:bg-red-700"
                onClick={handleDelete}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
