import React, { useState, useEffect } from "react";

const getEmoji = (prediction) => {
  const p = prediction.toLowerCase();
  if (p === "buy") return "🟢";
  if (p === "sell") return "🔴";
  return "🟡"; // hold or others
};

const fetchWithTimeout = (url, options = {}, timeout = 10000) =>
  Promise.race([
    fetch(url, options),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Request timed out")), timeout)
    ),
  ]);

export default function AgentPredictionCard() {
  const [allAgents, setAllAgents] = useState([]); // merged agents + predictions
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      setAllAgents([]);

      try {
        // 1. Fetch all agents - returns array like ["ADAUSDT", "AVAXUSDT", ...]
        const resAgents = await fetchWithTimeout("/api/agents");
        if (!resAgents.ok) {
          throw new Error(`Agents fetch failed (${resAgents.status})`);
        }
        const agentsList = await resAgents.json();

        // 2. Fetch predictions
        const resPred = await fetchWithTimeout(
          "/api/agent/predictions?page=1&limit=100"
        );
        if (!resPred.ok) {
          throw new Error(`Predictions fetch failed (${resPred.status})`);
        }
        const predData = await resPred.json();
        const predictions = Array.isArray(predData.data) ? predData.data : [];

        // 3. Map predictions by normalized agentName (remove " Agent" suffix, uppercase)
        const predMap = {};
        for (const pred of predictions) {
          if (pred.agentName) {
            // Normalize: remove trailing " Agent" and uppercase
            const normalizedName = pred.agentName.replace(/\s+Agent$/i, "").toUpperCase();
            predMap[normalizedName] = pred;
          }
        }

        // 4. Merge: for each agent symbol, add prediction or default values
        const merged = agentsList.map((agentSymbol) => {
          const key = agentSymbol.toUpperCase();
          const pred = predMap[key];

          const predictionText =
            pred && typeof pred.prediction === "string"
              ? pred.prediction.charAt(0).toUpperCase() +
                pred.prediction.slice(1).toLowerCase()
              : "Hold";

          return {
            agentName: agentSymbol,
            prediction: predictionText,
            confidence:
              pred && typeof pred.confidence === "number"
                ? pred.confidence.toFixed(2)
                : "0.00",
            tradeDetails: pred
              ? pred.tradeDetails || {
                  symbol: "N/A",
                  entryPrice: 0,
                  targetPrice: 0,
                  stopLoss: 0,
                }
              : {
                  symbol: "N/A",
                  entryPrice: 0,
                  targetPrice: 0,
                  stopLoss: 0,
                },
          };
        });

        setAllAgents(merged);
      } catch (err) {
        setError(err.message || "Unknown error occurred");
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [retry]);

  if (loading)
    return <p className="text-white">⏳ Loading agent predictions...</p>;

  if (error)
    return (
      <div className="text-red-400 p-4 bg-gray-800 border border-red-500 rounded-md text-sm max-w-md mx-auto">
        <p>⚠️ Error: {error}</p>
        <button
          onClick={() => setRetry((r) => r + 1)}
          className="mt-3 bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-white"
        >
          Retry
        </button>
      </div>
    );

  return (
    <div className="space-y-6 max-w-md mx-auto">
      <h3 className="text-2xl font-bold text-white border-b border-gray-600 pb-2 mb-4">
        Agent Predictions ({allAgents.length} agents)
      </h3>

      {allAgents.length === 0 && (
        <p className="text-gray-300">No agents available.</p>
      )}

      {allAgents.map((agent) => (
        <div
          key={agent.agentName}
          className="bg-gray-900 border border-gray-700 shadow-md rounded-xl p-6 transition-shadow duration-300 hover:shadow-lg"
        >
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-lg font-semibold text-white">{agent.agentName}</h4>
            <span
              className={`font-semibold text-sm flex items-center space-x-1 ${
                agent.prediction === "Buy"
                  ? "text-green-400"
                  : agent.prediction === "Sell"
                  ? "text-red-400"
                  : "text-yellow-400"
              }`}
            >
              <span aria-label={agent.prediction} role="img">
                {getEmoji(agent.prediction)}
              </span>
              <span>{agent.prediction}</span>
            </span>
          </div>

          <div className="text-sm text-gray-300 mb-4">
            <strong>Confidence:</strong> {agent.confidence}%
          </div>

          <div className="pt-3 border-t border-gray-700 text-sm text-gray-300">
            <strong className="block mb-2 text-white">Trade Details</strong>
            <ul className="list-disc list-inside space-y-1">
              <li>
                <strong>Symbol:</strong> {agent.tradeDetails.symbol}
              </li>
              <li>
                <strong>Entry:</strong> $
                {Number(agent.tradeDetails.entryPrice).toLocaleString()}
              </li>
              <li>
                <strong>Target:</strong> $
                {Number(agent.tradeDetails.targetPrice).toLocaleString()}
              </li>
              <li>
                <strong>Stop Loss:</strong> $
                {Number(agent.tradeDetails.stopLoss).toLocaleString()}
              </li>
            </ul>
          </div>
        </div>
      ))}
    </div>
  );
}
