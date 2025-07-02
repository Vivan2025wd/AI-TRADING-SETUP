import React, { useEffect, useState } from "react";

const getEmoji = (prediction) => {
  if (prediction === "Buy") return "🟢";
  if (prediction === "Sell") return "🔴";
  return "🟡";
};

export default function AgentPredictionCard() {
  const [agentData, setAgentData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchAgentData() {
      try {
        const res = await fetch("/api/agent-predictions");
        if (!res.ok) throw new Error("Failed to fetch agent predictions");
        const data = await res.json();
        setAgentData(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchAgentData();
  }, []);

  if (loading) return <p>Loading agent predictions...</p>;
  if (error) return <p className="text-red-500">Error: {error}</p>;

  return (
    <div className="space-y-6 max-w-md mx-auto">
      <h3 className="text-2xl font-bold text-white border-b border-gray-600 pb-2 mb-4">
        Agent Predictions
      </h3>

      {agentData.map((agent, index) => (
        <div
          key={index}
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
                <strong>Entry:</strong> ${agent.tradeDetails.entryPrice.toLocaleString()}
              </li>
              <li>
                <strong>Target:</strong> ${agent.tradeDetails.targetPrice.toLocaleString()}
              </li>
              <li>
                <strong>Stop Loss:</strong> ${agent.tradeDetails.stopLoss.toLocaleString()}
              </li>
            </ul>
          </div>
        </div>
      ))}
    </div>
  );
}
