import React from "react";

const mockAgentData = [
  {
    agentName: "BTC Agent",
    prediction: "Buy",
    confidence: 78,
    tradeDetails: {
      symbol: "BTCUSDT",
      entryPrice: 30100,
      targetPrice: 32500,
      stopLoss: 29200,
    },
  },
  {
    agentName: "ETH Agent",
    prediction: "Hold",
    confidence: 65,
    tradeDetails: {
      symbol: "ETHUSDT",
      entryPrice: 1980,
      targetPrice: 2100,
      stopLoss: 1900,
    },
  },
  {
    agentName: "Generic Agent",
    prediction: "Sell",
    confidence: 81,
    tradeDetails: {
      symbol: "SOLUSDT",
      entryPrice: 88,
      targetPrice: 77,
      stopLoss: 92,
    },
  },
];

const getEmoji = (prediction) => {
  if (prediction === "Buy") return "🟢";
  if (prediction === "Sell") return "🔴";
  return "🟡";
};

export default function AgentPredictionCard() {
  return (
    <div className="space-y-6 max-w-md mx-auto">
      <h3 className="text-2xl font-bold text-white border-b border-gray-600 pb-2 mb-4">
        Agent Predictions
      </h3>

      {mockAgentData.map((agent, index) => (
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
