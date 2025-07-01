// src/components/AgentPredictionCard.jsx
import React from "react";

const mockAgentData = {
  agentName: "GenericAgent",
  prediction: "Buy",
  confidence: 78, // in %
  tradeDetails: {
    symbol: "BTCUSDT",
    entryPrice: 30000,
    targetPrice: 33000,
    stopLoss: 29000,
  },
};

const mockMotherAIDecision = {
  decision: "Hold",
  rationale: "Waiting for confirmation from multiple agents.",
};

export default function AgentPredictionCard() {
  return (
    <div className="max-w-md mx-auto p-4 border rounded shadow bg-white space-y-4">
      <h3 className="text-xl font-semibold">Agent Prediction</h3>
      <div className="text-lg">
        <span className="font-bold">Agent:</span> {mockAgentData.agentName}
      </div>
      <div className="text-lg">
        <span className="font-bold">Prediction:</span>{" "}
        <span
          className={
            mockAgentData.prediction === "Buy"
              ? "text-green-600"
              : mockAgentData.prediction === "Sell"
              ? "text-red-600"
              : "text-yellow-600"
          }
        >
          {mockAgentData.prediction}
        </span>
      </div>
      <div>
        <span className="font-bold">Confidence:</span> {mockAgentData.confidence}%
      </div>

      <div className="border-t pt-3">
        <h4 className="font-semibold mb-2">Trade Details</h4>
        <ul className="list-disc list-inside text-sm">
          <li>
            <strong>Symbol:</strong> {mockAgentData.tradeDetails.symbol}
          </li>
          <li>
            <strong>Entry Price:</strong> ${mockAgentData.tradeDetails.entryPrice.toLocaleString()}
          </li>
          <li>
            <strong>Target Price:</strong> ${mockAgentData.tradeDetails.targetPrice.toLocaleString()}
          </li>
          <li>
            <strong>Stop Loss:</strong> ${mockAgentData.tradeDetails.stopLoss.toLocaleString()}
          </li>
        </ul>
      </div>

      <div className="border-t pt-3">
        <h3 className="text-xl font-semibold">Mother AI Decision</h3>
        <p className="italic">{mockMotherAIDecision.rationale}</p>
        <p>
          <strong>Decision:</strong>{" "}
          <span
            className={
              mockMotherAIDecision.decision === "Buy"
                ? "text-green-600"
                : mockMotherAIDecision.decision === "Sell"
                ? "text-red-600"
                : "text-yellow-600"
            }
          >
            {mockMotherAIDecision.decision}
          </span>
        </p>
      </div>
    </div>
  );
}
