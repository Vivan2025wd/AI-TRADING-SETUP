// src/components/MotherAIDecisionCard.jsx
import React from "react";

const mockMotherAIDecision = {
  tradePick: "Buy BTCUSDT",
  confidence: 85, // %
  rationale:
    "Consolidated signals from multiple agents indicate strong upward momentum. Risk factors minimal.",
  status: "Active",
  lastUpdated: "2025-07-01 12:00:00",
};

export default function MotherAIDecisionCard() {
  return (
    <div className="max-w-md mx-auto p-4 border rounded shadow bg-white space-y-4">
      <h2 className="text-2xl font-bold">Mother AI Trade Pick</h2>
      <p className="text-lg font-semibold">{mockMotherAIDecision.tradePick}</p>
      <p>
        <span className="font-bold">Confidence:</span>{" "}
        <span className="text-green-600">{mockMotherAIDecision.confidence}%</span>
      </p>
      <p className="italic">{mockMotherAIDecision.rationale}</p>
      <p>
        <span className="font-bold">Status:</span>{" "}
        <span
          className={
            mockMotherAIDecision.status === "Active"
              ? "text-green-600"
              : "text-gray-600"
          }
        >
          {mockMotherAIDecision.status}
        </span>
      </p>
      <p className="text-sm text-gray-500">
        Last updated: {mockMotherAIDecision.lastUpdated}
      </p>
    </div>
  );
}
