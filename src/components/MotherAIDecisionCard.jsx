import React from "react";
import { ArrowUpRight, AlertCircle, Clock } from "lucide-react";

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
    <div className="bg-gray-900 text-white shadow-md rounded-xl p-6 max-w-xl mx-auto space-y-6 border border-gray-700">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold flex items-center gap-2 text-white">
          <ArrowUpRight className="text-blue-400 w-5 h-5" /> Mother AI Signal
        </h2>
        <span
          className={`px-3 py-1 text-sm font-semibold rounded-full ${
            mockMotherAIDecision.status === "Active"
              ? "bg-green-700 text-green-100"
              : "bg-gray-600 text-gray-200"
          }`}
        >
          {mockMotherAIDecision.status}
        </span>
      </div>

      <div className="space-y-1">
        <p className="text-lg font-semibold text-blue-400">
          {mockMotherAIDecision.tradePick}
        </p>
        <p className="text-sm text-gray-400 flex items-center gap-1">
          <Clock className="w-4 h-4" /> Last Updated: {mockMotherAIDecision.lastUpdated}
        </p>
      </div>

      <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 text-sm text-gray-300">
        <p className="mb-2 font-semibold text-white">Rationale:</p>
        <p className="italic flex gap-2 items-start">
          <AlertCircle className="w-4 h-4 text-yellow-400 mt-1" />
          {mockMotherAIDecision.rationale}
        </p>
      </div>

      <div className="pt-2">
        <p className="text-sm text-gray-300">
          <span className="font-semibold text-white">Confidence:</span>{" "}
          <span className="text-green-400 font-semibold">
            {mockMotherAIDecision.confidence}%
          </span>
        </p>
      </div>
    </div>
  );
}
