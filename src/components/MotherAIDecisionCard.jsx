import React, { useEffect, useState } from "react";
import { ArrowUpRight, AlertCircle, Clock } from "lucide-react";

export default function MotherAIDecisionCard() {
  const [decisionData, setDecisionData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchDecision() {
      try {
        setError(null);
        const res = await fetch("http://localhost:8000/api/mother-ai/decision"); // or use /api if proxy set

        if (!res.ok) throw new Error("Failed to fetch Mother AI decision");

        const data = await res.json();

        if (!data || !data.decision || Object.keys(data.decision).length === 0) {
          // No valid decision data, set to inactive
          setDecisionData({
            status: "Inactive",
            tradePick: "No signal",
            lastUpdated: new Date().toLocaleString(),
            rationale: "No qualified trades met the confidence threshold.",
            confidence: 0,
          });
        } else {
          const decision = data.decision;
          setDecisionData({
            status: "Active",
            tradePick: `${decision.symbol} - ${decision.signal}`,
            lastUpdated: new Date(data.timestamp).toLocaleString(),
            rationale: `Confidence: ${(decision.confidence * 100).toFixed(2)}%, Win Rate: ${(decision.win_rate * 100).toFixed(2)}%, Score: ${decision.score}`,
            confidence: (decision.confidence * 100).toFixed(2),
          });
        }
      } catch (err) {
        console.error("Mother AI Fetch Error:", err);
        setError(err.message || "An unexpected error occurred.");
      } finally {
        setLoading(false);
      }
    }

    fetchDecision();
  }, []);

  if (loading) {
    return (
      <div className="text-gray-300 text-center p-4">
        Loading Mother AI decision...
      </div>
    );
  }

  return (
    <div className="bg-gray-900 text-white shadow-md rounded-xl p-6 max-w-xl mx-auto space-y-6 border border-gray-700">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold flex items-center gap-2 text-white">
          <ArrowUpRight className="text-blue-400 w-5 h-5" /> Mother AI Signal
        </h2>
        <span
          className={`px-3 py-1 text-sm font-semibold rounded-full ${
            decisionData?.status === "Active"
              ? "bg-green-700 text-green-100"
              : "bg-gray-600 text-gray-200"
          }`}
        >
          {decisionData?.status || "Unknown"}
        </span>
      </div>

      {error ? (
        <div className="bg-red-900 text-white p-4 rounded-lg text-center">
          <p className="font-bold">Error:</p>
          <p>{error}</p>
        </div>
      ) : (
        <>
          <div className="space-y-1">
            <p className="text-lg font-semibold text-blue-400">
              {decisionData?.tradePick}
            </p>
            <p className="text-sm text-gray-400 flex items-center gap-1">
              <Clock className="w-4 h-4" /> Last Updated: {decisionData?.lastUpdated}
            </p>
          </div>

          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 text-sm text-gray-300">
            <p className="mb-2 font-semibold text-white">Rationale:</p>
            <p className="italic flex gap-2 items-start">
              <AlertCircle className="w-4 h-4 text-yellow-400 mt-1" />
              {decisionData?.rationale}
            </p>
          </div>

          <div className="pt-2">
            <p className="text-sm text-gray-300">
              <span className="font-semibold text-white">Confidence:</span>{" "}
              <span className="text-green-400 font-semibold">
                {decisionData?.confidence}%
              </span>
            </p>
          </div>
        </>
      )}
    </div>
  );
}
