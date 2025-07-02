import React, { useEffect, useState } from "react";
import { ArrowUpRight, AlertCircle, Clock } from "lucide-react";

export default function MotherAIDecisionCard() {
  const [decision, setDecision] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchDecision() {
      try {
        const res = await fetch("/api/mother-ai/decision");
        if (!res.ok) throw new Error("Failed to fetch Mother AI decision");
        const data = await res.json();
        setDecision(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchDecision();
  }, []);

  if (loading) return <p>Loading Mother AI decision...</p>;
  if (error) return <p className="text-red-500">Error: {error}</p>;
  if (!decision)
    return <p className="text-gray-400">No active Mother AI decision available.</p>;

  return (
    <div className="bg-gray-900 text-white shadow-md rounded-xl p-6 max-w-xl mx-auto space-y-6 border border-gray-700">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold flex items-center gap-2 text-white">
          <ArrowUpRight className="text-blue-400 w-5 h-5" /> Mother AI Signal
        </h2>
        <span
          className={`px-3 py-1 text-sm font-semibold rounded-full ${
            decision.status === "Active"
              ? "bg-green-700 text-green-100"
              : "bg-gray-600 text-gray-200"
          }`}
        >
          {decision.status}
        </span>
      </div>

      <div className="space-y-1">
        <p className="text-lg font-semibold text-blue-400">{decision.tradePick}</p>
        <p className="text-sm text-gray-400 flex items-center gap-1">
          <Clock className="w-4 h-4" /> Last Updated: {decision.lastUpdated}
        </p>
      </div>

      <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 text-sm text-gray-300">
        <p className="mb-2 font-semibold text-white">Rationale:</p>
        <p className="italic flex gap-2 items-start">
          <AlertCircle className="w-4 h-4 text-yellow-400 mt-1" />
          {decision.rationale}
        </p>
      </div>

      <div className="pt-2">
        <p className="text-sm text-gray-300">
          <span className="font-semibold text-white">Confidence:</span>{" "}
          <span className="text-green-400 font-semibold">{decision.confidence}%</span>
        </p>
      </div>
    </div>
  );
}
