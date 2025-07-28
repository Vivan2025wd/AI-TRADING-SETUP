import React, { useEffect, useState, useCallback, useRef } from "react";
import { ArrowUpRight, AlertCircle, Clock, RefreshCw, Loader2 } from "lucide-react";

const CACHE_KEY = "mother_ai_decision_cache";
const CACHE_DURATION_MS = 2 * 60 * 60 * 1000; // 2 hours
const POLL_INTERVAL_MS = 15 * 60 * 1000; // 15 minutes

export default function MotherAIDecisionCard({ isLive }) {
  const [decisionData, setDecisionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);

  const fetchDecision = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const url = `http://localhost:8000/api/mother-ai/decision?is_live=${isLive}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to fetch Mother AI decision");

      const data = await res.json();

      let final = {};

      if (!data || typeof data !== "object") {
        final = {
          status: "Unknown",
          tradePick: "No signal",
          lastUpdated: null,
          rationale: "Invalid or empty response from Mother AI backend.",
          confidence: 0,
        };
      } else if (!data.decision || Object.keys(data.decision).length === 0) {
        final = {
          status: "Inactive",
          tradePick: "No signal",
          lastUpdated: data.timestamp ? new Date(data.timestamp).toLocaleString() : new Date().toLocaleString(),
          rationale: data.message || "No qualified trades met the confidence threshold.",
          confidence: 0,
        };
      } else {
        const d = data.decision;
        final = {
          status: "Active",
          tradePick: `${d.symbol} - ${d.signal?.toUpperCase() || "N/A"}`,
          lastUpdated: data.timestamp ? new Date(data.timestamp).toLocaleString() : new Date().toLocaleString(),
          rationale: `Confidence: ${(d.confidence * 100).toFixed(2)}%, Win Rate: ${(d.win_rate * 100).toFixed(2)}%, Score: ${d.score}`,
          confidence: (d.confidence * 100).toFixed(2),
        };
      }

      setDecisionData(final);

      // Cache the result in memory (localStorage not available in artifacts)
      console.log("Decision updated:", final);
    } catch (err) {
      console.error("Mother AI Fetch Error:", err);
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  }, [isLive]);

  useEffect(() => {
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    // Initial fetch on startup
    fetchDecision();

    // Set up interval to fetch every 10 minutes
    intervalRef.current = setInterval(() => {
      console.log("Polling Mother AI decision...");
      fetchDecision();
    }, POLL_INTERVAL_MS);

    // Cleanup function
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [fetchDecision]); // Depend on fetchDecision which includes isLive

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
              : decisionData?.status === "Inactive"
              ? "bg-gray-600 text-gray-200"
              : decisionData?.status === "Waiting"
              ? "bg-yellow-600 text-yellow-100"
              : "bg-red-700 text-red-200"
          }`}
        >
          {loading ? (
            <span className="flex items-center gap-1">
              <Loader2 className="animate-spin w-4 h-4" /> Loading...
            </span>
          ) : error ? (
            "Error"
          ) : (
            decisionData?.status || "Unknown"
          )}
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
            <p className="text-lg font-semibold text-blue-400">{decisionData?.tradePick}</p>
            <p className="text-sm text-gray-400 flex items-center gap-1">
              <Clock className="w-4 h-4" /> Last Updated: {decisionData?.lastUpdated || "N/A"}
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
              <span className="text-green-400 font-semibold">{decisionData?.confidence}%</span>
            </p>
          </div>

          {!loading && decisionData?.status === "Active" && (
            <p className="text-sm text-gray-400 italic mt-2 text-center">
              Next update in {Math.ceil(POLL_INTERVAL_MS / 60000)} minutes...
            </p>
          )}
        </>
      )}

      <button
        onClick={fetchDecision}
        className="mt-4 px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 flex items-center gap-2 mx-auto disabled:opacity-50"
        disabled={loading}
      >
        <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        Refresh Now
      </button>
    </div>
  );
}