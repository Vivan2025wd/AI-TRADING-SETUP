import React, { useEffect, useState } from "react";
import { ArrowUpRight, AlertCircle, Clock, RefreshCw, Loader2 } from "lucide-react";

const CACHE_KEY = "mother_ai_decision_cache";
const CACHE_DURATION_MS = 2 * 60 * 60 * 1000; // 2 hours

export default function MotherAIDecisionCard() {
  const [decisionData, setDecisionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function fetchDecision() {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("http://localhost:8000/api/mother-ai/latest-decision"); // or your endpoint
      if (!res.ok) throw new Error("Failed to fetch Mother AI decision");

      const data = await res.json();

      // Handle backend statuses explicitly
      if (data.status === "waiting") {
        setDecisionData({
          status: "Waiting",
          tradePick: "No signal yet",
          lastUpdated: null,
          rationale: data.message || "Mother AI is still evaluating. Please wait...",
          confidence: 0,
        });
      } else if (data.status === "no_signal") {
        setDecisionData({
          status: "Inactive",
          tradePick: "No signal",
          lastUpdated: data.timestamp ? new Date(data.timestamp).toLocaleString() : new Date().toLocaleString(),
          rationale: data.message || "No qualified trades met the confidence threshold.",
          confidence: 0,
        });
      } else if (data.status === "success" && data.decision) {
        const decision = data.decision;
        setDecisionData({
          status: "Active",
          tradePick: `${decision.symbol} - ${decision.signal.toUpperCase()}`,
          lastUpdated: data.timestamp ? new Date(data.timestamp).toLocaleString() : new Date().toLocaleString(),
          rationale: `Confidence: ${(decision.confidence * 100).toFixed(2)}%, Win Rate: ${(decision.win_rate * 100).toFixed(2)}%, Score: ${decision.score}`,
          confidence: (decision.confidence * 100).toFixed(2),
        });
      } else {
        // fallback for unknown or empty data
        setDecisionData({
          status: "Unknown",
          tradePick: "No signal",
          lastUpdated: null,
          rationale: "Unexpected response from Mother AI backend.",
          confidence: 0,
        });
      }

      localStorage.setItem(
        CACHE_KEY,
        JSON.stringify({
          timestamp: Date.now(),
          decisionData: data,
        })
      );
    } catch (err) {
      console.error("Mother AI Fetch Error:", err);
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const cached = localStorage.getItem(CACHE_KEY);
    if (cached) {
      try {
        const { timestamp, decisionData: cachedDecision } = JSON.parse(cached);
        const age = Date.now() - timestamp;

        if (age < CACHE_DURATION_MS) {
          // Handle cached data based on new backend status
          if (cachedDecision.status === "waiting") {
            setDecisionData({
              status: "Waiting",
              tradePick: "No signal yet",
              lastUpdated: null,
              rationale: cachedDecision.message || "Mother AI is still evaluating. Please wait...",
              confidence: 0,
            });
          } else if (cachedDecision.status === "no_signal") {
            setDecisionData({
              status: "Inactive",
              tradePick: "No signal",
              lastUpdated: cachedDecision.timestamp ? new Date(cachedDecision.timestamp).toLocaleString() : new Date().toLocaleString(),
              rationale: cachedDecision.message || "No qualified trades met the confidence threshold.",
              confidence: 0,
            });
          } else if (cachedDecision.status === "success" && cachedDecision.decision) {
            const decision = cachedDecision.decision;
            setDecisionData({
              status: "Active",
              tradePick: `${decision.symbol} - ${decision.signal.toUpperCase()}`,
              lastUpdated: cachedDecision.timestamp ? new Date(cachedDecision.timestamp).toLocaleString() : new Date().toLocaleString(),
              rationale: `Confidence: ${(decision.confidence * 100).toFixed(2)}%, Win Rate: ${(decision.win_rate * 100).toFixed(2)}%, Score: ${decision.score}`,
              confidence: (decision.confidence * 100).toFixed(2),
            });
          } else {
            setDecisionData({
              status: "Unknown",
              tradePick: "No signal",
              lastUpdated: null,
              rationale: "Unexpected cached data from Mother AI backend.",
              confidence: 0,
            });
          }

          setLoading(false);
          return;
        }
      } catch {
        // corrupted cache fallback
      }
    }

    fetchDecision();
  }, []);

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
            <p className="text-sm text-gray-400 italic mt-2 text-center">Waiting for next update...</p>
          )}
        </>
      )}

      {/* Manual refresh button */}
      <button
        onClick={() => {
          fetchDecision();
        }}
        className="mt-4 px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 flex items-center gap-2 mx-auto"
        disabled={loading}
      >
        <RefreshCw className="w-4 h-4" />
        Refresh Now
      </button>
    </div>
  );
}
