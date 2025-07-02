import React, { useEffect, useState } from "react";
import ChartDisplay from "./ChartDisplay";

export default function MotherAIBalanceChart() {
  const [balanceHistory, setBalanceHistory] = useState([]); // empty initially
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchTradeHistory() {
      try {
        const response = await fetch("/api/mother-ai/trades"); // Replace with your API
        if (!response.ok) throw new Error("Failed to fetch trade data");
        const data = await response.json();

        const formattedData = data.map(trade => ({
          time: trade.timestamp,
          value: trade.balance,
        }));

        setBalanceHistory(formattedData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    fetchTradeHistory();
  }, []);

  return (
    <div>
      {/* Always show the chart, even if no data yet */}
      <ChartDisplay balanceHistory={balanceHistory} />

      {/* Show loading or error messages below the chart */}
      {loading && <p className="text-white mt-2">Loading balance chart...</p>}
      {error && <p className="text-red-500 mt-2">Error: {error}</p>}
      {!loading && !error && balanceHistory.length === 0 && (
        <p className="text-gray-400 mt-2">No trade data available to display.</p>
      )}
    </div>
  );
}
