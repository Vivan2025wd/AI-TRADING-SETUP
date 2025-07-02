import React, { useEffect, useState } from "react";
import ChartDisplay from "../components/ChartDisplay";
import AgentPredictionCard from "../components/AgentPredictionCard";
import MotherAIDecisionCard from "../components/MotherAIDecisionCard";

export default function Dashboard() {
  const [trades, setTrades] = useState([]);
  const [balanceHistory, setBalanceHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchTrades() {
      try {
        const res = await fetch("/api/mother-ai/trades");
        if (!res.ok) throw new Error("Failed to fetch trades");
        const data = await res.json();

        setTrades(data);

        const balances = data.map((t) => ({
          time: t.timestamp,
          value: t.balance,
        }));
        setBalanceHistory(balances);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    fetchTrades();
  }, []);

  return (
    <div className="space-y-6 bg-gray-900 min-h-screen p-6 text-white">
      <h1 className="text-3xl font-bold mb-6">Trading Dashboard</h1>

      {loading && <p>Loading trade data...</p>}
      {error && <p className="text-red-500">Error: {error}</p>}

      {!loading && !error && (
        <>
          <ChartDisplay balanceHistory={balanceHistory} />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <AgentPredictionCard />
            <MotherAIDecisionCard />
          </div>

          <div className="bg-gray-800 rounded-xl shadow-lg p-6 mt-8 overflow-x-auto border border-gray-700">
            <h2 className="text-2xl font-semibold mb-4"> Trade History</h2>
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left border-b border-gray-700">
                  <th className="py-3 px-4">Time</th>
                  <th className="py-3 px-4">Type</th>
                  <th className="py-3 px-4">Agent</th>
                  <th className="py-3 px-4">Price</th>
                  <th className="py-3 px-4">Balance After</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((trade, i) => (
                  <tr
                    key={i}
                    className="border-b border-gray-700 hover:bg-gray-700 transition-colors"
                  >
                    <td className="py-2 px-4">{trade.timestamp}</td>
                    <td className="px-4 font-semibold text-blue-400">{trade.type}</td>
                    <td className="px-4">{trade.agent}</td>
                    <td className="px-4">${trade.price.toLocaleString()}</td>
                    <td className="px-4">${trade.balance.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
