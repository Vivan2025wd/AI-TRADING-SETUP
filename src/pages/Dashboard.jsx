import React, { useEffect, useState } from "react";
import ChartDisplay from "../components/ChartDisplay";
import AgentPredictionCard from "../components/AgentPredictionCard";
import MotherAIDecisionCard from "../components/MotherAIDecisionCard";

const mockTrades = [
  { timestamp: "2025-07-01", type: "BUY", agent: "BTC Agent", price: 30100, balance: 1020 },
  { timestamp: "2025-07-02", type: "SELL", agent: "BTC Agent", price: 30900, balance: 1095 },
  { timestamp: "2025-07-03", type: "BUY", agent: "ETH Agent", price: 1950, balance: 1095 },
  { timestamp: "2025-07-04", type: "SELL", agent: "ETH Agent", price: 2050, balance: 1145 },
];

export default function Dashboard() {
  const [trades, setTrades] = useState([]);
  const [balanceHistory, setBalanceHistory] = useState([]);

  useEffect(() => {
    // Normally fetch from backend
    setTrades(mockTrades);

    const balances = mockTrades.map((t) => ({
      time: t.timestamp,
      value: t.balance,
    }));
    setBalanceHistory(balances);
  }, []);

  return (
    <div className="space-y-6 bg-gray-900 min-h-screen p-6 text-white">
      <h1 className="text-3xl font-bold mb-6">Trading Dashboard</h1>

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
    </div>
  );
}
