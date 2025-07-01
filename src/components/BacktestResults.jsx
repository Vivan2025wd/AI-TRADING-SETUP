// src/components/BacktestResults.jsx
import React from "react";

const mockBacktestResults = {
  finalBalance: 1250.75,
  totalTrades: 12,
  winningTrades: 9,
  losingTrades: 3,
  trades: [
    { id: 1, type: "BUY", price: 100, timestamp: "2025-06-20T10:00:00Z" },
    { id: 2, type: "SELL", price: 110, timestamp: "2025-06-21T14:00:00Z", profitPercent: 10 },
    { id: 3, type: "BUY", price: 108, timestamp: "2025-06-22T09:30:00Z" },
    { id: 4, type: "SELL", price: 115, timestamp: "2025-06-23T11:00:00Z", profitPercent: 6.5 },
    // ...more trades
  ],
};

export default function BacktestResults() {
  const winRate = ((mockBacktestResults.winningTrades / mockBacktestResults.totalTrades) * 100).toFixed(1);

  return (
    <div className="max-w-3xl mx-auto p-4 border rounded shadow bg-white">
      <h3 className="text-lg font-semibold mb-4">Backtest Results</h3>
      
      <div className="mb-4 grid grid-cols-3 gap-4 text-center">
        <div>
          <p className="font-bold text-xl">${mockBacktestResults.finalBalance.toFixed(2)}</p>
          <p>Final Balance</p>
        </div>
        <div>
          <p className="font-bold text-xl">{mockBacktestResults.totalTrades}</p>
          <p>Total Trades</p>
        </div>
        <div>
          <p className="font-bold text-xl">{winRate}%</p>
          <p>Win Rate</p>
        </div>
      </div>

      <table className="w-full border-collapse text-left">
        <thead>
          <tr className="border-b">
            <th className="py-2 px-3">#</th>
            <th className="py-2 px-3">Type</th>
            <th className="py-2 px-3">Price</th>
            <th className="py-2 px-3">Timestamp</th>
            <th className="py-2 px-3">Profit %</th>
          </tr>
        </thead>
        <tbody>
          {mockBacktestResults.trades.map(({ id, type, price, timestamp, profitPercent }) => (
            <tr key={id} className="border-b hover:bg-gray-100">
              <td className="py-2 px-3">{id}</td>
              <td className="py-2 px-3">{type}</td>
              <td className="py-2 px-3">${price.toFixed(2)}</td>
              <td className="py-2 px-3">{new Date(timestamp).toLocaleString()}</td>
              <td className={`py-2 px-3 ${profitPercent && profitPercent > 0 ? "text-green-600" : "text-red-600"}`}>
                {profitPercent ? `${profitPercent}%` : "-"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
