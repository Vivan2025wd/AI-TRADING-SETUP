import React, { useState } from "react";

export default function StrategyBuilder() {
  const [strategyName, setStrategyName] = useState("");
  const [symbol, setSymbol] = useState("");
  const [buyCondition, setBuyCondition] = useState("");
  const [sellCondition, setSellCondition] = useState("");

  // This function generates the JSON internally when you need it
  const getStrategyJson = () => ({
    strategyName,
    symbol,
    conditions: {
      buy: buyCondition,
      sell: sellCondition,
    },
  });

  // Example usage: you could call getStrategyJson() when user clicks "Save" or "Submit"
  const handleSave = () => {
    const jsonToSend = getStrategyJson();
    console.log("Strategy JSON (hidden from user):", jsonToSend);
    // You can send this jsonToSend to backend here
  };

  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Strategy Builder</h2>

      <div className="space-y-3">
        <div>
          <label className="block font-semibold mb-1">Strategy Name</label>
          <input
            type="text"
            value={strategyName}
            onChange={(e) => setStrategyName(e.target.value)}
            placeholder="My Strategy"
            className="w-full border rounded px-2 py-1"
          />
        </div>

        <div>
          <label className="block font-semibold mb-1">Symbol</label>
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            placeholder="e.g. BTCUSDT"
            className="w-full border rounded px-2 py-1"
          />
        </div>

        <div>
          <label className="block font-semibold mb-1">Buy Condition</label>
          <input
            type="text"
            value={buyCondition}
            onChange={(e) => setBuyCondition(e.target.value)}
            placeholder="e.g. RSI < 30"
            className="w-full border rounded px-2 py-1"
          />
        </div>

        <div>
          <label className="block font-semibold mb-1">Sell Condition</label>
          <input
            type="text"
            value={sellCondition}
            onChange={(e) => setSellCondition(e.target.value)}
            placeholder="e.g. RSI > 70"
            className="w-full border rounded px-2 py-1"
          />
        </div>
      </div>

      <button
        onClick={handleSave}
        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded"
      >
        Save Strategy
      </button>
    </div>
  );
}
