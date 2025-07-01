import React, { useState } from "react";
import { PlusCircle, Save, Trash2 } from "lucide-react";

const indicators = ["RSI", "MACD", "SMA", "EMA"];
const conditions = ["<", "<=", ">", ">=", "=="];
const actions = ["BUY", "SELL", "HOLD"];

// Mock agents list - replace with real data if available
const agents = ["BTC Agent", "ETH Agent", "SOL Agent"];

export default function StrategyBuilder() {
  const [selectedAgent, setSelectedAgent] = useState(agents[0]);
  const [rules, setRules] = useState([
    { indicator: "RSI", condition: "<", value: "30", action: "BUY" },
  ]);

  const addRule = () => {
    setRules([
      ...rules,
      { indicator: "RSI", condition: "<", value: "30", action: "BUY" },
    ]);
  };

  const updateRule = (index, field, value) => {
    const newRules = [...rules];
    newRules[index][field] = value;
    setRules(newRules);
  };

  const removeRule = (index) => {
    const newRules = rules.filter((_, i) => i !== index);
    setRules(newRules);
  };

  const convertToJSON = () => {
    const strategyJSON = {
      agent: selectedAgent,
      strategy: rules.map((rule) => ({
        if: {
          indicator: rule.indicator,
          condition: rule.condition,
          value: parseFloat(rule.value),
        },
        then: rule.action,
      })),
    };
    console.log("JSON sent to backend (mocked):", strategyJSON);
    alert(`Strategy saved for agent "${selectedAgent}" (mocked)!`);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-gray-900 rounded-xl shadow-lg space-y-6 border border-gray-700 text-white">
      <h2 className="text-2xl font-bold flex items-center gap-2">
        Strategy Builder
      </h2>

      {/* Agent selection */}
      <div>
        <label className="block text-sm text-gray-300 font-medium mb-1">
          Assign Strategy to Agent
        </label>
        <select
          value={selectedAgent}
          onChange={(e) => setSelectedAgent(e.target.value)}
          className="border border-gray-600 rounded bg-gray-700 text-white px-3 py-2 w-full max-w-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {agents.map((agent) => (
            <option key={agent} className="bg-gray-700 text-white">
              {agent}
            </option>
          ))}
        </select>
      </div>

      {/* Rules */}
      {rules.map((rule, index) => (
        <div
          key={index}
          className="flex flex-wrap items-center gap-3 p-4 border rounded-md bg-gray-800 shadow-sm border-gray-700"
        >
          <div>
            <label className="block text-sm text-gray-300 font-medium">
              Indicator
            </label>
            <select
              value={rule.indicator}
              onChange={(e) => updateRule(index, "indicator", e.target.value)}
              className="border border-gray-600 rounded bg-gray-700 text-white px-2 py-1 mt-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {indicators.map((ind) => (
                <option key={ind} className="bg-gray-700 text-white">
                  {ind}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-300 font-medium">
              Condition
            </label>
            <select
              value={rule.condition}
              onChange={(e) => updateRule(index, "condition", e.target.value)}
              className="border border-gray-600 rounded bg-gray-700 text-white px-2 py-1 mt-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {conditions.map((c) => (
                <option key={c} className="bg-gray-700 text-white">
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-300 font-medium">
              Value
            </label>
            <input
              type="number"
              value={rule.value}
              onChange={(e) => updateRule(index, "value", e.target.value)}
              className="border border-gray-600 rounded bg-gray-700 text-white px-2 py-1 w-24 mt-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-300 font-medium">
              Action
            </label>
            <select
              value={rule.action}
              onChange={(e) => updateRule(index, "action", e.target.value)}
              className={`border border-gray-600 rounded px-2 py-1 mt-1 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                rule.action === "BUY"
                  ? "bg-green-700 text-green-300"
                  : rule.action === "SELL"
                  ? "bg-red-700 text-red-300"
                  : "bg-yellow-700 text-yellow-300"
              }`}
            >
              {actions.map((a) => (
                <option key={a} className="bg-gray-700 text-white">
                  {a}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={() => removeRule(index)}
            className="ml-auto text-red-500 hover:text-red-700"
            title="Remove this rule"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      ))}

      <div className="flex gap-4">
        <button
          onClick={addRule}
          className="flex items-center gap-2 bg-blue-700 text-white px-4 py-2 rounded hover:bg-blue-800 transition"
        >
          <PlusCircle className="w-5 h-5" />
          Add Rule
        </button>
        <button
          onClick={convertToJSON}
          className="flex items-center gap-2 bg-green-700 text-white px-4 py-2 rounded hover:bg-green-800 transition"
        >
          <Save className="w-5 h-5" />
          Save Strategy
        </button>
      </div>
    </div>
  );
}
