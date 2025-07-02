import React, { useState, useEffect } from "react";
import { PlusCircle, Save, Trash2 } from "lucide-react";

const indicators = ["RSI", "MACD", "SMA", "EMA"];
const conditions = ["<", "<=", ">", ">=", "=="];
const actions = ["BUY", "SELL", "HOLD"];

export default function StrategyBuilder() {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState("");
  const [strategyName, setStrategyName] = useState("");
  const [rules, setRules] = useState([
    { indicator: "RSI", condition: "<", value: "30", action: "BUY" },
  ]);
  const [loading, setLoading] = useState(false);

  // ✅ useEffect is inside component, no syntax errors
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const res = await fetch("/api/agents");
        if (!res.ok) throw new Error("Failed to fetch agents");
        const data = await res.json();
        setAgents(data);
        setSelectedAgent(data[0] || "");
      } catch (err) {
        console.error("Error fetching agents:", err.message);
        const fallback = ["BTC", "ETH", "SOL"];
        setAgents(fallback);
        setSelectedAgent(fallback[0]);
      }
    };
    fetchAgents();
  }, []);

  const addRule = () => {
    setRules([
      ...rules,
      { indicator: "RSI", condition: "<", value: "30", action: "BUY" },
    ]);
  };

  const updateRule = (index, field, value) => {
    const updated = [...rules];
    updated[index][field] = value;
    setRules(updated);
  };

  const removeRule = (index) => {
    setRules(rules.filter((_, i) => i !== index));
  };

  const convertRulesToIndicators = () => {
    const indicatorsObj = {};

    rules.forEach((rule) => {
      const key = rule.indicator.toLowerCase();
      const val = parseFloat(rule.value);
      const condition = rule.condition;
      const action = rule.action;

      if (!indicatorsObj[key]) indicatorsObj[key] = {};

      const add = (k) =>
        (indicatorsObj[key][k] = indicatorsObj[key][k] || []).push(val);

      if (key === "rsi") {
        if (action === "BUY") {
          if (["<", "<="].includes(condition)) add("buy_below");
          else if (condition === "==") add("buy_equals");
        } else if (action === "SELL") {
          if ([">", ">="].includes(condition)) add("sell_above");
          else if (condition === "==") add("sell_equals");
        }
      }

      if (["ema", "sma"].includes(key)) {
        if (action === "BUY") {
          if ([">", ">="].includes(condition)) add("buy_crosses_above");
          else if (condition === "==") add("buy_equals");
        } else if (action === "SELL") {
          if (["<", "<="].includes(condition)) add("sell_crosses_below");
          else if (condition === "==") add("sell_equals");
        }
      }

      if (key === "macd") {
        if (action === "BUY") {
          if (["<", "<="].includes(condition)) add("buy_below");
          else if (condition === "==") add("buy_equals");
        } else if (action === "SELL") {
          if ([">", ">="].includes(condition)) add("sell_above");
          else if (condition === "==") add("sell_equals");
        }
      }
    });

    return indicatorsObj;
  };

  const convertToJSON = async () => {
    if (!strategyName.trim()) return alert("Enter strategy name.");
    if (!selectedAgent) return alert("Select an agent.");
    if (rules.length === 0) return alert("Add at least one rule.");

    const payload = {
      strategy_id: strategyName.trim(),
      symbol: selectedAgent,
      strategy_json: convertRulesToIndicators(),
    };

    console.log("Submitting strategy:", payload);

    try {
      setLoading(true);
      const res = await fetch("/api/strategy/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Failed to save");
      }

      alert(`✅ Saved strategy "${strategyName}"`);
      setStrategyName("");
      setRules([
        { indicator: "RSI", condition: "<", value: "30", action: "BUY" },
      ]);
    } catch (err) {
      alert("Error saving: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-gray-900 rounded-xl shadow-lg space-y-6 border border-gray-700 text-white">
      <h2 className="text-2xl font-bold flex items-center gap-2">
        Strategy Builder
      </h2>

      {/* Strategy Name */}
      <div>
        <label className="block text-sm text-gray-300 mb-1">Strategy Name</label>
        <input
          type="text"
          value={strategyName}
          onChange={(e) => setStrategyName(e.target.value)}
          placeholder="Enter strategy name"
          className="w-full max-w-sm px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />
      </div>

      {/* Agent Selector */}
      <div>
        <label className="block text-sm text-gray-300 mb-1">Assign to Agent</label>
        <select
          value={selectedAgent}
          onChange={(e) => setSelectedAgent(e.target.value)}
          className="w-full max-w-sm px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        >
          {agents.map((agent) => (
            <option key={agent} value={agent}>
              {agent}
            </option>
          ))}
        </select>
      </div>

      {/* Rules */}
      {rules.map((rule, index) => (
        <div
          key={index}
          className="flex flex-wrap items-center gap-4 p-4 bg-gray-800 rounded border border-gray-700"
        >
          {/* Indicator */}
          <div>
            <label className="block text-sm text-gray-300">Indicator</label>
            <select
              value={rule.indicator}
              onChange={(e) => updateRule(index, "indicator", e.target.value)}
              className="px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
              disabled={loading}
            >
              {indicators.map((ind) => (
                <option key={ind}>{ind}</option>
              ))}
            </select>
          </div>

          {/* Condition */}
          <div>
            <label className="block text-sm text-gray-300">Condition</label>
            <select
              value={rule.condition}
              onChange={(e) => updateRule(index, "condition", e.target.value)}
              className="px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
              disabled={loading}
            >
              {conditions.map((cond) => (
                <option key={cond}>{cond}</option>
              ))}
            </select>
          </div>

          {/* Value */}
          <div>
            <label className="block text-sm text-gray-300">Value</label>
            <input
              type="number"
              value={rule.value}
              onChange={(e) => updateRule(index, "value", e.target.value)}
              className="w-24 px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
              disabled={loading}
            />
          </div>

          {/* Action */}
          <div>
            <label className="block text-sm text-gray-300">Action</label>
            <select
              value={rule.action}
              onChange={(e) => updateRule(index, "action", e.target.value)}
              className={`px-2 py-1 mt-1 border rounded focus:outline-none ${
                rule.action === "BUY"
                  ? "bg-green-700 text-green-300"
                  : rule.action === "SELL"
                  ? "bg-red-700 text-red-300"
                  : "bg-yellow-700 text-yellow-300"
              }`}
              disabled={loading}
            >
              {actions.map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </div>

          {/* Remove Rule */}
          <button
            onClick={() => removeRule(index)}
            className="ml-auto text-red-500 hover:text-red-700"
            title="Remove rule"
            disabled={loading}
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      ))}

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={addRule}
          className="flex items-center gap-2 bg-blue-700 hover:bg-blue-800 text-white px-4 py-2 rounded"
          disabled={loading}
        >
          <PlusCircle className="w-5 h-5" />
          Add Rule
        </button>
        <button
          onClick={convertToJSON}
          className="flex items-center gap-2 bg-green-700 hover:bg-green-800 text-white px-4 py-2 rounded"
          disabled={loading}
        >
          <Save className="w-5 h-5" />
          {loading ? "Saving..." : "Save Strategy"}
        </button>
      </div>
    </div>
  );
}
