import React, { useState, useEffect } from "react";
import { PlusCircle, Save, Trash2 } from "lucide-react";

const INDICATORS = ["rsi", "macd", "sma", "ema"];

const ACTIONS = ["BUY", "SELL"];

const CONDITIONS = {
  rsi: {
    BUY: ["buy_below", "buy_equals"],
    SELL: ["sell_above", "sell_equals"],
  },
  ema: {
    BUY: ["buy_crosses_above", "buy_equals"],
    SELL: ["sell_crosses_below", "sell_equals"],
  },
  sma: {
    BUY: ["buy_crosses_above", "buy_equals"],
    SELL: ["sell_crosses_below", "sell_equals"],
  },
  macd: {
    BUY: ["buy_below", "buy_equals"],
    SELL: ["sell_above", "sell_equals"],
  },
};

export default function StrategyBuilder() {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState("");
  const [strategyName, setStrategyName] = useState("");
  const [rules, setRules] = useState([
    {
      indicator: "rsi",
      period: 14,
      condition: "buy_below",
      value: 30,
      action: "BUY",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const res = await fetch("/api/agents");
        if (!res.ok) throw new Error("Failed to fetch agents");
        const data = await res.json();
        setAgents(data);
        setSelectedAgent(data[0] || "");
      } catch (err) {
        console.error(err);
        setError("Error loading agents. Using fallback list.");
        const fallback = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
        setAgents(fallback);
        setSelectedAgent(fallback[0]);
      }
    };

    fetchAgents();
  }, []);

  const addRule = () => {
    setRules([
      ...rules,
      {
        indicator: "rsi",
        period: 14,
        condition: "buy_below",
        value: 30,
        action: "BUY",
      },
    ]);
  };

  const updateRule = (index, field, value) => {
    setRules((prev) => {
      const newRules = [...prev];
      newRules[index] = { ...newRules[index], [field]: value };
      // Reset condition if indicator or action changed and condition is invalid now
      if (field === "indicator" || field === "action") {
        const conds = CONDITIONS[
          newRules[index].indicator.toLowerCase()
        ]?.[newRules[index].action];
        if (!conds || !conds.includes(newRules[index].condition)) {
          newRules[index].condition = conds ? conds[0] : "";
        }
      }
      // Reset period to default if indicator changed and no period specified
      if (field === "indicator" && !newRules[index].period) {
        newRules[index].period = 14;
      }
      return newRules;
    });
  };

  const removeRule = (index) => {
    setRules((prev) => prev.filter((_, i) => i !== index));
  };

  const convertRulesToIndicators = () => {
    const indicatorsObj = {};
    rules.forEach(({ indicator, period, condition, value }) => {
      const key = indicator.toLowerCase();
      if (!indicatorsObj[key]) indicatorsObj[key] = {};
      indicatorsObj[key]["period"] = parseInt(period) || 14;

      if (typeof value === "string" && (value === "true" || value === "false")) {
        indicatorsObj[key][condition] = value === "true";
      } else if (typeof value === "number" || !isNaN(parseFloat(value))) {
        indicatorsObj[key][condition] = parseFloat(value);
      } else {
        indicatorsObj[key][condition] = value;
      }
    });
    return indicatorsObj;
  };

  const convertToJSON = async () => {
    if (!strategyName.trim()) {
      alert("Please enter a strategy name.");
      return;
    }
    if (!selectedAgent) {
      alert("Please select an agent.");
      return;
    }
    if (rules.length === 0) {
      alert("Add at least one rule.");
      return;
    }

    const payload = {
      strategy_id: strategyName.trim(),
      symbol: selectedAgent,
      strategy_json: convertRulesToIndicators(),
    };

    try {
      setError(null);
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
        {
          indicator: "rsi",
          period: 14,
          condition: "buy_below",
          value: 30,
          action: "BUY",
        },
      ]);
    } catch (err) {
      setError("Error saving strategy: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Generate current JSON preview of strategy
  const currentStrategyJSON = {
    symbol: selectedAgent,
    indicators: convertRulesToIndicators(),
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-gray-900 rounded-xl shadow-lg space-y-6 border border-gray-700 text-white">
      <h2 className="text-2xl font-bold flex items-center gap-2">Strategy Builder</h2>

      {error && (
        <div className="bg-red-800 text-red-100 p-3 rounded border border-red-500">
          ⚠️ {error}
        </div>
      )}

      {/* Strategy Name */}
      <div>
        <label className="block text-sm text-gray-300 mb-1">Strategy Name</label>
        <input
          type="text"
          value={strategyName}
          onChange={(e) => setStrategyName(e.target.value)}
          placeholder="Enter strategy name"
          className="w-full max-w-sm px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
          disabled={loading}
        />
      </div>

      {/* Agent Selector */}
      <div>
        <label className="block text-sm text-gray-300 mb-1">Assign to Agent</label>
        <select
          value={selectedAgent}
          onChange={(e) => setSelectedAgent(e.target.value)}
          className="w-full max-w-sm px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
          disabled={loading}
        >
          {agents.map((agent) => (
            <option key={agent} value={agent}>
              {agent}
            </option>
          ))}
        </select>
      </div>

      {/* Rule Builder */}
      {rules.map((rule, index) => {
        const condOptions =
          CONDITIONS[rule.indicator.toLowerCase()]?.[rule.action] || [];

        return (
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
                {INDICATORS.map((ind) => (
                  <option key={ind} value={ind}>
                    {ind.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            {/* Period */}
            <div>
              <label className="block text-sm text-gray-300">Period</label>
              <input
                type="number"
                min={1}
                value={rule.period}
                onChange={(e) => updateRule(index, "period", e.target.value)}
                className="w-20 px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
                disabled={loading}
              />
            </div>

            {/* Action */}
            <div>
              <label className="block text-sm text-gray-300">Action</label>
              <select
                value={rule.action}
                onChange={(e) => updateRule(index, "action", e.target.value)}
                className={`px-2 py-1 mt-1 border rounded ${
                  rule.action === "BUY"
                    ? "bg-green-700 text-green-300"
                    : "bg-red-700 text-red-300"
                }`}
                disabled={loading}
              >
                {ACTIONS.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
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
                {condOptions.map((cond) => (
                  <option key={cond} value={cond}>
                    {cond.replace(/_/g, " ").toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            {/* Value */}
            <div>
              <label className="block text-sm text-gray-300">Value</label>
              {rule.condition.includes("crosses") ? (
                <select
                  value={rule.value === true ? "true" : "false"}
                  onChange={(e) =>
                    updateRule(index, "value", e.target.value === "true")
                  }
                  className="px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
                  disabled={loading}
                >
                  <option value="true">True</option>
                  <option value="false">False</option>
                </select>
              ) : (
                <input
                  type="number"
                  value={rule.value}
                  onChange={(e) => updateRule(index, "value", e.target.value)}
                  className="w-20 px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
                  disabled={loading}
                />
              )}
            </div>

            <button
              onClick={() => removeRule(index)}
              className="ml-auto text-red-500 hover:text-red-700"
              title="Remove rule"
              disabled={loading}
            >
              <Trash2 className="w-5 h-5" />
            </button>
          </div>
        );
      })}

      {/* Actions */}
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

      {/* JSON Preview */}
      <div className="mt-8 bg-gray-800 p-4 rounded border border-gray-700 whitespace-pre-wrap font-mono text-sm text-green-400 overflow-x-auto max-h-72">
        <h3 className="text-lg font-semibold mb-2">Generated Strategy JSON</h3>
        <pre>{JSON.stringify(currentStrategyJSON, null, 2)}</pre>
      </div>
    </div>
  );
}
