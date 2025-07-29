import React, { useState, useEffect } from "react";
import { PlusCircle, Save, Trash2 } from "lucide-react";

const INDICATORS = ["rsi", "macd", "sma", "ema"];
const ACTIONS = ["BUY", "SELL"];

const CONDITIONS = {
  rsi: { 
    BUY: ["buy_below", "buy_equals"], 
    SELL: ["sell_above", "sell_equals"] 
  },
  ema: { 
    BUY: [
      "price_crosses_above", 
      "price_above", 
      "ema_crosses_above_ema",
      "ema_above_value"
    ], 
    SELL: [
      "price_crosses_below", 
      "price_below", 
      "ema_crosses_below_ema",
      "ema_below_value"
    ] 
  },
  sma: { 
    BUY: [
      "price_crosses_above", 
      "price_above", 
      "sma_crosses_above_sma",
      "sma_above_value"
    ], 
    SELL: [
      "price_crosses_below", 
      "price_below", 
      "sma_crosses_below_sma",
      "sma_below_value"
    ] 
  },
  macd: { 
    BUY: [
      "macd_crosses_above_signal", 
      "macd_above_zero", 
      "histogram_positive",
      "macd_above_value"
    ], 
    SELL: [
      "macd_crosses_below_signal", 
      "macd_below_zero", 
      "histogram_negative",
      "macd_below_value"
    ] 
  },
};

// Conditions that use boolean values instead of numbers
const BOOLEAN_CONDITIONS = [
  "price_crosses_above", "price_crosses_below", "price_above", "price_below",
  "ema_crosses_above_ema", "ema_crosses_below_ema", 
  "sma_crosses_above_sma", "sma_crosses_below_sma",
  "macd_crosses_above_signal", "macd_crosses_below_signal",
  "macd_above_zero", "macd_below_zero", "histogram_positive", "histogram_negative"
];

// Conditions that need a second period/indicator for comparison
const COMPARISON_CONDITIONS = [
  "ema_crosses_above_ema", "ema_crosses_below_ema",
  "sma_crosses_above_sma", "sma_crosses_below_sma"
];

// Default periods for different indicators
const DEFAULT_PERIODS = {
  rsi: { period: 14 },
  sma: { period: 20 },
  ema: { period: 20 },
  macd: { fast_period: 12, slow_period: 26, signal_period: 9 }
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
      // MACD specific fields
      fast_period: 12,
      slow_period: 26,
      signal_period: 9,
      // Comparison period for MA crossovers
      compare_period: 50
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
    setRules((prev) => [
      ...prev,
      { 
        indicator: "rsi", 
        period: 14, 
        condition: "buy_below", 
        value: 30, 
        action: "BUY",
        fast_period: 12,
        slow_period: 26,
        signal_period: 9,
        compare_period: 50
      },
    ]);
  };

  const updateRule = (index, field, value) => {
    setRules((prev) => {
      const newRules = [...prev];
      const currentRule = { ...newRules[index], [field]: value };

      if (field === "indicator") {
        // Set default periods based on indicator
        const defaults = DEFAULT_PERIODS[value.toLowerCase()] || {};
        Object.assign(currentRule, defaults);
        
        // Reset condition to first available for new indicator/action combo
        const conds = CONDITIONS[value.toLowerCase()]?.[currentRule.action] || [];
        currentRule.condition = conds[0] || "";
        
        // Set default compare_period for moving averages
        if (value.toLowerCase() === 'sma' || value.toLowerCase() === 'ema') {
          currentRule.compare_period = currentRule.period === 20 ? 50 : 20;
        }
      }

      if (field === "action") {
        const conds = CONDITIONS[currentRule.indicator.toLowerCase()]?.[value] || [];
        if (!conds.includes(currentRule.condition)) {
          currentRule.condition = conds[0] || "";
        }
      }

      // Auto-set boolean value for boolean conditions
      if (field === "condition" && BOOLEAN_CONDITIONS.includes(value)) {
        currentRule.value = true;
      }

      newRules[index] = currentRule;
      return newRules;
    });
  };

  const removeRule = (index) => {
    setRules((prev) => prev.filter((_, i) => i !== index));
  };

  const convertRulesToIndicators = () => {
    const indicatorsObj = {};
    rules.forEach((rule) => {
      const { indicator, condition, value } = rule;
      const key = indicator.toLowerCase();
      
      if (!indicatorsObj[key]) indicatorsObj[key] = {};

      // Handle MACD with multiple periods
      if (key === 'macd') {
        indicatorsObj[key]["fast_period"] = parseInt(rule.fast_period) || 12;
        indicatorsObj[key]["slow_period"] = parseInt(rule.slow_period) || 26;
        indicatorsObj[key]["signal_period"] = parseInt(rule.signal_period) || 9;
      } else {
        indicatorsObj[key]["period"] = parseInt(rule.period) || 14;
      }

      // Add compare_period for comparison conditions
      if (COMPARISON_CONDITIONS.includes(condition)) {
        indicatorsObj[key]["compare_period"] = parseInt(rule.compare_period) || 50;
      }

      // Handle different value types
      if (typeof value === "boolean") {
        indicatorsObj[key][condition] = value;
      } else if (!isNaN(value)) {
        indicatorsObj[key][condition] = parseFloat(value);
      } else if (value === "true" || value === "false") {
        indicatorsObj[key][condition] = value === "true";
      } else {
        indicatorsObj[key][condition] = value;
      }
    });
    return indicatorsObj;
  };

  const convertToJSON = async () => {
    if (!strategyName.trim()) return alert("Please enter a strategy name.");
    if (!selectedAgent) return alert("Please select an agent.");
    if (rules.length === 0) return alert("Add at least one rule.");

    const payload = {
      strategy_id: strategyName.trim(),
      symbol: selectedAgent,
      strategy_json: convertRulesToIndicators(),
    };

    try {
      setLoading(true);
      setError(null);
      const res = await fetch("/api/strategy/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Failed to save");

      alert(`✅ Saved strategy "${strategyName}"`);
      setStrategyName("");
      setRules([{ 
        indicator: "rsi", 
        period: 14, 
        condition: "buy_below", 
        value: 30, 
        action: "BUY",
        fast_period: 12,
        slow_period: 26,
        signal_period: 9,
        compare_period: 50
      }]);
    } catch (err) {
      setError("Error saving strategy: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const currentStrategyJSON = {
    symbol: selectedAgent,
    indicators: convertRulesToIndicators(),
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-900 rounded-xl shadow-lg space-y-6 border border-gray-700 text-white">
      <h2 className="text-2xl font-bold flex items-center gap-2">Strategy Builder</h2>

      {error && <div className="bg-red-800 text-red-100 p-3 rounded border border-red-500">⚠️ {error}</div>}

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

      <div>
        <label className="block text-sm text-gray-300 mb-1">Assign to Agent</label>
        <select
          value={selectedAgent}
          onChange={(e) => setSelectedAgent(e.target.value)}
          className="w-full max-w-sm px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
          disabled={loading}
        >
          {agents.map((agent) => (
            <option key={agent} value={agent}>{agent}</option>
          ))}
        </select>
      </div>

      {rules.map((rule, index) => {
        const condOptions = CONDITIONS[rule.indicator.toLowerCase()]?.[rule.action] || [];
        const isBooleanCondition = BOOLEAN_CONDITIONS.includes(rule.condition);
        const isComparisonCondition = COMPARISON_CONDITIONS.includes(rule.condition);
        const isMACD = rule.indicator.toLowerCase() === 'macd';
        const isMovingAverage = ['sma', 'ema'].includes(rule.indicator.toLowerCase());

        return (
          <div key={index} className="p-4 bg-gray-800 rounded border border-gray-700">
            <div className="flex flex-wrap items-center gap-4 mb-4">
              <div>
                <label className="block text-sm text-gray-300">Indicator</label>
                <select
                  value={rule.indicator}
                  onChange={(e) => updateRule(index, "indicator", e.target.value)}
                  className="px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
                  disabled={loading}
                >
                  {INDICATORS.map((ind) => (
                    <option key={ind} value={ind}>{ind.toUpperCase()}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm text-gray-300">Action</label>
                <select
                  value={rule.action}
                  onChange={(e) => updateRule(index, "action", e.target.value)}
                  className={`px-2 py-1 mt-1 border rounded ${rule.action === "BUY" ? "bg-green-700 text-green-300" : "bg-red-700 text-red-300"}`}
                  disabled={loading}
                >
                  {ACTIONS.map((a) => (
                    <option key={a} value={a}>{a}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm text-gray-300">Condition</label>
                <select
                  value={rule.condition}
                  onChange={(e) => updateRule(index, "condition", e.target.value)}
                  className="px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white min-w-48"
                  disabled={loading}
                >
                  {condOptions.length ? condOptions.map((cond) => (
                    <option key={cond} value={cond}>{cond.replace(/_/g, " ").toUpperCase()}</option>
                  )) : <option>No Conditions</option>}
                </select>
              </div>

              <div>
                <label className="block text-sm text-gray-300">Value</label>
                {isBooleanCondition ? (
                  <select
                    value={rule.value === true ? "true" : "false"}
                    onChange={(e) => updateRule(index, "value", e.target.value === "true")}
                    className="px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
                    disabled={loading}
                  >
                    <option value="true">True</option>
                    <option value="false">False</option>
                  </select>
                ) : (
                  <input
                    type="number"
                    step="0.01"
                    value={rule.value}
                    onChange={(e) => updateRule(index, "value", parseFloat(e.target.value))}
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

            {/* Period configuration section */}
            <div className="pt-2 border-t border-gray-700">
              {isMACD ? (
                // MACD-specific period inputs
                <div className="flex gap-4">
                  <div>
                    <label className="block text-sm text-gray-300">Fast Period</label>
                    <input
                      type="number"
                      min={1}
                      value={rule.fast_period}
                      onChange={(e) => updateRule(index, "fast_period", e.target.value)}
                      className="w-20 px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
                      disabled={loading}
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300">Slow Period</label>
                    <input
                      type="number"
                      min={1}
                      value={rule.slow_period}
                      onChange={(e) => updateRule(index, "slow_period", e.target.value)}
                      className="w-20 px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
                      disabled={loading}
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300">Signal Period</label>
                    <input
                      type="number"
                      min={1}
                      value={rule.signal_period}
                      onChange={(e) => updateRule(index, "signal_period", e.target.value)}
                      className="w-20 px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
                      disabled={loading}
                    />
                  </div>
                </div>
              ) : (
                // Standard period input for other indicators
                <div className="flex gap-4">
                  <div>
                    <label className="block text-sm text-gray-300">
                      {isMovingAverage ? `${rule.indicator.toUpperCase()} Period` : "Period"}
                    </label>
                    <input
                      type="number"
                      min={1}
                      value={rule.period}
                      onChange={(e) => updateRule(index, "period", e.target.value)}
                      className="w-20 px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
                      disabled={loading}
                    />
                  </div>
                  
                  {/* Comparison period for moving average crossovers */}
                  {isComparisonCondition && (
                    <div>
                      <label className="block text-sm text-gray-300">
                        Compare {rule.indicator.toUpperCase()} Period
                      </label>
                      <input
                        type="number"
                        min={1}
                        value={rule.compare_period}
                        onChange={(e) => updateRule(index, "compare_period", e.target.value)}
                        className="w-20 px-2 py-1 mt-1 bg-gray-700 border border-gray-600 rounded text-white"
                        disabled={loading}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Condition explanation */}
            <div className="mt-2 text-xs text-gray-400">
              {rule.condition === "price_crosses_above" && "Price crosses above the moving average"}
              {rule.condition === "price_crosses_below" && "Price crosses below the moving average"}
              {rule.condition === "price_above" && "Price is above the moving average"}
              {rule.condition === "price_below" && "Price is below the moving average"}
              {rule.condition.includes("_crosses_above_") && `${rule.indicator.toUpperCase()}(${rule.period}) crosses above ${rule.indicator.toUpperCase()}(${rule.compare_period})`}
              {rule.condition.includes("_crosses_below_") && `${rule.indicator.toUpperCase()}(${rule.period}) crosses below ${rule.indicator.toUpperCase()}(${rule.compare_period})`}
              {rule.condition === "macd_crosses_above_signal" && "MACD line crosses above signal line"}
              {rule.condition === "macd_crosses_below_signal" && "MACD line crosses below signal line"}
              {rule.condition === "histogram_positive" && "MACD histogram is positive"}
              {rule.condition === "histogram_negative" && "MACD histogram is negative"}
            </div>
          </div>
        );
      })}

      <div className="flex gap-4">
        <button
          onClick={addRule}
          className={`flex items-center gap-2 px-4 py-2 rounded ${loading ? "bg-blue-500 cursor-not-allowed" : "bg-blue-700 hover:bg-blue-800"}`}
          disabled={loading}
        >
          <PlusCircle className="w-5 h-5" />
          Add Rule
        </button>
        <button
          onClick={convertToJSON}
          className={`flex items-center gap-2 px-4 py-2 rounded ${loading ? "bg-green-500 cursor-not-allowed" : "bg-green-700 hover:bg-green-800"}`}
          disabled={loading}
        >
          <Save className="w-5 h-5" />
          {loading ? "Saving..." : "Save Strategy"}
        </button>
      </div>

      <div className="mt-8 bg-gray-800 p-4 rounded border border-gray-700 whitespace-pre-wrap font-mono text-sm text-green-400 overflow-x-auto max-h-72">
        <h3 className="text-lg font-semibold mb-2">Generated Strategy JSON</h3>
        <pre>{JSON.stringify(currentStrategyJSON, null, 2)}</pre>
      </div>
    </div>
  );
}