import React, { useState, useEffect } from "react";

const BinanceAPISetup = () => {
  const [apiKey, setApiKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [tradingMode, setTradingMode] = useState("testnet");
  const [balances, setBalances] = useState([]);
  const [balanceLoading, setBalanceLoading] = useState(false);
  const [showBalances, setShowBalances] = useState(false);

  // Auto-sync connection state on mount
  useEffect(() => {
    const fetchConnectionStatus = async () => {
      try {
        const response = await fetch("/api/binance/status");
        const data = await response.json();
        if (data.connected) {
          setConnected(true);
          setTradingMode(data.mode === "mock" ? "testnet" : "live");
          setStatusMessage(`✅ Already connected in ${data.mode.toUpperCase()} MODE.`);
        } else {
          setConnected(false);
        }
      } catch (err) {
        console.error("Failed to fetch connection status", err);
      }
    };

    fetchConnectionStatus();
  }, []);

  const resetMessages = () => {
    setError(null);
    setStatusMessage("");
  };

  const fetchAccountBalance = async () => {
    if (!connected) {
      setError("❗ Please connect first.");
      return;
    }

    setBalanceLoading(true);
    resetMessages();

    try {
      const response = await fetch("/api/binance/account/balance");
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || "Failed to fetch balance");
      }
      
      setBalances(data.balances || []);
      setShowBalances(true);
      setStatusMessage(`✅ Balance retrieved! Found ${data.balances?.length || 0} assets.`);
    } catch (err) {
      console.error("Balance fetch error:", err);
      setError(
        err.message ||
        "❌ Failed to fetch account balance."
      );
    } finally {
      setBalanceLoading(false);
    }
  };

  const handleConnect = async () => {
    resetMessages();
    setLoading(true);

    if (!apiKey.trim() || !secretKey.trim()) {
      setError("❗ API Key and Secret Key are required.");
      setLoading(false);
      return;
    }

    if (tradingMode === "live") {
      const confirmLive = window.confirm(
        "⚠️ You are about to connect in LIVE MODE. Real trades will execute. Proceed?"
      );
      if (!confirmLive) {
        setLoading(false);
        return;
      }
    }

    try {
      const payload = {
        apiKey: apiKey.trim(),
        secretKey: secretKey.trim(),
        tradingMode: tradingMode === "testnet" ? "mock" : "live",
      };

      const response = await fetch("/api/binance/connect", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      
      const data = await response.json();

      if (response.ok && data.message) {
        setConnected(true);
        setStatusMessage(`✅ ${data.message}`);
      } else {
        throw new Error(data.detail || data.message || "Unexpected server response.");
      }
    } catch (err) {
      console.error("Binance connect error:", err);
      setConnected(false);
      setError(
        err.message ||
          "❌ Failed to connect to Binance."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDisconnect = async () => {
    resetMessages();
    setLoading(true);
    try {
      const response = await fetch("/api/binance/disconnect", {
        method: "POST",
      });
      
      if (response.ok) {
        setConnected(false);
        setApiKey("");
        setSecretKey("");
        setBalances([]);
        setShowBalances(false);
        setStatusMessage("🔌 Disconnected successfully.");
      } else {
        throw new Error("Failed to disconnect");
      }
    } catch (err) {
      console.error("Disconnect error:", err);
      setError("❌ Failed to disconnect.");
    } finally {
      setLoading(false);
    }
  };

  const handleTestConnection = async () => {
    if (!connected) {
      setError("❗ Please connect first.");
      return;
    }

    setLoading(true);
    resetMessages();

    try {
      const response = await fetch("/api/binance/test-connection");
      const data = await response.json();
      
      if (data.success) {
        setStatusMessage("✅ Account info retrieved successfully!");
      } else {
        throw new Error(data.error || "Failed to test connection.");
      }
    } catch (err) {
      console.error("Test connection error:", err);
      setError(err.message || "❌ Failed to retrieve account info.");
    } finally {
      setLoading(false);
    }
  };

  const formatBalance = (value) => {
    const num = parseFloat(value);
    if (num === 0) return "0";
    if (num < 0.001) return num.toExponential(2);
    if (num < 1) return num.toFixed(6);
    return num.toFixed(4);
  };

  return (
    <div className="p-6 bg-gray-900 text-white rounded-xl max-w-4xl mx-auto space-y-6 shadow-2xl border border-gray-700">
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">🔐 Binance API Setup</h2>
        <p className="text-gray-400 text-sm">Connect your Binance account for automated trading</p>
      </div>

      {/* Trading Mode */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-300">Trading Mode</label>
        <div className="flex space-x-4">
          {["testnet", "live"].map((mode) => (
            <label key={mode} className="flex items-center">
              <input
                type="radio"
                value={mode}
                checked={tradingMode === mode}
                onChange={(e) => setTradingMode(e.target.value)}
                className="mr-2"
                disabled={connected}
              />
              <span className={mode === "live" ? "text-red-400" : "text-green-400"}>
                {mode === "live" ? "⚠️ Live Trading" : "🧪 Testnet (Safe)"}
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* API Key Inputs */}
      {[
        { label: "API Key", value: apiKey, setter: setApiKey },
        { label: "Secret Key", value: secretKey, setter: setSecretKey, type: "password" },
      ].map(({ label, value, setter, type = "text" }) => (
        <div className="space-y-2" key={label}>
          <label className="block text-sm font-medium text-gray-300">{label}</label>
          <input
            className="w-full p-3 rounded-lg bg-gray-800 border border-gray-600 placeholder-gray-400 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder={`Enter your Binance ${label}`}
            value={value}
            onChange={(e) => setter(e.target.value)}
            type={type}
            disabled={connected}
          />
        </div>
      ))}

      {/* Messages */}
      {error && (
        <div className="bg-red-900/50 border border-red-500 text-red-300 p-3 rounded-lg text-sm flex items-center">
          <span className="mr-2">❌</span>{error}
        </div>
      )}
      {statusMessage && (
        <div className="bg-green-900/50 border border-green-500 text-green-300 p-3 rounded-lg text-sm flex items-center">
          <span className="mr-2">✅</span>{statusMessage}
        </div>
      )}

      {/* Live Mode Warning */}
      {tradingMode === "live" && !connected && (
        <div className="bg-yellow-900/50 border border-yellow-500 text-yellow-300 p-3 rounded-lg text-sm">
          <strong>⚠️ Warning:</strong> Live trading will place real orders. Understand the risks.
        </div>
      )}

      {/* Action Buttons */}
      <div className="space-y-3">
        {!connected ? (
          <button
            onClick={handleConnect}
            disabled={loading}
            className={`w-full px-4 py-3 rounded-lg font-semibold transition duration-200 flex items-center justify-center ${
              loading ? "bg-gray-600 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700 active:bg-blue-800"
            }`}
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Connecting...
              </>
            ) : (
              <>🔌 Connect to Binance</>
            )}
          </button>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <button
              onClick={handleTestConnection}
              disabled={loading}
              className="px-4 py-3 rounded-lg font-semibold transition duration-200 bg-green-600 hover:bg-green-700 disabled:bg-gray-600"
            >
              {loading ? "Testing..." : "🧪 Test Connection"}
            </button>
            <button
              onClick={fetchAccountBalance}
              disabled={balanceLoading}
              className="px-4 py-3 rounded-lg font-semibold transition duration-200 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 flex items-center justify-center"
            >
              {balanceLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Loading...
                </>
              ) : (
                <>💰 View Balance</>
              )}
            </button>
            <button
              onClick={handleDisconnect}
              disabled={loading}
              className="px-4 py-3 rounded-lg font-semibold transition duration-200 bg-red-600 hover:bg-red-700"
            >
              🔌 Disconnect
            </button>
          </div>
        )}
      </div>

      {/* Account Balance Display */}
      {showBalances && balances.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-xl font-semibold text-gray-200">💰 Account Balance</h3>
            <button
              onClick={() => setShowBalances(false)}
              className="text-gray-400 hover:text-white text-sm"
            >
              ✕ Hide
            </button>
          </div>
          
          <div className="grid gap-3 max-h-96 overflow-y-auto">
            {balances.map((balance, index) => {
              const freeAmount = parseFloat(balance.free);
              const lockedAmount = parseFloat(balance.locked);
              const totalAmount = freeAmount + lockedAmount;
              
              return (
                <div key={index} className="bg-gray-700 rounded-lg p-3 flex justify-between items-center">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-yellow-500 rounded-full flex items-center justify-center text-xs font-bold text-black">
                      {balance.asset.substring(0, 2)}
                    </div>
                    <div>
                      <div className="font-semibold text-white">{balance.asset}</div>
                      <div className="text-xs text-gray-400">
                        {lockedAmount > 0 && `${formatBalance(lockedAmount)} locked`}
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="font-mono text-white">{formatBalance(freeAmount)}</div>
                    <div className="text-xs text-gray-400">Available</div>
                  </div>
                </div>
              );
            })}
          </div>
          
          <div className="text-center text-sm text-gray-400 pt-2 border-t border-gray-600">
            Showing {balances.length} assets with non-zero balance
          </div>
        </div>
      )}

      {/* Empty Balance State */}
      {showBalances && balances.length === 0 && (
        <div className="bg-gray-800 rounded-lg p-6 text-center">
          <div className="text-4xl mb-3">🪙</div>
          <h3 className="text-lg font-semibold text-gray-300 mb-2">No Balance Found</h3>
          <p className="text-gray-400 text-sm">Your account appears to have zero balance in all assets.</p>
        </div>
      )}

      {/* Connection Status */}
      <div className="flex items-center justify-center space-x-2 pt-2">
        <div className={`w-3 h-3 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`}></div>
        <span className="text-sm text-gray-400">{connected ? "Connected" : "Disconnected"}</span>
      </div>
    </div>
  );
};

export default BinanceAPISetup;