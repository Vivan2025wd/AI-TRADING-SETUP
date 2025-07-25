import React, { useState, useEffect } from "react";
import axios from "axios";

const BinanceAPISetup = () => {
  const [apiKey, setApiKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [tradingMode, setTradingMode] = useState("testnet"); // "testnet" or "live"

  // Auto-sync connection state on mount
  useEffect(() => {
    const fetchConnectionStatus = async () => {
      try {
        const response = await axios.get("/api/binance/status");
        if (response.data.connected) {
          setConnected(true);
          setTradingMode(response.data.mode === "mock" ? "testnet" : "live");
          setStatusMessage(`✅ Already connected in ${response.data.mode.toUpperCase()} MODE.`);
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

      const response = await axios.post("/api/binance/connect", payload);

      if (response.status === 200 && response.data.message) {
        setConnected(true);
        setStatusMessage(`✅ ${response.data.message}`);
      } else {
        throw new Error(response.data.message || "Unexpected server response.");
      }
    } catch (err) {
      console.error("Binance connect error:", err);
      setConnected(false);
      setError(
        err.response?.data?.detail ||
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
      await axios.post("/api/binance/disconnect");
      setConnected(false);
      setApiKey("");
      setSecretKey("");
      setStatusMessage("🔌 Disconnected successfully.");
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
      const response = await axios.get("/api/binance/test-connection");
      if (response.data.success) {
        setStatusMessage("✅ Account info retrieved successfully!");
      } else {
        throw new Error(response.data.error || "Failed to test connection.");
      }
    } catch (err) {
      console.error("Test connection error:", err);
      setError(err.message || "❌ Failed to retrieve account info.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-gray-900 text-white rounded-xl max-w-lg mx-auto space-y-6 shadow-2xl border border-gray-700">
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
          <div className="flex space-x-3">
            <button
              onClick={handleTestConnection}
              disabled={loading}
              className="flex-1 px-4 py-3 rounded-lg font-semibold transition duration-200 bg-green-600 hover:bg-green-700 disabled:bg-gray-600"
            >
              {loading ? "Testing..." : "🧪 Test Connection"}
            </button>
            <button
              onClick={handleDisconnect}
              disabled={loading}
              className="flex-1 px-4 py-3 rounded-lg font-semibold transition duration-200 bg-red-600 hover:bg-red-700"
            >
              🔌 Disconnect
            </button>
          </div>
        )}
      </div>

      {/* Connection Status */}
      <div className="flex items-center justify-center space-x-2 pt-2">
        <div className={`w-3 h-3 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`}></div>
        <span className="text-sm text-gray-400">{connected ? "Connected" : "Disconnected"}</span>
      </div>
    </div>
  );
};

export default BinanceAPISetup;
