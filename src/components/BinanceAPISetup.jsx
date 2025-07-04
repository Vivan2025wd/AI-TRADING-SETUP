import React, { useState, useEffect } from "react";
import axios from "axios";

const BinanceAPISetup = () => {
  const [apiKey, setApiKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");

  useEffect(() => {
    try {
      const existing = localStorage.getItem("binance_api");
      if (existing) {
        const parsed = JSON.parse(existing);
        if (parsed.apiKey && parsed.secretKey) {
          setApiKey(parsed.apiKey);
          setSecretKey(parsed.secretKey);
          setConnected(true);
          setStatusMessage("Previously saved keys loaded.");
        }
      }
    } catch (err) {
      console.error("Failed to load API keys from localStorage", err);
      setError("Failed to read saved keys.");
    }
  }, []);

  const handleConnect = async () => {
    setError(null);
    setStatusMessage("");

    if (!apiKey.trim() || !secretKey.trim()) {
      setError("❗ API Key and Secret Key are required.");
      return;
    }

    try {
      const payload = { apiKey, secretKey };

      // Test connection to backend
      const response = await axios.post("/api/binance/connect", payload);

      if (response.status === 200) {
        localStorage.setItem("binance_api", JSON.stringify(payload));
        setConnected(true);
        setStatusMessage("✅ Connected successfully and keys saved.");
      } else {
        throw new Error("Unexpected response from server.");
      }
    } catch (err) {
      console.error("Binance connect error:", err);
      setConnected(false);
      setError(
        err.response?.data?.detail ||
          err.message ||
          "❌ Failed to connect to Binance."
      );
    }
  };

  return (
    <div className="p-6 bg-gray-800 text-white rounded-lg max-w-md mx-auto space-y-4 shadow-md">
      <h2 className="text-2xl font-bold text-white">🔐 Connect Binance API</h2>

      <input
        className="w-full p-2 rounded bg-gray-700 placeholder-gray-400 text-white"
        placeholder="API Key"
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
      />
      <input
        className="w-full p-2 rounded bg-gray-700 placeholder-gray-400 text-white"
        placeholder="Secret Key"
        value={secretKey}
        onChange={(e) => setSecretKey(e.target.value)}
        type="password"
      />

      {error && (
        <div className="bg-red-900 text-red-300 p-3 rounded-md text-sm">
          {error}
        </div>
      )}

      {statusMessage && (
        <div className="bg-green-900 text-green-300 p-3 rounded-md text-sm">
          {statusMessage}
        </div>
      )}

      <button
        onClick={handleConnect}
        className={`w-full px-4 py-2 mt-2 rounded font-semibold transition duration-200 ${
          connected
            ? "bg-green-600 hover:bg-green-700"
            : "bg-blue-600 hover:bg-blue-700"
        }`}
      >
        {connected ? "✅ Connected" : "🔌 Connect"}
      </button>
    </div>
  );
};

export default BinanceAPISetup;
