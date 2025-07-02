import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BinanceAPISetup = () => {
  const [apiKey, setApiKey] = useState('');
  const [secretKey, setSecretKey] = useState('');
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const existing = localStorage.getItem('binance_api');
    if (existing) {
      const { apiKey, secretKey } = JSON.parse(existing);
      setApiKey(apiKey);
      setSecretKey(secretKey);
      setConnected(true);
    }
  }, []);

  const handleConnect = async () => {
    try {
      const payload = { apiKey, secretKey };
      await axios.post('/api/binance/connect', payload);
      localStorage.setItem('binance_api', JSON.stringify(payload));
      setConnected(true);
    } catch (err) {
      alert('Connection failed: ' + err.response?.data?.detail || err.message);
    }
  };

  return (
    <div className="p-6 bg-gray-800 text-white rounded-lg max-w-md mx-auto">
      <h2 className="text-xl font-bold mb-4">Connect Binance API</h2>

      <input
        className="w-full p-2 mb-2 rounded bg-gray-700"
        placeholder="API Key"
        value={apiKey}
        onChange={e => setApiKey(e.target.value)}
      />
      <input
        className="w-full p-2 mb-4 rounded bg-gray-700"
        placeholder="Secret Key"
        value={secretKey}
        onChange={e => setSecretKey(e.target.value)}
        type="password"
      />

      <button
        onClick={handleConnect}
        className="bg-green-600 px-4 py-2 rounded hover:bg-green-700"
      >
        {connected ? '✅ Connected' : 'Connect'}
      </button>
    </div>
  );
};

export default BinanceAPISetup;
