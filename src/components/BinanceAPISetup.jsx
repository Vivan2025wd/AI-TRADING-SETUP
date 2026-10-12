import React, { useState, useEffect, useCallback } from "react";
import { 
  Shield, 
  Key, 
  Eye, 
  EyeOff, 
  Wifi, 
  WifiOff, 
  DollarSign, 
  AlertTriangle, 
  CheckCircle, 
  RefreshCw, 
  Settings, 
  Activity, 
  TrendingUp,
  Clock,
  Zap,
  Lock,
  Unlock,
  Server,
  Globe,
  Copy,
  Check,
  Info,
  ExternalLink
} from "lucide-react";

const BINANCE_DOCS_URL = "https://binance-docs.github.io/apidocs/spot/en/#change-log";

const EnhancedBinanceAPISetup = () => {
  // State Management
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
  const [showSecretKey, setShowSecretKey] = useState(false);
  const [connectionInfo, setConnectionInfo] = useState(null);
  const [permissions, setPermissions] = useState([]);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [copied, setCopied] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Auto-sync connection state on mount
  useEffect(() => {
    fetchConnectionStatus();
  }, []);

  // Auto-refresh balance if enabled
  useEffect(() => {
    if (!autoRefresh || !connected || !showBalances) return;
    
    const interval = setInterval(() => {
      fetchAccountBalance(true);
    }, 30000); // Refresh every 30 seconds
    
    return () => clearInterval(interval);
  }, [autoRefresh, connected, showBalances]);

  const fetchConnectionStatus = useCallback(async () => {
    try {
      const response = await fetch("/api/binance/status");
      const data = await response.json();
      
      if (data.connected) {
        setConnected(true);
        setTradingMode(data.mode === "mock" ? "testnet" : "live");
        setStatusMessage(`✅ Connected in ${data.mode.toUpperCase()} mode`);
        setConnectionInfo(data);
        setPermissions(data.permissions || []);
      } else {
        setConnected(false);
        setConnectionInfo(null);
        setPermissions([]);
      }
    } catch (err) {
      console.error("Failed to fetch connection status", err);
      setError("Failed to check connection status");
    }
  }, []);

  const resetMessages = () => {
    setError(null);
    setStatusMessage("");
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const fetchAccountBalance = async (silent = false) => {
    if (!connected) {
      setError("❗ Please connect first.");
      return;
    }

    if (!silent) {
      setBalanceLoading(true);
      resetMessages();
    }

    try {
      const response = await fetch("/api/binance/account/balance");
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || "Failed to fetch balance");
      }
      
      // Filter out zero balances and sort by value
      const nonZeroBalances = (data.balances || [])
        .filter(balance => parseFloat(balance.free) > 0 || parseFloat(balance.locked) > 0)
        .sort((a, b) => {
          const totalA = parseFloat(a.free) + parseFloat(a.locked);
          const totalB = parseFloat(b.free) + parseFloat(b.locked);
          return totalB - totalA;
        });
      
      setBalances(nonZeroBalances);
      setShowBalances(true);
      setLastUpdated(new Date());
      
      if (!silent) {
        setStatusMessage(`✅ Balance updated! Found ${nonZeroBalances.length} assets with balance.`);
      }
    } catch (err) {
      console.error("Balance fetch error:", err);
      if (!silent) {
        setError(err.message || "❌ Failed to fetch account balance.");
      }
    } finally {
      if (!silent) {
        setBalanceLoading(false);
      }
    }
  };

  const handleConnect = async () => {
    resetMessages();
    setLoading(true);

    if (!apiKey.trim() || !secretKey.trim()) {
      setError("❗ Both API Key and Secret Key are required.");
      setLoading(false);
      return;
    }

    // Validate API key format
    if (apiKey.length < 50) {
      setError("❗ API Key seems too short. Please check your key.");
      setLoading(false);
      return;
    }

    if (tradingMode === "live") {
      const confirmLive = window.confirm(
        "⚠️ LIVE TRADING MODE\n\nThis will connect to the real Binance exchange where actual trades will execute with real money. Are you absolutely sure you want to proceed?"
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
        await fetchConnectionStatus(); // Get updated connection info
      } else {
        throw new Error(data.detail || data.message || "Connection failed");
      }
    } catch (err) {
      console.error("Binance connect error:", err);
      setConnected(false);
      setError(err.message || "❌ Failed to connect to Binance API");
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
        setConnectionInfo(null);
        setPermissions([]);
        setStatusMessage("🔌 Disconnected successfully");
      } else {
        throw new Error("Failed to disconnect");
      }
    } catch (err) {
      console.error("Disconnect error:", err);
      setError("❌ Failed to disconnect");
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
        setStatusMessage("✅ Connection test successful! API is working properly.");
        await fetchConnectionStatus(); // Refresh connection info
      } else {
        throw new Error(data.error || "Connection test failed");
      }
    } catch (err) {
      console.error("Test connection error:", err);
      setError(err.message || "❌ Connection test failed");
    } finally {
      setLoading(false);
    }
  };

  const formatBalance = (value) => {
    const num = parseFloat(value);
    if (num === 0) return "0";
    if (num < 0.000001) return num.toExponential(2);
    if (num < 0.01) return num.toFixed(8);
    if (num < 1) return num.toFixed(6);
    return num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 });
  };

  const getTotalBalanceUSD = () => {
    // This would require price data - simplified for demo
    return balances.reduce((total, balance) => {
      const amount = parseFloat(balance.free) + parseFloat(balance.locked);
      // In real implementation, multiply by current price
      return total + amount;
    }, 0);
  };

  const getPermissionIcon = (permission) => {
    switch (permission.toLowerCase()) {
      case 'spot': return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'futures': return <Activity className="w-4 h-4 text-blue-400" />;
      case 'margin': return <DollarSign className="w-4 h-4 text-purple-400" />;
      default: return <Shield className="w-4 h-4 text-gray-400" />;
    }
  };

  return (
    <div className="p-6 bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white rounded-2xl max-w-5xl mx-auto space-y-6 shadow-2xl border border-gray-700 backdrop-blur-sm">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-3">
          <div className="relative">
            <Shield className="w-8 h-8 text-blue-400" />
            {connected && (
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse" />
            )}
          </div>
          <h2 className="text-3xl font-bold">Binance API Setup</h2>
        </div>
        <p className="text-gray-400">Secure connection to your Binance trading account</p>
        <div className="flex items-center justify-center gap-4 text-sm">
          <a 
            href={BINANCE_DOCS_URL} 
            target="_blank" 
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-blue-400 hover:text-blue-300 transition-colors"
          >
            <Info className="w-4 h-4" />
            API Documentation
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </div>

      {/* Connection Status Bar */}
      <div className={`flex items-center justify-between p-4 rounded-lg border ${
        connected 
          ? 'bg-green-900/30 border-green-700 text-green-100' 
          : 'bg-red-900/30 border-red-700 text-red-100'
      }`}>
        <div className="flex items-center gap-3">
          {connected ? <Wifi className="w-5 h-5" /> : <WifiOff className="w-5 h-5" />}
          <div>
            <div className="font-semibold">
              {connected ? `Connected to ${tradingMode.toUpperCase()}` : 'Disconnected'}
            </div>
            {connectionInfo && (
              <div className="text-xs opacity-80">
                Server Time: {new Date(connectionInfo.serverTime || Date.now()).toLocaleTimeString()}
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {connected && tradingMode === 'live' && (
            <div className="flex items-center gap-1 text-red-400">
              <AlertTriangle className="w-4 h-4" />
              <span className="text-xs font-semibold">LIVE MODE</span>
            </div>
          )}
          <div className={`w-3 h-3 rounded-full ${connected ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
        </div>
      </div>

      {/* Trading Mode Selection */}
      <div className="space-y-3">
        <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <Server className="w-4 h-4" />
          Trading Environment
        </label>
        <div className="grid grid-cols-2 gap-4">
          {[
            { 
              value: "testnet", 
              label: "Testnet (Safe)", 
              icon: <Shield className="w-4 h-4" />,
              color: "green",
              description: "Paper trading with fake money"
            },
            { 
              value: "live", 
              label: "Live Trading", 
              icon: <Globe className="w-4 h-4" />,
              color: "red",
              description: "Real trading with actual funds"
            }
          ].map(({ value, label, icon, color, description }) => (
            <div key={value} className={`relative ${connected ? 'opacity-50 cursor-not-allowed' : ''}`}>
              <input
                type="radio"
                id={value}
                value={value}
                checked={tradingMode === value}
                onChange={(e) => setTradingMode(e.target.value)}
                className="sr-only"
                disabled={connected}
              />
              <label
                htmlFor={value}
                className={`flex items-center p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  tradingMode === value
                    ? color === 'green' 
                      ? 'border-green-500 bg-green-900/30' 
                      : 'border-red-500 bg-red-900/30'
                    : 'border-gray-600 bg-gray-800 hover:border-gray-500'
                } ${connected ? 'cursor-not-allowed' : ''}`}
              >
                <div className="flex items-center gap-3 w-full">
                  <div className={`${color === 'green' ? 'text-green-400' : 'text-red-400'}`}>
                    {icon}
                  </div>
                  <div className="flex-1">
                    <div className="font-semibold">{label}</div>
                    <div className="text-xs text-gray-400">{description}</div>
                  </div>
                  {tradingMode === value && (
                    <CheckCircle className={`w-5 h-5 ${color === 'green' ? 'text-green-400' : 'text-red-400'}`} />
                  )}
                </div>
              </label>
            </div>
          ))}
        </div>
      </div>

      {/* API Credentials */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
            <Key className="w-4 h-4" />
            API Credentials
          </label>
          {connected && (
            <div className="flex items-center gap-1 text-green-400 text-xs">
              <Lock className="w-3 h-3" />
              Secured
            </div>
          )}
        </div>
        
        {/* API Key */}
        <div className="space-y-2">
          <label className="text-sm text-gray-400">API Key</label>
          <div className="relative">
            <input
              className="w-full p-3 pr-12 rounded-lg bg-gray-800 border border-gray-600 placeholder-gray-400 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              placeholder="Enter your Binance API Key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              disabled={connected}
              type="text"
            />
            {apiKey && (
              <button
                onClick={() => copyToClipboard(apiKey)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
              >
                {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
              </button>
            )}
          </div>
        </div>

        {/* Secret Key */}
        <div className="space-y-2">
          <label className="text-sm text-gray-400">Secret Key</label>
          <div className="relative">
            <input
              className="w-full p-3 pr-20 rounded-lg bg-gray-800 border border-gray-600 placeholder-gray-400 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              placeholder="Enter your Binance Secret Key"
              value={secretKey}
              onChange={(e) => setSecretKey(e.target.value)}
              type={showSecretKey ? "text" : "password"}
              disabled={connected}
            />
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 flex items-center gap-2">
              <button
                onClick={() => setShowSecretKey(!showSecretKey)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                {showSecretKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Messages */}
      {error && (
        <div className="bg-red-900/50 border border-red-500 text-red-100 p-4 rounded-lg flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
          <div>
            <div className="font-semibold">Error</div>
            <div className="text-sm mt-1">{error}</div>
          </div>
        </div>
      )}
      
      {statusMessage && (
        <div className="bg-green-900/50 border border-green-500 text-green-100 p-4 rounded-lg flex items-start gap-3">
          <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
          <div>
            <div className="font-semibold">Success</div>
            <div className="text-sm mt-1">{statusMessage}</div>
          </div>
        </div>
      )}

      {/* Live Mode Warning */}
      {tradingMode === "live" && !connected && (
        <div className="bg-yellow-900/50 border border-yellow-500 text-yellow-100 p-4 rounded-lg">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" />
            <div>
              <div className="font-semibold">Live Trading Warning</div>
              <div className="text-sm mt-1">
                You are about to connect to live trading. This will place real orders with actual money. 
                Make sure you understand the risks and have proper risk management in place.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="space-y-3">
        {!connected ? (
          <button
            onClick={handleConnect}
            disabled={loading || !apiKey.trim() || !secretKey.trim()}
            className={`w-full px-6 py-4 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center gap-2 ${
              loading || !apiKey.trim() || !secretKey.trim()
                ? "bg-gray-600 cursor-not-allowed text-gray-400" 
                : tradingMode === 'live'
                  ? "bg-red-600 hover:bg-red-700 active:bg-red-800 text-white"
                  : "bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white"
            }`}
          >
            {loading ? (
              <>
                <RefreshCw className="animate-spin w-5 h-5" />
                Connecting...
              </>
            ) : (
              <>
                <Wifi className="w-5 h-5" />
                Connect to Binance {tradingMode.toUpperCase()}
              </>
            )}
          </button>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            <button
              onClick={handleTestConnection}
              disabled={loading}
              className="px-4 py-3 rounded-lg font-semibold transition-all duration-200 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 flex items-center justify-center gap-2"
            >
              <Zap className="w-4 h-4" />
              {loading ? "Testing..." : "Test"}
            </button>
            <button
              onClick={() => fetchAccountBalance()}
              disabled={balanceLoading}
              className="px-4 py-3 rounded-lg font-semibold transition-all duration-200 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 flex items-center justify-center gap-2"
            >
              {balanceLoading ? (
                <RefreshCw className="animate-spin w-4 h-4" />
              ) : (
                <DollarSign className="w-4 h-4" />
              )}
              Balance
            </button>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="px-4 py-3 rounded-lg font-semibold transition-all duration-200 bg-gray-600 hover:bg-gray-700 flex items-center justify-center gap-2"
            >
              <Settings className="w-4 h-4" />
              Settings
            </button>
            <button
              onClick={handleDisconnect}
              disabled={loading}
              className="px-4 py-3 rounded-lg font-semibold transition-all duration-200 bg-red-600 hover:bg-red-700 flex items-center justify-center gap-2"
            >
              <WifiOff className="w-4 h-4" />
              Disconnect
            </button>
          </div>
        )}
      </div>

      {/* Advanced Settings */}
      {showAdvanced && connected && (
        <div className="bg-gray-800 rounded-lg p-4 space-y-4 border border-gray-600">
          <h3 className="font-semibold text-white flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Advanced Settings
          </h3>
          
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm text-gray-300">Auto-refresh Balance</span>
              <p className="text-xs text-gray-500">Automatically update balance every 30 seconds</p>
            </div>
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                autoRefresh ? 'bg-blue-600' : 'bg-gray-600'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  autoRefresh ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          {/* API Permissions */}
          {permissions.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-2">API Permissions</h4>
              <div className="flex flex-wrap gap-2">
                {permissions.map((permission, index) => (
                  <div key={index} className="flex items-center gap-1 bg-gray-700 px-2 py-1 rounded text-xs">
                    {getPermissionIcon(permission)}
                    {permission.toUpperCase()}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Account Balance Display */}
      {showBalances && (
        <div className="bg-gray-800 rounded-lg p-4 space-y-4 border border-gray-600">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <DollarSign className="w-5 h-5 text-green-400" />
              <h3 className="text-xl font-semibold text-white">Account Balance</h3>
              {autoRefresh && (
                <div className="flex items-center gap-1 text-xs text-blue-400">
                  <RefreshCw className="w-3 h-3 animate-spin" />
                  Auto-refresh
                </div>
              )}
            </div>
            <div className="flex items-center gap-3">
              {lastUpdated && (
                <div className="text-xs text-gray-400 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {lastUpdated.toLocaleTimeString()}
                </div>
              )}
              <button
                onClick={() => fetchAccountBalance()}
                className="text-blue-400 hover:text-blue-300 p-1"
                disabled={balanceLoading}
              >
                <RefreshCw className={`w-4 h-4 ${balanceLoading ? 'animate-spin' : ''}`} />
              </button>
              <button
                onClick={() => setShowBalances(false)}
                className="text-gray-400 hover:text-white text-sm"
              >
                ✕
              </button>
            </div>
          </div>
          
          {balances.length > 0 ? (
            <>
              <div className="grid gap-3 max-h-96 overflow-y-auto pr-2">
                {balances.map((balance, index) => {
                  const freeAmount = parseFloat(balance.free);
                  const lockedAmount = parseFloat(balance.locked);
                  const totalAmount = freeAmount + lockedAmount;
                  
                  return (
                    <div key={index} className="bg-gray-700 rounded-lg p-4 hover:bg-gray-650 transition-colors">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center space-x-3">
                          <div className="w-10 h-10 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-full flex items-center justify-center text-sm font-bold text-black">
                            {balance.asset.substring(0, balance.asset.length > 4 ? 2 : balance.asset.length)}
                          </div>
                          <div>
                            <div className="font-semibold text-white text-lg">{balance.asset}</div>
                            {lockedAmount > 0 && (
                              <div className="text-xs text-yellow-400 flex items-center gap-1">
                                <Lock className="w-3 h-3" />
                                {formatBalance(lockedAmount)} locked
                              </div>
                            )}
                          </div>
                        </div>
                        
                        <div className="text-right">
                          <div className="font-mono text-white text-lg font-semibold">
                            {formatBalance(freeAmount)}
                          </div>
                          <div className="text-xs text-gray-400">Available</div>
                          {totalAmount !== freeAmount && (
                            <div className="text-xs text-gray-500">
                              Total: {formatBalance(totalAmount)}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
              
              <div className="text-center text-sm text-gray-400 pt-3 border-t border-gray-600">
                Showing {balances.length} assets • Non-zero balances only
              </div>
            </>
          ) : (
            <div className="text-center py-8">
              <div className="text-4xl mb-3">💰</div>
              <h3 className="text-lg font-semibold text-gray-300 mb-2">No Balance Found</h3>
              <p className="text-gray-400 text-sm">Your account appears to have zero balance in all assets.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default EnhancedBinanceAPISetup;