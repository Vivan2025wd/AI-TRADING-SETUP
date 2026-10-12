import React, { Suspense, lazy, useState, useEffect, useCallback, useRef } from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import ErrorBoundary from "./components/ErrorBoundary";

// Lazy load each page/component
const AgentPredictionCard = lazy(() => import("./components/AgentPredictionCard"));
const BacktestResults = lazy(() => import("./components/BacktestResults"));
const BinanceAPISetup = lazy(() => import("./components/BinanceAPISetup"));
const ChartDisplay = lazy(() => import("./components/ChartDisplay"));
const MotherAIDecisionCard = lazy(() => import("./components/MotherAIDecisionCard"));
const StrategyBuilder = lazy(() => import("./components/StrategyBuilder"));
const StrategyPerformance = lazy(() => import("./components/StrategyPerformance"));

const DECISION_REFRESH_INTERVAL = 15 * 60 * 1000; // 15 minutes

const getNavClass = ({ isActive }) =>
  isActive
    ? "text-blue-400 border-b-2 border-blue-400 pb-1"
    : "text-gray-400 hover:text-blue-300 transition";

const SuspenseWrapper = ({ children }) => (
  <Suspense fallback={<div className="text-gray-300 text-center p-6">Loading...</div>}>
    {children}
  </Suspense>
);

const Logo = () => (
  <div className="mr-8">
    <div className="text-white font-bold text-2xl tracking-wide">
      TRADE<span className="text-cyan-400">HIVE</span><span className="text-yellow-400">.AI</span>
    </div>
  </div>
);

export default function App() {
  const [isLive, setIsLive] = useState(false);

  // ===== Mother AI Global State =====
  const [motherAIDecisionData, setMotherAIDecisionData] = useState(null);
  const [motherAILoading, setMotherAILoading] = useState(false);
  const [motherAIError, setMotherAIError] = useState(null);

  // ===== Persistent Timer State (moved from MotherAIDecisionCard) =====
  const [persistedDecision, setPersistedDecision] = useState(null);
  const [decisionTimestamp, setDecisionTimestamp] = useState(null);
  const [timeUntilNextFetch, setTimeUntilNextFetch] = useState(0);
  const [backgroundTimer, setBackgroundTimer] = useState(true);
  const [refreshHistory, setRefreshHistory] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('online');

  const countdownRef = useRef(null);

  // ===== Agent Predictions Global State =====
  const [allAgents, setAllAgents] = useState([]);
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentError, setAgentError] = useState(null);
  const [agentLastUpdate, setAgentLastUpdate] = useState(null);
  const [agentAutoRefresh, setAgentAutoRefresh] = useState(false);
  const [agentTimestamp, setAgentTimestamp] = useState(null);
  const [agentTimeUntilNextFetch, setAgentTimeUntilNextFetch] = useState(0);
  const [agentRefreshPaused, setAgentRefreshPaused] = useState(false);

  const agentIntervalRef = useRef(null);
  const agentFetchControllerRef = useRef(null);

  // Fetch Mother AI decision
  const fetchMotherAIDecision = useCallback(async (isManualRefresh = false) => {
    setMotherAILoading(true);
    setMotherAIError(null);

    try {
      const url = `http://localhost:8000/api/mother-ai/decision?is_live=${isLive}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to fetch Mother AI decision");

      const data = await res.json();

      let final = {};

      if (!data || typeof data !== "object") {
        final = {
          status: "Unknown",
          tradePick: "No signal",
          lastUpdated: null,
          rationale: "Invalid or empty response from Mother AI backend.",
          confidence: 0,
        };
      } else if (!data.decision || Object.keys(data.decision).length === 0) {
        final = {
          status: "Inactive",
          tradePick: "No signal",
          lastUpdated: data.timestamp ? new Date(data.timestamp).toLocaleString() : new Date().toLocaleString(),
          rationale: data.message || "No qualified trades met the confidence threshold.",
          confidence: 0,
        };
      } else {
        const d = data.decision;
        final = {
          status: "Active",
          tradePick: `${d.symbol} - ${d.signal?.toUpperCase() || "N/A"}`,
          lastUpdated: data.timestamp ? new Date(data.timestamp).toLocaleString() : new Date().toLocaleString(),
          rationale: `Confidence: ${(d.confidence * 100).toFixed(2)}%, Win Rate: ${(d.win_rate * 100).toFixed(2)}%, Score: ${d.score}`,
          confidence: (d.confidence * 100).toFixed(2),
        };
      }

      setMotherAIDecisionData(final);
      setPersistedDecision(final);

      // Update refresh history
      if (isManualRefresh) {
        const refreshTime = new Date();
        setRefreshHistory(prev => [
          {
            time: refreshTime,
            success: true,
            status: final.status,
            type: 'manual'
          },
          ...prev.slice(0, 4)
        ]);
      }

      console.log("Decision updated (Global):", final);
    } catch (err) {
      console.error("Mother AI Fetch Error:", err);
      setMotherAIError(err.message || "An unexpected error occurred.");
      
      // Update refresh history on error
      if (isManualRefresh) {
        const refreshTime = new Date();
        setRefreshHistory(prev => [
          {
            time: refreshTime,
            success: false,
            status: 'Error',
            type: 'manual'
          },
          ...prev.slice(0, 4)
        ]);
      }
    } finally {
      setMotherAILoading(false);
    }
  }, [isLive]);

  // Fetch agents data
  const fetchAgentData = useCallback(async (isManualRefresh = false) => {
    // Don't fetch if tab is not visible and it's auto-refresh
    if (!document.hasFocus() && !isManualRefresh && agentAutoRefresh) {
      setAgentRefreshPaused(true);
      return;
    }

    setAgentRefreshPaused(false);
    setAgentLoading(true);
    setAgentError(null);

    // Cancel previous request if exists
    if (agentFetchControllerRef.current) {
      agentFetchControllerRef.current.abort();
    }

    // Create new AbortController for this request
    agentFetchControllerRef.current = new AbortController();
    const signal = agentFetchControllerRef.current.signal;

    try {
      const fetchWithTimeout = (url, options = {}, timeout = 100000) =>
        Promise.race([
          fetch(url, options),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error("Request timed out")), timeout)
          ),
        ]);

      // 1. Fetch all agents
      const resAgents = await fetchWithTimeout("/api/agents", { signal });
      if (!resAgents.ok) {
        throw new Error(`Agents fetch failed (${resAgents.status})`);
      }
      const agentsList = await resAgents.json();

      // Check if request was aborted
      if (signal.aborted) return;

      // 2. Fetch predictions
      const resPred = await fetchWithTimeout(
        "/api/agent/predictions?page=1&limit=100",
        { signal }
      );
      if (!resPred.ok) {
        throw new Error(`Predictions fetch failed (${resPred.status})`);
      }
      const predData = await resPred.json();

      // Check if request was aborted
      if (signal.aborted) return;

      const predictions = Array.isArray(predData.data) ? predData.data : [];

      // 3. Map predictions by normalized agentName
      const predMap = {};
      for (const pred of predictions) {
        if (pred.agentName) {
          const normalizedName = pred.agentName.replace(/\s+Agent$/i, "").toUpperCase();
          predMap[normalizedName] = pred;
        }
      }

      // 4. Merge agents with predictions
      const merged = agentsList.map((agentSymbol) => {
        const key = agentSymbol.toUpperCase();
        const pred = predMap[key];

        const predictionText =
          pred && typeof pred.prediction === "string"
            ? pred.prediction.charAt(0).toUpperCase() +
              pred.prediction.slice(1).toLowerCase()
            : "Hold";

        return {
          agentName: agentSymbol,
          prediction: predictionText,
          confidence:
            pred && typeof pred.confidence === "number"
              ? pred.confidence.toFixed(2)
              : "0.00",
          tradeDetails: pred
            ? pred.tradeDetails || {
                symbol: "N/A",
                entryPrice: 0,
                targetPrice: 0,
                stopLoss: 0,
              }
            : {
                symbol: "N/A",
                entryPrice: 0,
                targetPrice: 0,
                stopLoss: 0,
              },
          timestamp: pred?.timestamp || new Date().toISOString(),
          status: pred ? "active" : "inactive"
        };
      });

      // Only update state if request wasn't aborted
      if (!signal.aborted) {
        setAllAgents(merged);
        setAgentLastUpdate(new Date().toLocaleString());
        console.log("Agent data updated (Global):", merged.length, "agents");
      }
    } catch (err) {
      // Don't show error if request was aborted (normal behavior when tab changes)
      if (!signal.aborted) {
        setAgentError(err.message || "Unknown error occurred");
        console.error("Agent Data Fetch Error:", err);
      }
    } finally {
      if (!signal.aborted) {
        setAgentLoading(false);
      }
      agentFetchControllerRef.current = null;
    }
  }, [agentAutoRefresh]);

  // Manual refresh function for Mother AI (resets timer)
  const handleManualRefresh = useCallback(() => {
    setDecisionTimestamp(new Date());
    setTimeUntilNextFetch(DECISION_REFRESH_INTERVAL / 1000);
    fetchMotherAIDecision(true);
  }, [fetchMotherAIDecision]);

  // Auto-fetch for Mother AI when timer expires
  const handleAutoFetch = useCallback(() => {
    setDecisionTimestamp(new Date());
    setTimeUntilNextFetch(DECISION_REFRESH_INTERVAL / 1000);
    fetchMotherAIDecision(false);
  }, [fetchMotherAIDecision]);

  // Manual refresh function for agents
  const handleAgentManualRefresh = useCallback(() => {
    setAgentTimestamp(new Date());
    setAgentTimeUntilNextFetch(30); // 30 seconds for agents
    fetchAgentData(true);
  }, [fetchAgentData]);

  // Auto-fetch for agents when timer expires
  const handleAgentAutoFetch = useCallback(() => {
    setAgentTimestamp(new Date());
    setAgentTimeUntilNextFetch(30);
    fetchAgentData(false);
  }, [fetchAgentData]);

  // Background countdown timer for Mother AI (persistent across navigation)
  useEffect(() => {
    if (decisionTimestamp && backgroundTimer) {
      const updateCountdown = () => {
        const now = new Date();
        const elapsed = now - decisionTimestamp;
        const remaining = Math.max(0, DECISION_REFRESH_INTERVAL - elapsed);
        setTimeUntilNextFetch(Math.ceil(remaining / 1000));
        
        // Auto-fetch when timer expires
        if (remaining <= 0 && connectionStatus === 'online') {
          handleAutoFetch();
        }
      };

      updateCountdown();
      countdownRef.current = setInterval(updateCountdown, 1000);

      return () => {
        if (countdownRef.current) {
          clearInterval(countdownRef.current);
        }
      };
    }
  }, [decisionTimestamp, backgroundTimer, connectionStatus, handleAutoFetch]);

  // Agent auto-refresh timer (30 seconds)
  useEffect(() => {
    if (agentAutoRefresh && agentTimestamp && !agentRefreshPaused) {
      const updateCountdown = () => {
        const now = new Date();
        const elapsed = (now - agentTimestamp) / 1000;
        const remaining = Math.max(0, 30 - elapsed);
        setAgentTimeUntilNextFetch(Math.ceil(remaining));
        
        // Auto-fetch when timer expires
        if (remaining <= 0 && connectionStatus === 'online' && document.hasFocus()) {
          handleAgentAutoFetch();
        }
      };

      updateCountdown();
      agentIntervalRef.current = setInterval(updateCountdown, 1000);

      return () => {
        if (agentIntervalRef.current) {
          clearInterval(agentIntervalRef.current);
        }
      };
    }
  }, [agentAutoRefresh, agentTimestamp, agentRefreshPaused, connectionStatus, handleAgentAutoFetch]);

  // Handle tab visibility for agent predictions
  useEffect(() => {
    const handleVisibilityChange = () => {
      const isVisible = !document.hidden;
      
      if (!isVisible) {
        // Tab is not visible - pause agent refresh and cancel ongoing requests
        setAgentRefreshPaused(true);
        
        // Cancel any ongoing agent fetch request
        if (agentFetchControllerRef.current) {
          agentFetchControllerRef.current.abort();
          agentFetchControllerRef.current = null;
        }
        
        // Clear the agent auto-refresh interval
        if (agentIntervalRef.current) {
          clearInterval(agentIntervalRef.current);
          agentIntervalRef.current = null;
        }
      } else {
        // Tab is visible again - resume refresh if auto-refresh is enabled
        setAgentRefreshPaused(false);
        
        // If auto-refresh was enabled, restart the timer
        if (agentAutoRefresh && agentTimestamp) {
          setAgentTimestamp(new Date());
          setAgentTimeUntilNextFetch(30);
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      
      // Clean up on unmount
      if (agentIntervalRef.current) {
        clearInterval(agentIntervalRef.current);
      }
      if (agentFetchControllerRef.current) {
        agentFetchControllerRef.current.abort();
      }
    };
  }, [agentAutoRefresh, agentTimestamp]);

  // Monitor connection status
  useEffect(() => {
    const handleOnline = () => setConnectionStatus('online');
    const handleOffline = () => setConnectionStatus('offline');
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    setConnectionStatus(navigator.onLine ? 'online' : 'offline');
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Initial fetch and set timer on first load
  useEffect(() => {
    fetchMotherAIDecision();
    setDecisionTimestamp(new Date());
    setTimeUntilNextFetch(DECISION_REFRESH_INTERVAL / 1000);
  }, [fetchMotherAIDecision]);

  return (
    <BrowserRouter>
      <header className="bg-gray-900 text-white shadow-sm sticky top-0 z-50">
        <nav className="flex items-center px-6 py-4 text-sm font-medium overflow-x-auto">
          <Logo />
          <div className="flex items-center space-x-6 whitespace-nowrap">
            <NavLink to="/binance-api-setup" className={getNavClass}>
              Binance API Setup
            </NavLink>
            <NavLink to="/backtest-results" className={getNavClass}>
              Backtest Results
            </NavLink>
            <NavLink to="/agent-predictions" className={getNavClass}>
              Agent Predictions
            </NavLink>
            <NavLink to="/chart-display" className={getNavClass}>
              Chart Display
            </NavLink>
            <NavLink to="/mother-ai-decision" className={getNavClass}>
              Mother AI Decision
            </NavLink>
            <NavLink to="/strategy-builder" className={getNavClass}>
              Strategy Builder
            </NavLink>
            <NavLink to="/strategy-performance" className={getNavClass}>
              Strategy Performance
            </NavLink>
          </div>
        </nav>
      </header>

      <main className="p-6 bg-gray-950 min-h-screen text-white">
        <ErrorBoundary>
          <Routes>
            <Route
              path="/binance-api-setup"
              element={
                <SuspenseWrapper>
                  <BinanceAPISetup />
                </SuspenseWrapper>
              }
            />
            <Route
              path="/backtest-results"
              element={
                <SuspenseWrapper>
                  <BacktestResults />
                </SuspenseWrapper>
              }
            />
            <Route
              path="/agent-predictions"
              element={
                <SuspenseWrapper>
                  <AgentPredictionCard
                    // Pass agent prediction state from App.jsx
                    allAgents={allAgents}
                    loading={agentLoading}
                    error={agentError}
                    lastUpdate={agentLastUpdate}
                    autoRefresh={agentAutoRefresh}
                    setAutoRefresh={setAgentAutoRefresh}
                    refreshPaused={agentRefreshPaused}
                    timeUntilNextFetch={agentTimeUntilNextFetch}
                    onManualRefresh={handleAgentManualRefresh}
                    connectionStatus={connectionStatus}
                  />
                </SuspenseWrapper>
              }
            />
            <Route
              path="/chart-display"
              element={
                <SuspenseWrapper>
                  <ChartDisplay />
                </SuspenseWrapper>
              }
            />
            <Route
              path="/mother-ai-decision"
              element={
                <SuspenseWrapper>
                  <MotherAIDecisionCard
                    isLive={isLive}
                    decisionData={motherAIDecisionData}
                    loading={motherAILoading}
                    error={motherAIError}
                    refreshDecision={handleManualRefresh}
                    // Pass persistent timer state
                    persistedDecision={persistedDecision}
                    decisionTimestamp={decisionTimestamp}
                    timeUntilNextFetch={timeUntilNextFetch}
                    backgroundTimer={backgroundTimer}
                    setBackgroundTimer={setBackgroundTimer}
                    refreshHistory={refreshHistory}
                    connectionStatus={connectionStatus}
                  />
                </SuspenseWrapper>
              }
            />
            <Route
              path="/strategy-builder"
              element={
                <SuspenseWrapper>
                  <StrategyBuilder />
                </SuspenseWrapper>
              }
            />
            <Route
              path="/strategy-performance"
              element={
                <SuspenseWrapper>
                  <StrategyPerformance />
                </SuspenseWrapper>
              }
            />

            {/* Default route */}
            <Route
              path="/"
              element={
                <div className="text-center text-gray-400 mt-20">
                  <h2 className="text-2xl mb-4">Welcome to TradeHive.AI</h2>
                  <p>Please select a page from the navigation above to get started.</p>
                </div>
              }
            />
          </Routes>
        </ErrorBoundary>
      </main>
    </BrowserRouter>
  );
}