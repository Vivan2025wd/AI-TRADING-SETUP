import React, { Suspense, lazy, useState, useEffect, useCallback } from "react";
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

  const fetchMotherAIDecision = useCallback(async () => {
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
      console.log("Decision updated (Global):", final);
    } catch (err) {
      console.error("Mother AI Fetch Error:", err);
      setMotherAIError(err.message || "An unexpected error occurred.");
    } finally {
      setMotherAILoading(false);
    }
  }, [isLive]);

  useEffect(() => {
    fetchMotherAIDecision();
    const interval = setInterval(fetchMotherAIDecision, 10 * 60 * 1000); // 10 min polling
    return () => clearInterval(interval);
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
                  <AgentPredictionCard />
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
                    refreshDecision={fetchMotherAIDecision}
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
