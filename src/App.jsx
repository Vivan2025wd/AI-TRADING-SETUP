import React, { Suspense, lazy, useState } from "react";
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

// Simple Suspense wrapper with fallback
const SuspenseWrapper = ({ children }) => (
  <Suspense fallback={<div className="text-gray-300 text-center p-6">Loading...</div>}>
    {children}
  </Suspense>
);

export default function App() {
  const [isLive, setIsLive] = useState(false);

  return (
    <BrowserRouter>
      <header className="bg-gray-900 text-white shadow-sm sticky top-0 z-50">
        <nav className="flex items-center space-x-6 px-6 py-4 text-sm font-medium overflow-x-auto whitespace-nowrap">
          <NavLink to="/agent-predictions" className={getNavClass}>
            Binance API Setup
          </NavLink>
          <NavLink to="/backtest-results" className={getNavClass}>
            Backtest Results
          </NavLink>
          <NavLink to="/binance-api-setup" className={getNavClass}>
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
          <div className="flex items-center space-x-2">
            <span>Mock Trading</span>
            <label className="switch">
              <input type="checkbox" checked={isLive} onChange={() => setIsLive(!isLive)} />
              <span className="slider round"></span>
            </label>
            <span>Live Trading</span>
          </div>
        </nav>
      </header>

      <main className="p-6 bg-gray-950 min-h-screen text-white">
        <ErrorBoundary>
          <Routes>
            <Route
              path="/agent-predictions"
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
              path="/binance-api-setup"
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
                  <MotherAIDecisionCard isLive={isLive} />
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

            {/* Optional: Redirect root or fallback 
            <Route
              path="/"
              element={
                <div className="text-center text-gray-400 mt-20">
                  <h2>Welcome! Please select a page from the navigation.</h2>
                </div>
              }
            />*/}
          </Routes>
        </ErrorBoundary>
      </main>
    </BrowserRouter>
  );
}
