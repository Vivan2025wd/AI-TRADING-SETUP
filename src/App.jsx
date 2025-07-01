// src/App.jsx
import React from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import StrategyBuilder from "./components/StrategyBuilder";
import BacktestResults from "./components/BacktestResults";
import StrategyPerformance from "./components/StrategyPerformance"; // <-- Added import

export default function App() {
  return (
    <BrowserRouter>
      <header className="bg-gray-900 text-white shadow-sm sticky top-0 z-50">
        <nav className="flex items-center space-x-6 px-6 py-4 text-sm font-medium">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              isActive
                ? "text-blue-400 border-b-2 border-blue-400 pb-1"
                : "text-gray-400 hover:text-blue-300 transition"
            }
          >
            Dashboard
          </NavLink>
          <NavLink
            to="/strategy-builder"
            className={({ isActive }) =>
              isActive
                ? "text-blue-400 border-b-2 border-blue-400 pb-1"
                : "text-gray-400 hover:text-blue-300 transition"
            }
          >
            Strategy Builder
          </NavLink>
          <NavLink
            to="/backtest-results"
            className={({ isActive }) =>
              isActive
                ? "text-blue-400 border-b-2 border-blue-400 pb-1"
                : "text-gray-400 hover:text-blue-300 transition"
            }
          >
            Backtest Results
          </NavLink>
          <NavLink
            to="/strategy-performance"
            className={({ isActive }) =>
              isActive
                ? "text-blue-400 border-b-2 border-blue-400 pb-1"
                : "text-gray-400 hover:text-blue-300 transition"
            }
          >
            Strategy Performance
          </NavLink>
        </nav>
      </header>

      <main className="p-6 bg-gray-950 min-h-screen text-white">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/strategy-builder" element={<StrategyBuilder />} />
          <Route path="/backtest-results" element={<BacktestResults />} />
          <Route path="/strategy-performance" element={<StrategyPerformance />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
