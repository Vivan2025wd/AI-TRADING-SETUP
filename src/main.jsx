import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import StrategyBuilder from "./components/StrategyBuilder";
import BacktestResults from "./components/BacktestResults";

import "./styles/index.css";

function App() {
  return (
    <BrowserRouter>
      <nav className="bg-white shadow px-6 py-4 flex space-x-6">
        <Link to="/" className="text-blue-600 hover:underline">Dashboard</Link>
        <Link to="/strategy-builder" className="text-blue-600 hover:underline">Strategy Builder</Link>
        <Link to="/backtest-results" className="text-blue-600 hover:underline">Backtest Results</Link>
      </nav>

      <main className="p-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/strategy-builder" element={<StrategyBuilder />} />
          <Route path="/backtest-results" element={<BacktestResults />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
